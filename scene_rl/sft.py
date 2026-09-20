"""Etap pierwszy: uczenie nadzorowane (supervised fine-tuning).

Model uczy się przewidywać kolejny token łańcucha rekonstrukcji
przy pomocy entropii krzyżowej. Etap ten stanowi punkt odniesienia
dla eksperymentu oraz politykę początkową dla algorytmu PPO.

Poprawki względem wersji notatnikowej:

1. Tempo uczenia jest faktycznie ustawiane w optymalizatorze.
   Poprzednio zmienna `learning_rate` była nadpisywana w komórce
   pętli treningowej i zapisywana do historii, ale optymalizator
   pozostawał przy wartości podanej przy jego utworzeniu, więc
   raportowane tempo uczenia nie odpowiadało rzeczywistemu.
2. Dane nie są generowane w całości przed każdą epoką. Poprzednio
   jedna epoka wymagała zbudowania około trzydziestu milionów
   tokenów, co zajmowało kilka minut samego przygotowania danych.
   Tutaj partie losowane są na bieżąco, więc świeżość przykładów
   jest zachowana bez kosztu pamięciowego.
3. Metryki podobieństwa liczone są równolegle z funkcją straty
   i zapisywane w tym samym pliku historii.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from .config import ExperimentConfig
from .data import (
    AugmentationConfig,
    ExampleSampler,
    SceneDataset,
    required_block_size,
)
from .evaluation import evaluate_model, format_summary
from .model import SceneTransformer, cosine_learning_rate
from .utils import (
    HistoryLogger,
    count_parameters_table,
    describe_device,
    resolve_device,
    save_checkpoint,
    set_seed,
)


def build_optimizer(
    model: torch.nn.Module,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """AdamW z regularyzacją nakładaną tylko na macierze wag.

    Parametry jednowymiarowe, czyli obciążenia oraz współczynniki
    normalizacji warstwowej, wyłączone są z regularyzacji, ponieważ
    ich zmniejszanie nie ogranicza pojemności modelu, a zaburza
    statystyki normalizacji.
    """
    decay, no_decay = [], []
    for _, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.dim() >= 2:
            decay.append(parameter)
        else:
            no_decay.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=(0.9, 0.95),
    )


@torch.no_grad()
def estimate_loss(
    model: SceneTransformer,
    samplers: dict[str, ExampleSampler],
    batch_size: int,
    num_batches: int,
    device: torch.device,
) -> dict[str, float]:
    """Średnia strata na zbiorze uczącym i walidacyjnym."""
    was_training = model.training
    model.eval()
    output: dict[str, float] = {}
    for split, sampler in samplers.items():
        losses = []
        for _ in range(num_batches):
            x, y = sampler.batch(batch_size)
            x = torch.from_numpy(x).to(device)
            y = torch.from_numpy(y).to(device)
            _, loss = model(x, y)
            losses.append(loss.item())
        output[f"{split}_loss"] = float(np.mean(losses))
        output[f"{split}_perplexity"] = float(np.exp(np.mean(losses)))
    if was_training:
        model.train()
    return output


def train_sft(config: ExperimentConfig) -> str:
    """Przeprowadza etap uczenia nadzorowanego. Zwraca ścieżkę best.pt."""
    set_seed(config.seed)
    device = resolve_device(config.device)
    os.makedirs(config.output_dir, exist_ok=True)

    dataset = SceneDataset(config.data.path)
    stats = dataset.statistics()
    config.model.vocab_size = dataset.vocab_size

    augmentation = AugmentationConfig(
        min_elements=config.data.min_elements,
        max_remove_per_step=config.data.max_remove_per_step,
        swap_probability=config.data.swap_probability,
    )

    needed = required_block_size(dataset, augmentation, num_draws=500)
    if config.model.block_size < needed:
        raise ValueError(
            f"block_size = {config.model.block_size} jest za małe. "
            f"Cały proces rekonstrukcji jednej sceny zajmuje do "
            f"{needed - 1} tokenów, więc block_size musi wynosić co "
            f"najmniej {needed}."
        )

    print(f"Urządzenie: {describe_device(device)}")
    print(f"Zbiór danych: {config.data.path}")
    print(f"  scen: {stats['num_scenes']}, "
          f"obiektów w scenie: {stats['scene_length']}, "
          f"unikalnych obiektów: {stats['num_unique_objects']}")
    print(f"  słownik: {stats['vocab_size']} tokenów "
          f"(w tym 3 specjalne: STOP, SEP, PAD)")
    print(f"  minimalny block_size: {needed}, "
          f"użyty: {config.model.block_size}")

    train_sampler = ExampleSampler(
        dataset,
        config.model.block_size,
        augmentation,
        seed=config.train_data_seed,
        loss_on_prompt=config.data.loss_on_prompt,
        start_mode=config.data.start_mode,
    )
    val_sampler = ExampleSampler(
        dataset,
        config.model.block_size,
        augmentation,
        seed=config.val_data_seed,
        loss_on_prompt=config.data.loss_on_prompt,
        start_mode=config.data.start_mode,
    )
    samplers = {"train": train_sampler, "val": val_sampler}

    model = SceneTransformer(config.model).to(device)
    print(f"\nLiczba parametrów: {model.num_parameters():,}")
    print(count_parameters_table(model))
    print(f"\nStrata modelu losowego: ln({config.model.vocab_size}) = "
          f"{np.log(config.model.vocab_size):.4f}\n")

    optimizer = build_optimizer(
        model, config.train.learning_rate, config.train.weight_decay
    )
    logger = HistoryLogger(config.output_dir, "history_sft")
    config.to_json(os.path.join(config.output_dir, "config.json"))

    total_steps = config.train.num_epochs * config.train.steps_per_epoch
    best_val = float("inf")
    global_step = 0

    for epoch in range(config.train.num_epochs):
        model.train()
        epoch_losses = []

        for _ in range(config.train.steps_per_epoch):
            learning_rate = cosine_learning_rate(
                global_step,
                config.train.warmup_steps,
                total_steps,
                config.train.learning_rate,
                config.train.min_learning_rate,
            )
            # Tempo uczenia trafia do optymalizatora, a nie tylko
            # do zmiennej lokalnej. To była przyczyna rozbieżności
            # między raportowanym a rzeczywistym tempem uczenia.
            for group in optimizer.param_groups:
                group["lr"] = learning_rate

            x, y = train_sampler.batch(config.train.batch_size)
            x = torch.from_numpy(x).to(device)
            y = torch.from_numpy(y).to(device)

            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.train.grad_clip
            )
            optimizer.step()

            epoch_losses.append(loss.item())
            global_step += 1

        record = {
            "epoch": epoch,
            "step": global_step,
            "learning_rate": learning_rate,
            "batch_size": config.train.batch_size,
            "grad_norm": float(grad_norm),
            "running_train_loss": float(np.mean(epoch_losses)),
        }

        if epoch % config.train.eval_every == 0:
            record.update(
                estimate_loss(
                    model,
                    samplers,
                    config.train.batch_size,
                    config.train.eval_batches,
                    device,
                )
            )

        if epoch % config.train.metrics_every == 0:
            summary = evaluate_model(
                model,
                dataset,
                device,
                num_scenes=config.train.metrics_samples,
                prompt_length=config.train.prompt_length,
                seed=config.eval_seed,
            )
            record.update({f"eval_{k}": v for k, v in summary.items()})
            print(f"Epoka {epoch:4d} | strata {record.get('val_loss', 0):.4f} "
                  f"| {format_summary(summary)}")
        else:
            print(f"Epoka {epoch:4d} | strata treningowa "
                  f"{record['running_train_loss']:.4f} | walidacyjna "
                  f"{record.get('val_loss', float('nan')):.4f} | "
                  f"lr {learning_rate:.2e}")

        logger.log(record)
        logger.flush_csv()

        save_checkpoint(
            os.path.join(config.output_dir, "last.pt"),
            model, optimizer, config, epoch, logger.records,
        )
        save_checkpoint(
            os.path.join(config.output_dir, f"epoch_{epoch:04d}.pt"),
            model, optimizer, config, epoch, logger.records,
        )
        current_val = record.get("val_loss")
        if current_val is not None and current_val < best_val:
            best_val = current_val
            save_checkpoint(
                os.path.join(config.output_dir, "best.pt"),
                model, optimizer, config, epoch, logger.records,
                extra={"best_val_loss": best_val},
            )

    if train_sampler.truncated_examples:
        print(f"UWAGA: {train_sampler.truncated_examples} przykładów "
              f"nie zmieściło się w oknie kontekstowym.")

    print(f"\nNajlepsza strata walidacyjna: {best_val:.4f}")
    return os.path.join(config.output_dir, "best.pt")
