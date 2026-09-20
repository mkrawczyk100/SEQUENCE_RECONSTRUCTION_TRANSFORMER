"""Narzędzia wspólne: powtarzalność, checkpointy, logowanie."""

from __future__ import annotations

import csv
import json
import os
import random
import time

import numpy as np
import torch

from .config import ExperimentConfig


def set_seed(seed: int) -> None:
    """Ustala ziarna wszystkich używanych generatorów losowych.

    Poprzednia wersja kodu nie ustalała żadnego ziarna, przez co
    dwa uruchomienia tego samego eksperymentu dawały różne wyniki
    i nie dało się zweryfikować raportowanych liczb.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str = "auto") -> torch.device:
    """Wybiera urządzenie obliczeniowe."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        memory_gb = properties.total_memory / (1024 ** 3)
        return (
            f"{properties.name}, pamięć {memory_gb:.1f} GB, "
            f"CUDA {torch.version.cuda}, PyTorch {torch.__version__}"
        )
    return f"CPU, PyTorch {torch.__version__}"


class HistoryLogger:
    """Zapisuje historię treningu w formacie JSONL oraz CSV.

    Poprzednia wersja zapisywała historię jako tekstową
    reprezentację tensorów, w postaci `tensor(0.4253, device='cuda:0')`,
    co wymagało parsowania przed narysowaniem jakiegokolwiek wykresu.
    Tutaj zapisywane są zwykłe liczby zmiennoprzecinkowe.
    """

    def __init__(self, output_dir: str, name: str = "history"):
        os.makedirs(output_dir, exist_ok=True)
        self.jsonl_path = os.path.join(output_dir, f"{name}.jsonl")
        self.csv_path = os.path.join(output_dir, f"{name}.csv")
        self.records: list[dict] = []
        self.start_time = time.time()

    def log(self, record: dict) -> None:
        record = dict(record)
        record["wall_time_s"] = round(time.time() - self.start_time, 3)
        self.records.append(record)
        with open(self.jsonl_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def flush_csv(self) -> None:
        if not self.records:
            return
        columns: list[str] = []
        for record in self.records:
            for key in record:
                if key not in columns:
                    columns.append(key)
        with open(self.csv_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(self.records)


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    config: ExperimentConfig,
    epoch: int,
    history: list[dict],
    extra: dict | None = None,
) -> None:
    """Zapisuje pełny stan eksperymentu w jednym pliku.

    Checkpoint zawiera konfigurację w całości, dzięki czemu dowolną
    nową metrykę można policzyć bez powtarzania treningu, a przebieg
    da się odtworzyć bez zgadywania użytych hiperparametrów.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict() if optimizer is not None else None
        ),
        "config": config.to_dict(),
        "history": history,
        "extra": extra or {},
        "torch_version": torch.__version__,
    }
    torch.save(payload, path)


def load_checkpoint(path: str, device: torch.device) -> dict:
    """Wczytuje checkpoint zapisany funkcją `save_checkpoint`."""
    return torch.load(path, map_location=device, weights_only=False)


def build_model_from_checkpoint(
    path: str,
    device: torch.device,
    with_value_head: bool = False,
):
    """Odtwarza model i konfigurację bezpośrednio z checkpointu."""
    from .config import ExperimentConfig as _Config
    from .model import SceneTransformer

    payload = load_checkpoint(path, device)
    config = _Config.from_dict(payload["config"])
    model = SceneTransformer(config.model, with_value_head=with_value_head)
    missing, unexpected = model.load_state_dict(
        payload["model_state_dict"], strict=False
    )
    # Głowica wartości występuje wyłącznie w checkpointach zapisanych
    # w trakcie PPO. Może więc brakować w state_dict (checkpoint z
    # samego SFT, ładowany z with_value_head=True) albo odwrotnie,
    # występować w state_dict, gdy model budowany jest bez niej
    # (checkpoint z PPO, ładowany domyślnie z with_value_head=False,
    # tak jak robią to polecenia evaluate i compare). Obie sytuacje są
    # oczekiwane i nie powinny przerywać wczytywania.
    value_head_keys = {"value_head.weight", "value_head.bias"}
    if set(unexpected) - value_head_keys:
        raise RuntimeError(f"Nieoczekiwane wagi w checkpoincie: {unexpected}")
    if set(missing) - value_head_keys:
        raise RuntimeError(f"Brakujące wagi w checkpoincie: {missing}")
    model.to(device)
    return model, config, payload


def disable_dropout(model: torch.nn.Module) -> int:
    """Wyłącza dropout, ustawiając prawdopodobieństwo zerowania na 0.

    W algorytmie PPO iloraz prawdopodobieństw polityki nowej
    i starej musi wynosić dokładnie jeden przed pierwszą
    aktualizacją. Aktywny dropout losowo zeruje część aktywacji,
    więc to samo przejście w przód dwukrotnie daje różne
    log-prawdopodobieństwa, co zaburza próbkowanie ważone leżące
    u podstaw metody. Z tego powodu regularyzację tę wyłącza się
    na czas dostrajania.
    """
    count = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
            count += 1
    return count


def count_parameters_table(model: torch.nn.Module) -> str:
    """Zestawienie liczby parametrów w podziale na komponenty."""
    groups: dict[str, int] = {}
    for name, parameter in model.named_parameters():
        top = name.split(".")[0]
        groups[top] = groups.get(top, 0) + parameter.numel()
    total = sum(groups.values())
    lines = [f"{'komponent':<28}{'parametry':>12}{'udział':>9}"]
    for name, count in sorted(groups.items(), key=lambda kv: -kv[1]):
        lines.append(f"{name:<28}{count:>12,}{100 * count / total:>8.1f}%")
    lines.append(f"{'RAZEM':<28}{total:>12,}")
    return "\n".join(lines)
