"""Rekonstrukcja scen i ocena jakości modelu.

Najważniejsza poprawka względem wersji notatnikowej dotyczy funkcji
wycinającej odzyskaną scenę z odpowiedzi modelu. Poprzednia wersja
używała wyrażenia

    restored = answer[last_sep + 1 : stop_position - 1]

które systematycznie obcinało ostatni obiekt sceny. Zaniżało to
wszystkie metryki podobieństwa i zawyżało liczbę obiektów
brakujących o jeden w każdej ocenianej scenie. Wyniki policzone
przed tą poprawką nie nadają się do zamieszczenia w pracy.
"""

from __future__ import annotations

import random
from collections import defaultdict

import numpy as np
import torch

from .data import SceneDataset, extract_restored_scene, make_prompt
from .metrics import ALL_METRICS, TARGET_MEASURE_KEYS, target_measure

__all__ = [
    "extract_restored_scene",
    "reconstruct",
    "evaluate_model",
    "evaluate_by_prompt_length",
    "format_summary",
]


@torch.no_grad()
def reconstruct(
    model,
    prompts: list[list[int]],
    device: torch.device,
    max_new_tokens: int | None = None,
    temperature: float = 1.0,
    top_k: int = 0,
    greedy: bool = False,
) -> list[list[int] | None]:
    """Rekonstruuje sceny dla listy fragmentów wejściowych.

    Fragmenty o jednakowej długości przetwarzane są wsadowo, dzięki
    czemu ocena całego zbioru jest o rząd wielkości szybsza niż
    generowanie scena po scenie.
    """
    block_size = model.config.block_size
    results: list[list[int] | None] = [None] * len(prompts)

    grouped: dict[int, list[int]] = defaultdict(list)
    for index, prompt in enumerate(prompts):
        grouped[len(prompt)].append(index)

    for length, indices in grouped.items():
        limit = block_size - length - 1
        if max_new_tokens is not None:
            limit = min(limit, max_new_tokens)
        if limit <= 0:
            continue

        batch = torch.tensor(
            [prompts[i] for i in indices], dtype=torch.long, device=device
        )
        answers, _ = model.generate(
            batch,
            max_new_tokens=limit,
            temperature=temperature,
            top_k=top_k,
            greedy=greedy,
        )
        for row, index in enumerate(indices):
            results[index] = extract_restored_scene(
                answers[row].tolist(), prompt_length=length
            )

    return results


@torch.no_grad()
def evaluate_model(
    model,
    dataset: SceneDataset,
    device: torch.device,
    num_scenes: int = 200,
    prompt_length: int = 6,
    seed: int = 3,
    temperature: float = 1.0,
    top_k: int = 0,
    greedy: bool = False,
    batch_size: int = 128,
) -> dict[str, float]:
    """Ocena modelu na próbie scen.

    Zwraca łącznie: metryki podobieństwa użyte w pracy jako nagrody
    oraz niezależną miarę docelową. Rozdzielenie obu grup jest
    konieczne, ponieważ model dostrajany daną metryką wygrywa w niej
    z definicji, więc porównanie modeli musi opierać się na mierze
    trzeciej, nieużywanej w treningu żadnego z nich.
    """
    rng = random.Random(seed)
    scene_ids = list(range(dataset.num_scenes))
    rng.shuffle(scene_ids)
    scene_ids = scene_ids[:num_scenes]

    accumulator: dict[str, list[float]] = defaultdict(list)
    failures = 0

    for start in range(0, len(scene_ids), batch_size):
        chunk = scene_ids[start:start + batch_size]
        references = [dataset.scenes[i] for i in chunk]
        prompts = [
            make_prompt(scene, prompt_length, rng) for scene in references
        ]
        restored = reconstruct(
            model,
            prompts,
            device,
            temperature=temperature,
            top_k=top_k,
            greedy=greedy,
        )

        for reference, candidate in zip(references, restored):
            if candidate is None:
                failures += 1
                candidate = []
            for name, function in ALL_METRICS.items():
                accumulator[name].append(function(reference, candidate))
            for key, value in target_measure(reference, candidate).items():
                accumulator[key].append(value)

    summary = {
        name: float(np.mean(values)) for name, values in accumulator.items()
    }
    summary["failure_rate"] = failures / max(len(scene_ids), 1)
    summary["num_scenes"] = float(len(scene_ids))
    summary["prompt_length"] = float(prompt_length)
    return summary


@torch.no_grad()
def evaluate_by_prompt_length(
    model,
    dataset: SceneDataset,
    device: torch.device,
    lengths: tuple[int, ...] = (3, 4, 5, 6, 7, 8),
    num_scenes: int = 200,
    seed: int = 3,
    greedy: bool = False,
) -> dict[int, dict[str, float]]:
    """Ocena w funkcji długości fragmentu wejściowego.

    Odpowiada na pytanie badawcze o wpływ długości kontekstu na
    skuteczność rekonstrukcji. Fragment jednoelementowy jest
    niejednoznaczny w 99,3 procent przypadków, dwuelementowy
    w 4,3 procent, a od trzech elementów jednoznacznie identyfikuje
    scenę, więc oczekiwany jest wyraźny próg w okolicy trzech
    obiektów.
    """
    return {
        length: evaluate_model(
            model,
            dataset,
            device,
            num_scenes=num_scenes,
            prompt_length=length,
            seed=seed,
            greedy=greedy,
        )
        for length in lengths
    }


def format_summary(summary: dict[str, float]) -> str:
    """Zwięzły zapis wyników do wypisania w konsoli."""
    return (
        f"lev {summary['levenshtein']:.4f} | "
        f"bleu {summary['bleu']:.4f} | "
        f"rouge {summary['rouge_l']:.4f} | "
        f"F1 tresci {summary['content_f1']:.4f} | "
        f"tau {summary['order_tau']:.4f} | "
        f"dokladne {summary['exact_match']:.3f} | "
        f"halucynacje {summary['hallucinated']:.2f} | "
        f"braki {summary['missing']:.2f} | "
        f"bez STOP {summary['failure_rate']:.3f}"
    )


TABLE_COLUMNS = ("levenshtein", "bleu", "rouge_l") + TARGET_MEASURE_KEYS
