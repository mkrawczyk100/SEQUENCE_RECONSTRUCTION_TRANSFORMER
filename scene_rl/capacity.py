"""Pomiar pojemności modelu i memoryzacji wg Morris i in. (2025).

Moduł implementuje metodykę z pracy "How much do language models
memorize?" (arXiv:2505.24832), zastosowaną do zadania rekonstrukcji
sceny.

Podstawowe pojęcia z tej pracy:

  memoryzacja niezamierzona (unintended memorization)
      informacja, jaką model przechowuje o konkretnym zbiorze danych,
      liczona jako różnica między długością kodu danych bez modelu
      a długością kodu danych przy dostępie do modelu,

  generalizacja (intended memorization)
      informacja, jaką model przechowuje o procesie generującym dane,

  pojemność (capacity)
      całkowita liczba bitów, jaką model jest w stanie zapamiętać;
      autorzy szacują ją empirycznie na około 3,6 bita na parametr
      dla modeli rodziny GPT trenowanych w połowicznej precyzji.

Kluczowa uwaga dotycząca niniejszego zadania. Morris i in. mierzą
pojemność właśnie na danych losowych (rozdział 3 ich pracy), ponieważ
przy danych pozbawionych struktury generalizacja jest niemożliwa
i cała nauczona informacja jest memoryzacją. Zbiór scen użyty w tej
pracy ma dokładnie tę własność, co wykazano w rozdziale 3.3: nie
istnieje reguła współwystępowania obiektów, którą można by uogólnić.
Wynika stąd, że memoryzacji nie da się w tym zadaniu wyeliminować,
ponieważ jest ona jedynym dostępnym mechanizmem rozwiązania zadania.
Celem doboru rozmiaru modelu nie jest zatem zerowa memoryzacja, lecz
doprowadzenie stosunku pojemności modelu do zawartości informacyjnej
zbioru w okolice jedności, czyli praca modelu na granicy pojemności
zamiast wielokrotnie powyżej niej.
"""

from __future__ import annotations

import math

# Empiryczne oszacowanie z pracy Morris i in. (2025), Tabela 1.
# Wartość 3,6 dotyczy precyzji bfloat16, 3,83 precyzji float32.
BITS_PER_PARAMETER_BF16 = 3.51
BITS_PER_PARAMETER_FP32 = 3.83
BITS_PER_PARAMETER_HEADLINE = 3.6


def count_parameters_analytic(
    vocab_size: int,
    n_embd: int,
    block_size: int,
    n_layer: int,
) -> dict[str, int]:
    """Analityczna liczba parametrów, w rozbiciu na komponenty.

    Wzór wyprowadzony z definicji modelu w module `model`:
      - tablica embeddingów tokenów:      V * E
      - tablica embeddingów pozycji:      B * E
      - blok transformera:                12 * E^2 + 10 * E
      - końcowa normalizacja warstwowa:   2 * E
      - głowica językowa:                 E * V + V
    """
    token_embedding = vocab_size * n_embd
    position_embedding = block_size * n_embd
    per_block = 12 * n_embd * n_embd + 10 * n_embd
    blocks = n_layer * per_block
    final_norm = 2 * n_embd
    lm_head = n_embd * vocab_size + vocab_size

    total = (
        token_embedding + position_embedding + blocks + final_norm + lm_head
    )
    embeddings = token_embedding + lm_head

    return {
        "token_embedding": token_embedding,
        "position_embedding": position_embedding,
        "blocks": blocks,
        "final_norm": final_norm,
        "lm_head": lm_head,
        "total": total,
        "embedding_share": embeddings / total,
    }


def dataset_information_content(
    num_scenes: int,
    scene_length: int,
    num_distinct_objects: int,
) -> float:
    """Zawartość informacyjna zbioru w bitach, wzór H(x) = N*S*log2(V).

    Wzór pochodzi z rozdziału 3.2 pracy Morris i in. i obowiązuje dla
    danych losowanych jednostajnie i niezależnie, co odpowiada
    charakterystyce zbioru scen wykazanej w rozdziale 3.3 niniejszej
    pracy. Każda z N scen zawiera S pozycji, a na każdej pozycji może
    wystąpić jeden z V obiektów.
    """
    return num_scenes * scene_length * math.log2(num_distinct_objects)


def model_capacity_bits(
    num_parameters: int,
    bits_per_parameter: float = BITS_PER_PARAMETER_HEADLINE,
) -> float:
    """Szacowana pojemność modelu w bitach."""
    return num_parameters * bits_per_parameter


def capacity_report(
    vocab_size: int,
    n_embd: int,
    block_size: int,
    n_layer: int,
    num_scenes: int,
    scene_length: int,
    num_distinct_objects: int,
    bits_per_parameter: float = BITS_PER_PARAMETER_HEADLINE,
) -> dict[str, float]:
    """Zestawienie pojemności modelu i zawartości informacyjnej danych.

    Interpretacja stosunku danych do pojemności, za Morris i in.:
      znacznie poniżej 1  - model działa daleko poniżej swojej
                            pojemności, zapamiętuje zbiór w całości
                            i nie jest zmuszony do żadnej kompresji,
      w okolicach 1       - granica pojemności, w tym rejonie autorzy
                            obserwują zjawisko podwójnego opadania
                            i początek generalizacji,
      znacznie powyżej 1  - zbiór nie mieści się w modelu, memoryzacja
                            pojedynczych przykładów staje się
                            niemożliwa.
    """
    counts = count_parameters_analytic(
        vocab_size, n_embd, block_size, n_layer
    )
    data_bits = dataset_information_content(
        num_scenes, scene_length, num_distinct_objects
    )
    capacity = model_capacity_bits(counts["total"], bits_per_parameter)

    return {
        "n_embd": float(n_embd),
        "n_layer": float(n_layer),
        "parameters": float(counts["total"]),
        "embedding_share": counts["embedding_share"],
        "data_bits": data_bits,
        "capacity_bits": capacity,
        "capacity_to_data": capacity / data_bits,
        "data_to_capacity": data_bits / capacity,
        "bits_per_scene_available": capacity / num_scenes,
        "bits_per_scene_required": data_bits / num_scenes,
    }


def sweep_configurations(
    vocab_size: int,
    block_size: int,
    num_scenes: int,
    scene_length: int,
    num_distinct_objects: int,
    candidates: tuple[tuple[int, int], ...] | None = None,
    bits_per_parameter: float = BITS_PER_PARAMETER_HEADLINE,
) -> list[dict[str, float]]:
    """Zestawienie kandydujących konfiguracji architektury."""
    if candidates is None:
        candidates = (
            (180, 6), (128, 4), (96, 4), (64, 4), (64, 2),
            (48, 2), (32, 2), (24, 2), (16, 2), (16, 1),
        )
    return [
        capacity_report(
            vocab_size, n_embd, block_size, n_layer,
            num_scenes, scene_length, num_distinct_objects,
            bits_per_parameter,
        )
        for n_embd, n_layer in candidates
    ]
