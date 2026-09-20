"""Rekonstrukcja sekwencji transformerem z uczeniem ze wzmocnieniem.

Pakiet towarzyszący pracy magisterskiej "Rekonstrukcja sekwencji za
pomocą sztucznej sieci neuronowej o architekturze transformera
z wykorzystaniem metryk podobieństwa i uczenia ze wzmocnieniem".

Moduły:
    config      - hiperparametry eksperymentu
    data        - wczytanie scen, tokenizacja, przykłady uczące
    metrics     - BLEU, ROUGE-L, podobieństwo edycyjne, miara docelowa
    model       - transformer decoder-only z głowicą wartości
    sft         - etap 1: uczenie nadzorowane
    ppo         - etap 2: dostrajanie algorytmem PPO
    evaluation  - rekonstrukcja scen i ocena jakości
    utils       - ziarna losowości, checkpointy, logowanie
    selftest    - autotesty implementacji

Import pakietu nie wciąga biblioteki PyTorch. Moduły `model`, `sft`,
`ppo`, `evaluation` i `utils` importuje się jawnie, dzięki czemu
metryki i analizę danych można uruchomić na maszynie bez torcha.
"""

__version__ = "1.0.0"

from .config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    PPOConfig,
    TrainConfig,
)
from .data import (
    PAD,
    SEP,
    STOP,
    AugmentationConfig,
    ExampleSampler,
    SceneDataset,
    build_chain,
    encode_chain,
    extract_restored_scene,
    make_prompt,
    required_block_size,
)
from .metrics import (
    ALL_METRICS,
    REWARD_METRICS,
    bleu_score,
    kendall_tau_order,
    levenshtein_distance,
    levenshtein_f_score,
    levenshtein_similarity,
    rouge_l_score,
    target_measure,
)

__all__ = [
    "__version__",
    "ExperimentConfig",
    "ModelConfig",
    "DataConfig",
    "TrainConfig",
    "PPOConfig",
    "SceneDataset",
    "AugmentationConfig",
    "ExampleSampler",
    "build_chain",
    "encode_chain",
    "extract_restored_scene",
    "make_prompt",
    "required_block_size",
    "STOP",
    "SEP",
    "PAD",
    "bleu_score",
    "rouge_l_score",
    "levenshtein_distance",
    "levenshtein_similarity",
    "levenshtein_f_score",
    "kendall_tau_order",
    "target_measure",
    "REWARD_METRICS",
    "ALL_METRICS",
]
