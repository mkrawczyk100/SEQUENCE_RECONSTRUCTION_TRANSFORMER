"""Konfiguracja eksperymentu.

Wszystkie hiperparametry zebrano w jawnych strukturach danych,
zamiast rozpraszać je po zmiennych globalnych notatnika. Dzięki
temu konfiguracja zapisywana jest w całości w checkpoincie
i przebieg eksperymentu daje się odtworzyć bez odgadywania,
która komórka została uruchomiona jako ostatnia.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class ModelConfig:
    """Hiperparametry architektury transformera."""

    vocab_size: int = 995
    block_size: int = 64
    n_embd: int = 180
    n_head: int = 6
    n_layer: int = 6
    dropout: float = 0.3

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) musi dzielić się przez "
                f"n_head ({self.n_head}) bez reszty."
            )

    @property
    def head_size(self) -> int:
        return self.n_embd // self.n_head


@dataclass
class DataConfig:
    """Parametry zbioru danych i generowania przykładów."""

    path: str = "sequences_przemek.txt"
    min_elements: int = 3
    max_remove_per_step: int = 1
    swap_probability: float = 0.15
    start_mode: str = "sep"
    loss_on_prompt: bool = False


@dataclass
class TrainConfig:
    """Parametry etapu pierwszego, uczenia nadzorowanego."""

    batch_size: int = 256
    steps_per_epoch: int = 500
    num_epochs: int = 60
    learning_rate: float = 5e-4
    min_learning_rate: float = 5e-5
    warmup_steps: int = 200
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_batches: int = 20
    eval_every: int = 1
    metrics_every: int = 5
    metrics_samples: int = 100
    prompt_length: int = 6


@dataclass
class PPOConfig:
    """Parametry etapu drugiego, dostrajania algorytmem PPO."""

    reward_metric: str = "levenshtein"
    iterations: int = 300
    episodes_per_iteration: int = 256
    ppo_epochs: int = 4
    minibatch_size: int = 64
    learning_rate: float = 1e-5
    clip_range: float = 0.2
    value_clip_range: float = 0.2
    value_coef: float = 0.5
    # Premia entropijna wyłączona domyślnie. Polityka po etapie
    # nadzorowanym jest już bliska optimum, więc wymuszanie
    # eksploracji podnosi entropię i pogarsza rekonstrukcje dokładne,
    # nie wnosząc nic w zamian.
    entropy_coef: float = 0.0
    gamma: float = 1.0
    gae_lambda: float = 0.95
    kl_coef: float = 0.02
    grad_clip: float = 1.0
    temperature: float = 1.0
    top_k: int = 0
    prompt_min: int = 3
    prompt_max: int = 8
    no_stop_penalty: float = 0.0
    whiten_advantages: bool = True
    eval_every: int = 10
    eval_samples: int = 200
    # Ewaluacja zachłanna usuwa z pomiaru szum losowania tokenów.
    # Przy próbkowaniu z temperaturą 1,0 model o wyższej entropii
    # wypada gorzej w mierze exact_match nawet wtedy, gdy jego
    # najbardziej prawdopodobna rekonstrukcja jest identyczna.
    greedy_eval: bool = False


@dataclass
class ExperimentConfig:
    """Pełna konfiguracja eksperymentu wraz z ziarnami losowości."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)

    seed: int = 1234
    train_data_seed: int = 1
    val_data_seed: int = 2
    eval_seed: int = 3
    device: str = "auto"
    output_dir: str = "runs/sft"
    run_name: str = "sft"

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, payload: dict) -> "ExperimentConfig":
        return cls(
            model=ModelConfig(**payload["model"]),
            data=DataConfig(**payload["data"]),
            train=TrainConfig(**payload["train"]),
            ppo=PPOConfig(**payload["ppo"]),
            **{
                key: value
                for key, value in payload.items()
                if key not in ("model", "data", "train", "ppo")
            },
        )

    @classmethod
    def from_json(cls, path: str) -> "ExperimentConfig":
        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))
