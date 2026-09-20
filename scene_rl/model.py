"""Transformer typu decoder-only wraz z głowicą wartości dla PPO.

Zmiany względem wersji notatnikowej:

1. Nazwa klasy zmieniona z `BigramLanguageModel` na `SceneTransformer`.
   Model nie jest modelem bigramowym: przewiduje kolejny token na
   podstawie całego widocznego prefiksu, a nie jednego poprzednika.
2. Poprawiono skalowanie iloczynu skalarnego uwagi. Poprzednio
   dzielono przez pierwiastek z `n_embd`, podczas gdy wzór (2.x)
   za Vaswanim i innymi wymaga pierwiastka z wymiaru głowy `d_k`.
   Przy n_embd = 180 i n_head = 6 współczynnik był zaniżony
   pierwiastkiem z sześciu, co nadmiernie wygładzało rozkład uwagi.
3. Hiperparametry przekazywane są przez obiekt konfiguracji, a nie
   przez zmienne globalne, dzięki czemu można trzymać w pamięci
   kilka modeli o różnych ustawieniach jednocześnie, co jest
   niezbędne w PPO (polityka bieżąca i polityka odniesienia).
4. Generowanie kończy się na tokenie STOP, obsługuje temperaturę,
   obcinanie do k najlepszych tokenów oraz wariant zachłanny,
   a token wypełnienia PAD nigdy nie może zostać wygenerowany.
5. Dodano opcjonalną głowicę wartości, wykorzystywaną jako krytyk
   w algorytmie PPO.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from .config import ModelConfig
from .data import IGNORE_INDEX, PAD, STOP


class Head(nn.Module):
    """Pojedyncza głowica uwagi własnej z maskowaniem przyczynowym."""

    def __init__(self, config: ModelConfig, head_size: int):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        # Macierz dolnotrójkątna nie jest parametrem uczonym, lecz
        # stałym elementem modelu, dlatego rejestrowana jest jako bufor.
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(config.block_size, config.block_size)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, T, _ = x.shape
        k = self.key(x)      # (B, T, head_size)
        q = self.query(x)    # (B, T, head_size)

        # Skalowanie pierwiastkiem z wymiaru głowy, zgodnie ze wzorem
        # Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V.
        scale = k.shape[-1] ** -0.5
        weights = q @ k.transpose(-2, -1) * scale       # (B, T, T)
        weights = weights.masked_fill(
            self.tril[:T, :T] == 0, float("-inf")
        )
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        v = self.value(x)
        return weights @ v   # (B, T, head_size)


class MultiHeadAttention(nn.Module):
    """Kilka głowic uwagi liczonych równolegle."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(config, config.head_size) for _ in range(config.n_head)]
        )
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """Warstwa gęsta stosowana niezależnie do każdego tokenu."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """Blok transformera: komunikacja, następnie obliczenia."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.sa = MultiHeadAttention(config)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class SceneTransformer(nn.Module):
    """Model rekonstrukcji sceny, architektura decoder-only.

    Parametr `with_value_head` włącza dodatkową głowicę liniową
    przewidującą wartość stanu. Głowica ta jest nieaktywna podczas
    uczenia nadzorowanego i wykorzystywana wyłącznie jako krytyk
    w etapie dostrajania algorytmem PPO.
    """

    def __init__(self, config: ModelConfig, with_value_head: bool = False):
        super().__init__()
        # Zabezpieczenie niezależne od walidacji w ModelConfig.
        # __post_init__ dataclassy uruchamia się wyłącznie w chwili
        # tworzenia obiektu, więc jeśli którykolwiek fragment kodu
        # podmienił n_embd albo n_head na polu już istniejącego
        # obiektu konfiguracji, niepodzielność nie zostałaby wykryta
        # aż do pierwszego mnożenia macierzy w warstwie uwagi. Model
        # sprawdza więc ten warunek jeszcze raz, tuż przed zbudowaniem
        # jakichkolwiek warstw.
        if config.n_embd % config.n_head != 0:
            raise ValueError(
                f"n_embd ({config.n_embd}) musi dzielić się przez "
                f"n_head ({config.n_head}) bez reszty."
            )
        self.config = config
        self.token_embedding_table = nn.Embedding(
            config.vocab_size, config.n_embd
        )
        self.position_embedding_table = nn.Embedding(
            config.block_size, config.n_embd
        )
        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        self.value_head = (
            nn.Linear(config.n_embd, 1) if with_value_head else None
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Inicjalizacja wag rozkładem normalnym o małej wariancji.

        Brak jawnej inicjalizacji był jedną z przyczyn bardzo wysokiej
        wartości funkcji straty w pierwszych krokach treningu.
        Dla słownika liczącego 995 tokenów strata modelu losowego
        wynosi ln(995), czyli około 6,90.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def features(self, idx: torch.Tensor) -> torch.Tensor:
        """Zwraca reprezentacje ukryte o kształcie (B, T, n_embd)."""
        _, T = idx.shape
        if T > self.config.block_size:
            raise ValueError(
                f"Sekwencja o długości {T} przekracza block_size "
                f"({self.config.block_size})."
            )
        positions = torch.arange(T, device=idx.device)
        x = self.token_embedding_table(idx)
        x = x + self.position_embedding_table(positions)
        x = self.blocks(x)
        return self.ln_f(x)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Zwraca logity oraz, opcjonalnie, wartość funkcji straty.

        Logit jest surową, nieznormalizowaną oceną przypisaną przez
        sieć każdemu tokenowi słownika. Rozkład prawdopodobieństwa
        otrzymuje się dopiero po zastosowaniu funkcji softmax.

        Pozycje oznaczone wartością IGNORE_INDEX, odpowiadające
        wypełnieniu oraz fragmentowi wejściowemu, nie wnoszą wkładu
        do funkcji straty.
        """
        x = self.features(idx)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )
        return logits, loss

    def forward_with_value(
        self, idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Zwraca logity oraz oszacowanie wartości stanu."""
        if self.value_head is None:
            raise RuntimeError("Model utworzono bez głowicy wartości.")
        x = self.features(idx)
        logits = self.lm_head(x)
        values = self.value_head(x).squeeze(-1)   # (B, T)
        return logits, values

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 0,
        greedy: bool = False,
        stop_token: int = STOP,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generuje tokeny aż do STOP albo wyczerpania limitu.

        Zwraca parę: pełną sekwencję oraz maskę logiczną wskazującą,
        które przykłady w partii zakończyły się tokenem STOP.
        Sekwencje zakończone są dopełniane tokenem PAD, żeby nie
        generować śmieci po zakończeniu epizodu.

        Wartości logitów tokenu PAD ustawiane są na minus
        nieskończoność, ponieważ wypełnienie nie jest elementem
        przestrzeni akcji modelu.
        """
        was_training = self.training
        self.eval()

        batch_size = idx.shape[0]
        finished = torch.zeros(
            batch_size, dtype=torch.bool, device=idx.device
        )

        for _ in range(max_new_tokens):
            context = idx[:, -self.config.block_size:]
            logits, _ = self(context)
            logits = logits[:, -1, :].float()
            logits[:, PAD] = float("-inf")

            if greedy:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / max(temperature, 1e-6)
                if top_k > 0:
                    k = min(top_k, logits.size(-1))
                    threshold = torch.topk(logits, k, dim=-1).values[:, -1:]
                    logits = logits.masked_fill(
                        logits < threshold, float("-inf")
                    )
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            next_token = torch.where(
                finished.unsqueeze(1),
                torch.full_like(next_token, PAD),
                next_token,
            )
            idx = torch.cat((idx, next_token), dim=1)
            finished = finished | (next_token.squeeze(1) == stop_token)

            if bool(finished.all()):
                break

            if idx.shape[1] >= self.config.block_size:
                break

        if was_training:
            self.train()
        return idx, finished


def cosine_learning_rate(
    step: int,
    warmup_steps: int,
    total_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """Harmonogram tempa uczenia: rozgrzewka i wygaszanie cosinusem."""
    if step < warmup_steps:
        return max_lr * (step + 1) / max(warmup_steps, 1)
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    coefficient = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coefficient * (max_lr - min_lr)
