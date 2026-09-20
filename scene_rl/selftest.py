"""Autotesty części kodu wymagającej PyTorcha.

Uruchomienie: python -m scene_rl selftest

Testy sprawdzają własności, które w razie błędu nie objawiają się
awarią programu, lecz cichym pogorszeniem wyników: zgodność
maskowania przyczynowego, skalowanie uwagi, pomijanie wypełnienia
w funkcji straty, wyrównanie indeksów log-prawdopodobieństw w PPO
oraz zdolność modelu do przeuczenia pojedynczej partii danych.
"""

from __future__ import annotations

import math
import random
import tempfile
import os

import numpy as np
import torch
import torch.nn.functional as F

from .config import ExperimentConfig, ModelConfig
from .data import (
    IGNORE_INDEX,
    PAD,
    STOP,
    AugmentationConfig,
    ExampleSampler,
    SceneDataset,
)
from .model import SceneTransformer, cosine_learning_rate
from .ppo import _score_sequence, compute_gae, masked_mean
from .utils import disable_dropout, resolve_device, save_checkpoint, set_seed

PASSED: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def report(message: str) -> None:
    PASSED.append(message)
    print(f"  OK: {message}")


def small_config(vocab_size: int = 40) -> ModelConfig:
    return ModelConfig(
        vocab_size=vocab_size,
        block_size=16,
        n_embd=24,
        n_head=4,
        n_layer=2,
        dropout=0.0,
    )


# ---------------------------------------------------------------

def test_liczba_parametrow() -> None:
    config = ModelConfig(
        vocab_size=995, block_size=64, n_embd=180,
        n_head=6, n_layer=6, dropout=0.3,
    )
    model = SceneTransformer(config)

    v, b, e = config.vocab_size, config.block_size, config.n_embd
    h, layers = config.n_head, config.n_layer
    head_size = e // h
    per_head = 3 * e * head_size
    attention = h * per_head + (e * e + e)
    feedforward = (e * 4 * e + 4 * e) + (4 * e * e + e)
    norms = 4 * e
    per_block = attention + feedforward + norms
    expected = (
        v * e + b * e + layers * per_block + 2 * e + (e * v + v)
    )
    check(
        model.num_parameters() == expected,
        f"liczba parametrów {model.num_parameters()} != {expected}",
    )
    report(f"liczba parametrów zgodna ze wzorem analitycznym "
           f"({expected:,})")


def test_ksztalty_i_strata() -> None:
    config = small_config()
    model = SceneTransformer(config)
    x = torch.randint(3, config.vocab_size, (5, config.block_size))
    y = torch.randint(3, config.vocab_size, (5, config.block_size))

    logits, loss = model(x, y)
    check(logits.shape == (5, config.block_size, config.vocab_size),
          f"zły kształt logitów: {logits.shape}")
    check(loss.ndim == 0, "strata musi być skalarem")

    expected = math.log(config.vocab_size)
    check(abs(loss.item() - expected) < 0.6,
          f"strata modelu losowego {loss.item():.3f}, oczekiwano "
          f"okolic ln(V) = {expected:.3f}")
    report(f"kształty poprawne, strata początkowa {loss.item():.3f} "
           f"~ ln(V) = {expected:.3f}")


def test_maskowanie_przyczynowe() -> None:
    config = small_config()
    model = SceneTransformer(config).eval()
    x = torch.randint(3, config.vocab_size, (1, config.block_size))

    with torch.no_grad():
        base, _ = model(x)
        modified = x.clone()
        modified[0, -1] = (modified[0, -1] + 7) % config.vocab_size
        other, _ = model(modified)

    check(torch.allclose(base[0, :-1], other[0, :-1], atol=1e-6),
          "zmiana ostatniego tokenu wpłynęła na wcześniejsze logity")
    check(not torch.allclose(base[0, -1], other[0, -1]),
          "zmiana ostatniego tokenu nie wpłynęła na jego własne logity")
    report("maskowanie przyczynowe działa: przyszłość nie przecieka")


def test_skalowanie_uwagi() -> None:
    """Uwaga musi być skalowana pierwiastkiem z d_k, nie z n_embd."""
    config = small_config()
    model = SceneTransformer(config).eval()
    head = model.blocks[0].sa.heads[0]

    x = torch.randn(1, config.block_size, config.n_embd)
    with torch.no_grad():
        k = head.key(x)
        q = head.query(x)
        v = head.value(x)
        expected_weights = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        expected_weights = expected_weights.masked_fill(
            head.tril[:config.block_size, :config.block_size] == 0,
            float("-inf"),
        )
        expected = F.softmax(expected_weights, dim=-1) @ v
        got = head(x)

    check(torch.allclose(got, expected, atol=1e-6),
          "wynik głowicy nie zgadza się ze wzorem ze skalowaniem 1/sqrt(d_k)")

    wrong = q @ k.transpose(-2, -1) * (config.n_embd ** -0.5)
    check(not torch.allclose(
        q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5), wrong),
        "test bezużyteczny: oba skalowania dają ten sam wynik")
    report(f"skalowanie uwagi 1/sqrt(d_k) = 1/sqrt({config.head_size}), "
           f"nie 1/sqrt({config.n_embd})")


def test_strata_ignoruje_wypelnienie() -> None:
    config = small_config()
    model = SceneTransformer(config).eval()

    x = torch.randint(3, config.vocab_size, (4, config.block_size))
    y = torch.randint(3, config.vocab_size, (4, config.block_size))
    y[:, 8:] = IGNORE_INDEX

    with torch.no_grad():
        _, loss_a = model(x, y)
        x_modified = x.clone()
        x_modified[:, 12:] = PAD
        _, loss_b = model(x_modified, y)

    check(abs(loss_a.item() - loss_b.item()) < 1e-6,
          "zmiana tokenów w obszarze wypełnienia wpłynęła na stratę")
    report("wypełnienie nie wnosi wkładu do funkcji straty")


def test_generowanie() -> None:
    config = small_config()
    model = SceneTransformer(config).eval()
    prompt = torch.randint(3, config.vocab_size, (8, 4))

    torch.manual_seed(0)
    output, finished = model.generate(prompt, max_new_tokens=10)
    check(output.shape[0] == 8, "zmieniła się liczba przykładów")
    check(output.shape[1] <= config.block_size,
          "wygenerowano sekwencję dłuższą niż okno kontekstowe")
    generated = output[:, 4:]
    check(not bool((generated == PAD).any() & ~finished.any()),
          "wygenerowano token wypełnienia w nieukończonym epizodzie")

    for row in range(8):
        tail = generated[row].tolist()
        if STOP in tail:
            after = tail[tail.index(STOP) + 1:]
            check(all(t == PAD for t in after),
                  "po tokenie STOP pojawiły się tokeny inne niż PAD")

    greedy_a, _ = model.generate(prompt, max_new_tokens=6, greedy=True)
    greedy_b, _ = model.generate(prompt, max_new_tokens=6, greedy=True)
    check(torch.equal(greedy_a, greedy_b),
          "generowanie zachłanne nie jest deterministyczne")
    report("generowanie: STOP kończy epizod, PAD nie jest generowany, "
           "wariant zachłanny deterministyczny")


def test_wyrownanie_logprob() -> None:
    """Indeks k musi odpowiadać akcji na pozycji k+1."""
    config = small_config()
    model = SceneTransformer(config, with_value_head=True).eval()
    tokens = torch.randint(3, config.vocab_size, (3, 10))

    with torch.no_grad():
        logprobs, values = _score_sequence(model, tokens, with_value=True)
        logits, _ = model(tokens)
        manual = F.log_softmax(logits.float(), dim=-1)

    check(logprobs.shape == (3, 9), f"zły kształt: {logprobs.shape}")
    check(values.shape == (3, 9), f"zły kształt wartości: {values.shape}")
    for row in range(3):
        for k in range(9):
            expected = manual[row, k, tokens[row, k + 1]]
            check(abs(logprobs[row, k].item() - expected.item()) < 1e-5,
                  f"przesunięcie indeksów w _score_sequence ({row}, {k})")
    report("log-prawdopodobieństwa i wartości wyrównane o jeden krok")


def test_iloraz_ppo_wynosi_jeden() -> None:
    """Bez dropoutu polityka nowa i stara muszą dać iloraz 1."""
    config = small_config()
    config.dropout = 0.3
    model = SceneTransformer(config, with_value_head=True)
    disable_dropout(model)
    model.train()

    tokens = torch.randint(3, config.vocab_size, (4, 12))
    with torch.no_grad():
        old_logprobs, _ = _score_sequence(model, tokens, with_value=True)
        new_logprobs, _ = _score_sequence(model, tokens, with_value=True)

    ratio = torch.exp(new_logprobs - old_logprobs)
    check(torch.allclose(ratio, torch.ones_like(ratio), atol=1e-5),
          f"iloraz PPO wynosi {ratio.mean().item():.6f}, a nie 1. "
          f"Dropout nie został wyłączony.")
    report("iloraz prawdopodobieństw przed aktualizacją wynosi 1")


def test_gae_i_maski() -> None:
    rewards = np.array([0.0, 0.0, 1.0])
    values = np.array([0.5, 0.5, 0.5])
    adv, ret = compute_gae(rewards, values, 1.0, 1.0)
    check(np.allclose(ret, [1.0, 1.0, 1.0]), f"zły zwrot: {ret}")

    tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mask = torch.tensor([[True, True, False], [False, True, True]])
    check(abs(masked_mean(tensor, mask).item() - 3.5) < 1e-6,
          "masked_mean liczy źle")
    check(masked_mean(tensor, torch.zeros_like(mask)).item() == 0.0,
          "masked_mean przy pustej masce musi zwrócić zero")
    report("GAE i uśrednianie po masce policzone poprawnie")


def test_harmonogram_lr() -> None:
    values = [
        cosine_learning_rate(s, 10, 100, 1e-3, 1e-5) for s in range(120)
    ]
    check(values[0] < values[9] <= 1e-3, "rozgrzewka nie rośnie")
    check(abs(values[9] - 1e-3) < 1e-9, "szczyt rozgrzewki != max_lr")
    check(all(values[i] >= values[i + 1] - 1e-12
              for i in range(10, 99)), "wygaszanie nie jest malejące")
    check(abs(values[110] - 1e-5) < 1e-12, "po końcu lr != min_lr")
    report("harmonogram tempa uczenia: rozgrzewka i wygaszanie poprawne")


def test_przeuczenie_jednej_partii() -> None:
    """Model musi umieć zapamiętać jedną partię danych.

    Jeżeli strata nie spada blisko zera, oznacza to błąd w przepływie
    gradientu, maskowaniu albo w konstrukcji celu.
    """
    set_seed(0)
    config = small_config(vocab_size=30)
    model = SceneTransformer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    x = torch.randint(3, config.vocab_size, (8, config.block_size))
    y = torch.randint(3, config.vocab_size, (8, config.block_size))
    y[:, -3:] = IGNORE_INDEX

    first = None
    for step in range(300):
        _, loss = model(x, y)
        if first is None:
            first = loss.item()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    check(loss.item() < 0.05,
          f"model nie przeuczył jednej partii: strata {loss.item():.4f}")
    report(f"przeuczenie jednej partii: strata {first:.3f} -> "
           f"{loss.item():.4f}")


def test_potok_danych_i_checkpoint(data_path: str | None) -> None:
    if data_path is None or not os.path.exists(data_path):
        print("  POMINIETO: brak pliku danych, test potoku danych")
        return

    dataset = SceneDataset(data_path)
    augmentation = AugmentationConfig()
    sampler = ExampleSampler(dataset, 64, augmentation, seed=0)
    x, y = sampler.batch(32)

    config = ExperimentConfig()
    config.model = ModelConfig(
        vocab_size=dataset.vocab_size, block_size=64,
        n_embd=48, n_head=4, n_layer=2, dropout=0.0,
    )
    model = SceneTransformer(config.model)
    logits, loss = model(
        torch.from_numpy(x), torch.from_numpy(y)
    )
    check(torch.isfinite(loss), "strata nie jest skończona")

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "test.pt")
        save_checkpoint(path, model, None, config, 0, [])
        from .utils import build_model_from_checkpoint
        restored, restored_config, _ = build_model_from_checkpoint(
            path, torch.device("cpu")
        )
        restored.eval()
        model.eval()
        with torch.no_grad():
            a, _ = model(torch.from_numpy(x))
            b, _ = restored(torch.from_numpy(x))
        check(torch.allclose(a, b, atol=1e-6),
              "model po wczytaniu daje inne wyniki")
        check(restored_config.model.vocab_size == dataset.vocab_size,
              "konfiguracja nie przetrwała zapisu")

        head, _, _ = build_model_from_checkpoint(
            path, torch.device("cpu"), with_value_head=True
        )
        check(head.value_head is not None,
              "nie udało się dodać głowicy wartości do wczytanego modelu")

    report("potok danych, zapis i wczytanie checkpointu działają")


def run(data_path: str | None = None) -> None:
    print("Autotesty implementacji\n")
    device = resolve_device("auto")
    print(f"Urządzenie: {device}, PyTorch {torch.__version__}\n")

    torch.manual_seed(0)
    random.seed(0)

    test_liczba_parametrow()
    test_ksztalty_i_strata()
    test_maskowanie_przyczynowe()
    test_skalowanie_uwagi()
    test_strata_ignoruje_wypelnienie()
    test_generowanie()
    test_wyrownanie_logprob()
    test_iloraz_ppo_wynosi_jeden()
    test_gae_i_maski()
    test_harmonogram_lr()
    test_przeuczenie_jednej_partii()
    test_potok_danych_i_checkpoint(data_path)

    print(f"\nWSZYSTKIE AUTOTESTY PRZESZŁY ({len(PASSED)})")
