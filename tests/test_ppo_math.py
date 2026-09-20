"""Testy części algorytmu PPO niezależnych od PyTorcha."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# compute_gae importujemy bez wciągania torcha
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_gae", os.path.join(os.path.dirname(__file__), "..",
                         "scene_rl", "ppo.py")
)


def _load_compute_gae():
    """Wyciąga samą funkcję GAE z pliku, bez importu torcha."""
    path = os.path.join(
        os.path.dirname(__file__), "..", "scene_rl", "ppo.py"
    )
    source = open(path, encoding="utf-8").read()
    start = source.index("def compute_gae(")
    end = source.index("def masked_mean(")
    namespace = {"np": np, "annotations": None}
    exec("from __future__ import annotations\n" + source[start:end],
         namespace)
    return namespace["compute_gae"]


compute_gae = _load_compute_gae()


def test_gae_monte_carlo():
    """Dla gamma=1 i lambda=1 przewaga to zwrot minus wartość."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = rng.integers(1, 30)
        rewards = rng.normal(size=n)
        values = rng.normal(size=n)
        adv, ret = compute_gae(rewards, values, 1.0, 1.0)
        expected = np.array(
            [rewards[t:].sum() - values[t] for t in range(n)]
        )
        assert np.allclose(adv, expected), (adv, expected)
        assert np.allclose(ret, adv + values)
    print("OK: GAE przy lambda=1 równa się zwrotowi Monte Carlo")


def test_gae_td_zero():
    """Dla lambda=0 przewaga to błąd czasowo-różnicowy TD(0)."""
    rng = np.random.default_rng(1)
    for _ in range(200):
        n = int(rng.integers(1, 20))
        rewards = rng.normal(size=n)
        values = rng.normal(size=n)
        gamma = float(rng.uniform(0.5, 1.0))
        adv, _ = compute_gae(rewards, values, gamma, 0.0)
        for t in range(n):
            next_value = values[t + 1] if t + 1 < n else 0.0
            expected = rewards[t] + gamma * next_value - values[t]
            assert abs(adv[t] - expected) < 1e-10
    print("OK: GAE przy lambda=0 równa się TD(0)")


def test_gae_nagroda_terminalna():
    """Nagroda przyznana tylko na końcu propaguje się wstecz."""
    n = 5
    rewards = np.zeros(n)
    rewards[-1] = 1.0
    values = np.zeros(n)
    adv, ret = compute_gae(rewards, values, 1.0, 1.0)
    assert np.allclose(adv, np.ones(n)), adv
    assert np.allclose(ret, np.ones(n))
    print("OK: rzadka nagroda terminalna propaguje się na cały epizod")


def test_wycinanie_sceny():
    """Poprawka błędu o jeden: ostatni obiekt nie może zginąć."""
    from scene_rl.data import PAD, SEP, STOP  # noqa: E402
    from scene_rl.data import extract_restored_scene  # noqa: E402

    prompt = [10, 11, 12]
    generated = [SEP, 10, 11, 12, 13, SEP, 10, 11, 12, 13, 14, STOP]
    tokens = prompt + generated
    got = extract_restored_scene(tokens, prompt_length=len(prompt))
    assert got == [10, 11, 12, 13, 14], got

    # stara, błędna formuła obcięłaby ostatni obiekt
    stop = tokens.index(STOP)
    last_sep = max(i for i, t in enumerate(tokens[:stop]) if t == SEP)
    stara = tokens[last_sep + 1: stop - 1]
    assert stara == [10, 11, 12, 13], stara
    assert len(stara) == len(got) - 1

    # brak STOP oznacza epizod nieudany
    assert extract_restored_scene(prompt + [SEP, 10, 11], 3) is None

    # brak separatora: fragment wejściowy nie może trafić do wyniku
    got = extract_restored_scene([10, 11, 12, 77, 88, STOP], 3)
    assert got == [77, 88], got

    # wypełnienie po tokenie STOP jest ignorowane
    got = extract_restored_scene(
        [10, 11, 12, SEP, 5, 6, 7, STOP, PAD, PAD], 3
    )
    assert got == [5, 6, 7], got

    # pusta rekonstrukcja jest dozwolona i daje pustą listę
    assert extract_restored_scene([10, 11, 12, SEP, STOP], 3) == []
    print("OK: wycinanie sceny naprawione, ostatni obiekt zachowany")


if __name__ == "__main__":
    test_gae_monte_carlo()
    test_gae_td_zero()
    test_gae_nagroda_terminalna()
    test_wycinanie_sceny()
    print("\nWSZYSTKIE TESTY PPO/EWALUACJI PRZESZLY")
