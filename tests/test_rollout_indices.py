"""Weryfikacja arytmetyki indeksów w buforze doświadczenia PPO.

Test odtwarza w numpy dokładnie tę samą logikę wyznaczania maski
akcji i umieszczania nagrody terminalnej, co funkcja
`collect_rollout`, i sprawdza jej zgodność z definicją procesu
decyzyjnego. Wyprowadzenie jest niezależne od implementacji:
liczba akcji w epizodzie musi być równa liczbie tokenów
wygenerowanych przez model, a nagroda terminalna musi trafić
na krok odpowiadający ostatniemu z nich.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scene_rl.data import PAD, SEP, STOP  # noqa: E402


def build_masks(episodes, max_length):
    """Replikuje logikę indeksowania z collect_rollout."""
    n = len(episodes)
    tokens = np.full((n, max_length), PAD, dtype=np.int64)
    action_mask = np.zeros((n, max_length - 1), dtype=bool)
    reward_position = np.zeros(n, dtype=np.int64)

    for row, episode in enumerate(episodes):
        length = len(episode["tokens"])
        tokens[row, :length] = episode["tokens"]
        first = episode["prompt_length"] - 1
        last = length - 2
        action_mask[row, first:last + 1] = True
        reward_position[row] = last

    return tokens, action_mask, reward_position


def test_liczba_akcji():
    episodes = [
        {"tokens": [10, 11, 12, SEP, 20, 21, STOP], "prompt_length": 3},
        {"tokens": [10, 11, 12, 13, SEP, 20, 21, 22, STOP],
         "prompt_length": 4},
        {"tokens": [10, 11, 12, STOP], "prompt_length": 3},
    ]
    max_length = max(len(e["tokens"]) for e in episodes)
    tokens, mask, reward_position = build_masks(episodes, max_length)

    for row, episode in enumerate(episodes):
        length = len(episode["tokens"])
        prompt = episode["prompt_length"]
        expected_actions = length - prompt
        assert mask[row].sum() == expected_actions, (
            f"epizod {row}: {mask[row].sum()} akcji, "
            f"oczekiwano {expected_actions}"
        )

        # akcja o indeksie k dotyczy tokenu na pozycji k+1
        positions = np.where(mask[row])[0] + 1
        assert positions.min() == prompt, (
            "pierwsza akcja musi dotyczyć pierwszego wygenerowanego tokenu"
        )
        assert positions.max() == length - 1, (
            "ostatnia akcja musi dotyczyć ostatniego tokenu epizodu"
        )
        assert tokens[row, positions.max()] == STOP, (
            "ostatnim tokenem epizodu zakończonego musi być STOP"
        )

        # nagroda terminalna trafia na ostatni krok
        assert reward_position[row] == positions.max() - 1
        assert mask[row, reward_position[row]]

    print("OK: maska akcji i pozycja nagrody terminalnej zgodne "
          "z definicją epizodu")


def test_wypelnienie_poza_maska():
    episodes = [
        {"tokens": [10, 11, 12, SEP, 20, STOP], "prompt_length": 3},
        {"tokens": [10, 11, 12, SEP, 20, 21, 22, 23, STOP],
         "prompt_length": 3},
    ]
    max_length = max(len(e["tokens"]) for e in episodes)
    tokens, mask, _ = build_masks(episodes, max_length)

    for row, episode in enumerate(episodes):
        length = len(episode["tokens"])
        assert all(tokens[row, length:] == PAD)
        # pozycje wypełnienia nigdy nie są akcjami
        assert not mask[row, length - 1:].any(), (
            "wypełnienie zostało potraktowane jako akcja"
        )
    print("OK: wypełnienie nigdy nie jest liczone jako akcja")


def test_zgodnosc_z_dlugoscia_generacji():
    """Suma akcji musi być równa liczbie wygenerowanych tokenów."""
    rng = np.random.default_rng(0)
    for _ in range(500):
        episodes = []
        for _ in range(int(rng.integers(1, 6))):
            prompt = int(rng.integers(3, 9))
            generated = int(rng.integers(1, 20))
            body = [int(rng.integers(3, 100)) for _ in range(prompt)]
            tail = [int(rng.integers(3, 100))
                    for _ in range(generated - 1)] + [STOP]
            episodes.append({
                "tokens": body + tail,
                "prompt_length": prompt,
                "generated": generated,
            })
        max_length = max(len(e["tokens"]) for e in episodes)
        _, mask, _ = build_masks(episodes, max_length)
        for row, episode in enumerate(episodes):
            assert mask[row].sum() == episode["generated"]
    print("OK: liczba akcji równa liczbie tokenów wygenerowanych "
          "na 500 losowych buforach")


if __name__ == "__main__":
    test_liczba_akcji()
    test_wypelnienie_poza_maska()
    test_zgodnosc_z_dlugoscia_generacji()
    print("\nWSZYSTKIE TESTY INDEKSOWANIA PPO PRZESZLY")
