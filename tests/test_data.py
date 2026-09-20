"""Testy przygotowania danych."""

import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scene_rl.data import (  # noqa: E402
    IGNORE_INDEX,
    PAD,
    SEP,
    STOP,
    AugmentationConfig,
    ExampleSampler,
    SceneDataset,
    build_chain,
    chain_start_positions,
    encode_chain,
    make_prompt,
    required_block_size,
)

DATA = "/mnt/project/sequences_przemek.txt"


def test_wczytanie_i_statystyki():
    ds = SceneDataset(DATA)
    st = ds.statistics()
    print("statystyki:", st)
    assert st["num_scenes"] == 500
    assert st["scene_length"] == 10
    assert st["num_unique_objects"] == 992
    assert st["vocab_size"] == 995
    assert st["scenes_with_duplicate"] == 21
    assert st["sorted_scenes"] == 0

    # tokenizacja odwracalna
    for i in range(50):
        original = [int(v) for v in ds.raw[i]]
        tokens = ds.scenes[i]
        assert all(t >= 3 for t in tokens)
        assert [ds.itos[t] for t in tokens] == original
    print("OK: tokenizacja odwracalna, tokeny specjalne nie kolidują")


def test_jednoznacznosc_kontekstu():
    ds = SceneDataset(DATA)
    wyniki = {}
    for k in (1, 2, 3):
        wyniki[k] = ds.context_ambiguity(k)["ambiguous_fraction"]
        print(f"  kontekst {k}: niejednoznaczny w "
              f"{100 * wyniki[k]:.2f}% przypadków")
    assert wyniki[1] > 0.99
    assert 0.03 < wyniki[2] < 0.05
    assert wyniki[3] < 0.001
    co = ds.cooccurrence_statistics()
    print("  współwystępowanie:", co)
    assert co["distinct_pairs"] == 21828
    assert co["pairs_occurring_more_than_once"] == 477
    print("OK: jednoznaczność kontekstu uzasadnia min_elements = 3")


def test_lancuch_i_kodowanie():
    ds = SceneDataset(DATA)
    cfg = AugmentationConfig()
    rng = random.Random(0)
    scene = ds.scenes[0]

    for _ in range(200):
        versions = build_chain(scene, rng, cfg)
        # długości rosną o jeden, od min_elements do pełnej sceny
        lengths = [len(v) for v in versions]
        assert lengths == list(range(cfg.min_elements, len(scene) + 1)), lengths
        # ostatnia wersja to dokładnie scena oryginalna, bez zamiany
        assert versions[-1] == scene
        # każda wersja jest wielozbiorem zawartym w scenie
        for v in versions[:-1]:
            assert set(v).issubset(set(scene))

        tokens = encode_chain(versions)
        assert tokens[-1] == STOP
        assert tokens.count(SEP) == len(versions) - 1
        assert PAD not in tokens

    # scena nie została zmodyfikowana w miejscu
    assert ds.scenes[0] == scene
    print(f"OK: łańcuch ma {len(versions)} wersji, "
          f"{len(tokens)} tokenów, scena nienaruszona")


def test_pozycje_startowe():
    tokens = [5, 6, 7, SEP, 5, 6, 7, 8, SEP, 5, 6, 7, 8, 9, STOP]
    starts = chain_start_positions(tokens)
    assert starts == [0, 4, 9], starts
    for s in starts:
        assert s == 0 or tokens[s - 1] == SEP
    print("OK: okno startuje na początku łańcucha albo tuż za SEP")


def test_required_block_size():
    ds = SceneDataset(DATA)
    cfg = AugmentationConfig()
    need = required_block_size(ds, cfg, num_draws=500)
    # 3+4+...+10 = 52 tokeny obiektów, 7 separatorów, 1 STOP = 60
    assert need == 61, need
    print(f"OK: minimalny block_size wynosi {need} "
          f"(łańcuch 60 tokenów + 1 na przesunięcie celu)")


def test_batch_padding_i_maskowanie():
    ds = SceneDataset(DATA)
    cfg = AugmentationConfig()
    block = 64
    sampler = ExampleSampler(ds, block, cfg, seed=123)
    x, y = sampler.batch(256)

    assert x.shape == (256, block) and y.shape == (256, block)
    assert sampler.truncated_examples == 0, "przykład nie zmieścił się w oknie"

    for row in range(x.shape[0]):
        xr, yr = x[row].tolist(), y[row].tolist()

        # wypełnienie występuje wyłącznie na końcu
        if PAD in xr:
            first_pad = xr.index(PAD)
            assert all(v == PAD for v in xr[first_pad:])
            assert all(v == IGNORE_INDEX for v in yr[first_pad:])

        # cel jest wejściem przesuniętym o jeden token
        content = [v for v in xr if v != PAD]
        assert content[-1] != STOP  # STOP jest tylko w celu
        for i in range(len(content) - 1):
            if yr[i] != IGNORE_INDEX:
                assert yr[i] == xr[i + 1], (row, i)

        # maskowany jest fragment wejściowy; pierwszą pozycją uczącą
        # jest ta, na której model ma wygenerować SEP
        content_len = len(content)
        first_sep = xr.index(SEP) if SEP in xr else content_len
        mask_until = max(0, first_sep - 1)
        assert all(v == IGNORE_INDEX for v in yr[:mask_until])
        assert yr[mask_until] != IGNORE_INDEX
        if SEP in xr:
            assert yr[mask_until] == SEP
        else:
            assert yr[mask_until] == STOP

        # cel zawiera dokładnie jeden STOP
        assert yr.count(STOP) == 1

    udzial = float(np.mean(y != IGNORE_INDEX))
    print(f"OK: padding i maskowanie poprawne, "
          f"pozycji uczących {100 * udzial:.1f}%")


def test_rozne_ziarna_daja_rozne_dane():
    ds = SceneDataset(DATA)
    cfg = AugmentationConfig()
    a = ExampleSampler(ds, 64, cfg, seed=1).batch(16)[0]
    b = ExampleSampler(ds, 64, cfg, seed=2).batch(16)[0]
    c = ExampleSampler(ds, 64, cfg, seed=1).batch(16)[0]
    assert not np.array_equal(a, b), "różne ziarna dały te same dane"
    assert np.array_equal(a, c), "to samo ziarno dało różne dane"
    print("OK: powtarzalność i rozdzielność strumieni losowych")


def test_start_mode_chain():
    ds = SceneDataset(DATA)
    cfg = AugmentationConfig()
    s = ExampleSampler(ds, 64, cfg, seed=7, start_mode="chain")
    x, y = s.batch(64)
    for row in range(x.shape[0]):
        xr = x[row].tolist()
        # w trybie 'chain' pierwsze min_elements tokenów to fragment
        assert xr[cfg.min_elements] == SEP, xr[:6]
    print("OK: tryb 'chain' zawsze startuje od pełnego fragmentu")


def test_make_prompt():
    ds = SceneDataset(DATA)
    rng = random.Random(0)
    scene = ds.scenes[3]
    for k in (3, 5, 6, 10):
        p = make_prompt(scene, k, rng)
        assert len(p) == k
        # fragment jest podciągiem sceny, kolejność zachowana
        it = iter(scene)
        assert all(tok in it for tok in p)
    print("OK: fragment wejściowy jest podciągiem sceny")


if __name__ == "__main__":
    test_wczytanie_i_statystyki()
    test_jednoznacznosc_kontekstu()
    test_lancuch_i_kodowanie()
    test_pozycje_startowe()
    test_required_block_size()
    test_batch_padding_i_maskowanie()
    test_rozne_ziarna_daja_rozne_dane()
    test_start_mode_chain()
    test_make_prompt()
    print("\nWSZYSTKIE TESTY DANYCH PRZESZLY")
