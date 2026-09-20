"""Testy metryk. Sprawdzają zgodność z Tabelą 2.2 z pracy."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scene_rl.metrics import (  # noqa: E402
    bleu_score,
    content_scores,
    kendall_tau_order,
    lcs_length,
    levenshtein_distance,
    levenshtein_f_score,
    levenshtein_similarity,
    position_accuracy,
    rouge_l_score,
    target_measure,
)

X = [742, 118, 903, 55, 671, 284, 12, 460, 837, 199]

WARIANTY = {
    "rekonstrukcja idealna": list(X),
    # Uwaga: wartosc 0,880 z Tabeli 2.2 odpowiada bledowi na pozycji
    # skrajnej. Blad na pozycji srodkowej daje BLEU 0,658 (patrz
    # test_wrazliwosc_bleu_na_pozycje_bledu).
    "jeden bledny obiekt": [742, 118, 903, 55, 671, 284, 12, 460, 837, 999],
    "zamiana dwoch sasiednich": [
        742, 118, 903, 55, 284, 671, 12, 460, 837, 199
    ],
    "odwrocona kolejnosc": list(reversed(X)),
    "brak trzech ostatnich": X[:7],
    "trzy nadmiarowe": X + [901, 902, 904],
    "piec ostatnich blednych": X[:5] + [991, 992, 993, 994, 995],
    "brak wspolnych": [901, 902, 903 + 1000, 904, 905,
                       906, 907, 908, 909, 910],
}

# Wartości z Tabeli 2.2: (|Y|, d, s_Lev, BLEU, ROUGE-L, F-score k=2)
OCZEKIWANE = {
    "rekonstrukcja idealna": (10, 0, 1.000, 1.000, 1.000, 1.000),
    "jeden bledny obiekt": (10, 1, 0.900, 0.880, 0.900, 0.900),
    "zamiana dwoch sasiednich": (10, 2, 0.800, 0.556, 0.900, 0.900),
    "odwrocona kolejnosc": (10, 10, 0.000, 0.000, 0.100, 0.100),
    "brak trzech ostatnich": (7, 3, 0.700, 0.651, 0.824, 0.824),
    "trzy nadmiarowe": (13, 3, 0.769, 0.736, 0.870, 0.870),
    "piec ostatnich blednych": (10, 5, 0.500, 0.393, 0.500, 0.500),
    "brak wspolnych": (10, 10, 0.000, 0.000, 0.000, 0.000),
}


def test_tabela_2_2():
    bledy = []
    print(f"{'wariant':<26}{'|Y|':>5}{'d':>4}{'s_Lev':>8}"
          f"{'BLEU':>8}{'ROUGE-L':>9}{'F(k=2)':>9}")
    for nazwa, Y in WARIANTY.items():
        d = levenshtein_distance(X, Y, 1)
        s = levenshtein_similarity(X, Y)
        b = bleu_score(X, Y, smoothing="none")
        r = rouge_l_score(X, Y)
        f = levenshtein_f_score(X, Y)
        print(f"{nazwa:<26}{len(Y):>5}{d:>4}{s:>8.3f}"
              f"{b:>8.3f}{r:>9.3f}{f:>9.3f}")

        exp = OCZEKIWANE[nazwa]
        for got, want, etykieta in (
            (len(Y), exp[0], "|Y|"),
            (d, exp[1], "d"),
            (s, exp[2], "s_Lev"),
            (b, exp[3], "BLEU"),
            (r, exp[4], "ROUGE-L"),
            (f, exp[5], "F(k=2)"),
        ):
            if abs(got - want) > 0.0006:
                bledy.append(f"{nazwa}/{etykieta}: {got} != {want}")

    assert not bledy, "\n".join(bledy)
    print("OK: wszystkie wartosci zgodne z Tabela 2.2")


def test_wrazliwosc_bleu_na_pozycje_bledu():
    """BLEU zalezy od tego, gdzie w sekwencji lezy blad."""
    wyniki = []
    for i in range(len(X)):
        Y = list(X)
        Y[i] = 999
        wyniki.append(bleu_score(X, Y))
    assert abs(wyniki[0] - 0.8801) < 5e-4
    assert abs(wyniki[-1] - 0.8801) < 5e-4
    assert abs(min(wyniki) - 0.6580) < 5e-4
    print(f"OK: BLEU dla jednego bledu waha sie od {min(wyniki):.3f} "
          f"(srodek) do {max(wyniki):.3f} (skraj)")


def test_tozsamosc_f_score_i_rouge():
    """F-score przy koszcie 2 musi być tożsamy z ROUGE-L."""
    import random
    rng = random.Random(0)
    for _ in range(2000):
        a = [rng.randrange(20) for _ in range(rng.randrange(0, 9))]
        b = [rng.randrange(20) for _ in range(rng.randrange(0, 9))]
        if not a or not b:
            continue
        assert abs(levenshtein_f_score(a, b) - rouge_l_score(a, b)) < 1e-9
    print("OK: F-score (koszt 2) tozsamy z ROUGE-L na 2000 par losowych")


def test_wlasnosci_podstawowe():
    assert levenshtein_similarity([], []) == 1.0
    assert levenshtein_similarity([1, 2], []) == 0.0
    assert rouge_l_score([1, 2], []) == 0.0
    assert bleu_score([1, 2, 3], []) == 0.0
    assert lcs_length([1, 2, 3], [3, 2, 1]) == 1
    assert lcs_length([1, 2, 3, 4], [1, 3, 4]) == 3
    assert levenshtein_distance([1, 2, 3], [1, 2, 3]) == 0
    assert levenshtein_distance([1, 2, 3], [1, 4, 3]) == 1
    assert levenshtein_distance([1, 2, 3], [1, 4, 3], 2) == 2

    # symetria odległości
    import random
    rng = random.Random(1)
    for _ in range(500):
        a = [rng.randrange(8) for _ in range(rng.randrange(1, 8))]
        b = [rng.randrange(8) for _ in range(rng.randrange(1, 8))]
        assert levenshtein_distance(a, b) == levenshtein_distance(b, a)

    # nierówność trójkąta dla surowej odległości
    for _ in range(500):
        a = [rng.randrange(5) for _ in range(rng.randrange(1, 6))]
        b = [rng.randrange(5) for _ in range(rng.randrange(1, 6))]
        c = [rng.randrange(5) for _ in range(rng.randrange(1, 6))]
        assert (levenshtein_distance(a, c)
                <= levenshtein_distance(a, b) + levenshtein_distance(b, c))
    print("OK: wlasnosci podstawowe i nierownosc trojkata dla d")


def test_kontrprzyklad_nierownosc_trojkata_dla_s():
    """s_Lev nie jest metryką: pokazujemy kontrprzykład."""
    import random
    rng = random.Random(2)
    znaleziony = None
    for _ in range(200000):
        a = [rng.randrange(3) for _ in range(rng.randrange(1, 5))]
        b = [rng.randrange(3) for _ in range(rng.randrange(1, 5))]
        c = [rng.randrange(3) for _ in range(rng.randrange(1, 5))]
        dac = 1 - levenshtein_similarity(a, c)
        dab = 1 - levenshtein_similarity(a, b)
        dbc = 1 - levenshtein_similarity(b, c)
        if dac > dab + dbc + 1e-12:
            znaleziony = (a, b, c, dac, dab + dbc)
            break
    assert znaleziony is not None
    a, b, c, lewa, prawa = znaleziony
    print(f"OK: kontrprzyklad dla s_Lev: X={a} Y={b} Z={c}, "
          f"{lewa:.4f} > {prawa:.4f}")


def test_miara_docelowa():
    # scena z duplikatem obiektu
    ref = [5, 7, 5, 9]
    cand = [5, 7, 9]
    s = content_scores(ref, cand)
    assert s["restored"] == 3, s
    assert s["missing"] == 1, s
    assert s["hallucinated"] == 0, s

    # kolejność odwrócona: zawartość idealna, porządek najgorszy
    ref = [1, 2, 3, 4]
    cand = [4, 3, 2, 1]
    s = target_measure(ref, cand)
    assert s["content_f1"] == 1.0
    assert s["order_tau"] == 0.0
    assert s["position_accuracy"] == 0.0
    assert s["exact_match"] == 0.0

    # zgodność idealna
    s = target_measure([1, 2, 3], [1, 2, 3])
    assert s["order_tau"] == 1.0 and s["exact_match"] == 1.0

    # tau niezależne od halucynacji: dokładanie obcych obiektów
    assert kendall_tau_order([1, 2, 3], [1, 99, 2, 98, 3]) == 1.0
    assert position_accuracy([1, 2, 3], [1, 2]) == 2 / 3
    print("OK: miara docelowa rozdziela zawartosc od uporzadkowania")


if __name__ == "__main__":
    test_tabela_2_2()
    test_wrazliwosc_bleu_na_pozycje_bledu()
    test_tozsamosc_f_score_i_rouge()
    test_wlasnosci_podstawowe()
    test_kontrprzyklad_nierownosc_trojkata_dla_s()
    test_miara_docelowa()
    print("\nWSZYSTKIE TESTY METRYK PRZESZLY")
