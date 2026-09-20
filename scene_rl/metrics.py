"""Metryki podobieństwa sekwencji oraz niezależna miara docelowa.

Moduł zawiera wyłącznie implementacje własne, bez zależności od
bibliotek zewnętrznych. Każda funkcja przyjmuje dwie sekwencje
tokenów całkowitoliczbowych: `reference` (scena oryginalna) oraz
`candidate` (scena zrekonstruowana przez model), i zwraca wartość
z przedziału [0, 1], gdzie 1 oznacza zgodność pełną.

Rozdzielono tu dwie grupy miar:

1. Metryki podobieństwa (BLEU, ROUGE-L, podobieństwo edycyjne),
   używane w etapie drugim jako funkcja nagrody algorytmu PPO.
2. Miara docelowa (`target_measure`), niezależna od powyższych,
   służąca do porównywania modeli. Rozdzielenie jest konieczne,
   ponieważ model dostrajany nagrodą BLEU wygrałby w BLEU z samej
   definicji, co czyniłoby porównanie cyrkularnym.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

Seq = Sequence[int]


# ---------------------------------------------------------------
# 1. Odległość Levenshteina i znormalizowane podobieństwo edycyjne
# ---------------------------------------------------------------

def levenshtein_distance(
    reference: Seq,
    candidate: Seq,
    substitution_cost: int = 1,
) -> int:
    """Odległość edycyjna liczona programowaniem dynamicznym.

    Zwraca minimalny łączny koszt operacji wstawienia, usunięcia
    i substytucji przekształcających `reference` w `candidate`.
    Koszt wstawienia i usunięcia wynosi 1, koszt substytucji jest
    parametrem: wartość 1 odpowiada wariantowi przyjętemu w pracy,
    wartość 2 wariantowi tożsamemu z ROUGE-L (patrz podrozdz. 2.5.3).

    Złożoność czasowa i pamięciowa wynosi O(n*m). Implementacja
    używa dwóch wierszy macierzy zamiast pełnej tablicy, ponieważ
    wiersz i-ty zależy wyłącznie od wiersza i-1.
    """
    n, m = len(reference), len(candidate)
    if n == 0:
        return m
    if m == 0:
        return n

    previous = list(range(m + 1))
    current = [0] * (m + 1)

    for i in range(1, n + 1):
        current[0] = i
        ref_symbol = reference[i - 1]
        for j in range(1, m + 1):
            if ref_symbol == candidate[j - 1]:
                current[j] = previous[j - 1]
            else:
                current[j] = min(
                    previous[j] + 1,                     # usunięcie
                    current[j - 1] + 1,                  # wstawienie
                    previous[j - 1] + substitution_cost,  # substytucja
                )
        previous, current = current, previous

    return previous[m]


def levenshtein_similarity(reference: Seq, candidate: Seq) -> float:
    """Znormalizowane podobieństwo edycyjne s_Lev ze wzoru (2.8).

    Definicja: 1 - d(X, Y) / max(|X|, |Y|), przy jednostkowym
    koszcie substytucji. Wartość 1 oznacza sekwencje identyczne,
    wartość 0 brak jakiegokolwiek dopasowania.

    Uwaga: miara ta NIE jest metryką w sensie przestrzeni
    metrycznej, ponieważ nie spełnia nierówności trójkąta.
    Nierówność trójkąta zachowuje wyłącznie surowa odległość d.
    """
    n, m = len(reference), len(candidate)
    max_len = max(n, m)
    if max_len == 0:
        return 1.0
    distance = levenshtein_distance(reference, candidate, 1)
    return 1.0 - distance / max_len


def levenshtein_f_score(reference: Seq, candidate: Seq) -> float:
    """Wariant F-score przy koszcie substytucji 2 (wzór 2.11).

    Funkcja zachowana wyłącznie w celu udokumentowania tożsamości
    tego wariantu z ROUGE-L, wykazanej w podrozdziale 2.5.3.
    NIE należy jej używać jako nagrody w eksperymentach, ponieważ
    duplikowałaby ROUGE-L.
    """
    n, m = len(reference), len(candidate)
    if n == 0 and m == 0:
        return 1.0
    if n + m == 0:
        return 0.0
    distance = levenshtein_distance(reference, candidate, 2)
    matches = (n + m - distance) / 2.0
    return 2.0 * matches / (n + m)


# ---------------------------------------------------------------
# 2. ROUGE-L
# ---------------------------------------------------------------

def lcs_length(a: Seq, b: Seq) -> int:
    """Długość najdłuższego wspólnego podciągu (LCS).

    Podciąg nie musi być spójny, ale musi zachowywać kolejność
    występowania elementów w obu sekwencjach. Implementacja
    używa programowania dynamicznego z dwoma wierszami.
    """
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0

    previous = [0] * (m + 1)
    current = [0] * (m + 1)

    for i in range(1, n + 1):
        a_symbol = a[i - 1]
        for j in range(1, m + 1):
            if a_symbol == b[j - 1]:
                current[j] = previous[j - 1] + 1
            else:
                current[j] = max(previous[j], current[j - 1])
        previous, current = current, [0] * (m + 1)

    return previous[m]


def rouge_l_score(
    reference: Seq,
    candidate: Seq,
    beta: float = 1.0,
) -> float:
    """ROUGE-L w wariancie F-score o współczynniku beta.

    Czułość i precyzja liczone są na podstawie długości LCS:
    R = LCS / |X|, P = LCS / |Y|. Dla beta = 1 miara sprowadza
    się do średniej harmonicznej, czyli 2*LCS / (|X| + |Y|).
    """
    n, m = len(reference), len(candidate)
    if n == 0 or m == 0:
        return 0.0

    lcs = lcs_length(reference, candidate)
    if lcs == 0:
        return 0.0

    recall = lcs / n
    precision = lcs / m
    beta_sq = beta * beta
    numerator = (1.0 + beta_sq) * recall * precision
    denominator = recall + beta_sq * precision
    return numerator / denominator


# ---------------------------------------------------------------
# 3. BLEU
# ---------------------------------------------------------------

def _ngram_counts(sequence: Seq, order: int) -> Counter:
    """Zlicza n-gramy rzędu `order` w sekwencji."""
    counts: Counter = Counter()
    for i in range(len(sequence) - order + 1):
        counts[tuple(sequence[i:i + order])] += 1
    return counts


def modified_precision(
    reference: Seq,
    candidate: Seq,
    order: int,
) -> tuple[int, int]:
    """Licznik i mianownik zmodyfikowanej precyzji n-gramowej.

    Zwraca parę (liczba trafień, liczba n-gramów kandydata).
    Obcięcie liczby trafień do liczby wystąpień danego n-gramu
    w referencji zapobiega sztucznemu zawyżaniu wyniku przez
    wielokrotne powtarzanie tego samego elementu.
    """
    candidate_counts = _ngram_counts(candidate, order)
    total = sum(candidate_counts.values())
    if total == 0:
        return 0, 0
    reference_counts = _ngram_counts(reference, order)
    matches = 0
    for ngram, count in candidate_counts.items():
        matches += min(count, reference_counts.get(ngram, 0))
    return matches, total


def brevity_penalty(reference_len: int, candidate_len: int) -> float:
    """Kara za zwięzłość BP ze wzoru (2.4)."""
    if candidate_len == 0:
        return 0.0
    if candidate_len > reference_len:
        return 1.0
    return math.exp(1.0 - reference_len / candidate_len)


def bleu_score(
    reference: Seq,
    candidate: Seq,
    max_order: int = 4,
    weights: Sequence[float] | None = None,
    smoothing: str = "none",
) -> float:
    """BLEU dla pojedynczej pary sekwencji, implementacja własna.

    Parametry:
        max_order: najwyższy rząd n-gramów (domyślnie 4).
        weights:   wagi w_n, domyślnie jednakowe 1/max_order.
        smoothing: "none" (bez wygładzania, wariant użyty
                   w tabeli 2.2) albo "add-one" (odpowiednik
                   metody 1 z biblioteki NLTK, w której do
                   licznika i mianownika precyzji rzędu n > 1
                   dodaje się jedynkę).

    Bez wygładzania zerowa precyzja dowolnego rzędu zeruje całą
    wartość BLEU, ponieważ logarytm zera nie jest określony.
    Zjawisko to występuje często dla sekwencji krótkich i jest
    głównym powodem, dla którego w treningu PPO należy używać
    wariantu wygładzanego.
    """
    if weights is None:
        weights = [1.0 / max_order] * max_order
    if len(weights) != max_order:
        raise ValueError("Liczba wag musi być równa max_order.")

    n, m = len(reference), len(candidate)
    if n == 0 or m == 0:
        return 0.0

    log_precision_sum = 0.0
    for order in range(1, max_order + 1):
        matches, total = modified_precision(reference, candidate, order)
        if smoothing == "add-one" and order > 1:
            matches += 1
            total += 1
        if total == 0:
            return 0.0
        if matches == 0:
            return 0.0
        log_precision_sum += weights[order - 1] * math.log(matches / total)

    return brevity_penalty(n, m) * math.exp(log_precision_sum)


# ---------------------------------------------------------------
# 4. Rejestr metryk używanych jako funkcja nagrody
# ---------------------------------------------------------------

def bleu_reward(reference: Seq, candidate: Seq) -> float:
    """BLEU z wygładzaniem, wariant przeznaczony do nagrody PPO."""
    return bleu_score(reference, candidate, smoothing="add-one")


REWARD_METRICS = {
    "bleu": bleu_reward,
    "rouge_l": rouge_l_score,
    "levenshtein": levenshtein_similarity,
}

ALL_METRICS = {
    "bleu": lambda r, c: bleu_score(r, c, smoothing="none"),
    "bleu_smooth": bleu_reward,
    "rouge_l": rouge_l_score,
    "levenshtein": levenshtein_similarity,
}


# ---------------------------------------------------------------
# 5. Miara docelowa, niezależna od metryk użytych jako nagroda
# ---------------------------------------------------------------

def content_scores(reference: Seq, candidate: Seq) -> dict[str, float]:
    """Zgodność zawartości sceny z pominięciem kolejności.

    Porównanie prowadzone jest na wielozbiorach, a nie na zbiorach,
    ponieważ w zbiorze danych 21 scen zawiera ten sam obiekt
    dwukrotnie. Użycie zbioru zliczałoby taki obiekt jeden raz
    i zaniżałoby liczbę obiektów brakujących.

    Zwraca liczby obiektów odtworzonych, brakujących
    i halucynowanych oraz odpowiadające im precyzję, czułość i F1.
    """
    reference_counts = Counter(reference)
    candidate_counts = Counter(candidate)
    common = reference_counts & candidate_counts

    restored = sum(common.values())
    missing = len(reference) - restored
    hallucinated = len(candidate) - restored

    precision = restored / len(candidate) if candidate else 0.0
    recall = restored / len(reference) if reference else 0.0
    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)

    return {
        "restored": float(restored),
        "missing": float(missing),
        "hallucinated": float(hallucinated),
        "content_precision": precision,
        "content_recall": recall,
        "content_f1": f1,
    }


def kendall_tau_order(reference: Seq, candidate: Seq) -> float:
    """Zgodność uporządkowania mierzona współczynnikiem tau.

    Miara liczona jest wyłącznie na obiektach obecnych w obu
    sekwencjach, dzięki czemu jest niezależna od zgodności
    zawartości mierzonej funkcją `content_scores`. Rozdzielenie
    obu aspektów jest wymagane przez pytanie badawcze dotyczące
    tego, czy poprawa dotyczy doboru obiektów czy ich kolejności.

    Dla każdej pary wspólnych obiektów sprawdza się, czy ich
    wzajemna kolejność jest w obu sekwencjach taka sama. Wynik
    normalizowany jest do przedziału [0, 1], gdzie 1 oznacza
    porządek identyczny, a 0 porządek dokładnie odwrócony.
    Wartość 1.0 zwracana jest, gdy wspólnych obiektów jest mniej
    niż dwa, ponieważ pojedynczy obiekt nie definiuje porządku.
    """
    reference_positions: dict[int, int] = {}
    for index, token in enumerate(reference):
        reference_positions.setdefault(token, index)

    seen: set[int] = set()
    common_order: list[int] = []
    for token in candidate:
        if token in reference_positions and token not in seen:
            seen.add(token)
            common_order.append(reference_positions[token])

    k = len(common_order)
    if k < 2:
        return 1.0

    concordant = 0
    total_pairs = k * (k - 1) // 2
    for i in range(k):
        for j in range(i + 1, k):
            if common_order[i] < common_order[j]:
                concordant += 1

    tau = 2.0 * concordant / total_pairs - 1.0
    return (tau + 1.0) / 2.0


def position_accuracy(reference: Seq, candidate: Seq) -> float:
    """Udział pozycji, na których oba ciągi mają ten sam token."""
    if not reference:
        return 1.0 if not candidate else 0.0
    hits = sum(
        1
        for i in range(min(len(reference), len(candidate)))
        if reference[i] == candidate[i]
    )
    return hits / len(reference)


def target_measure(reference: Seq, candidate: Seq) -> dict[str, float]:
    """Pełna miara docelowa użyta do porównywania modeli.

    Łączy zgodność zawartości, zgodność uporządkowania, zgodność
    pozycyjną, zgodność długości oraz wskaźnik rekonstrukcji
    dokładnej. Żaden z tych składników nie jest równy ani
    monotonicznie zależny od BLEU, ROUGE-L czy podobieństwa
    edycyjnego, dzięki czemu porównanie modeli dostrajanych
    różnymi nagrodami pozostaje uczciwe.
    """
    scores = content_scores(reference, candidate)
    scores["order_tau"] = kendall_tau_order(reference, candidate)
    scores["position_accuracy"] = position_accuracy(reference, candidate)
    scores["length_error"] = float(len(candidate) - len(reference))
    scores["exact_match"] = float(list(reference) == list(candidate))
    return scores


TARGET_MEASURE_KEYS = (
    "restored",
    "missing",
    "hallucinated",
    "content_precision",
    "content_recall",
    "content_f1",
    "order_tau",
    "position_accuracy",
    "length_error",
    "exact_match",
)
