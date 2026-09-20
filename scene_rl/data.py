"""Wczytanie zbioru scen, tokenizacja i budowa przykładów uczących.

Moduł jest niezależny od biblioteki PyTorch. Zwraca tablice numpy,
dzięki czemu cała logika przygotowania danych daje się testować bez
uruchamiania sieci.

Zmiany względem wersji notatnikowej, wymuszone uwagami promotora:

1. Przykładem uczącym jest kompletny łańcuch rekonstrukcji jednej
   sceny, a nie losowe okno wycięte ze sklejonego ciągu wszystkich
   scen. Okno rozpoczyna się zawsze na granicy podsekwencji, czyli
   tuż za tokenem SEP albo na początku łańcucha.
2. Przykład krótszy od okna uzupełniany jest tokenem wypełnienia
   PAD, pomijanym przy obliczaniu funkcji straty. Model nie uczy się
   więc rekonstrukcji urwanych.
3. Wprowadzono trzeci token specjalny PAD o indeksie 2. Tokeny
   specjalne nadal zajmują najniższe indeksy słownika, więc ich
   numery nie zależą od liczby obiektów w zbiorze danych.
"""

from __future__ import annotations

import itertools
import random
from collections import Counter
from dataclasses import dataclass

import numpy as np

# Tokeny specjalne. Muszą zajmować najniższe indeksy słownika, aby
# ich numeracja była niezależna od liczby obiektów w zbiorze danych.
STOP = 0   # koniec procesu rekonstrukcji
SEP = 1    # granica między kolejnymi przybliżeniami sceny
PAD = 2    # wypełnienie, pomijane w funkcji straty
NUM_SPECIAL_TOKENS = 3

# Wartość ignorowana przez funkcję straty entropii krzyżowej.
IGNORE_INDEX = -100


@dataclass
class AugmentationConfig:
    """Parametry generowania łańcucha rekonstrukcji."""

    min_elements: int = 3
    max_remove_per_step: int = 1
    swap_probability: float = 0.15


class SceneDataset:
    """Zbiór scen wraz ze słownikiem tokenów.

    Scena jest sekwencją uporządkowaną, a nie zbiorem. Kolejność
    obiektów niesie informację i podlega rekonstrukcji na równi
    z samym doborem obiektów.
    """

    def __init__(self, path: str):
        raw = np.loadtxt(path, dtype=np.int64)
        if raw.ndim != 2:
            raise ValueError("Plik danych musi mieć układ tabelaryczny.")

        self.path = path
        self.raw = raw
        self.num_scenes, self.scene_length = raw.shape

        unique_ids = np.unique(raw)
        self.unique_object_ids = unique_ids

        # stoi: oryginalny identyfikator obiektu -> token sieci.
        # Przenumerowanie jest konieczne, ponieważ neuron wyjściowy
        # o indeksie i musi odpowiadać dokładnie i-temu tokenowi
        # słownika. Luki w numeracji identyfikatorów prowadziłyby do
        # błędnego dekodowania predykcji.
        self.stoi = {
            int(object_id): index + NUM_SPECIAL_TOKENS
            for index, object_id in enumerate(unique_ids)
        }
        self.itos = {token: obj for obj, token in self.stoi.items()}

        self.vocab_size = len(self.stoi) + NUM_SPECIAL_TOKENS
        self.scenes = [
            [self.stoi[int(value)] for value in row] for row in raw
        ]

    # -----------------------------------------------------------
    # Statystyki zbioru, wykorzystywane w rozdziale 3
    # -----------------------------------------------------------

    def statistics(self) -> dict:
        """Podstawowa charakterystyka zbioru danych."""
        counts = Counter(int(v) for v in self.raw.ravel())
        scenes_with_duplicate = sum(
            1 for row in self.raw if len(set(row.tolist())) < self.scene_length
        )
        sorted_scenes = sum(
            1 for row in self.raw
            if list(row) == sorted(row.tolist())
        )
        return {
            "num_scenes": self.num_scenes,
            "scene_length": self.scene_length,
            "num_unique_objects": len(self.unique_object_ids),
            "min_object_id": int(self.raw.min()),
            "max_object_id": int(self.raw.max()),
            "mean_occurrences_per_object": float(np.mean(list(counts.values()))),
            "scenes_with_duplicate": scenes_with_duplicate,
            "sorted_scenes": sorted_scenes,
            "vocab_size": self.vocab_size,
        }

    def cooccurrence_statistics(self) -> dict:
        """Sprawdza, czy pary obiektów współwystępują nielosowo.

        Brak struktury statystycznej oznacza, że nie istnieje reguła
        postaci "obiekt A występuje z obiektem B", którą model mógłby
        uogólnić na sceny niewidziane. Zadanie sprowadza się wówczas
        do pamięci asocjacyjnej adresowanej treścią.
        """
        pair_counts: Counter = Counter()
        for row in self.raw:
            for pair in itertools.combinations(sorted(set(row.tolist())), 2):
                pair_counts[pair] += 1
        total_pairs = sum(pair_counts.values())
        repeated_pairs = sum(1 for c in pair_counts.values() if c > 1)
        return {
            "distinct_pairs": len(pair_counts),
            "total_pair_occurrences": total_pairs,
            "pairs_occurring_more_than_once": repeated_pairs,
            "max_pair_count": max(pair_counts.values()) if pair_counts else 0,
        }

    def context_ambiguity(self, context_size: int) -> dict:
        """Udział fragmentów, które nie identyfikują sceny jednoznacznie.

        Wyznacza dolną granicę sensownej długości kontekstu. Fragment
        zbyt krótki pasuje do wielu scen naraz, więc zadanie staje się
        nierozwiązywalne niezależnie od jakości modelu.
        """
        subset_counts: Counter = Counter()
        for row in self.raw:
            objects = sorted(set(row.tolist()))
            for subset in itertools.combinations(objects, context_size):
                subset_counts[subset] += 1
        total = sum(subset_counts.values())
        ambiguous = sum(c for c in subset_counts.values() if c > 1)
        return {
            "context_size": context_size,
            "ambiguous_fraction": ambiguous / total if total else 0.0,
            "distinct_contexts": len(subset_counts),
        }


# ---------------------------------------------------------------
# Generowanie łańcucha rekonstrukcji ("piramidki")
# ---------------------------------------------------------------

def remove_random_elements(
    sequence: list[int],
    how_many: int,
    rng: random.Random,
) -> list[int]:
    """Usuwa `how_many` losowych elementów, zachowując kolejność."""
    if how_many <= 0:
        return list(sequence)
    indices = set(rng.sample(range(len(sequence)), how_many))
    return [v for i, v in enumerate(sequence) if i not in indices]


def build_chain(
    scene: list[int],
    rng: random.Random,
    config: AugmentationConfig,
) -> list[list[int]]:
    """Buduje łańcuch wersji sceny, od najkrótszej do pełnej.

    Wersja najkrótsza pełni rolę fragmentu wejściowego, a wersja
    ostatnia jest pełną sceną docelową. Zamiana miejscami dwóch
    elementów, wykonywana z prawdopodobieństwem `swap_probability`,
    uczy model korygowania kolejności zamiast wiązania rekonstrukcji
    z jednym tylko uporządkowaniem wejścia.

    Zamiana nigdy nie dotyka wersji pełnej, ponieważ ta stanowi cel
    rekonstrukcji i musi zachować kolejność oryginalną.
    """
    versions = [list(scene)]
    current = list(scene)

    while len(current) > config.min_elements:
        how_many = rng.randint(1, config.max_remove_per_step)
        if len(current) - how_many < config.min_elements:
            how_many = len(current) - config.min_elements

        current = remove_random_elements(current, how_many, rng)

        if len(current) >= 2 and rng.random() < config.swap_probability:
            i = rng.randrange(len(current))
            j = rng.randrange(len(current))
            current[i], current[j] = current[j], current[i]

        versions.append(list(current))

    versions.reverse()
    return versions


def encode_chain(versions: list[list[int]]) -> list[int]:
    """Zamienia łańcuch wersji na płaską sekwencję tokenów.

    Format: fragment SEP wersja SEP ... SEP pełna_scena STOP.
    """
    tokens: list[int] = []
    for index, version in enumerate(versions):
        if index > 0:
            tokens.append(SEP)
        tokens.extend(version)
    tokens.append(STOP)
    return tokens


def chain_start_positions(tokens: list[int]) -> list[int]:
    """Pozycje, od których wolno rozpocząć okno kontekstowe.

    Są to: początek łańcucha oraz każda pozycja tuż za tokenem SEP.
    Dzięki temu okno nigdy nie zaczyna się w środku podsekwencji,
    co było zarzutem wobec poprzedniej wersji funkcji get_batch.
    """
    positions = [0]
    for index, token in enumerate(tokens):
        if token == SEP and index + 1 < len(tokens):
            positions.append(index + 1)
    return positions


def required_block_size(
    dataset: SceneDataset,
    config: AugmentationConfig,
    num_draws: int = 200,
    seed: int = 0,
) -> int:
    """Empirycznie wyznacza minimalną wystarczającą długość okna.

    Wartość `block_size` musi przekraczać długość całego procesu
    rekonstrukcji jednej sceny, a nie długość samej sceny. Funkcja
    zwraca maksymalną zaobserwowaną długość łańcucha powiększoną
    o jeden, ponieważ para wejście-cel wymaga o jeden token więcej.
    """
    rng = random.Random(seed)
    longest = 0
    for _ in range(num_draws):
        scene = dataset.scenes[rng.randrange(dataset.num_scenes)]
        tokens = encode_chain(build_chain(scene, rng, config))
        longest = max(longest, len(tokens))
    return longest + 1


# ---------------------------------------------------------------
# Budowa partii treningowych
# ---------------------------------------------------------------

class ExampleSampler:
    """Losuje partie przykładów uczących o stałej długości okna.

    Podział na zbiór uczący i walidacyjny przebiega po przestrzeni
    KONTEKSTÓW, a nie po przestrzeni scen. Rozłączny podział scen
    byłby błędem metodologicznym, ponieważ scena spoza zbioru
    uczącego jest z definicji nieodtwarzalna: zbiór danych nie ma
    struktury statystycznej pozwalającej ją uogólnić. Zbiór
    walidacyjny stanowią zatem nowe losowe fragmenty tych samych
    scen, generowane niezależnym strumieniem liczb losowych.
    """

    def __init__(
        self,
        dataset: SceneDataset,
        block_size: int,
        config: AugmentationConfig,
        seed: int,
        loss_on_prompt: bool = False,
        start_mode: str = "sep",
    ):
        if start_mode not in ("sep", "chain"):
            raise ValueError("start_mode musi być 'sep' albo 'chain'.")
        self.dataset = dataset
        self.block_size = block_size
        self.config = config
        self.rng = random.Random(seed)
        self.loss_on_prompt = loss_on_prompt
        self.start_mode = start_mode
        self.truncated_examples = 0

    def _one_example(self) -> tuple[list[int], list[int]]:
        """Zwraca parę (wejście, cel) dla jednego przykładu."""
        scene = self.dataset.scenes[self.rng.randrange(self.dataset.num_scenes)]
        tokens = encode_chain(build_chain(scene, self.rng, self.config))

        if self.start_mode == "sep":
            starts = chain_start_positions(tokens)
            start = starts[self.rng.randrange(len(starts))]
        else:
            start = 0
        window = tokens[start:]

        if len(window) > self.block_size + 1:
            # Nie powinno wystąpić przy poprawnie dobranym block_size.
            self.truncated_examples += 1
            window = window[: self.block_size + 1]

        x = window[:-1]
        y = window[1:]

        if not self.loss_on_prompt:
            # Maskujemy część odpowiadającą fragmentowi wejściowemu.
            # Przewidywanie tokenów samego fragmentu nie jest zadaniem
            # modelu: identyfikatory obiektów są losowe, więc składnik
            # ten jest nieuczalny i jedynie zaszumia funkcję straty.
            #
            # Pierwszą pozycją uczącą jest ta, na której model ma
            # wygenerować token SEP, ponieważ decyzja o zakończeniu
            # fragmentu i rozpoczęciu rekonstrukcji należy do modelu.
            # Gdy okno nie zawiera separatora, jedynym celem jest STOP.
            first_sep = x.index(SEP) if SEP in x else len(x)
            mask_until = max(0, first_sep - 1)
            y = [IGNORE_INDEX] * mask_until + y[mask_until:]

        pad_length = self.block_size - len(x)
        x = x + [PAD] * pad_length
        y = y + [IGNORE_INDEX] * pad_length
        return x, y

    def batch(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        """Zwraca parę tablic o kształcie (batch_size, block_size)."""
        xs = np.empty((batch_size, self.block_size), dtype=np.int64)
        ys = np.empty((batch_size, self.block_size), dtype=np.int64)
        for row in range(batch_size):
            x, y = self._one_example()
            xs[row] = x
            ys[row] = y
        return xs, ys


def extract_restored_scene(
    tokens: list[int],
    prompt_length: int = 0,
) -> list[int] | None:
    """Wycina finalną wersję sceny z odpowiedzi modelu.

    Odpowiedź ma postać: fragment SEP wersja SEP ... SEP pełna STOP.
    Sceną odzyskaną jest to, co znajduje się między ostatnim
    separatorem a tokenem STOP. Zwraca None, gdy model nie wygenerował
    tokenu STOP, co traktowane jest jako epizod nieudany.

    Argument `prompt_length` pozwala odciąć fragment wejściowy. Bez
    tego, gdy model nie wygeneruje ani jednego separatora, fragment
    wejściowy zostałby błędnie potraktowany jako część rekonstrukcji.
    """
    tokens = tokens[prompt_length:]
    if STOP not in tokens:
        return None

    stop_position = tokens.index(STOP)

    last_sep = -1
    for position in range(stop_position):
        if tokens[position] == SEP:
            last_sep = position

    restored = tokens[last_sep + 1: stop_position]
    return [token for token in restored if token not in (PAD, SEP)]


def make_prompt(
    scene: list[int],
    prompt_length: int,
    rng: random.Random,
) -> list[int]:
    """Buduje fragment wejściowy przez usunięcie losowych obiektów.

    Kolejność pozostałych obiektów jest zachowana, ponieważ fragment
    ma być podciągiem sceny, a nie jej dowolną permutacją.
    """
    to_remove = len(scene) - prompt_length
    if to_remove <= 0:
        return list(scene)
    return remove_random_elements(scene, to_remove, rng)
