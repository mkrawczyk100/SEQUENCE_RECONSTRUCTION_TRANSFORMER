# Rekonstrukcja sekwencji transformerem z uczeniem ze wzmocnieniem

Kod źródłowy i dane do pracy magisterskiej „Rekonstrukcja sekwencji za pomocą
sztucznej sieci neuronowej o architekturze transformera".

- Autor: Mateusz Krawczyk
- Promotor: dr inż. Marcin Kowalik
- Politechnika Rzeszowska im. Ignacego Łukasiewicza, Wydział Matematyki
  i Fizyki Stosowanej, kierunek Inżynieria i analiza danych, 2026

## O zadaniu

Zadanie ma charakter pamięci asocjacyjnej adresowanej treścią. Model otrzymuje
niepełny fragment sceny, czyli uporządkowanej sekwencji dziesięciu
identyfikatorów obiektów, i ma odtworzyć całą zapamiętaną scenę wraz
z kolejnością obiektów.

Model, transformer typu decoder-only, uczony jest w dwóch etapach.
Etap pierwszy to uczenie nadzorowane. Etap drugi to dostrajanie algorytmem PPO,
w którym nagrodą jest jedna z trzech metryk podobieństwa sekwencji (BLEU,
ROUGE-L albo podobieństwo edycyjne Levenshteina). Jakość modeli ocenia osobna
miara docelowa, niezależna od metryk użytych jako nagroda.

## Wymagania i instalacja

- Python 3.10 lub nowszy
- PyTorch 2.0 lub nowszy, NumPy
- Matplotlib (tylko do wykresów)
- nltk, rouge-score, python-Levenshtein (tylko do porównania własnych
  implementacji metryk z bibliotecznymi w poleceniu `metrics-table`)

```bash
pip install -r requirements.txt
```

Plik `requirements.txt` obejmuje wszystkie powyższe biblioteki. Bez bibliotek
porównawczych polecenie `metrics-table` pomija kolumny porównawcze, a reszta
kodu działa bez zmian.

Wszystkie polecenia uruchamia się z katalogu głównego repozytorium. Przykłady
podano w składni powłoki bash. W PowerShellu znak kontynuacji linii `\` należy
zastąpić znakiem `` ` ``.

## Struktura repozytorium

```
scene_rl/                     pakiet z implementacją modelu, treningu i metryk
tests/                        testy jednostkowe (nie wymagają PyTorcha)
dane_wynikowe/                dane liczbowe leżące u podstaw tabel i rysunków
sequences_przemek.txt         zbiór danych, 500 scen po 10 obiektów
requirements.txt              wymagane biblioteki
uruchom_eksperyment.sh        pełna sekwencja poleceń odtwarzająca eksperyment
uruchom_ppo_wieloziarnowo.py  przebiegi PPO dla siatki metryk nagrody i ziaren
zbierz_krzywe_ppo.py          uśrednianie krzywych PPO po ziarnach, test wypłaszczenia
tabela_koncowa_ppo.py         ocena checkpointów w mierze docelowej, tabele końcowe
wybor_modelu_ppo.py           zachłanna ocena wybranych checkpointów uczenia nadzorowanego
```

W tekście pracy zbiór danych nazwany jest `sequences.txt`. W repozytorium
plik nazywa się `sequences_przemek.txt` i pod tą nazwą zapisany jest
w konfiguracji każdego checkpointu, dlatego nazwy pliku nie należy zmieniać.

### Pakiet `scene_rl`

| Plik | Zawartość |
|---|---|
| `config.py` | hiperparametry w postaci struktur danych, zapisywane w checkpointach |
| `data.py` | wczytanie scen, tokenizacja, łańcuch rekonstrukcji, partie danych |
| `metrics.py` | BLEU, ROUGE-L, podobieństwo edycyjne, miara docelowa |
| `model.py` | transformer decoder-only, głowica wartości, generowanie |
| `sft.py` | etap pierwszy, uczenie nadzorowane |
| `ppo.py` | etap drugi, dostrajanie algorytmem PPO |
| `evaluation.py` | rekonstrukcja wsadowa i ocena jakości |
| `capacity.py` | pojemność modelu wobec zawartości informacyjnej danych |
| `utils.py` | ziarna losowości, checkpointy, logowanie historii |
| `selftest.py` | autotesty poprawności implementacji |
| `__main__.py` | interfejs wiersza poleceń |

Moduły `config`, `data`, `metrics` i `capacity` nie importują PyTorcha.

## Odtworzenie wyników pracy

Cały eksperyment w kolejności z pracy uruchamia skrypt:

```bash
bash uruchom_eksperyment.sh
```

Poniżej te same kroki opisano osobno. Podane czasy zmierzono na karcie
NVIDIA GeForce RTX 4050 Laptop GPU.

### Krok 1. Autotesty, analiza danych, tabela metryk

```bash
python -m scene_rl selftest --data sequences_przemek.txt
python -m scene_rl analyze --data sequences_przemek.txt    # rozdział 3
python -m scene_rl metrics-table                           # podrozdział 2.5
```

### Krok 2. Uczenie nadzorowane trzech architektur

Trzy modele różnią się wyłącznie parametrami architektury. Pozostałe
hiperparametry i ziarna losowości są identyczne (podrozdział 4.3, Tabela 4.4).
Jeden przebieg modelu e96 trwa około 24 minut. Model e180 zapisywany jest
w katalogu `runs/sft`.

```bash
python -m scene_rl train-sft --data sequences_przemek.txt \
    --n-embd 64  --n-head 4 --n-layer 4 --seed 1234 --out runs/sft_e64
python -m scene_rl train-sft --data sequences_przemek.txt \
    --n-embd 96  --n-head 4 --n-layer 4 --seed 1234 --out runs/sft_e96
python -m scene_rl train-sft --data sequences_przemek.txt \
    --n-embd 180 --n-head 6 --n-layer 6 --seed 1234 --out runs/sft
```

Checkpoint każdej epoki zapisywany jest jako `epoch_XXXX.pt`, a checkpoint
o najniższej stracie walidacyjnej jako `best.pt`. Punktem startowym dostrajania
jest celowo niedouczony checkpoint `runs/sft_e96/epoch_0004.pt`
(podrozdział 4.5).

Trajektorię modelu e96 w kolejnych epokach oraz jego wynik po pełnym treningu
(Tabela 5.4, Rys. 5.6) wyznacza skrypt `tabela_koncowa_ppo.py`, opisany
w kroku 7. Każda epoka uczenia nadzorowanego trwa około 24 sekund.

```bash
python tabela_koncowa_ppo.py \
    --checkpoint "SFT e96 epoka 3=runs/sft_e96/epoch_0003.pt" \
    --checkpoint "SFT e96 epoka 4 (start PPO)=runs/sft_e96/epoch_0004.pt" \
    --checkpoint "SFT e96 epoka 5=runs/sft_e96/epoch_0005.pt" \
    --checkpoint "SFT e96 epoka 7=runs/sft_e96/epoch_0007.pt" \
    --checkpoint "SFT e96 epoka 10=runs/sft_e96/epoch_0010.pt" \
    --checkpoint "SFT e96 pelny trening (sufit)=runs/sft_e96/best.pt" \
    --prompt-lengths 3 4 5 6 8 --num-scenes 500 --seed 3 \
    --csv wyniki_odniesienia_z5.csv --json wyniki_odniesienia_z5.json
```

Do przeglądu dowolnych checkpointów przy jednej długości fragmentu służy
`wybor_modelu_ppo.py`. Ocenia on checkpointy dekodowaniem zachłannym na 500
scenach, bez ponownego treningu.

```bash
python wybor_modelu_ppo.py --epoch-range runs/sft_e96:0:11 \
    --checkpoint runs/sft_e96/best.pt --prompt-length 6 --num-scenes 500
```

### Krok 3. Dobór współczynnika kary za rozbieżność

Przebieg wstępny, jedno ziarno, 150 iteracji, ewaluacja co 10 iteracji na 200
scenach (Tabela 4.7, Rys. 4.2).

```bash
for KL in 000:0.0 001:0.01 005:0.05; do
  python uruchom_ppo_wieloziarnowo.py --sft runs/sft_e96/epoch_0004.pt \
      --metrics levenshtein --seeds 1234 --iterations 150 \
      --kl-coef "${KL#*:}" --eval-every 10 --eval-samples 200 \
      --tag "pilotkl${KL%%:*}"
done
```

### Krok 4. Seria główna PPO

Trzy metryki nagrody razy trzy ziarna, czyli dziewięć przebiegów. Łącznie
około 12,7 godziny.

```bash
python uruchom_ppo_wieloziarnowo.py --sft runs/sft_e96/epoch_0004.pt \
    --metrics levenshtein bleu rouge_l --seeds 1234 2025 7 \
    --iterations 1000 --kl-coef 0.01 --eval-every 20 --eval-samples 500 \
    --tag glowny
```

Seria główna uruchamiana jest tym skryptem, a nie poleceniem `train-ppo`,
ponieważ `train-ppo` nie udostępnia parametrów `eval_every` i `eval_samples`
i domyślnie ocenia model próbkowaniem. Skrypt ma własne wartości domyślne
(1500 iteracji, ewaluacja co 10 iteracji na 200 scenach), dlatego wartości
użyte w pracy podano powyżej jawnie.

Pełne parametry przebiegu zapisywane są w pliku `config.json` w katalogu
przebiegu. Dla serii głównej wynoszą one:

| Parametr | Wartość |
|---|---|
| checkpoint startowy | `runs/sft_e96/epoch_0004.pt` |
| iteracje | 1000 |
| epizody na iterację | 256 |
| tempo uczenia | 1e-5 |
| współczynnik kary KL | 0,01 |
| współczynnik entropijny | 0 |
| epoki aktualizacji na buforze, minipartia | 4, 64 |
| zakres obcięcia, współczynnik funkcji wartości | 0,2, 0,5 |
| gamma, lambda GAE | 1,0, 0,95 |
| długość fragmentu startowego w treningu | od 3 do 8 |
| ewaluacja okresowa | zachłanna, co 20 iteracji, 500 scen, fragment 6 |
| ewaluacja okresowa treningu wydłużonego | zachłanna, co 10 iteracji, 200 scen, fragment 6 |
| ziarna | 1234, 2025, 7 |

### Krok 5. Trening wydłużony do 3000 iteracji

Tylko nagroda Levenshteina, trzy ziarna (podrozdział 5.4, Rys. 5.10,
Tabela 5.5). Jeden przebieg trwa około 6,4 godziny. Ewaluacja okresowa tych
przebiegów odbywała się co 10 iteracji na 200 scenach, czyli rzadziej
i na mniejszej próbie niż w serii głównej.

```bash
python uruchom_ppo_wieloziarnowo.py --sft runs/sft_e96/epoch_0004.pt \
    --metrics levenshtein --seeds 1234 2025 7 \
    --iterations 3000 --kl-coef 0.01 --eval-every 10 --eval-samples 200 \
    --tag glowny_long
```

### Krok 6. Krzywe uczenia uśrednione po ziarnach

```bash
python zbierz_krzywe_ppo.py --tag glowny      --eval-scenes 500 --csv krzywe_ppo_glowny.csv
python zbierz_krzywe_ppo.py --tag glowny_long --eval-scenes 200 --csv krzywe_ppo_sonda3000.csv
python zbierz_krzywe_ppo.py --tag pilotkl000  --eval-scenes 200 --csv krzywe_kl000.csv
python zbierz_krzywe_ppo.py --tag pilotkl001  --eval-scenes 200 --csv krzywe_kl001.csv
python zbierz_krzywe_ppo.py --tag pilotkl005  --eval-scenes 200 --csv krzywe_kl005.csv
```

Skrypt wyszukuje katalogi według wzorca `ppo_<tag>_*_seed*`. Przy tagu
`glowny` wzorzec obejmuje również katalogi `ppo_glowny_long_...`
i `ppo_glowny_pilot_...`, które trafiają do pliku CSV jako osobne metryki
(`long_levenshtein`, `pilot_levenshtein`). Do wykresów serii głównej należy
brać wyłącznie wiersze z metrykami `levenshtein`, `bleu` i `rouge_l`.

### Krok 7. Tabele końcowe

Skrypt ocenia checkpointy od nowa dekodowaniem zachłannym na pełnym zbiorze
500 scen, przy ziarnie ewaluacji 3, osobno dla każdej długości fragmentu.
Każdy checkpoint podaje się jako `ETYKIETA=ŚCIEŻKA`.

```bash
python tabela_koncowa_ppo.py \
    --checkpoint "SFT e96 epoka 4 (start)=runs/sft_e96/epoch_0004.pt" \
    --checkpoint "SFT e96 sufit=runs/sft_e96/best.pt" \
    --checkpoint "PPO Levenshtein z1234 (1000 iter)=runs/ppo_glowny_levenshtein_seed1234/best.pt" \
    --prompt-lengths 3 4 5 6 8 --num-scenes 500 --seed 3 \
    --csv wyniki_tabela_koncowa_z5.csv --json wyniki_tabela_koncowa_z5.json
```

Pełna lista czternastu checkpointów (punkt startowy, sufit, dziewięć
przebiegów głównych, trzy przebiegi wydłużone) znajduje się w skrypcie
`uruchom_eksperyment.sh`. Tym samym skryptem, dla checkpointów `last.pt`
serii głównej, wyznacza się porównanie z Tabeli 5.9, a dla checkpointów
`runs/sft_e64/best.pt` i `runs/sft/best.pt` wyniki modeli e64 i e180 po pełnym
treningu (Tabela 5.3).

Checkpoint `best.pt` przebiegu PPO wybierany jest według sumy F1 zawartości
i współczynnika tau w ewaluacji okresowej, a `last.pt` to stan po ostatniej
iteracji.

### Krok 8. Przykłady rekonstrukcji

Po sześć rekonstrukcji dekodowaniem zachłannym, przed dostrajaniem i po nim
(podrozdział 5.5, Tabele 5.10 i 5.11). Sceny i fragmenty wejściowe losowane
są ziarnem polecenia `demo` (domyślnie 0), więc oba modele otrzymują te same
przykłady. Tryb dekodowania zapisywany jest w nagłówku wyjścia.

```bash
for K in 3 4 5; do
  python -m scene_rl demo runs/sft_e96/epoch_0004.pt \
      --n 6 --prompt-length $K --greedy > demo_start_len$K.txt
  python -m scene_rl demo runs/ppo_glowny_long_levenshtein_seed1234/best.pt \
      --n 6 --prompt-length $K --greedy > demo_ppo3000_len$K.txt
done
```

### Krok 9. Pojemność modeli

```bash
python -m scene_rl capacity
```

Polecenie wypisuje zawartość informacyjną zbioru (49 771 bitów) oraz tabelę
pojemności obejmującą między innymi modele e64, e96 i e180, zgodną
z Tabelą 5.12 (podrozdział 5.6).

## Pozostałe polecenia

```bash
# ocena jednego modelu w funkcji długości fragmentu
python -m scene_rl evaluate runs/sft_e96/best.pt --by-length --greedy --num-scenes 500

# porównanie kilku checkpointów, jedna długość fragmentu
python -m scene_rl compare --checkpoints \
    runs/sft_e96/epoch_0004.pt runs/ppo_glowny_levenshtein_seed1234/best.pt \
    --greedy --save porownanie.json

# wykresy z historii treningu
python -m scene_rl plots runs/sft_e96/history_sft.jsonl --out figures/sft

# pojedynczy przebieg PPO z parametrami serii głównej
# (ewaluacja okresowa pozostaje przy wartościach domyślnych: co 10 iteracji, 200 scen)
python -m scene_rl train-ppo --sft runs/sft_e96/epoch_0004.pt \
    --reward levenshtein --iterations 1000 --kl-coef 0.01 --greedy-eval \
    --seed 1234 --out runs/ppo_levenshtein_seed1234
```

Polecenie `train-ppo` ma domyślny współczynnik kary 0,02 oraz 300 iteracji.
Wartości użyte w pracy trzeba podać jawnie, jak wyżej.

## Testy

Testy nie wymagają PyTorcha i uruchamia się je bezpośrednio:

```bash
python tests/test_metrics.py
python tests/test_data.py
python tests/test_ppo_math.py
python tests/test_rollout_indices.py
python tests/static_check.py
```

## Zawartość folderu `dane_wynikowe`

| Plik | Materiał do |
|---|---|
| `history_e64.jsonl`, `history_e96.jsonl`, `history_e180.jsonl` | Tabela 4.4, Rys. 5.1–5.3, przebieg uczenia nadzorowanego trzech architektur |
| `wyniki_odniesienia_z5.csv` | Tabela 5.4, Rys. 5.6, trajektoria modelu e96 w epokach 3, 4, 5, 7, 10 i jego wynik po pełnym treningu, pięć długości fragmentu |
| `wyniki_odniesienia.csv`, `krok0.txt` | pierwsza ocena tych samych checkpointów i jej wyjście konsoli, cztery długości fragmentu (3, 4, 6, 8); wartości wspólne z plikiem `wyniki_odniesienia_z5.csv` są identyczne |
| `wyniki_e64_e180_sufit.csv` | Tabela 5.3, modele e64 i e180 po pełnym treningu |
| `krzywe_kl000.csv`, `krzywe_kl001.csv`, `krzywe_kl005.csv` | Tabela 4.7, Rys. 4.2, dobór współczynnika kary za rozbieżność |
| `krzywe_ppo_glowny.csv` | Rys. 5.7–5.8, Tabela 5.8, krzywe uczenia serii głównej |
| `krzywe_ppo_sonda3000.csv` | Rys. 5.10, trening wydłużony do 3000 iteracji |
| `wyniki_levenshtein_3000_tabela.csv` | Tabela 5.5, porównanie 1000 i 3000 iteracji |
| `wyniki_tabela_koncowa_z5.csv` | Tabele 5.6–5.7, Rys. 5.9 i 5.11, wyniki końcowe wszystkich wariantów |
| `wyniki_tabela_koncowa_last.csv` | Tabela 5.9, porównanie checkpointów `best.pt` i `last.pt` |
| `demo_start_len*.txt`, `demo_ppo3000_len*.txt` | Tabele 5.10–5.11, przykłady rekonstrukcji przed dostrajaniem i po nim |

Wartości w Tabeli 5.12 nie wymagają pliku danych, ponieważ wyznacza je
polecenie `capacity`. Parametry każdego przebiegu zapisane są w pliku
`config.json` w jego katalogu.

## Uwaga o wersji sprzed pakietu

Wcześniejsza, notatnikowa wersja kodu różniła się od niniejszego pakietu
w dwóch miejscach wpływających na wartości liczbowe.

1. Iloczyn skalarny w mechanizmie uwagi skalowano pierwiastkiem z `n_embd`
   zamiast z wymiaru pojedynczej głowy, niezgodnie ze wzorem Vaswaniego et al.
   Przy `n_embd = 180` i sześciu głowach iloczyny były zaniżone około
   dwuipółkrotnie, co nadmiernie wygładzało rozkład wag uwagi.
2. Obiekty brakujące i dodatkowe zliczano operatorem `in`, czyli bez
   uwzględnienia powtórzeń. Dla scen zawierających ten sam obiekt dwukrotnie
   zaniżało to liczbę obiektów brakujących i dodatkowych.

Niniejszy pakiet skaluje uwagę zgodnie ze wzorem i zlicza obiekty na
wielozbiorach (podrozdziały 3.2 i 3.6 pracy). Wszystkie wyniki przedstawione
w pracy policzono wyłącznie tym pakietem.
