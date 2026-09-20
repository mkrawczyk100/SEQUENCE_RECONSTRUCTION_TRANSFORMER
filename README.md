# Rekonstrukcja sekwencji transformerem z uczeniem ze wzmocnieniem

Kod źródłowy do pracy magisterskiej „Rekonstrukcja sekwencji za pomocą sztucznej sieci neuronowej o architekturze transformera", Politechnika Rzeszowska,
2026.

Zadanie ma charakter pamięci asocjacyjnej adresowanej treścią. Model
otrzymuje niepełny fragment sceny (uporządkowanej sekwencji dziesięciu
identyfikatorów obiektów) i ma odtworzyć całą zapamiętaną scenę, wraz
z kolejnością obiektów. Model uczony jest dwuetapowo: uczeniem
nadzorowanym, a następnie dostrajany algorytmem PPO, z nagrodą opartą
na jednej z trzech metryk podobieństwa (BLEU, ROUGE-L, podobieństwo
edycyjne Levenshteina).

## Wymagania

python >= 3.10

torch >= 2.0

numpy

matplotlib (tylko do wykresów)

nltk, rouge-score, python-Levenshtein (opcjonalnie, do porównania

implementacji własnych metryk
z bibliotecznymi)


Instalacja:

```bash
pip install -r requirements.txt
pip install nltk rouge-score python-Levenshtein   # opcjonalnie
```

Bez bibliotek opcjonalnych polecenie `metrics-table` pomija kolumny
porównawcze, reszta kodu działa bez zmian.

## Struktura repozytorium

scene_rl/ pakiet z implementacją modelu, treningu i metryk
tests/ testy jednostkowe (nie wymagają PyTorcha)
dane_wynikowe/ dane liczbowe i wykresy leżące u podstaw rozdziału 5
sequences_przemek.txt zbiór danych, 500 scen po 10 obiektów
uruchom_eksperyment.sh pełna sekwencja poleceń odtwarzająca eksperyment
uruchom_ppo_wieloziarnowo.py uruchamia train-ppo dla siatki metryk x ziaren
zbierz_krzywe_ppo.py zbiera historie PPO w zbiorcze pliki CSV
tabela_koncowa_ppo.py buduje tabele końcowe (rozdział 5) z historii
wybor_modelu_ppo.py analiza wyboru rozmiaru modelu (podrozdział 4.3)


### Pakiet `scene_rl`

| Plik | Zawartość |
|---|---|
| `config.py` | hiperparametry jako struktury danych, zapisywane w checkpointach |
| `data.py` | wczytanie scen, tokenizacja, łańcuch rekonstrukcji, partie danych |
| `metrics.py` | BLEU, ROUGE-L, podobieństwo edycyjne, miara docelowa |
| `model.py` | transformer decoder-only, głowica wartości, generowanie |
| `sft.py` | etap 1: uczenie nadzorowane |
| `ppo.py` | etap 2: dostrajanie algorytmem PPO |
| `evaluation.py` | rekonstrukcja wsadowa i ocena jakości |
| `capacity.py` | oszacowanie pojemności informacyjnej modelu (podrozdział 5.6) |
| `utils.py` | ziarna losowości, checkpointy |
| `selftest.py` | autotesty implementacji |
| `__main__.py` | interfejs wiersza poleceń |

## Interfejs wiersza poleceń

Cały eksperyment daje się odtworzyć bez notatnika, poleceniami modułu
`scene_rl`:

```bash
# Statystyki zbioru danych (rozdział 3)
python -m scene_rl analyze --data sequences_przemek.txt

# Tabela porównawcza metryk podobieństwa (podrozdział 2.5)
python -m scene_rl metrics-table

# Autotesty implementacji
python -m scene_rl selftest --data sequences_przemek.txt

# Etap 1: uczenie nadzorowane
python -m scene_rl train-sft --data sequences_przemek.txt \
    --epochs 60 --n-embd 96 --n-head 4 --n-layer 4 \
    --seed 1234 --out runs/sft_e96

# Etap 2: dostrajanie PPO, osobno dla każdej metryki nagrody
python -m scene_rl train-ppo --sft runs/sft_e96/best.pt \
    --reward levenshtein --iterations 1000 --seed 1234 \
    --out runs/ppo_levenshtein_seed1234

# Ocena pojedynczego modelu w funkcji długości fragmentu
python -m scene_rl evaluate runs/sft_e96/best.pt \
    --by-length --greedy --num-scenes 500

# Porównanie kilku checkpointów w mierze docelowej
python -m scene_rl compare --checkpoints \
    runs/sft_e96/best.pt runs/ppo_levenshtein_seed1234/best.pt \
    --greedy --save wyniki/porownanie.json

# Kilka przykładowych rekonstrukcji "na żywo"
python -m scene_rl demo runs/sft_e96/best.pt --n 6 --prompt-length 3

# Wykresy z historii treningu
python -m scene_rl plots runs/sft_e96/history_sft.jsonl --out figures/sft

# Oszacowanie pojemności modelu (podrozdział 5.6)
python -m scene_rl capacity --n-embd 96 --n-layer 4 --block-size 64
```

Pełny, wieloziarnowy eksperyment PPO (dziewięć przebiegów: trzy metryki
razy trzy ziarna) uruchamia `uruchom_ppo_wieloziarnowo.py`, cienka
nakładka na `train-ppo` z pętlą po obu listach.

### Testy

```bash
python tests/test_metrics.py
python tests/test_data.py
python tests/test_ppo_math.py
python tests/test_rollout_indices.py
python tests/static_check.py
```

## Zawartość folderu `dane_wynikowe`

Pliki leżące u podstaw tabel i rysunków rozdziału 5.

| Plik | Materiał do |
|---|---|
| `history_e64.jsonl`, `history_e96.jsonl`, `history_e180.jsonl` | Rys. 5.1–5.3, przebieg uczenia nadzorowanego trzech architektur |
| `wyniki_odniesienia.csv`, `krok0.txt` | Tab. 5.3–5.4, trajektoria modelu e96 i sufit uczenia nadzorowanego |
| `krzywe_ppo_glowny.csv` | Rys. 5.7–5.8, krzywe uczenia PPO (seria główna) |
| `krzywe_ppo_sonda3000.csv` | Rys. 5.10, sonda wydłużona do 3000 iteracji |
| `wyniki_levenshtein_3000_tabela.csv` | Tab. 5.5, porównanie 1000 i 3000 iteracji |
| `wyniki_tabela_koncowa_z5.csv` | Tab. 5.6–5.7, 5.9–5.10, wyniki końcowe wszystkich wariantów |
| `wyniki_tabela_koncowa_last.csv` | Tab. 5.9, porównanie checkpointów best.pt i last.pt |
| `wyniki_e64_e180_sufit.csv` | Tab. 5.3, sufit modeli e64 i e180 |
| `krzywe_kl000.csv`, `krzywe_kl001.csv`, `krzywe_kl005.csv` | Tab. 4.7, dobór współczynnika kary za rozbieżność |
| `demo_start_len*.txt`, `demo_ppo3000_len*.txt` | Tab. 5.10–5.11, przykłady rekonstrukcji przed i po dostrajaniu |

## Uwaga o wersji sprzed pakietu

Wcześniejsza, notatnikowa wersja tego kodu zawierała kilka błędów
wpływających na wartości liczbowe (m.in. błędne skalowanie iloczynu
skalarnego uwagi, nienaktualizowane tempo uczenia w optymalizatorze,
zliczanie na zbiorach zamiast wielozbiorach), opisanych szczegółowo
w podrozdziałach 4.1–4.3 pracy. Wyników policzonych tamtą wersją nie
należy traktować jako miarodajnych; niniejszy pakiet jest wersją
poprawioną i jedyną, na której oparto wyniki przedstawione w pracy.
