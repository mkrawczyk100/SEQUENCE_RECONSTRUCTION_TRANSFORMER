# Rekonstrukcja sekwencji transformerem z uczeniem ze wzmocnieniem

Kod do pracy magisterskiej „Rekonstrukcja sekwencji za pomocą sztucznej
sieci neuronowej o architekturze transformera z wykorzystaniem metryk
podobieństwa i uczenia ze wzmocnieniem".

Zadanie ma charakter pamięci asocjacyjnej adresowanej treścią: model
otrzymuje niepełny fragment sceny i ma odtworzyć całą zapamiętaną scenę,
wraz z kolejnością obiektów.

## Wymagania

```
python >= 3.10
torch >= 2.0
numpy
matplotlib          (tylko do wykresów)
nltk, rouge-score, python-Levenshtein   (opcjonalnie, do porównania
                                         implementacji własnych
                                         z bibliotecznymi)
```

Instalacja bibliotek opcjonalnych:

```bash
pip install nltk rouge-score python-Levenshtein
```

Bez nich polecenie `metrics-table` pominie kolumny porównawcze,
a reszta kodu działa bez zmian.

## Szybki start

```bash
# 1. Sprawdzenie poprawności implementacji
python -m scene_rl selftest --data sequences_przemek.txt

# 2. Statystyki zbioru (materiał do rozdziału 3)
python -m scene_rl analyze --data sequences_przemek.txt

# 3. Tabela porównawcza metryk (materiał do rozdziału 2.5)
python -m scene_rl metrics-table

# 4. Etap 1: uczenie nadzorowane
python -m scene_rl train-sft --data sequences_przemek.txt \
    --epochs 60 --out runs/sft --seed 1234

# 5. Etap 2: dostrajanie PPO, osobno każdą metryką
python -m scene_rl train-ppo --sft runs/sft/best.pt \
    --reward levenshtein --out runs/ppo_lev --seed 1234
python -m scene_rl train-ppo --sft runs/sft/best.pt \
    --reward bleu --out runs/ppo_bleu --seed 1234
python -m scene_rl train-ppo --sft runs/sft/best.pt \
    --reward rouge_l --out runs/ppo_rouge --seed 1234

# 6. Porównanie modeli w mierze docelowej (tabela do rozdziału 5)
python -m scene_rl compare --checkpoints \
    runs/sft/best.pt runs/ppo_lev/best.pt \
    runs/ppo_bleu/best.pt runs/ppo_rouge/best.pt \
    --num-scenes 500 --save wyniki/porownanie.json

# 7. Ocena w funkcji długości fragmentu wejściowego
python -m scene_rl evaluate runs/sft/best.pt --by-length

# 8. Wykresy przebiegu treningu
python -m scene_rl plots runs/sft/history_sft.jsonl --out figures/sft
```

Trzy przebiegi PPO startują z tego samego checkpointu, mają ten sam
budżet obliczeniowy i to samo ziarno losowości. Różnią się wyłącznie
metryką pełniącą funkcję nagrody, co jest warunkiem uczciwości
porównania.

## Struktura pakietu

| Plik | Zawartość |
|---|---|
| `config.py` | hiperparametry w postaci struktur danych, zapisywane w checkpointach |
| `data.py` | wczytanie scen, tokenizacja, łańcuch rekonstrukcji, partie danych |
| `metrics.py` | BLEU, ROUGE-L, podobieństwo edycyjne, miara docelowa |
| `model.py` | transformer decoder-only, głowica wartości, generowanie |
| `sft.py` | etap 1: uczenie nadzorowane |
| `ppo.py` | etap 2: dostrajanie algorytmem PPO |
| `evaluation.py` | rekonstrukcja wsadowa i ocena jakości |
| `utils.py` | ziarna losowości, checkpointy, logowanie historii |
| `selftest.py` | autotesty implementacji |
| `__main__.py` | interfejs wiersza poleceń |

Testy niewymagające PyTorcha znajdują się w katalogu `tests/`
i uruchamia się je bezpośrednio:

```bash
python tests/test_metrics.py
python tests/test_data.py
python tests/test_ppo_math.py
python tests/test_rollout_indices.py
python tests/static_check.py
```

## Poprawki względem wersji notatnikowej

Poniższa lista dokumentuje różnice między tym kodem a notatnikiem
`19_07_2026.ipynb`. Punkty od 1 do 4 wpływają na wartości liczbowe,
więc wyników policzonych poprzednią wersją nie należy raportować.

**1. Błąd o jeden przy wycinaniu odzyskanej sceny.** Poprzednio
`answer[last_sep + 1 : stop - 1]` obcinało ostatni obiekt każdej
rekonstrukcji, co zaniżało wszystkie metryki i zawyżało liczbę obiektów
brakujących. Poprawnie: `answer[last_sep + 1 : stop]`.

**2. Skalowanie iloczynu skalarnego uwagi.** Poprzednio dzielono przez
pierwiastek z `n_embd` zamiast z wymiaru głowy `d_k`. Przy `n_embd = 180`
i `n_head = 6` współczynnik był zaniżony pierwiastkiem z sześciu, przez
co rozkład uwagi był nadmiernie wygładzony. Wzór jest teraz zgodny
z pracą Vaswaniego i innych.

**3. Pozycja fragmentu wejściowego.** Poprzednio łańcuch rekonstrukcji
zawsze zaczynał się od fragmentu trzyelementowego, a okno kontekstowe
wycinano z losowego miejsca sklejonego ciągu wszystkich scen. Fragment
sześcioelementowy, używany przy ewaluacji, nigdy nie występował na
początku okna, więc embedding pozycyjny przy inferencji nie odpowiadał
temu, co model widział w treningu. Obecnie okno startuje na granicy
podsekwencji, dzięki czemu fragment dowolnej długości pojawia się na
pozycji zerowej.

**4. Tempo uczenia.** Poprzednio zmienna `learning_rate` była nadpisywana
w komórce pętli treningowej i zapisywana do historii, ale optymalizator
zachowywał wartość podaną przy jego utworzeniu. Raportowane tempo
uczenia nie odpowiadało rzeczywistemu. Obecnie wartość trafia do
`optimizer.param_groups`, a harmonogram obejmuje rozgrzewkę i wygaszanie
cosinusem.

**5. Wypełnienie i maskowanie straty.** Wprowadzono token `PAD` o
indeksie 2. Przykłady krótsze od okna są dopełniane, a pozycje
wypełnienia oraz fragmentu wejściowego nie wnoszą wkładu do entropii
krzyżowej. Model nie uczy się rekonstrukcji urwanych.

**6. Zliczanie na wielozbiorach.** Poprzednio używano operatora `in`
i funkcji `set`, co przy 21 scenach zawierających ten sam obiekt
dwukrotnie dawało błędne liczby obiektów odtworzonych i brakujących.

**7. Ziarna losowości.** Poprzednio nie ustalano żadnego ziarna, więc
dwa uruchomienia tego samego eksperymentu dawały różne wyniki.

**8. Implementacja BLEU.** Poprzednio używano biblioteki NLTK, podczas
gdy praca deklaruje implementację własną. Obecnie BLEU liczony jest
własnym kodem, a polecenie `metrics-table` zestawia go z wynikiem
bibliotecznym.

**9. Nazwa klasy modelu.** `BigramLanguageModel` zmieniono na
`SceneTransformer`. Model nie jest bigramowy: przewiduje kolejny token
na podstawie całego widocznego prefiksu.

**10. Generowanie.** Kończy się na tokenie `STOP` zamiast generować
stałą liczbę tokenów i szukać `STOP` po fakcie. Obsługuje temperaturę,
obcinanie do k najlepszych tokenów oraz wariant zachłanny. Token `PAD`
nigdy nie może zostać wygenerowany.

**11. Wydajność generowania.** Rekonstrukcje liczone są wsadowo, po
grupach fragmentów o jednakowej długości, zamiast scena po scenie.

**12. Format historii treningu.** Poprzednio zapisywano tekstową
reprezentację tensorów, postaci `tensor(0.4253, device='cuda:0')`, którą
trzeba było parsować przed narysowaniem wykresu. Obecnie historia
zapisywana jest jako JSONL oraz CSV z liczbami.

**13. Checkpointy.** Zawierają pełną konfigurację eksperymentu, dzięki
czemu dowolną nową metrykę można policzyć bez powtarzania treningu,
a model odtworzyć bez zgadywania hiperparametrów.

## Elementy dodane, nieobecne w wersji notatnikowej

**Miara docelowa.** Weryfikacja hipotezy wymaga miary niezależnej od
metryk użytych jako nagroda. Model dostrajany nagrodą BLEU wygrałby
w BLEU z samej definicji, więc porównanie oparte na metrykach nagrody
byłoby cyrkularne. Miara docelowa raportuje rozdzielnie:

- zgodność zawartości sceny: liczby obiektów odtworzonych, brakujących
  i halucynowanych oraz precyzja, czułość i F1 liczone na wielozbiorach,
- zgodność uporządkowania: współczynnik tau liczony wyłącznie na
  obiektach wspólnych, dzięki czemu jest niezależny od zgodności
  zawartości,
- zgodność pozycyjna, błąd długości i udział rekonstrukcji dokładnych.

Rozdzielenie zawartości od uporządkowania odpowiada wprost na pytanie
badawcze o to, czy poprawa dotyczy doboru obiektów, czy ich kolejności.

**Pełna implementacja PPO.** Stanem jest prefiks sekwencji, akcją wybór
tokenu, epizod kończy się na `STOP`. Nagroda metryczna jest rzadka,
przyznawana w kroku terminalnym, ponieważ metryki podobieństwa są
zdefiniowane dla całych sekwencji. Zaimplementowano uogólnioną estymację
przewagi, obcinanie ilorazu prawdopodobieństw, obcinanie funkcji
wartości, premię entropijną oraz karę za rozbieżność względem polityki
odniesienia, czyli zamrożonej kopii modelu po etapie pierwszym. Bez tej
kary polityka degeneruje się do sekwencji maksymalizujących metrykę.

**Wyłączanie dropoutu w PPO.** Iloraz prawdopodobieństw polityki nowej
i starej musi wynosić dokładnie jeden przed pierwszą aktualizacją.
Aktywny dropout losowo zeruje aktywacje, więc to samo przejście w przód
dwukrotnie daje różne log-prawdopodobieństwa, co zaburza próbkowanie
ważone leżące u podstaw metody.

**Weryfikacja długości okna.** `required_block_size` wyznacza empirycznie
minimalną wystarczającą wartość `block_size`, a trening przerywa się
komunikatem błędu, gdy podana wartość jest za mała. Dla scen
dziesięcioelementowych cały proces rekonstrukcji zajmuje 60 tokenów,
więc `block_size` musi wynosić co najmniej 61.

**Autotesty.** Polecenie `selftest` sprawdza własności, które w razie
błędu nie objawiają się awarią programu, lecz cichym pogorszeniem
wyników: szczelność maskowania przyczynowego, poprawność skalowania
uwagi, pomijanie wypełnienia w funkcji straty, wyrównanie indeksów
log-prawdopodobieństw, iloraz PPO równy jedności przed aktualizacją,
harmonogram tempa uczenia oraz zdolność modelu do przeuczenia
pojedynczej partii danych.

## Uwaga o wynikach z poprzednich przebiegów

Pliki `loss_history.txt`, `val_history.txt`, `lr_history.txt`
i `similarity_metrics_history.txt` pochodzą z konfiguracji, której nie da
się odtworzyć z notatnika: przy zapisanych tam parametrach liczba
zarejestrowanych pomiarów nie zgadza się z liczbą epok. Wartości
`lr_history.txt` wynoszą 0,0005, podczas gdy optymalizator działał
z tempem 0,001. Do pracy należy użyć wyników z nowych przebiegów.
