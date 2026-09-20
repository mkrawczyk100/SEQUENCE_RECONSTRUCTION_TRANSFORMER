#!/usr/bin/env python3
"""Uśrednianie krzywych PPO po ziarnach losowości i test wypłaszczenia.

Odpowiada na dwa zarzuty promotora z komentarzy do wersji pierwszej:

- pojedynczy przebieg na metrykę nie pozwala odróżnić efektu od szumu,
  więc krzywe trzeba uśrednić po kilku niezależnych uruchomieniach;
- nie widać, czy uczenie stanęło w miejscu, więc trzeba jawnie zbadać,
  czy końcowy odcinek krzywej ma nachylenie nieodróżnialne od zera.

Skrypt czyta pliki history_ppo.jsonl z katalogów przebiegów, grupuje je po
metryce nagrody, uśrednia po ziarnach dla każdej iteracji ewaluacji i liczy
regresję liniową na końcowym fragmencie krzywej. Nie wymaga PyTorcha ani
żadnej biblioteki spoza standardowej.

Test wypłaszczenia porównuje zmianę przewidywaną przez dopasowaną prostą na
całym analizowanym ogonie z niepewnością samego pomiaru. Wskaźnik
rekonstrukcji dokładnej jest proporcją wyznaczaną na skończonej próbie scen,
więc ma błąd standardowy rzędu pierwiastka z 0,25 podzielonego przez liczbę
scen (0,035 dla 200 scen, 0,022 dla 500). Nachylenie mniejsze niż ta
niepewność nie jest trendem, tylko szumem, i nie wolno na jego podstawie
twierdzić, że trening nie zdążył się wypłaszczyć. Liczbę scen podaje się
flagą --eval-scenes i musi ona odpowiadać wartości eval_samples użytej
w przebiegu.

Użycie:

    python zbierz_krzywe_ppo.py --runs-root runs --tag glowny --eval-scenes 500
    python zbierz_krzywe_ppo.py --run levenshtein:1234=runs/ppo_x/history_ppo.jsonl

Wyjście: tabela na konsolę (do wklejenia w rozmowie) oraz plik CSV
`wyniki_ppo_krzywe.csv` z kolumnami metric, iteration, pole, mean, std, n.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import statistics
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


POLA = [
    ("eval_exact_match", "rekonstrukcje dokładne"),
    ("eval_content_f1", "F1 zawartości"),
    ("reward_mean", "średnia nagroda"),
    ("kl_to_reference", "KL do odniesienia"),
    ("entropy", "entropia polityki"),
]

# Pola będące proporcjami, dla których test wypłaszczenia ma sens.
POLA_PROPORCJI = {"eval_exact_match", "eval_content_f1"}


def wczytaj(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def znajdz_przebiegi(runs_root: str, tag: str) -> list[tuple[str, int, str]]:
    """Zwraca listę (metryka, ziarno, sciezka) na podstawie nazw katalogów.

    Zakładany wzorzec nazwy, tworzony przez uruchom_ppo_wieloziarnowo.py:
    ppo_<tag>_<metryka>_seed<ziarno>
    """
    wynik = []
    wzorzec = os.path.join(runs_root, f"ppo_{tag}_*_seed*")
    for katalog in sorted(glob.glob(wzorzec)):
        nazwa = os.path.basename(katalog)
        reszta = nazwa[len(f"ppo_{tag}_"):]
        if "_seed" not in reszta:
            continue
        metryka, _, ziarno = reszta.rpartition("_seed")
        history = os.path.join(katalog, "history_ppo.jsonl")
        if os.path.exists(history):
            try:
                wynik.append((metryka, int(ziarno), history))
            except ValueError:
                continue
    return wynik


def regresja_liniowa(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Zwraca (nachylenie, wyraz wolny) metodą najmniejszych kwadratów."""
    n = len(xs)
    if n < 2:
        return 0.0, ys[0] if ys else 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    licznik = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    mianownik = sum((x - mean_x) ** 2 for x in xs)
    if mianownik == 0:
        return 0.0, mean_y
    nachylenie = licznik / mianownik
    return nachylenie, mean_y - nachylenie * mean_x


def rozrzut_wokol_prostej(
    xs: list[float], ys: list[float], nachylenie: float, wyraz: float
) -> float:
    """Odchylenie standardowe reszt względem dopasowanej prostej."""
    if len(xs) < 3:
        return 0.0
    reszty = [y - (nachylenie * x + wyraz) for x, y in zip(xs, ys)]
    return statistics.pstdev(reszty)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Uśrednia krzywe PPO po ziarnach i bada, czy końcówka "
                     "treningu jest wypłaszczona.")
    )
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--tag", default="glowny")
    parser.add_argument("--run", action="append", default=[],
                        metavar="METRYKA:ZIARNO=SCIEZKA",
                        help=("Ręczne wskazanie przebiegu, jeśli nazwy "
                              "katalogów odbiegają od wzorca."))
    parser.add_argument("--ogon", type=float, default=0.25,
                        help=("Jaka część końcowa krzywej wchodzi do testu "
                              "wypłaszczenia (domyślnie ostatnie 25 procent)."))
    parser.add_argument("--eval-scenes", type=int, default=500,
                        help=("Liczba scen użyta w ewaluacji okresowej "
                              "(eval_samples przebiegu). Wyznacza próg "
                              "szumu w teście wypłaszczenia."))
    parser.add_argument("--csv", default="wyniki_ppo_krzywe.csv")
    args = parser.parse_args()

    blad_standardowy = math.sqrt(0.25 / max(args.eval_scenes, 1))

    przebiegi = znajdz_przebiegi(args.runs_root, args.tag)
    for entry in args.run:
        if "=" not in entry or ":" not in entry.split("=")[0]:
            raise SystemExit(
                f"Zły format --run: '{entry}'. Oczekiwano METRYKA:ZIARNO=SCIEZKA"
            )
        klucz, sciezka = entry.split("=", 1)
        metryka, ziarno = klucz.split(":", 1)
        przebiegi.append((metryka, int(ziarno), sciezka))

    if not przebiegi:
        raise SystemExit(
            f"Nie znaleziono żadnych przebiegów we wzorcu "
            f"{os.path.join(args.runs_root, f'ppo_{args.tag}_*_seed*')}. "
            f"Sprawdź --runs-root i --tag albo wskaż przebiegi flagą --run."
        )

    print("Znalezione przebiegi:")
    for metryka, ziarno, sciezka in przebiegi:
        print(f"  nagroda={metryka:<12} ziarno={ziarno:<6} {sciezka}")
    print(f"\nZałożona liczba scen w ewaluacji okresowej: {args.eval_scenes}")
    print(f"Błąd standardowy pojedynczego pomiaru proporcji: "
          f"{blad_standardowy:.4f} (przedział ufności 95 procent ma "
          f"szerokość do {2 * 1.96 * blad_standardowy:.4f})")

    # dane[metryka][pole][iteracja] = lista wartości po ziarnach
    dane: dict[str, dict[str, dict[int, list[float]]]] = {}
    punkty_startowe: dict[str, list[float]] = {}

    for metryka, _ziarno, sciezka in przebiegi:
        records = wczytaj(sciezka)
        dane.setdefault(metryka, {})
        punkty_startowe.setdefault(metryka, [])
        for record in records:
            iteracja = record.get("iteration")
            if iteracja is None:
                continue
            if iteracja == -1:
                # wpis odniesienia sprzed pierwszej aktualizacji
                wartosc = record.get("eval_exact_match")
                if wartosc is not None:
                    punkty_startowe[metryka].append(float(wartosc))
                continue
            for pole, _etykieta in POLA:
                if pole in record and record[pole] is not None:
                    dane[metryka].setdefault(pole, {})
                    dane[metryka][pole].setdefault(int(iteracja), [])
                    dane[metryka][pole][int(iteracja)].append(
                        float(record[pole])
                    )

    wiersze_csv = []

    for metryka in sorted(dane):
        print(f"\n=== Nagroda: {metryka} ===")
        starty = punkty_startowe.get(metryka, [])
        if starty:
            print(
                f"  Punkt startowy (przed pierwszą aktualizacją), "
                f"rekonstrukcje dokładne: średnia "
                f"{statistics.mean(starty):.4f}"
                + (f", odchylenie {statistics.pstdev(starty):.4f}"
                   if len(starty) > 1 else "")
                + f" (n={len(starty)})"
            )

        for pole, etykieta in POLA:
            seria = dane[metryka].get(pole)
            if not seria:
                continue
            iteracje = sorted(seria)
            srednie = [statistics.mean(seria[i]) for i in iteracje]
            odchylenia = [
                statistics.pstdev(seria[i]) if len(seria[i]) > 1 else 0.0
                for i in iteracje
            ]
            liczebnosci = [len(seria[i]) for i in iteracje]

            for i, iteracja in enumerate(iteracje):
                wiersze_csv.append({
                    "metric": metryka,
                    "iteration": iteracja,
                    "pole": pole,
                    "mean": round(srednie[i], 6),
                    "std": round(odchylenia[i], 6),
                    "n": liczebnosci[i],
                })

            ile_ogon = max(5, int(len(iteracje) * args.ogon))
            ile_ogon = min(ile_ogon, len(iteracje))
            xs = [float(i) for i in iteracje[-ile_ogon:]]
            ys = srednie[-ile_ogon:]
            nachylenie, wyraz = regresja_liniowa(xs, ys)
            rozrzut = rozrzut_wokol_prostej(xs, ys, nachylenie, wyraz)
            rozpietosc = xs[-1] - xs[0] if len(xs) > 1 else 0.0
            zmiana_na_ogonie = nachylenie * rozpietosc
            zmiana_na_1000 = nachylenie * 1000.0

            print(f"\n  {etykieta} ({pole}):")
            print(f"    początek: {srednie[0]:.4f} (iteracja {iteracje[0]})")
            print(f"    koniec:   {srednie[-1]:.4f} (iteracja {iteracje[-1]})")
            print(f"    maksimum: {max(srednie):.4f} "
                  f"(iteracja {iteracje[srednie.index(max(srednie))]})")
            print(f"    zmiana koniec minus początek: "
                  f"{srednie[-1] - srednie[0]:+.4f}")
            print(f"    ogon: ostatnie {ile_ogon} pomiarów, "
                  f"iteracje {int(xs[0])}-{int(xs[-1])}")
            print(f"    nachylenie na ogonie: {zmiana_na_1000:+.4f} "
                  f"na 1000 iteracji, czyli {zmiana_na_ogonie:+.4f} "
                  f"na całym ogonie")

            if pole in POLA_PROPORCJI:
                print(f"    rozrzut pomiarów wokół prostej: {rozrzut:.4f}")
                prog = 2.0 * blad_standardowy
                if abs(zmiana_na_ogonie) < prog:
                    werdykt = (
                        f"WYPŁASZCZONA. Zmiana przewidywana na ogonie "
                        f"({zmiana_na_ogonie:+.4f}) jest mniejsza od progu "
                        f"szumu pomiaru ({prog:.4f}), więc trend jest "
                        f"nieodróżnialny od zera. Dalsze wydłużanie treningu "
                        f"nie jest uzasadnione tymi danymi."
                    )
                else:
                    werdykt = (
                        f"NADAL SIĘ ZMIENIA. Zmiana przewidywana na ogonie "
                        f"({zmiana_na_ogonie:+.4f}) przekracza próg szumu "
                        f"pomiaru ({prog:.4f}), więc trening przerwano przed "
                        f"wypłaszczeniem i budżet warto wydłużyć."
                    )
                print(f"    wniosek: {werdykt}")
                if odchylenia and max(odchylenia) > 0:
                    print(f"    największe odchylenie między ziarnami: "
                          f"{max(odchylenia):.4f}")

    if wiersze_csv:
        with open(args.csv, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["metric", "iteration", "pole", "mean", "std", "n"],
            )
            writer.writeheader()
            writer.writerows(wiersze_csv)
        print(f"\nZapisano {len(wiersze_csv)} wierszy do {args.csv}")


if __name__ == "__main__":
    main()
