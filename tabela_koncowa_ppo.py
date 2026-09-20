#!/usr/bin/env python3
"""Tabela końcowa: porównanie modeli w mierze docelowej, po długościach fragmentu.

Odtwarza protokół rozdziału 5 pracy (dekodowanie zachłanne, pełny zbiór 500
scen, ustalone ziarno ewaluacji) dla dowolnej listy checkpointów i dowolnego
zestawu długości fragmentu startowego, i wypisuje wynik w postaci tabeli
gotowej do przeniesienia do pracy (format markdown z przecinkiem dziesiętnym).

W odróżnieniu od polecenia `python -m scene_rl compare` pozwala nadać każdemu
checkpointowi własną etykietę (compare zgaduje ją z nazwy katalogu) oraz
liczy wszystkie długości fragmentu w jednym przebiegu.

Użycie:

    python tabela_koncowa_ppo.py \\
        --checkpoint "SFT e96, epoka 4 (punkt startowy)=runs/sft_e96/epoch_0004.pt" \\
        --checkpoint "SFT e96, pelny trening (sufit)=runs/sft_e96/best.pt" \\
        --checkpoint "PPO Levenshtein, ziarno 1234=runs/ppo_glowny_levenshtein_seed1234/best.pt" \\
        --prompt-lengths 3 4 6 8 --num-scenes 500 --seed 3 \\
        --csv wyniki_tabela_koncowa.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def pl(value: float, miejsca: int = 4) -> str:
    """Liczba z przecinkiem dziesiętnym, zgodnie z konwencją pracy."""
    return f"{value:.{miejsca}f}".replace(".", ",")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Porównanie checkpointów w mierze docelowej, dekodowanie "
                     "zachłanne, w funkcji długości fragmentu startowego.")
    )
    parser.add_argument("--checkpoint", action="append", required=True,
                        metavar="ETYKIETA=SCIEZKA",
                        help="Można podać wielokrotnie, kolejność zachowana.")
    parser.add_argument("--prompt-lengths", nargs="+", type=int,
                        default=[3, 4, 6, 8])
    parser.add_argument("--num-scenes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--data", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--csv", default="wyniki_tabela_koncowa.csv")
    parser.add_argument("--json", default="wyniki_tabela_koncowa.json")
    args = parser.parse_args()

    pary = []
    for entry in args.checkpoint:
        if "=" not in entry:
            raise SystemExit(
                f"Zły format --checkpoint: '{entry}'. Oczekiwano ETYKIETA=SCIEZKA"
            )
        etykieta, sciezka = entry.split("=", 1)
        if not os.path.exists(sciezka):
            raise SystemExit(f"Nie istnieje checkpoint: {sciezka}")
        pary.append((etykieta.strip(), sciezka))

    try:
        from scene_rl.data import SceneDataset
        from scene_rl.evaluation import evaluate_model
        from scene_rl.utils import (
            build_model_from_checkpoint,
            resolve_device,
            set_seed,
        )
    except ImportError as exc:
        raise SystemExit(
            f"Nie udało się zaimportować scene_rl ({exc}). Uruchom skrypt "
            f"z katalogu, w którym działa 'python -m scene_rl ...'."
        )

    device = resolve_device(args.device)
    wyniki = []

    for etykieta, sciezka in pary:
        model, config, payload = build_model_from_checkpoint(sciezka, device)
        model.eval()
        dataset = SceneDataset(args.data or config.data.path)
        for prompt_length in args.prompt_lengths:
            set_seed(args.seed)
            summary = evaluate_model(
                model, dataset, device,
                num_scenes=args.num_scenes,
                prompt_length=prompt_length,
                seed=args.seed,
                greedy=True,
            )
            wiersz = {
                "etykieta": etykieta,
                "checkpoint": sciezka,
                "epoka_lub_iteracja": payload.get("epoch", ""),
                "fragment": prompt_length,
                "exact_match": summary["exact_match"],
                "content_f1": summary["content_f1"],
                "order_tau": summary["order_tau"],
                "missing": summary["missing"],
                "hallucinated": summary["hallucinated"],
                "levenshtein": summary["levenshtein"],
                "bleu": summary["bleu"],
                "rouge_l": summary["rouge_l"],
                "failure_rate": summary["failure_rate"],
            }
            wyniki.append(wiersz)
            print(
                f"{etykieta:<45} fragment {prompt_length}: "
                f"dokładne={summary['exact_match']:.4f} "
                f"F1={summary['content_f1']:.4f} "
                f"tau={summary['order_tau']:.4f} "
                f"braki={summary['missing']:.3f}"
            )

    print("\n\nTabela w formacie do wklejenia do pracy\n")
    print("| Fragment | Model | F1 | Tau | Dokładne | Braki | Halucynacje |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for prompt_length in args.prompt_lengths:
        for wiersz in wyniki:
            if wiersz["fragment"] != prompt_length:
                continue
            print(
                f"| {prompt_length} | {wiersz['etykieta']} | "
                f"{pl(wiersz['content_f1'])} | {pl(wiersz['order_tau'])} | "
                f"{pl(wiersz['exact_match'], 3)} | "
                f"{pl(wiersz['missing'], 3)} | "
                f"{pl(wiersz['hallucinated'], 3)} |"
            )

    # Próg istotności różnic proporcji przy danej liczbie scen, potrzebny
    # do interpretacji tabeli w tekście pracy. Najszerszy przedział wypada
    # przy wartości 0,5, więc podawana jest wartość zachowawcza.
    n = args.num_scenes
    blad_max = (0.25 / n) ** 0.5
    print(
        f"\nPrzy {n} scenach błąd standardowy proporcji wynosi najwyżej "
        f"{blad_max:.4f}, więc przedział ufności 95 procent ma szerokość do "
        f"{2 * 1.96 * blad_max:.4f}. Różnice mniejsze nie są istotne."
    )

    if wyniki:
        with open(args.csv, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(wyniki[0]))
            writer.writeheader()
            writer.writerows(wyniki)
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(wyniki, handle, indent=2, ensure_ascii=False)
        print(f"\nZapisano wyniki do {args.csv} oraz {args.json}")


if __name__ == "__main__":
    main()
