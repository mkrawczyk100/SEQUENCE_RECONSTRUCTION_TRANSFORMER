#!/usr/bin/env python3
"""Wlasciwa (zachlanna, pelnozbiorowa) ocena wybranych checkpointow SFT.

Po co ten skrypt: `history_sft.jsonl` analizowany przez
`wybor_modelu_ppo.py` zawiera metryki liczone W TRAKCIE treningu, czyli
przez PROBKOWANIE z rozkladu modelu (temperatura 1.0) na zaledwie 100
scenach (parametr `metrics_samples` w TrainConfig). To wystarcza do
ogladania trendu na biezaco, ale jest zbyt szumne, zeby na tej podstawie
stwierdzic, czy model faktycznie ma sufit w okolicach 1.0, czy nie -
widac to po nagle skaczacych wartosciach exact_match miedzy sasiednimi
zmierzonymi epokami.

Ten skrypt uzywa tego samego protokolu, co rozdzial 5 pracy: dekodowanie
zachlanne (deterministyczne) na pelnym zbiorze 500 scen, przez
`scene_rl.evaluation.evaluate_model`. Dziala na dowolnej liscie
checkpointow (moga to byc pliki epoch_XXXX.pt, ktore juz masz na dysku -
nie trzeba niczego trenowac od nowa, zeby je ocenic).

Uzycie (z katalogu, w ktorym normalnie odpalasz `python -m scene_rl`,
tak zeby import scene_rl dzialal):

    python sprawdz_checkpointy.py \\
        --checkpoint runs/sft_e96/best.pt \\
        --checkpoint runs/sft/epoch_0001.pt \\
        --checkpoint runs/sft/epoch_0002.pt \\
        --checkpoint runs/sft/epoch_0003.pt \\
        --checkpoint runs/sft/epoch_0004.pt \\
        --prompt-length 6 --num-scenes 500

Mozna tez podac caly zakres epok jednym zamachem flaga --epoch-range:

    python sprawdz_checkpointy.py \\
        --epoch-range runs/sft:0:5 \\
        --checkpoint runs/sft_e96/best.pt \\
        --prompt-length 6 --num-scenes 500

`--epoch-range KATALOG:OD:DO` doklada do listy pliki
KATALOG/epoch_OD.pt ... KATALOG/epoch_(DO-1).pt (zakres jak w range()),
zakladajac standardowa nazwe plikow zapisywana przez `train_sft`
(`epoch_{epoch:04d}.pt`).

Wymaga tego samego srodowiska (torch + scene_rl), w ktorym trenowales
model - w przeciwienstwie do wybor_modelu_ppo.py, ktory dziala na samej
bibliotece standardowej.
"""

from __future__ import annotations

import argparse
import os
import sys


def add_epoch_range(paths: list[str], spec: str) -> None:
    try:
        directory, start, end = spec.split(":")
        start = int(start)
        end = int(end)
    except ValueError:
        raise SystemExit(
            f"Zly format dla --epoch-range: '{spec}'. Oczekiwano "
            f"KATALOG:OD:DO, np. runs/sft:0:5"
        )
    for epoch in range(start, end):
        candidate = os.path.join(directory, f"epoch_{epoch:04d}.pt")
        paths.append(candidate)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Ocena zachlanna (greedy, 500 scen) wybranych checkpointow SFT, "
            "do weryfikacji prawdziwego sufitu modelu i lokalizacji "
            "checkpointu 'tak sobie' bez ponownego treningu."
        )
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="SCIEZKA",
        help="Sciezka do pojedynczego checkpointu .pt. Mozna podac wiele razy.",
    )
    parser.add_argument(
        "--epoch-range",
        action="append",
        default=[],
        metavar="KATALOG:OD:DO",
        help=(
            "Dolozenie zakresu epok z jednego katalogu, np. runs/sft:0:5 "
            "doda epoch_0000.pt .. epoch_0004.pt z tego katalogu."
        ),
    )
    parser.add_argument("--data", default=None, help="Nadpisanie pliku danych (domyslnie z konfiguracji checkpointu).")
    parser.add_argument("--num-scenes", type=int, default=500)
    parser.add_argument("--prompt-length", type=int, default=6)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    checkpoints: list[str] = list(args.checkpoint)
    for spec in args.epoch_range:
        add_epoch_range(checkpoints, spec)

    if not checkpoints:
        raise SystemExit(
            "Nie podano zadnego checkpointu. Uzyj --checkpoint lub --epoch-range."
        )

    try:
        from scene_rl.data import SceneDataset
        from scene_rl.evaluation import evaluate_model
        from scene_rl.utils import build_model_from_checkpoint, resolve_device, set_seed
    except ImportError as exc:
        raise SystemExit(
            f"Nie udalo sie zaimportowac scene_rl ({exc}). Uruchom ten "
            f"skrypt z katalogu, w ktorym dziala 'python -m scene_rl ...', "
            f"i w tym samym srodowisku (z zainstalowanym torch)."
        )

    device = resolve_device(args.device)
    rows = []

    for path in checkpoints:
        if not os.path.exists(path):
            print(f"POMINIETO (nie istnieje): {path}", file=sys.stderr)
            continue
        set_seed(args.seed)
        model, config, payload = build_model_from_checkpoint(path, device)
        model.eval()
        dataset = SceneDataset(args.data or config.data.path)
        summary = evaluate_model(
            model,
            dataset,
            device,
            num_scenes=args.num_scenes,
            prompt_length=args.prompt_length,
            seed=args.seed,
            greedy=True,
        )
        epoch = payload.get("epoch", "?")
        rows.append((path, epoch, summary))
        print(
            f"{path:<40} epoka {epoch!s:>5}  "
            f"exact_match={summary['exact_match']:.4f}  "
            f"content_f1={summary['content_f1']:.4f}  "
            f"order_tau={summary['order_tau']:.4f}  "
            f"braki={summary['missing']:.3f}  "
            f"halucynacje={summary['hallucinated']:.3f}"
        )

    if not rows:
        print("Brak wynikow - zadna sciezka nie istniala.")
        return

    print(
        f"\nProtokol: dekodowanie zachlanne, {args.num_scenes} scen, "
        f"fragment startowy {args.prompt_length}, ziarno {args.seed}."
    )
    print(
        "Ta tabela jest wiarygodnym zrodlem do decyzji o sufi cie modelu "
        "oraz o wyborze checkpointu 'tak sobie' - w odroznieniu od "
        "wybor_modelu_ppo.py, ktory czyta zaszumione metryki liczone "
        "probkowaniem w trakcie treningu."
    )


if __name__ == "__main__":
    main()