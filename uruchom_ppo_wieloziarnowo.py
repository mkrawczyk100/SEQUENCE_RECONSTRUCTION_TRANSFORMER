#!/usr/bin/env python3
"""Sterownik przebiegów PPO: wiele metryk nagrody x wiele ziaren losowości.

Po co ten skrypt: obecne polecenie `python -m scene_rl train-ppo` uruchamia
dokładnie jeden przebieg, przy jednym ziarnie, i nie wystawia w wierszu
poleceń kilku parametrów PPOConfig, które są tu potrzebne (eval_samples,
eval_every, greedy_eval, prompt_min, prompt_max). Ten skrypt nie zmienia
niczego w pakiecie scene_rl, tylko buduje ExperimentConfig bezpośrednio
i wywołuje `scene_rl.ppo.train_ppo` w pętli po metrykach i ziarnach.

Trzy rzeczy, które robi inaczej niż dotychczasowe przebiegi:

1. Ewaluacja okresowa w trakcie PPO jest ZACHŁANNA (greedy_eval=True).
   Domyślnie scene_rl liczy ją próbkowaniem z temperaturą 1,0, przez co
   krzywe treningu są zdominowane przez szum losowania tokenów. To jest
   dokładnie zarzut z komentarza promotora o nieczytelnych wykresach.
2. Każda kombinacja (metryka, ziarno) trafia do osobnego katalogu, więc
   krzywe da się potem uśrednić po ziarnach zamiast pokazywać pojedynczy
   przebieg.
3. Tryb `--pilot` pozwala zmierzyć czas jednej iteracji na małej liczbie
   iteracji, zanim uruchomi się pełny, wielogodzinny eksperyment.

Przykłady użycia opisano w instrukcji, w skrócie:

    # pomiar czasu iteracji
    python uruchom_ppo_wieloziarnowo.py --sft runs/sft_e96/epoch_0004.pt \\
        --pilot --iterations 20

    # pełna siatka
    python uruchom_ppo_wieloziarnowo.py --sft runs/sft_e96/epoch_0004.pt \\
        --metrics levenshtein bleu rouge_l --seeds 1234 2025 7 \\
        --iterations 1500 --kl-coef 0.01 --tag glowny
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

try:  # zabezpieczenie przed konsolą Windows w cp1250
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def build_config(args, metric: str, seed: int, output_dir: str):
    from scene_rl.config import ExperimentConfig

    config = ExperimentConfig()
    config.seed = seed
    config.eval_seed = args.eval_seed
    config.device = args.device
    config.output_dir = output_dir
    config.run_name = f"ppo_{metric}_seed{seed}"

    # Długość fragmentu startowego używana przez ewaluację okresową
    # wewnątrz train_ppo (train_ppo czyta config.train.prompt_length).
    config.train.prompt_length = args.prompt_length

    ppo = config.ppo
    ppo.reward_metric = metric
    ppo.iterations = args.iterations
    ppo.episodes_per_iteration = args.episodes
    ppo.learning_rate = args.lr
    ppo.kl_coef = args.kl_coef
    ppo.entropy_coef = args.entropy_coef
    ppo.eval_every = args.eval_every
    ppo.eval_samples = args.eval_samples
    ppo.prompt_min = args.prompt_min
    ppo.prompt_max = args.prompt_max
    ppo.greedy_eval = not args.sampled_eval
    return config


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Uruchamia przebiegi PPO dla wielu metryk nagrody i wielu ziaren "
            "losowości, startując z jednego wspólnego checkpointu SFT."
        )
    )
    parser.add_argument("--sft", required=True,
                        help="Checkpoint startowy, np. runs/sft_e96/epoch_0004.pt")
    parser.add_argument("--metrics", nargs="+",
                        default=["levenshtein", "bleu", "rouge_l"],
                        choices=["levenshtein", "bleu", "rouge_l"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1234, 2025, 7])
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--kl-coef", type=float, default=0.01)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-samples", type=int, default=200)
    parser.add_argument("--prompt-min", type=int, default=3)
    parser.add_argument("--prompt-max", type=int, default=8)
    parser.add_argument("--prompt-length", type=int, default=6,
                        help="Fragment startowy używany w ewaluacji okresowej.")
    parser.add_argument("--eval-seed", type=int, default=3)
    parser.add_argument("--sampled-eval", action="store_true",
                        help=("Wraca do ewaluacji okresowej przez próbkowanie "
                              "(domyślnie zachłanna). Nie zalecane."))
    parser.add_argument("--out-root", default="runs")
    parser.add_argument("--tag", default="glowny",
                        help="Etykieta serii, wchodzi w nazwę katalogu.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pilot", action="store_true",
                        help=("Tryb pomiaru czasu: jeden przebieg, pierwsza "
                              "metryka i pierwsze ziarno, katalog z sufiksem "
                              "_pilot."))
    parser.add_argument("--skip-existing", action="store_true",
                        help=("Pomija kombinacje, dla których istnieje już "
                              "history_ppo.jsonl. Pozwala wznowić przerwaną "
                              "serię bez powtarzania gotowych przebiegów."))
    parser.add_argument("--dry-run", action="store_true",
                        help="Wypisuje plan przebiegów i kończy działanie.")
    args = parser.parse_args()

    if not os.path.exists(args.sft):
        raise SystemExit(f"Nie istnieje checkpoint startowy: {args.sft}")

    metrics = args.metrics[:1] if args.pilot else args.metrics
    seeds = args.seeds[:1] if args.pilot else args.seeds
    tag = f"{args.tag}_pilot" if args.pilot else args.tag

    plan = []
    for metric in metrics:
        for seed in seeds:
            output_dir = os.path.join(
                args.out_root, f"ppo_{tag}_{metric}_seed{seed}"
            )
            plan.append((metric, seed, output_dir))

    print("Plan przebiegów:")
    for metric, seed, output_dir in plan:
        print(f"  nagroda={metric:<12} ziarno={seed:<6} -> {output_dir}")
    print(f"  iteracji na przebieg: {args.iterations}")
    print(f"  epizodów na iterację: {args.episodes}")
    print(f"  tempo uczenia: {args.lr}, kara KL: {args.kl_coef}, "
          f"entropia: {args.entropy_coef}")
    print(f"  ewaluacja okresowa: "
          f"{'próbkowanie' if args.sampled_eval else 'zachłanna'}, "
          f"co {args.eval_every} iteracji, {args.eval_samples} scen")
    print(f"  checkpoint startowy: {args.sft}\n")

    if args.dry_run:
        return

    try:
        from scene_rl.ppo import train_ppo
    except ImportError as exc:
        raise SystemExit(
            f"Nie udało się zaimportować scene_rl ({exc}). Uruchom skrypt "
            f"z katalogu, w którym działa 'python -m scene_rl ...', w tym "
            f"samym środowisku."
        )

    summary = []
    total_start = time.time()

    for index, (metric, seed, output_dir) in enumerate(plan, start=1):
        history_path = os.path.join(output_dir, "history_ppo.jsonl")
        if args.skip_existing and os.path.exists(history_path):
            print(f"[{index}/{len(plan)}] POMINIETO (istnieje): {output_dir}")
            continue

        if os.path.exists(history_path):
            # HistoryLogger dopisuje do pliku, więc pozostawienie starej
            # historii zmieszałoby dwa przebiegi w jednym pliku.
            backup = history_path + f".stary_{int(time.time())}"
            os.replace(history_path, backup)
            print(f"  (poprzednia historia przeniesiona do {backup})")

        print(f"\n[{index}/{len(plan)}] START nagroda={metric} ziarno={seed}")
        print(f"    katalog wyjściowy: {output_dir}")
        start = time.time()

        config = build_config(args, metric, seed, output_dir)
        best_path = train_ppo(config, args.sft)

        elapsed = time.time() - start
        per_iteration = elapsed / max(args.iterations, 1)
        print(f"[{index}/{len(plan)}] KONIEC po {elapsed / 60:.1f} min "
              f"({per_iteration:.2f} s na iterację)")
        print(f"    najlepszy checkpoint: {best_path}")

        summary.append({
            "metric": metric,
            "seed": seed,
            "output_dir": output_dir,
            "best_checkpoint": best_path,
            "iterations": args.iterations,
            "elapsed_s": round(elapsed, 1),
            "seconds_per_iteration": round(per_iteration, 3),
        })

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    total_elapsed = time.time() - total_start
    summary_path = os.path.join(args.out_root, f"podsumowanie_ppo_{tag}.json")
    os.makedirs(args.out_root, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "sft_checkpoint": args.sft,
                "iterations": args.iterations,
                "episodes_per_iteration": args.episodes,
                "learning_rate": args.lr,
                "kl_coef": args.kl_coef,
                "entropy_coef": args.entropy_coef,
                "greedy_eval": not args.sampled_eval,
                "total_elapsed_s": round(total_elapsed, 1),
                "runs": summary,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nCałość zajęła {total_elapsed / 60:.1f} min.")
    print(f"Podsumowanie zapisane w {summary_path}")
    if summary:
        mean_per_iteration = sum(
            r["seconds_per_iteration"] for r in summary
        ) / len(summary)
        print(f"Średni czas iteracji: {mean_per_iteration:.2f} s")
        print(
            f"Szacunek dla 1000 iteracji na przebieg: "
            f"{mean_per_iteration * 1000 / 3600:.2f} h na przebieg."
        )


if __name__ == "__main__":
    main()
