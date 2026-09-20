"""Interfejs wiersza poleceń.

Cały eksperyment daje się uruchomić bez notatnika, jednym poleceniem
na przebieg.

Przykłady:

    python -m scene_rl analyze --data sequences_przemek.txt
    python -m scene_rl metrics-table
    python -m scene_rl selftest --data sequences_przemek.txt
    python -m scene_rl train-sft --data sequences_przemek.txt \\
        --epochs 60 --out runs/sft
    python -m scene_rl train-ppo --sft runs/sft/best.pt \\
        --reward levenshtein --out runs/ppo_lev
    python -m scene_rl compare --checkpoints runs/sft/best.pt \\
        runs/ppo_lev/best.pt runs/ppo_bleu/best.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def cmd_analyze(args: argparse.Namespace) -> None:
    """Statystyki zbioru danych, materiał do rozdziału trzeciego."""
    from .data import AugmentationConfig, SceneDataset, required_block_size

    dataset = SceneDataset(args.data)
    stats = dataset.statistics()

    print("Charakterystyka zbioru danych")
    print(f"  plik                          {args.data}")
    print(f"  liczba scen                   {stats['num_scenes']}")
    print(f"  obiektów w scenie             {stats['scene_length']}")
    print(f"  unikalnych identyfikatorów    "
          f"{stats['num_unique_objects']}")
    print(f"  zakres identyfikatorów        "
          f"{stats['min_object_id']} do {stats['max_object_id']}")
    print(f"  średnia liczba wystąpień      "
          f"{stats['mean_occurrences_per_object']:.3f}")
    print(f"  sceny z powtórzonym obiektem  "
          f"{stats['scenes_with_duplicate']}")
    print(f"  sceny posortowane rosnąco     {stats['sorted_scenes']}")
    print(f"  rozmiar słownika              {stats['vocab_size']}")

    print("\nStruktura statystyczna par obiektów")
    cooccurrence = dataset.cooccurrence_statistics()
    print(f"  różnych par                   "
          f"{cooccurrence['distinct_pairs']}")
    print(f"  par występujących wielokrotnie "
          f"{cooccurrence['pairs_occurring_more_than_once']}")
    print(f"  najczęstsza para występuje     "
          f"{cooccurrence['max_pair_count']} razy")
    print("  Wniosek: brak struktury pozwalającej uogólnić")
    print("  współwystępowanie na sceny niewidziane.")

    print("\nJednoznaczność fragmentu wejściowego")
    for size in range(1, 5):
        result = dataset.context_ambiguity(size)
        print(f"  {size} obiektów: niejednoznaczny w "
              f"{100 * result['ambiguous_fraction']:.2f}% przypadków")

    augmentation = AugmentationConfig(
        min_elements=args.min_elements,
        max_remove_per_step=1,
        swap_probability=0.15,
    )
    needed = required_block_size(dataset, augmentation, num_draws=1000)
    print(f"\nDługość procesu rekonstrukcji jednej sceny")
    print(f"  maksymalna liczba tokenów     {needed - 1}")
    print(f"  minimalny block_size          {needed}")


def cmd_metrics_table(args: argparse.Namespace) -> None:
    """Odtwarza tabelę porównawczą metryk z rozdziału 2.5."""
    from .metrics import (
        bleu_score,
        levenshtein_distance,
        levenshtein_f_score,
        levenshtein_similarity,
        rouge_l_score,
    )

    reference = [742, 118, 903, 55, 671, 284, 12, 460, 837, 199]
    variants = {
        "rekonstrukcja idealna": list(reference),
        "jeden błędny obiekt (skraj)":
            reference[:9] + [999],
        "jeden błędny obiekt (środek)":
            reference[:4] + [999] + reference[5:],
        "zamiana dwóch sąsiednich":
            reference[:4] + [reference[5], reference[4]] + reference[6:],
        "odwrócona kolejność": list(reversed(reference)),
        "brak trzech ostatnich": reference[:7],
        "trzy obiekty nadmiarowe": reference + [901, 902, 904],
        "pięć ostatnich błędnych":
            reference[:5] + [991, 992, 993, 994, 995],
        "brak wspólnych obiektów":
            [901, 902, 1903, 904, 905, 906, 907, 908, 909, 910],
    }

    header = (f"{'wariant':<30}{'|Y|':>5}{'d':>4}{'s_Lev':>9}"
              f"{'BLEU':>9}{'ROUGE-L':>9}{'F(k=2)':>9}")
    print(header)
    print("-" * len(header))
    for name, candidate in variants.items():
        print(
            f"{name:<30}"
            f"{len(candidate):>5}"
            f"{levenshtein_distance(reference, candidate):>4}"
            f"{levenshtein_similarity(reference, candidate):>9.3f}"
            f"{bleu_score(reference, candidate):>9.3f}"
            f"{rouge_l_score(reference, candidate):>9.3f}"
            f"{levenshtein_f_score(reference, candidate):>9.3f}"
        )

    print("\nPorównanie z implementacjami bibliotecznymi")
    _compare_with_libraries(reference, variants)


def _compare_with_libraries(reference, variants) -> None:
    """Zestawia implementacje własne z bibliotecznymi."""
    from .metrics import bleu_score, levenshtein_distance, rouge_l_score

    try:
        from nltk.translate.bleu_score import sentence_bleu
    except ImportError:
        sentence_bleu = None
        print("  nltk niedostępny, pomijam porównanie BLEU")

    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    except ImportError:
        scorer = None
        print("  rouge_score niedostępny, pomijam porównanie ROUGE-L")

    try:
        import Levenshtein
    except ImportError:
        Levenshtein = None
        print("  python-Levenshtein niedostępny, pomijam porównanie")

    maximum_difference = 0.0
    for name, candidate in variants.items():
        line = f"  {name:<30}"
        if sentence_bleu is not None:
            library = sentence_bleu(
                [[str(t) for t in reference]],
                [str(t) for t in candidate],
            )
            own = bleu_score(reference, candidate)
            maximum_difference = max(maximum_difference, abs(own - library))
            line += f" BLEU {own:.4f}/{library:.4f}"
        if scorer is not None:
            mapping = {}
            def encode(sequence):
                words = []
                for token in sequence:
                    if token not in mapping:
                        mapping[token] = f"t{len(mapping)}"
                    words.append(mapping[token])
                return " ".join(words)
            library = scorer.score(
                encode(reference), encode(candidate)
            )["rougeL"].fmeasure
            own = rouge_l_score(reference, candidate)
            maximum_difference = max(maximum_difference, abs(own - library))
            line += f" | ROUGE {own:.4f}/{library:.4f}"
        if Levenshtein is not None:
            alphabet = {}
            def to_text(sequence):
                return "".join(
                    chr(0x100 + alphabet.setdefault(t, len(alphabet)))
                    for t in sequence
                )
            library = Levenshtein.distance(
                to_text(reference), to_text(candidate)
            )
            own = levenshtein_distance(reference, candidate)
            maximum_difference = max(
                maximum_difference, abs(own - library)
            )
            line += f" | d {own}/{library}"
        print(line)

    print(f"\n  Największa różnica własne kontra biblioteczne: "
          f"{maximum_difference:.6f}")


def cmd_capacity(args: argparse.Namespace) -> None:
    """Pojemność modelu wobec zawartości informacyjnej zbioru."""
    from .capacity import (
        BITS_PER_PARAMETER_HEADLINE,
        capacity_report,
        sweep_configurations,
    )
    from .data import SceneDataset

    dataset = SceneDataset(args.data)
    stats = dataset.statistics()

    print("Zawartość informacyjna zbioru danych")
    print(f"  wzór H(x) = N * S * log2(V), za Morris i in. (2025)")
    print(f"  scen N                        {stats['num_scenes']}")
    print(f"  obiektów w scenie S           {stats['scene_length']}")
    print(f"  różnych obiektów V            "
          f"{stats['num_unique_objects']}")
    data_bits = capacity_report(
        dataset.vocab_size, args.n_embd, args.block_size, args.n_layer,
        stats["num_scenes"], stats["scene_length"],
        stats["num_unique_objects"],
    )["data_bits"]
    print(f"  H(x)                          {data_bits:,.0f} bitów "
          f"({data_bits / 8 / 1024:.2f} KB)")

    print(f"\nPojemność modelu przy {BITS_PER_PARAMETER_HEADLINE} "
          f"bita na parametr")
    header = (f"{'n_embd':>7}{'n_layer':>8}{'parametry':>12}"
              f"{'udz.emb':>9}{'pojemność':>13}{'poj/dane':>10}"
              f"{'dane/poj':>10}")
    print(header)
    print("-" * len(header))
    rows = sweep_configurations(
        dataset.vocab_size, args.block_size,
        stats["num_scenes"], stats["scene_length"],
        stats["num_unique_objects"],
    )
    for row in rows:
        print(f"{row['n_embd']:>7.0f}{row['n_layer']:>8.0f}"
              f"{row['parameters']:>12,.0f}"
              f"{100 * row['embedding_share']:>8.0f}%"
              f"{row['capacity_bits']:>13,.0f}"
              f"{row['capacity_to_data']:>10.1f}"
              f"{row['data_to_capacity']:>10.3f}")

    print("\nInterpretacja stosunku dane/pojemność:")
    print("  << 1  model daleko poniżej pojemności, pełna memoryzacja")
    print("   ~ 1  granica pojemności, początek kompresji")
    print("  >> 1  zbiór nie mieści się w modelu")
    print("\nUwaga metodologiczna: empiryczny pomiar memoryzacji na")
    print("konkretnym checkpoincie został z tego polecenia usunięty.")
    print("Trzy niezależne próby wykazały, że model nie ma żadnej")
    print("ścieżki bezpośredniej z fragmentu do pełnej sceny: uczy się")
    print("wyłącznie procesu wieloetapowego (SEP po każdym dodanym")
    print("obiekcie). Każde zapytanie pomijające ten proces wypada")
    print("poza rozkład treningowy i daje niemiarodajny wynik. Jako")
    print("praktyczny odpowiednik 'ile model faktycznie zapamiętał'")
    print("służy rzeczywista dokładność rekonstrukcji, patrz polecenie:")
    print("  python -m scene_rl evaluate <checkpoint>")


def cmd_selftest(args: argparse.Namespace) -> None:
    from . import selftest
    selftest.run(args.data)


def _config_from_args(args: argparse.Namespace):
    from .config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig

    config = ExperimentConfig()
    config.data = DataConfig(path=args.data)
    # ModelConfig budowany jest jednym wywołaniem konstruktora, a nie
    # przez nadpisywanie pól po fakcie. Dataclassy w Pythonie nie
    # uruchamiają ponownie __post_init__ przy zwykłym przypisaniu do
    # atrybutu, więc wcześniejsza wersja tej funkcji budowała najpierw
    # poprawną konfigurację domyślną (n_embd=180, n_head=6, podzielne),
    # a dopiero potem cichcem nadpisywała n_embd wartością z wiersza
    # poleceń, omijając w ten sposób sprawdzenie podzielności n_embd
    # przez n_head. Błąd ujawniał się dopiero w środku mnożenia
    # macierzy, z komunikatem nieczytelnym dla kogokolwiek spoza
    # implementacji.
    config.model = ModelConfig(
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
    )
    config.train = TrainConfig(
        num_epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
    config.seed = args.seed
    config.device = args.device
    config.output_dir = args.out
    return config


def cmd_train_sft(args: argparse.Namespace) -> None:
    from .sft import train_sft

    config = _config_from_args(args)
    path = train_sft(config)
    print(f"\nNajlepszy checkpoint: {path}")


def cmd_train_ppo(args: argparse.Namespace) -> None:
    from .config import ExperimentConfig
    from .ppo import train_ppo

    config = ExperimentConfig()
    config.ppo.reward_metric = args.reward
    config.ppo.iterations = args.iterations
    config.ppo.episodes_per_iteration = args.episodes
    config.ppo.learning_rate = args.lr
    config.ppo.kl_coef = args.kl_coef
    config.ppo.entropy_coef = args.entropy_coef
    config.ppo.greedy_eval = args.greedy_eval
    config.seed = args.seed
    config.device = args.device
    config.output_dir = args.out
    path = train_ppo(config, args.sft)
    print(f"\nNajlepszy checkpoint: {path}")


def cmd_demo(args: argparse.Namespace) -> None:
    """Pokazuje konkretne przykłady rekonstrukcji, obok siebie."""
    import random

    from .data import SceneDataset, make_prompt
    from .evaluation import reconstruct
    from .utils import build_model_from_checkpoint, resolve_device

    device = resolve_device(args.device)
    model, config, _ = build_model_from_checkpoint(args.checkpoint, device)
    model.eval()
    dataset = SceneDataset(args.data or config.data.path)

    rng = random.Random(args.seed)
    scene_ids = rng.sample(range(dataset.num_scenes), args.n)
    references = [dataset.scenes[i] for i in scene_ids]
    prompts = [
        make_prompt(scene, args.prompt_length, rng) for scene in references
    ]
    restored = reconstruct(
        model, prompts, device, greedy=args.greedy, temperature=1.0
    )

    def to_objects(tokens):
        return [dataset.itos.get(t, t) for t in tokens]

    hits = 0
    print(f"Model: {args.checkpoint}")
    print(f"Tryb: {'zachłanny (deterministyczny)' if args.greedy else 'próbkowanie (losowy)'}")
    print(f"Fragment wejściowy: {args.prompt_length} obiektów\n")

    for i, (scene_id, prompt, reference, candidate) in enumerate(
        zip(scene_ids, prompts, references, restored), start=1
    ):
        candidate = candidate or []
        ok = candidate == reference
        hits += ok
        znak = "✓ ZGADZA SIĘ" if ok else "✗ RÓŻNI SIĘ"
        reference_obj = to_objects(reference)
        candidate_obj = to_objects(candidate)
        print(f"--- Scena #{scene_id}  [{znak}] ---")
        print(f"  Fragment podany modelowi: {to_objects(prompt)}")
        print(f"  Prawdziwa scena:          {reference_obj}")
        print(f"  Model wygenerował:        {candidate_obj}")
        if not ok:
            # Liczone na numerach obiektów (po przeliczeniu przez itos),
            # nie na wewnętrznych tokenach modelu, żeby liczby zgadzały
            # się z tym, co widać w dwóch liniach wyżej.
            missing = [o for o in reference_obj if o not in candidate_obj]
            extra = [o for o in candidate_obj if o not in reference_obj]
            if missing:
                print(f"    brakuje: {missing}")
            if extra:
                print(f"    dodatkowo (nie powinno tam być): {extra}")
        print()

    print(f"Podsumowanie: {hits}/{args.n} scen odtworzonych bezbłędnie")


def cmd_evaluate(args: argparse.Namespace) -> None:
    from .data import SceneDataset
    from .evaluation import evaluate_by_prompt_length, evaluate_model
    from .utils import build_model_from_checkpoint, resolve_device, set_seed

    device = resolve_device(args.device)
    set_seed(args.seed)
    model, config, _ = build_model_from_checkpoint(args.checkpoint, device)
    model.eval()
    dataset = SceneDataset(args.data or config.data.path)

    summary = evaluate_model(
        model, dataset, device,
        num_scenes=args.num_scenes,
        prompt_length=args.prompt_length,
        seed=args.seed,
        greedy=args.greedy,
    )
    print(f"Model: {args.checkpoint}")
    for key, value in summary.items():
        print(f"  {key:<22}{value:>10.4f}")

    if args.by_length:
        print("\nWyniki w funkcji długości fragmentu wejściowego")
        results = evaluate_by_prompt_length(
            model, dataset, device,
            num_scenes=args.num_scenes, seed=args.seed, greedy=args.greedy
        )
        print(f"{'k':>3}{'F1 treści':>12}{'tau':>10}"
              f"{'dokładne':>11}{'braki':>9}{'halucyn.':>10}")
        for length, result in results.items():
            print(f"{length:>3}{result['content_f1']:>12.4f}"
                  f"{result['order_tau']:>10.4f}"
                  f"{result['exact_match']:>11.4f}"
                  f"{result['missing']:>9.2f}"
                  f"{result['hallucinated']:>10.2f}")

    if args.save:
        with open(args.save, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)


def cmd_compare(args: argparse.Namespace) -> None:
    """Zestawia modele w mierze docelowej. Tabela do rozdziału 5."""
    from .data import SceneDataset
    from .evaluation import evaluate_model
    from .utils import build_model_from_checkpoint, resolve_device, set_seed

    device = resolve_device(args.device)
    rows = []
    for path in args.checkpoints:
        set_seed(args.seed)
        model, config, payload = build_model_from_checkpoint(path, device)
        model.eval()
        dataset = SceneDataset(args.data or config.data.path)
        summary = evaluate_model(
            model, dataset, device,
            num_scenes=args.num_scenes,
            prompt_length=args.prompt_length,
            seed=args.seed,
            greedy=args.greedy,
        )
        label = config.ppo.reward_metric if "ppo" in path else "SFT"
        rows.append((os.path.basename(os.path.dirname(path)), label, summary))

    print(f"\nPorównanie modeli, {args.num_scenes} scen, fragment "
          f"{args.prompt_length} obiektów, ziarno {args.seed}\n")
    header = (f"{'przebieg':<16}{'nagroda':<13}{'F1 treści':>11}"
              f"{'tau':>9}{'dokładne':>10}{'braki':>8}{'halucyn.':>10}"
              f"{'lev':>8}{'bleu':>8}{'rouge':>8}")
    print(header)
    print("-" * len(header))
    for name, label, summary in rows:
        print(f"{name:<16}{label:<13}{summary['content_f1']:>11.4f}"
              f"{summary['order_tau']:>9.4f}{summary['exact_match']:>10.4f}"
              f"{summary['missing']:>8.2f}{summary['hallucinated']:>10.2f}"
              f"{summary['levenshtein']:>8.4f}{summary['bleu']:>8.4f}"
              f"{summary['rouge_l']:>8.4f}")

    print("\nMiara docelowa (F1 treści, tau, rekonstrukcje dokładne) jest")
    print("niezależna od metryk użytych jako nagroda, więc porównanie")
    print("nie faworyzuje żadnego z modeli z definicji.")

    if args.save:
        with open(args.save, "w", encoding="utf-8") as handle:
            json.dump(
                [{"run": n, "reward": l, **s} for n, l, s in rows],
                handle, indent=2, ensure_ascii=False,
            )


def cmd_plots(args: argparse.Namespace) -> None:
    """Rysuje wykresy przebiegu treningu na podstawie historii."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = []
    with open(args.history, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    os.makedirs(args.out, exist_ok=True)

    def series(key):
        xs, ys = [], []
        for record in records:
            if key in record:
                xs.append(record.get("epoch", record.get("iteration", 0)))
                ys.append(record[key])
        return xs, ys

    figure, axis = plt.subplots(figsize=(7, 4.2))
    for key, label in (("train_loss", "zbiór uczący"),
                       ("val_loss", "zbiór walidacyjny")):
        xs, ys = series(key)
        if xs:
            axis.plot(xs, ys, label=label)
    axis.set_xlabel("epoka")
    axis.set_ylabel("entropia krzyżowa")
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(os.path.join(args.out, "strata.png"), dpi=200)

    figure, axis = plt.subplots(figsize=(7, 4.2))
    for key, label in (("eval_levenshtein", "podobieństwo edycyjne"),
                       ("eval_bleu", "BLEU"),
                       ("eval_rouge_l", "ROUGE-L")):
        xs, ys = series(key)
        if xs:
            axis.plot(xs, ys, marker="o", markersize=3, label=label)
    axis.set_xlabel("epoka")
    axis.set_ylabel("wartość metryki")
    axis.set_ylim(0, 1)
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(os.path.join(args.out, "metryki.png"), dpi=200)

    figure, axis = plt.subplots(figsize=(7, 4.2))
    for key, label in (("eval_content_f1", "F1 zawartości"),
                       ("eval_order_tau", "zgodność kolejności"),
                       ("eval_exact_match", "rekonstrukcje dokładne")):
        xs, ys = series(key)
        if xs:
            axis.plot(xs, ys, marker="o", markersize=3, label=label)
    axis.set_xlabel("epoka")
    axis.set_ylabel("wartość miary docelowej")
    axis.set_ylim(0, 1)
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(os.path.join(args.out, "miara_docelowa.png"), dpi=200)

    print(f"Zapisano wykresy w katalogu {args.out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scene_rl",
        description="Rekonstrukcja sekwencji transformerem, "
                    "metryki podobieństwa i uczenie ze wzmocnieniem.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p = subparsers.add_parser("analyze", help="statystyki zbioru danych")
    p.add_argument("--data", default="sequences_przemek.txt")
    p.add_argument("--min-elements", type=int, default=3)
    p.set_defaults(func=cmd_analyze)

    p = subparsers.add_parser(
        "metrics-table", help="tabela porównawcza metryk"
    )
    p.set_defaults(func=cmd_metrics_table)

    p = subparsers.add_parser(
        "capacity", help="pojemność modelu a zawartość informacyjna danych"
    )
    p.add_argument("--data", default="sequences_przemek.txt")
    p.add_argument("--n-embd", type=int, default=180)
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--block-size", type=int, default=64)
    p.set_defaults(func=cmd_capacity)

    p = subparsers.add_parser("selftest", help="autotesty implementacji")
    p.add_argument("--data", default=None)
    p.set_defaults(func=cmd_selftest)

    p = subparsers.add_parser("train-sft", help="etap 1: uczenie nadzorowane")
    p.add_argument("--data", default="sequences_przemek.txt")
    p.add_argument("--out", default="runs/sft")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--steps-per-epoch", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--n-embd", type=int, default=180)
    p.add_argument("--n-head", type=int, default=6)
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="auto")
    p.set_defaults(func=cmd_train_sft)

    p = subparsers.add_parser("train-ppo", help="etap 2: dostrajanie PPO")
    p.add_argument("--sft", required=True, help="checkpoint po etapie 1")
    p.add_argument("--reward", default="levenshtein",
                   choices=["levenshtein", "bleu", "rouge_l"])
    p.add_argument("--out", default="runs/ppo")
    p.add_argument("--iterations", type=int, default=300)
    p.add_argument("--episodes", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--kl-coef", type=float, default=0.02)
    p.add_argument("--entropy-coef", type=float, default=0.0,
                   help="premia entropijna; 0 wyłącza eksplorację")
    p.add_argument("--greedy-eval", action="store_true",
                   help="ewaluacja zachłanna, bez losowania tokenów")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="auto")
    p.set_defaults(func=cmd_train_ppo)

    p = subparsers.add_parser(
        "demo", help="pokaż konkretne przykłady rekonstrukcji"
    )
    p.add_argument("checkpoint")
    p.add_argument("--data", default=None)
    p.add_argument("--n", type=int, default=10,
                   help="ile przykładowych scen pokazać")
    p.add_argument("--prompt-length", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--greedy", action="store_true",
                   help="tryb deterministyczny zamiast losowego")
    p.add_argument("--device", default="auto")
    p.set_defaults(func=cmd_demo)

    p = subparsers.add_parser("evaluate", help="ocena jednego modelu")
    p.add_argument("checkpoint")
    p.add_argument("--data", default=None)
    p.add_argument("--num-scenes", type=int, default=500)
    p.add_argument("--prompt-length", type=int, default=6)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--by-length", action="store_true")
    p.add_argument("--save", default=None)
    p.add_argument("--device", default="auto")
    p.set_defaults(func=cmd_evaluate)

    p = subparsers.add_parser("compare", help="porównanie modeli")
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--data", default=None)
    p.add_argument("--num-scenes", type=int, default=500)
    p.add_argument("--prompt-length", type=int, default=6)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--save", default=None)
    p.add_argument("--device", default="auto")
    p.set_defaults(func=cmd_compare)

    p = subparsers.add_parser("plots", help="wykresy z historii treningu")
    p.add_argument("history")
    p.add_argument("--out", default="figures")
    p.set_defaults(func=cmd_plots)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
