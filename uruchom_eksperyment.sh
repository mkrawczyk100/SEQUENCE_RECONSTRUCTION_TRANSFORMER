#!/usr/bin/env bash
# Pelny eksperyment: etap 1, trzy przebiegi PPO, porownanie.
# Uzycie: bash uruchom_eksperyment.sh [sciezka_do_danych]
set -e

DANE="${1:-sequences_przemek.txt}"
ZIARNO=1234

echo "== Autotesty implementacji =="
python -m scene_rl selftest --data "$DANE"

echo "== Statystyki zbioru danych =="
python -m scene_rl analyze --data "$DANE" | tee wyniki_analiza.txt

echo "== Tabela porownawcza metryk =="
python -m scene_rl metrics-table | tee wyniki_metryki.txt

echo "== Etap 1: uczenie nadzorowane =="
python -m scene_rl train-sft --data "$DANE" \
    --epochs 60 --out runs/sft --seed "$ZIARNO"

echo "== Etap 2: dostrajanie PPO =="
for METRYKA in levenshtein bleu rouge_l; do
    python -m scene_rl train-ppo --sft runs/sft/best.pt \
        --reward "$METRYKA" --out "runs/ppo_$METRYKA" --seed "$ZIARNO"
done

echo "== Porownanie modeli =="
python -m scene_rl compare --checkpoints \
    runs/sft/best.pt runs/ppo_levenshtein/best.pt \
    runs/ppo_bleu/best.pt runs/ppo_rouge_l/best.pt \
    --num-scenes 500 --save wyniki_porownanie.json | tee wyniki_porownanie.txt

echo "== Wykresy =="
python -m scene_rl plots runs/sft/history_sft.jsonl --out figures/sft
for METRYKA in levenshtein bleu rouge_l; do
    python -m scene_rl plots "runs/ppo_$METRYKA/history_ppo.jsonl" \
        --out "figures/ppo_$METRYKA"
done

echo "Gotowe."
