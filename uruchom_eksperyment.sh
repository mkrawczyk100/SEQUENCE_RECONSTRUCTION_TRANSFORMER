#!/usr/bin/env bash
# Pełny eksperyment z pracy magisterskiej, w kolejności z rozdziałów 4 i 5.
# Użycie: bash uruchom_eksperyment.sh [sciezka_do_danych]
#
# Czas zmierzony na karcie RTX 4050 Laptop GPU:
#   uczenie nadzorowane modelu e96 (60 epok)   ~24 min
#   seria główna PPO (9 x 1000 iteracji)       ~12,7 h
#   trening wydłużony (3 x 3000 iteracji)      ~6,4 h na przebieg
#
# Model e180 zapisywany jest w katalogu runs/sft.
# Plik danych nazywa się sequences_przemek.txt (w tekście pracy: sequences.txt).
set -e

DANE="${1:-sequences_przemek.txt}"
ZIARNO=1234
START=runs/sft_e96/epoch_0004.pt
mkdir -p dane_wynikowe

echo "== Krok 1: autotesty, analiza danych, tabela metryk =="
python -m scene_rl selftest --data "$DANE"
python -m scene_rl analyze --data "$DANE" | tee dane_wynikowe/analiza.txt
python -m scene_rl metrics-table | tee dane_wynikowe/metryki.txt

echo "== Krok 2: uczenie nadzorowane trzech architektur =="
python -m scene_rl train-sft --data "$DANE" \
    --n-embd 64  --n-head 4 --n-layer 4 --seed "$ZIARNO" --out runs/sft_e64
python -m scene_rl train-sft --data "$DANE" \
    --n-embd 96  --n-head 4 --n-layer 4 --seed "$ZIARNO" --out runs/sft_e96
python -m scene_rl train-sft --data "$DANE" \
    --n-embd 180 --n-head 6 --n-layer 6 --seed "$ZIARNO" --out runs/sft
cp runs/sft_e64/history_sft.jsonl dane_wynikowe/history_e64.jsonl
cp runs/sft_e96/history_sft.jsonl dane_wynikowe/history_e96.jsonl
cp runs/sft/history_sft.jsonl     dane_wynikowe/history_e180.jsonl

# Tabela 5.4, Rys. 5.6: trajektoria modelu e96 i jego wynik po pełnym treningu
python tabela_koncowa_ppo.py \
    --checkpoint "SFT e96 epoka 3=runs/sft_e96/epoch_0003.pt" \
    --checkpoint "SFT e96 epoka 4 (start PPO)=$START" \
    --checkpoint "SFT e96 epoka 5=runs/sft_e96/epoch_0005.pt" \
    --checkpoint "SFT e96 epoka 7=runs/sft_e96/epoch_0007.pt" \
    --checkpoint "SFT e96 epoka 10=runs/sft_e96/epoch_0010.pt" \
    --checkpoint "SFT e96 pelny trening (sufit)=runs/sft_e96/best.pt" \
    --prompt-lengths 3 4 5 6 8 --num-scenes 500 --seed 3 \
    --csv dane_wynikowe/wyniki_odniesienia_z5.csv \
    --json dane_wynikowe/wyniki_odniesienia_z5.json

echo "== Krok 3: dobór współczynnika kary za rozbieżność =="
for KL in 000:0.0 001:0.01 005:0.05; do
  python uruchom_ppo_wieloziarnowo.py --sft "$START" \
      --metrics levenshtein --seeds "$ZIARNO" --iterations 150 \
      --kl-coef "${KL#*:}" --eval-every 10 --eval-samples 200 \
      --tag "pilotkl${KL%%:*}"
  python zbierz_krzywe_ppo.py --tag "pilotkl${KL%%:*}" --eval-scenes 200 \
      --csv "dane_wynikowe/krzywe_kl${KL%%:*}.csv"
done

echo "== Krok 4: seria główna PPO =="
python uruchom_ppo_wieloziarnowo.py --sft "$START" \
    --metrics levenshtein bleu rouge_l --seeds 1234 2025 7 \
    --iterations 1000 --kl-coef 0.01 --eval-every 20 --eval-samples 500 \
    --tag glowny

echo "== Krok 5: trening wydłużony do 3000 iteracji =="
# Ewaluacja okresowa tych przebiegów: co 10 iteracji, 200 scen.
python uruchom_ppo_wieloziarnowo.py --sft "$START" \
    --metrics levenshtein --seeds 1234 2025 7 \
    --iterations 3000 --kl-coef 0.01 --eval-every 10 --eval-samples 200 \
    --tag glowny_long

echo "== Krok 6: krzywe uczenia uśrednione po ziarnach =="
# Tag 'glowny' obejmuje również katalogi glowny_long; do wykresów serii
# głównej brać tylko metryki levenshtein, bleu, rouge_l.
python zbierz_krzywe_ppo.py --tag glowny --eval-scenes 500 \
    --csv dane_wynikowe/krzywe_ppo_glowny.csv
python zbierz_krzywe_ppo.py --tag glowny_long --eval-scenes 200 \
    --csv dane_wynikowe/krzywe_ppo_sonda3000.csv

echo "== Krok 7: tabele końcowe =="
ARGS_PPO=()
for METRYKA in levenshtein:Levenshtein bleu:BLEU rouge_l:ROUGE-L; do
  for Z in 1234 2025 7; do
    ARGS_PPO+=(--checkpoint "PPO ${METRYKA#*:} z$Z (1000 iter)=runs/ppo_glowny_${METRYKA%%:*}_seed$Z/best.pt")
  done
done
ARGS_LONG=()
for Z in 1234 2025 7; do
  ARGS_LONG+=(--checkpoint "PPO Levenshtein z$Z (3000 iter)=runs/ppo_glowny_long_levenshtein_seed$Z/best.pt")
done

# Tabele 5.6-5.7, Rys. 5.9 i 5.11: wszystkie warianty, pięć długości fragmentu
python tabela_koncowa_ppo.py \
    --checkpoint "SFT e96 epoka 4 (start)=$START" \
    --checkpoint "SFT e96 sufit=runs/sft_e96/best.pt" \
    "${ARGS_PPO[@]}" "${ARGS_LONG[@]}" \
    --prompt-lengths 3 4 5 6 8 --num-scenes 500 --seed 3 \
    --csv dane_wynikowe/wyniki_tabela_koncowa_z5.csv \
    --json dane_wynikowe/wyniki_tabela_koncowa_z5.json

# Tabela 5.9: checkpointy last.pt serii głównej, cztery długości fragmentu
ARGS_LAST=()
for METRYKA in levenshtein:Levenshtein bleu:BLEU rouge_l:ROUGE-L; do
  for Z in 1234 2025 7; do
    ARGS_LAST+=(--checkpoint "PPO ${METRYKA#*:} z$Z (1000 iter, last)=runs/ppo_glowny_${METRYKA%%:*}_seed$Z/last.pt")
  done
done
python tabela_koncowa_ppo.py "${ARGS_LAST[@]}" \
    --prompt-lengths 3 4 6 8 --num-scenes 500 --seed 3 \
    --csv dane_wynikowe/wyniki_tabela_koncowa_last.csv \
    --json dane_wynikowe/wyniki_tabela_koncowa_last.json

# Tabela 5.3: modele e64 i e180 po pełnym treningu
python tabela_koncowa_ppo.py \
    --checkpoint "SFT e64 sufit=runs/sft_e64/best.pt" \
    --checkpoint "SFT e180 sufit=runs/sft/best.pt" \
    --prompt-lengths 3 4 6 8 --num-scenes 500 --seed 3 \
    --csv dane_wynikowe/wyniki_e64_e180_sufit.csv \
    --json dane_wynikowe/wyniki_e64_e180_sufit.json

echo "== Krok 8: przykłady rekonstrukcji =="
for K in 3 4 5; do
  python -m scene_rl demo "$START" --n 6 --prompt-length "$K" --greedy \
      > "dane_wynikowe/demo_start_len$K.txt"
  python -m scene_rl demo runs/ppo_glowny_long_levenshtein_seed1234/best.pt \
      --n 6 --prompt-length "$K" --greedy \
      > "dane_wynikowe/demo_ppo3000_len$K.txt"
done

echo "== Krok 9: pojemność modeli (Tabela 5.12) =="
python -m scene_rl capacity | tee dane_wynikowe/pojemnosc.txt

echo "Gotowe."
