#!/bin/bash
# Тот же полный прогон, но режимом archive: гейт обойдён, забывание
# выключено. Разница с normal и есть цена политики забывания, и мерить
# её надо на той же выборке — иначе сравниваются разные вещи.
cd /var/www/mindnumbness
OUT=storage/bench/full_run_archive.txt
: > "$OUT"
for f in storage/bench/shards/*.json; do
  echo "=== $(basename $f) ===" >> "$OUT"
  ./venv/bin/python tools/bench_longmemeval.py --data "$f" \
      --mode archive --encoder potion --threshold 0.0 2>/dev/null >> "$OUT"
done
echo "ГОТОВО" >> "$OUT"
