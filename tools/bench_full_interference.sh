#!/bin/bash
# Полный перемер на модели интерференции. Ни одно прежнее число не
# переносится: важность больше не производная от возраста.
cd /var/www/mindnumbness
OUT=storage/bench/full_run_interference.txt
: > "$OUT"
for f in storage/bench/shards/*.json; do
  echo "=== $(basename $f) ===" >> "$OUT"
  ./venv/bin/python tools/bench_longmemeval.py --data "$f" \
      --encoder potion --threshold 0.0 --keep-all --interference 2>/dev/null >> "$OUT"
done
echo "ГОТОВО" >> "$OUT"
