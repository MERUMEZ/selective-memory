#!/bin/bash
# Полный прогон НА УМОЛЧАНИЯХ. Никаких флагов кроме кодировщика: числа
# для документов должны быть получены на том, что реально ставится
# по умолчанию, а не на конфигурации из ручек.
cd /var/www/mindnumbness
OUT=storage/bench/full_run_defaults.txt
: > "$OUT"
for f in storage/bench/shards/*.json; do
  echo "=== $(basename $f) ===" >> "$OUT"
  ./venv/bin/python tools/bench_longmemeval.py --data "$f" \
      --encoder potion --threshold 0.0 2>/dev/null >> "$OUT"
done
echo "ГОТОВО" >> "$OUT"
