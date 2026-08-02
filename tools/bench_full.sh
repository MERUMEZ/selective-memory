#!/bin/bash
# Полный прогон LongMemEval по шардам: файл целиком не влезает в память
# вместе с моделью эмбеддингов (2.4 ГБ на загрузку JSON + 351 МБ модель
# при 3.7 ГБ на машине). Шарды по 50 вопросов грузятся по 26 МБ.
cd /var/www/mindnumbness
OUT=storage/bench/full_run.txt
: > "$OUT"
for f in storage/bench/shards/*.json; do
  echo "=== $(basename $f) ===" >> "$OUT"
  ./venv/bin/python tools/bench_longmemeval.py --data "$f" \
      --encoder potion --threshold 0.0 2>/dev/null >> "$OUT"
done
echo "ГОТОВО" >> "$OUT"
