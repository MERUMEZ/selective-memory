#!/bin/bash
# Полный прогон с НОВОЙ политикой: удаления по возрасту нет.
# На одном шарде дало +46 пунктов R@5; проверяем, держится ли на 500.
cd /var/www/mindnumbness
OUT=storage/bench/full_run_keepall.txt
: > "$OUT"
for f in storage/bench/shards/*.json; do
  echo "=== $(basename $f) ===" >> "$OUT"
  ./venv/bin/python tools/bench_longmemeval.py --data "$f" \
      --encoder potion --threshold 0.0 --keep-all 2>/dev/null >> "$OUT"
done
echo "ГОТОВО" >> "$OUT"
