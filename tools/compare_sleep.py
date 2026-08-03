"""
================================================================================
 TOOLS/COMPARE_SLEEP.PY — Сон обещает сжатие. Проверяем, чем оно оплачено
================================================================================
Сон делает две вещи: подрезает слабые связи и осиротевшие узлы, и
сворачивает плотный кластер в один абстрактный узел взамен многих.

Обещание, стало быть, такое: ПАМЯТЬ СТАНОВИТСЯ МЕНЬШЕ, А НАХОДИМОСТЬ НЕ
ПАДАЕТ. Сжатие без потери. Если узлов стало меньше и найти по-прежнему
можно всё — сон полезен. Если вместе с узлами пропали ответы — это не
консолидация, а потеря данных под красивым именем.

ПОЧЕМУ ОТДЕЛЬНЫЙ СТЕНД. Вклад сна нельзя померить ни на LongMemEval, ни
на compare_retention: они не вызывают sleep() вовсе. За эту работу
дважды подряд был построен замер, который по устройству не мог
сработать, и оба раза числа выходили одинаковыми — механизм просто не
включался. Отсюда правило: сначала убедиться, что стенд создаёт условия
для срабатывания, и только потом сравнивать.

Здесь условия создаются явно: разговор достаточно плотный, чтобы
образовались хабы, связывание включено (без рёбер кластеров не бывает),
и sleep() вызывается прямо.

Запуск:
    python tools/compare_sleep.py
    python tools/compare_sleep.py --seeds 1,7,13,42,99 --rounds 3
================================================================================
"""

import argparse
import os
import random
import statistics
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if "--logs" not in sys.argv:
    os.environ["LOG_LEVEL"] = "ERROR"

from selectivemem import Memory, MemorySettings  # noqa: E402

DAY = 86400.0

# Темы плотные и пересекающиеся: сон сворачивает КЛАСТЕРЫ, поэтому
# разговор должен возвращаться к одному кругу вещей разными словами.
# На разрозненных репликах хабов не образуется и мерить будет нечего.
TOPICS = [
    "у меня аллергия на пенициллин",
    "врач выписал азитромицин вместо пенициллина",
    "аптека на углу азитромицин не держит",
    "рецепт на азитромицин действует месяц",
    "моя дочь учится в третьем классе",
    "дочь ходит в школу пешком через парк",
    "в школе перенесли родительское собрание",
    "собрание в школе будет в пятницу вечером",
    "моя машина стоит в подземном паркинге",
    "шлагбаум паркинга заедает на выезде",
    "паркинг подорожал с начала года",
    "в паркинге поставили новые камеры",
]


def run_once(seed: int, rounds: int, do_sleep: bool) -> Dict[str, float]:
    rng = random.Random(seed)
    now = [1_700_000_000.0]
    memory = Memory(
        ":memory:",
        settings=MemorySettings(
            associate_recalled_limit=3,   # без связей кластеров не бывает
            delete_on_decay=False,        # мерим сон, а не удаление по возрасту
        ),
        clock=lambda: now[0],
    )

    stream = TOPICS * rounds
    rng.shuffle(stream)
    for text in stream:
        memory.context_for(text, top_k=3)
        memory.observe(text, response="понятно", emotion=0.6)
        now[0] += 180.0

    now[0] += DAY
    memory.forget(now=now[0])

    before = memory.graph.db.count_nodes_by_type("episodic")
    if do_sleep:
        memory.sleep(timestamp=now[0])
    after = memory.graph.db.count_nodes_by_type("episodic")

    found = 0
    for topic in TOPICS:
        matches = memory.recall(topic, top_k=5, timestamp=now[0])
        text = " | ".join(f"{m.context} {m.response}" for m in matches)
        if topic.lower() in text.lower():
            found += 1

    memory.close()
    return {
        "nodes_before": before,
        "nodes_after": after,
        "recall": found / len(TOPICS),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Сжимает ли сон без потери")
    parser.add_argument("--seeds", default="1,7,13,42,99")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--logs", action="store_true")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]

    print("=" * 74)
    print(" СЖИМАЕТ ЛИ СОН БЕЗ ПОТЕРИ")
    print("=" * 74)
    print(f" Тем: {len(TOPICS)} (плотных, пересекающихся), кругов: {args.rounds},")
    print(f" сидов: {len(seeds)}. Связывание включено — без рёбер нет кластеров.")
    print("-" * 74)
    print(f" {'':16} {'узлов до':>10} {'узлов после':>13} {'найдено тем':>13}")

    results = {}
    for do_sleep in (False, True):
        rows = [run_once(seed, args.rounds, do_sleep) for seed in seeds]
        before = statistics.mean(r["nodes_before"] for r in rows)
        after = statistics.mean(r["nodes_after"] for r in rows)
        recall = statistics.mean(r["recall"] for r in rows)
        results[do_sleep] = (after, recall)
        label = "со сном" if do_sleep else "без сна"
        print(f" {label:16} {before:10.1f} {after:13.1f} {recall*100:12.1f}%")

    print("=" * 74)
    (nodes_no, recall_no) = results[False]
    (nodes_yes, recall_yes) = results[True]
    saved = (nodes_no - nodes_yes) / nodes_no * 100 if nodes_no else 0.0
    lost = (recall_no - recall_yes) * 100
    print(f" СЖАТИЕ: {saved:+.1f}% узлов     ПОТЕРЯ НАХОДИМОСТИ: {lost:+.1f} пунктов")
    if saved < 1.0:
        print(" Сон ничего не сжал: кластеров не нашлось. Смотреть на плотность")
        print(" разговора и на то, образуются ли рёбра вообще.")
    elif lost > 5:
        print(" ВРЕДИТЕЛЬ: сжатие оплачено потерей ответов.")
    else:
        print(" Сон оправдан: память меньше, находимость та же.")


if __name__ == "__main__":
    main()
