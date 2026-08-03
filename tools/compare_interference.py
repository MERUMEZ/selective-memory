"""
================================================================================
 TOOLS/COMPARE_INTERFERENCE.PY — Есть ли у хранения цена
================================================================================
Замер, от которого зависит, чем является продукт.

В мозге забывание существует НЕ ради места. Долговременная память не
переполняется; забывание нужно, чтобы извлечение оставалось возможным.
Чем больше похожих следов отзывается на один и тот же ключ, тем труднее
достать любой — это называется перегрузкой признака.

У нас поиск оценивает каждого кандидата НЕЗАВИСИМО и берёт лучших.
Значит лишний узел, возможно, ничего не стоит. А если хранение
бесплатно, то забывание может только вредить — что все наши замеры и
показывали: минус 18.6 пункта, как только перестали удалять.

    ЕСЛИ ХРАНЕНИЕ БЕСПЛАТНО, ПРОДАВАТЬ ЗАБЫВАНИЕ НЕЛЬЗЯ.

Здесь это проверяется прямо: одни и те же вопросы, растущий стог.

ОГОВОРКА, БЕЗ КОТОРОЙ ЗАМЕР СОВРЁТ. Перегрузка признака — про ПОХОЖИХ
конкурентов, а не про любых. Тысяча реплик про погоду не мешает
вспомнить про аллергию; мешают двадцать реплик про лекарства. Поэтому
отвлекающие узлы здесь строятся ИЗ СЛОВ САМИХ ФАКТОВ и делят с вопросом
ключи, но ответа не содержат.

Прогон со случайным шумом показал бы ровно ничего и был бы пятым пустым
замером за две сессии.

Запуск:
    python tools/compare_interference.py
    python tools/compare_interference.py --levels 0,100,400,1600
================================================================================
"""

import argparse
import os
import random
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if "--logs" not in sys.argv:
    os.environ["LOG_LEVEL"] = "ERROR"

from selectivemem import Memory, MemorySettings  # noqa: E402

# Факт -> вопрос о нём. Вопрос делит с фактом часть слов, как в жизни.
TARGETS: List[Tuple[str, str]] = [
    ("у меня аллергия на пенициллин", "на какое лекарство у меня аллергия"),
    ("моя собака рекс боится грозы", "чего боится моя собака"),
    ("я работаю программистом в банке", "кем я работаю"),
    ("мой отпуск начинается в июле", "когда у меня отпуск"),
    ("я живу на улице пушкина", "на какой улице я живу"),
    ("моя дочь учится в третьем классе", "в каком классе моя дочь"),
]

# Слова, из которых строится БЛИЗКИЙ стог: они взяты из самих фактов и
# вопросов, поэтому отвлекающие узлы отзываются на те же ключи.
# ПОЧТИ ДВОЙНИКИ. Первая версия стенда делала отвлекающие узлы из
# одного-двух общих слов, и замер упёрся в потолок: 100% уже без стога.
# Настоящая перегрузка признака — это когда конкурент делит с фактом
# ПОЧТИ ВСЕ слова и отличается только тем, к кому относится.
_WHO = ["у соседа", "у коллеги", "у знакомого", "у брата", "у врача",
        "у продавца", "у водителя", "у племянника", "у сослуживца",
        "у попутчика", "у соседки", "у товарища"]

_SHADOWS = [
    "{who} аллергия на пенициллин",
    "{who} собака боится грозы",
    "{who} работа программистом в банке",
    "{who} отпуск начинается в июле",
    "{who} квартира на улице пушкина",
    "{who} дочь учится в третьем классе",
]


def build_distractors(rng: random.Random, count: int) -> List[str]:
    """
    Двойники фактов: те же слова, другой носитель. Именно они и создают
    перегрузку признака — на ключ "аллергия пенициллин" отзывается
    двадцать узлов, и нужный тонет среди них.
    """
    pool = [tpl.format(who=who) for tpl in _SHADOWS for who in _WHO]
    extra = [f"{who} тоже {tpl.format(who='').strip()}"
             for tpl in _SHADOWS for who in _WHO]
    pool += extra
    rng.shuffle(pool)
    out = []
    while len(out) < count:
        out.extend(pool)
    return out[:count]


def run_once(seed: int, distractors: int, top_k: int) -> Dict[str, float]:
    rng = random.Random(seed)
    now = [1_700_000_000.0]
    memory = Memory(
        ":memory:",
        settings=MemorySettings(delete_on_decay=False),
        clock=lambda: now[0],
    )

    # Факты записываются ПЕРВЫМИ, дальше их заваливает похожим.
    for fact, _ in TARGETS:
        memory.graph.save_connection(fact, "понятно", weight=0.7, timestamp=now[0])
        now[0] += 60.0
    for text in build_distractors(rng, distractors):
        memory.graph.save_connection(text, "понятно", weight=0.7, timestamp=now[0])
        now[0] += 60.0

    hits_1 = hits_k = 0
    for fact, question in TARGETS:
        found = memory.recall(question, top_k=top_k, timestamp=now[0],
                              with_associations=False)
        texts = [f"{m.context}".lower() for m in found]
        if texts and fact.lower() in texts[0]:
            hits_1 += 1
        if any(fact.lower() in t for t in texts):
            hits_k += 1

    nodes = memory.graph.db.count_nodes_by_type("episodic")
    memory.close()
    return {"r1": hits_1 / len(TARGETS), "rk": hits_k / len(TARGETS), "nodes": nodes}


def main() -> None:
    parser = argparse.ArgumentParser(description="Дорожает ли извлечение с ростом памяти")
    parser.add_argument("--levels", default="0,50,200,800")
    parser.add_argument("--seeds", default="1,7,13")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--logs", action="store_true")
    args = parser.parse_args()

    levels = [int(x) for x in args.levels.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    print("=" * 70)
    print(" ЕСТЬ ЛИ У ХРАНЕНИЯ ЦЕНА")
    print("=" * 70)
    print(f" {len(TARGETS)} фактов, стог из БЛИЗКИХ по словам узлов, сидов {len(seeds)}.")
    print(" Если извлечение не проседает — хранение бесплатно, и забывание")
    print(" в нынешней архитектуре продавать нельзя.")
    print("-" * 70)
    print(f" {'отвлекающих':>12} {'узлов':>8} {'R@1':>8} {f'R@{args.top_k}':>8}")

    first = None
    last = None
    for level in levels:
        rows = [run_once(seed, level, args.top_k) for seed in seeds]
        r1 = statistics.mean(r["r1"] for r in rows)
        rk = statistics.mean(r["rk"] for r in rows)
        nodes = statistics.mean(r["nodes"] for r in rows)
        if first is None:
            first = (r1, rk)
        last = (r1, rk)
        print(f" {level:>12} {nodes:8.0f} {r1*100:7.1f}% {rk*100:7.1f}%")

    print("=" * 70)
    drop_1 = (first[0] - last[0]) * 100
    drop_k = (first[1] - last[1]) * 100
    print(f" ПРОСАДКА: R@1 {drop_1:+.1f} пунктов, R@{args.top_k} {drop_k:+.1f} пунктов")
    if drop_k < 5 and drop_1 < 5:
        print(" Хранение БЕСПЛАТНО: конкуренции при извлечении нет, её надо")
        print(" строить с нуля — либо честно продавать дешёвую запись.")
    else:
        print(" Конкуренция при извлечении ЕСТЬ. Избирательность окупается на")
        print(" объёмах, и её надо усиливать, а не изобретать.")


if __name__ == "__main__":
    main()
