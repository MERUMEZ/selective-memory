"""
================================================================================
 TOOLS/COMPARE_CONSEQUENCE.PY — Учится ли память на своих ошибках
================================================================================
Вопрос стенда: ЕСЛИ ЧЕЛОВЕК ПЕРЕСПРОСИЛ, СТАНЕТ ЛИ ОТВЕТ ЛУЧШЕ.

Сила у нас растёт от извлечения — но растёт у ВСЕГО, что попало в выдачу,
у верного и неверного одинаково. Пока это так, ускорение подкрепления
ускоряет и закрепление ошибок: замерено, подъём шага с 0.05 до 0.15
выигрывал на трёх стендах и проигрывал на четвёртом.

Недостаёт не скорости, а ОТРИЦАТЕЛЬНОЙ ветви. У дофамина она есть: всплеск
на неожиданную награду и провал ниже фона на обещанную и не полученную.

ЧЕСТНОСТЬ КОНСТРУКЦИИ. Повтор вопроса здесь происходит НЕ МЕХАНИЧЕСКИ, и
это главное. Если переспрашивать всегда, наказание достанется и верным
ответам, и стенд покажет улучшение там, где его нет — механизм окажется
просто общим ослаблением.

Поэтому человек здесь ведёт себя как человек: получил не то — переспросил
теми же словами; получил то — пошёл дальше. Наказание достаётся только
неверным выдачам, и если оно не помогает, стенд это покажет.

ЧТО ИЗМЕРЯЕТСЯ. Один и тот же набор вопросов задаётся несколько кругов.
Смотрим, растёт ли доля верных ответов от круга к кругу — то есть учится
ли память на собственных промахах, БЕЗ ЕДИНОГО ВЫЗОВА feedback().

Запуск:
    python tools/compare_consequence.py
    python tools/compare_consequence.py --rounds 6 --distractors 400
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
from tools.compare_interference import TARGETS, build_distractors  # noqa: E402


def run_once(seed: int, distractors: int, rounds: int, penalty: float) -> Dict[str, object]:
    rng = random.Random(seed)
    now = [1_700_000_000.0]
    memory = Memory(
        ":memory:",
        settings=MemorySettings(
            delete_on_decay=False,
            consequence_penalty=penalty,
        ),
        clock=lambda: now[0],
    )

    for fact, _ in TARGETS:
        memory.graph.save_connection(fact, "понятно", weight=0.7, timestamp=now[0])
        now[0] += 60.0
    for text in build_distractors(rng, distractors):
        memory.graph.save_connection(text, "понятно", weight=0.7, timestamp=now[0])
        now[0] += 60.0

    per_round: List[float] = []
    for _ in range(rounds):
        hits = 0
        for fact, question in TARGETS:
            found = memory.recall(question, top_k=3, timestamp=now[0],
                                  with_associations=False)
            ok = bool(found) and fact.lower() in found[0].context.lower()
            hits += ok
            if ok:
                # Получил что хотел — идёт дальше. Время сдвигается сильно,
                # чтобы следующий вопрос не сошёл за переспрашивание.
                now[0] += 3600.0
            else:
                # НЕ ПОЛУЧИЛ — переспрашивает теми же словами, в том же
                # окне. Именно этот повтор и судит предыдущую выдачу.
                memory.recall(question, top_k=3, timestamp=now[0] + 30.0,
                              with_associations=False)
                now[0] += 120.0
        per_round.append(hits / len(TARGETS))

    strengths = [
        row["strength"] for row in memory.graph.db.fetch_all_nodes()
        if row["node_type"] == "episodic" and row["strength"] is not None
    ]
    memory.close()
    return {
        "rounds": per_round,
        "mean_strength": statistics.mean(strengths) if strengths else 0.0,
        "max_strength": max(strengths) if strengths else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Учится ли память на переспрашивании, без вызова feedback()"
    )
    parser.add_argument("--seeds", default="1,7,13")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--distractors", type=int, default=200)
    parser.add_argument("--penalties", default="0.0,0.1,0.3")
    parser.add_argument("--logs", action="store_true")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    penalties = [float(p) for p in args.penalties.split(",")]

    print("=" * 76)
    print(" УЧИТСЯ ЛИ ПАМЯТЬ НА СВОИХ ОШИБКАХ")
    print("=" * 76)
    print(f" {len(TARGETS)} фактов, {args.distractors} почти-двойников, "
          f"{args.rounds} круга вопросов, сидов {len(seeds)}.")
    print(" Переспрашивают ТОЛЬКО после неверного ответа — иначе наказание")
    print(" досталось бы и верным, и стенд показал бы улучшение там, где его нет.")
    print("-" * 76)
    header = " ".join(f"{f'круг {i+1}':>9}" for i in range(args.rounds))
    print(f" {'наказание':<11}{header} {'ср.сила':>9} {'макс':>7}")

    for penalty in penalties:
        rows = [run_once(s, args.distractors, args.rounds, penalty) for s in seeds]
        cells = []
        for i in range(args.rounds):
            cells.append(f"{statistics.mean(r['rounds'][i] for r in rows) * 100:8.1f}%")
        print(f" {penalty:<11.2f}{' '.join(cells)}"
              f" {statistics.mean(r['mean_strength'] for r in rows):>9.3f}"
              f" {statistics.mean(r['max_strength'] for r in rows):>7.2f}")

    print("=" * 76)
    print(" Растёт ли доля верного от круга к кругу — вот и весь вопрос.")
    print(" Если растёт и БЕЗ наказания, дело в обычном подкреплении от")
    print(" извлечения, а не в отрицательной ветви: смотреть на первую строку.")
    print(" Если вместе с попаданиями уехала средняя сила — это не отбор,")
    print(" а общий сдвиг, и засчитывать его нельзя.")
    print("=" * 76)


if __name__ == "__main__":
    main()
