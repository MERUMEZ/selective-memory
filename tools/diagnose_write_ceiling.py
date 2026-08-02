"""
================================================================================
 DIAGNOSE_WRITE_CEILING.PY — Что теряет гейт, а что теряет поиск
================================================================================
Полный прогон LongMemEval (500 вопросов) дал R@5 64.8% в среднем и 20.8%
на категории knowledge-update — вопросах, где факт со временем изменился.
Это худшая категория с двойным отрывом, и она бьёт ровно по механизму,
который заявлен как отличительный: вытеснение устаревшего.

Но «плохо» — это не диагноз. Провал возможен в двух РАЗНЫХ местах:

  1. ЗАПИСЬ. Гейт не счёл обновление достойным памяти. Тогда нужной
     реплики в памяти просто нет, и поиск бессилен по построению.
  2. ПОИСК. Реплика записана, но ранжирование поднимает наверх старую
     версию или посторонние узлы.

Чинятся они противоположным: первое — порогом записи, второе —
ранжированием. Не разделив, правку выбирать наугад.

Здесь считается ПОТОЛОК: доля вопросов, где хоть одна реплика из нужной
сессии вообще попала в память. Выше потолка никакой поиск не прыгнет.
Разрыв между потолком и R@5 — это и есть цена поиска.

Кодировщик не нужен: гейт решает по эмоции и удивлению, эмбеддинги на
запись не влияют. Поэтому прогон лёгкий.

    python tools/diagnose_write_ceiling.py --data storage/bench/shards/s00.json
================================================================================
"""

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from selectivemem import Memory, MemorySettings
from tools.bench_longmemeval import parse_date


def measure(instance, settings_kwargs):
    evidence = set(instance.get("answer_session_ids") or [])
    if not evidence:
        return None

    sessions = instance["haystack_sessions"]
    session_ids = instance["haystack_session_ids"]
    dates = instance.get("haystack_dates") or []

    memory = Memory(":memory:", settings=MemorySettings(**settings_kwargs))
    written_sessions = set()
    evidence_turns = 0
    evidence_written = 0

    for index, session in enumerate(sessions):
        session_id = session_ids[index]
        when = parse_date(dates[index]) if index < len(dates) else float(index * 86400)
        is_evidence = session_id in evidence

        pending_user = None
        for turn in session:
            content = (turn.get("content") or "").strip()
            if not content:
                continue
            if turn.get("role") == "user":
                pending_user = content
                continue
            if pending_user is None:
                continue

            if is_evidence:
                evidence_turns += 1
            node_id = memory.observe(pending_user, response=content, timestamp=when).node_id
            if node_id is not None:
                written_sessions.add(session_id)
                if is_evidence:
                    evidence_written += 1
            pending_user = None

    return {
        "type": instance["question_type"],
        # Потолок: нужная сессия представлена в памяти хоть одним узлом.
        "reachable": bool(written_sessions & evidence),
        "evidence_turns": evidence_turns,
        "evidence_written": evidence_written,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--plasticity", type=float, default=None)
    # Набор ОТСОРТИРОВАН по типу вопроса: knowledge-update лежит целиком в
    # шардах 7-8, temporal-reasoning в других. Без отбора легко померить
    # не ту категорию и не заметить этого — что со мной и случилось.
    ap.add_argument("--type", default=None, help="только этот тип вопроса")
    ap.add_argument("--limit", type=int, default=0, help="0 = все")
    args = ap.parse_args()

    settings_kwargs = {}
    if args.plasticity is not None:
        settings_kwargs["base_plasticity_threshold"] = args.plasticity

    data = json.load(open(args.data, encoding="utf-8"))
    if args.type:
        data = [i for i in data if i["question_type"] == args.type]
    if args.limit:
        data = data[: args.limit]
    print(f"инстансов к замеру: {len(data)}", flush=True)
    by_type = collections.defaultdict(lambda: collections.Counter())

    for instance in data:
        r = measure(instance, settings_kwargs)
        if r is None:
            continue
        c = by_type[r["type"]]
        c["n"] += 1
        c["reachable"] += int(r["reachable"])
        c["ev_turns"] += r["evidence_turns"]
        c["ev_written"] += r["evidence_written"]

    print(f"{'тип вопроса':30} {'n':>4} {'потолок':>9} {'реплик из нужной сессии':>26}")
    total = collections.Counter()
    for t, c in sorted(by_type.items(), key=lambda x: -x[1]["n"]):
        total.update(c)
        share = c["ev_written"] / c["ev_turns"] * 100 if c["ev_turns"] else 0.0
        print(f"{t:30} {c['n']:>4} {c['reachable']/c['n']*100:8.1f}% "
              f"{c['ev_written']:>10}/{c['ev_turns']:<6} ({share:.1f}%)")
    share = total["ev_written"] / total["ev_turns"] * 100 if total["ev_turns"] else 0.0
    print(f"\n{'ВСЕГО':30} {total['n']:>4} {total['reachable']/total['n']*100:8.1f}% "
          f"{total['ev_written']:>10}/{total['ev_turns']:<6} ({share:.1f}%)")


if __name__ == "__main__":
    main()
