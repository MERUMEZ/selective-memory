"""
================================================================================
 TOOLS/COMPARE_DIALOGUE.PY — Стенд, работающий как живой ассистент
================================================================================
Все прежние стенды устроены одинаково: сначала загрузить всё, потом один
раз спросить. LongMemEval тоже — он вызывает observe() двести раз и
recall() РОВНО ОДИН РАЗ, в конце.

Живой ассистент так себя не ведёт. Он достаёт контекст ПЕРЕД КАЖДЫМ
ответом, то есть вспоминание и запись у него чередуются сотни раз.

Разница не косметическая. Целый класс механизмов существует только в
чередовании и на прежних стендах невидим:

  - ассоциации между воспоминаниями. Связь возникает от совместной
    активности: приложение что-то достало из памяти, и следующая запись
    связывается с этим. Без вспоминаний рёбер не образуется вовсе —
    замерено, 201 узел и ноль рёбер;
  - растекающаяся активация, которой без рёбер не по чему идти;
  - использование как сигнал важности: стабильность растёт от обращений,
    а обращений в прежних стендах почти нет.

ЧТО ИМЕННО ЗДЕСЬ ПРОВЕРЯЕТСЯ — многошаговое извлечение. Факт называется
один раз. Позже разговор возвращается к теме ДРУГИМИ СЛОВАМИ — эта
реплика связывается с фактом, потому что факт был вспомнен в тот момент.
Вопрос в конце сформулирован под ВТОРУЮ реплику, а ответ нужен из первой.

Прямым поиском такое не берётся: у вопроса нет общих слов с фактом.
Взять его можно только по связи. Поэтому стенд честно отвечает на вопрос
"стоят ли ассоциации того, что стоят".

Запуск:
    python tools/compare_dialogue.py
    python tools/compare_dialogue.py --seeds 1,7,13,42,99 --silence-days 7
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

DAY = 86400.0

# Тройки: факт -> мостик (та же тема другими словами) -> вопрос под мостик.
#
# Вопрос НАРОЧНО не делит слов с фактом. Проверяется не поиск, а связь:
# найти факт можно только через мостик, с которым он связался в момент,
# когда мостик записывался.
TRIPLETS: List[Tuple[str, str, str]] = [
    ("у меня аллергия на пенициллин",
     "врач выписал мне азитромицин вместо привычного",
     "что назначил доктор"),
    ("моя дочь учится в третьем классе",
     "родительское собрание перенесли на пятницу",
     "когда собрание"),
    ("я работаю удалённо из дома",
     "интернет вчера отключали на четыре часа",
     "надолго ли пропадала связь"),
    ("моя машина стоит в подземном паркинге",
     "шлагбаум опять заедает на выезде",
     "что заедает"),
    ("я не ем мясо уже пять лет",
     "в новом кафе на углу неплохой выбор блюд",
     "что за заведение открылось"),
    ("мой отпуск начинается в середине июля",
     "билеты подорожали почти вдвое к сезону",
     "что подорожало"),
]

FILLER = [
    "сосед опять шумит по вечерам",
    "погода на выходных обещает быть ясной",
    "надо бы разобрать наконец кладовку",
    "цены в магазине у дома растут",
    "во дворе поставили новую лавочку",
    "почта работает до семи вечера",
    "лифт починили только к среде",
    "в парке зацвели каштаны",
]


def run_once(seed: int, associate: int, silence_days: float, top_k: int) -> Dict[str, float]:
    """
    Один прогон разговора. Возвращает долю вопросов, на которые нужный
    факт нашёлся, и среднюю степень узлов — чтобы видеть, образовался
    граф или нет.
    """
    rng = random.Random(seed)
    now = [1_700_000_000.0]

    memory = Memory(
        ":memory:",
        settings=MemorySettings(associate_recalled_limit=associate),
        clock=lambda: now[0],
    )

    # Разговор: факты, мостики и наполнитель вперемешку. Мостик всегда
    # ПОЗЖЕ своего факта — иначе связывать нечего.
    stream: List[str] = []
    for fact, bridge, _ in TRIPLETS:
        stream.append(("факт", fact))
    rng.shuffle(stream)
    stream = [text for _, text in stream]

    filler_first = [rng.choice(FILLER) for _ in range(6)]
    bridges = [bridge for _, bridge, _ in TRIPLETS]
    rng.shuffle(bridges)
    filler_last = [rng.choice(FILLER) for _ in range(6)]

    conversation = stream + filler_first + bridges + filler_last

    fact_ids: Dict[str, int] = {}
    for text in conversation:
        # ТАК ВЕДЁТ СЕБЯ АССИСТЕНТ: сначала достаёт контекст, потом пишет.
        memory.context_for(text, top_k=3)
        node_id = memory.observe(text, response="понятно", emotion=0.6).node_id
        if node_id is not None:
            fact_ids.setdefault(text, node_id)
        now[0] += 120.0

    now[0] += silence_days * DAY
    memory.forget(now=now[0])

    hits = 0
    for fact, _bridge, question in TRIPLETS:
        found = memory.recall(question, top_k=top_k, with_associations=True)
        texts = " | ".join(f"{m.context} {m.response}" for m in found)
        if fact.lower() in texts.lower():
            hits += 1

    ids = [r["id"] for r in memory.graph.db.fetch_all_nodes() if r["node_type"] == "episodic"]
    degrees = memory.graph.db.get_degrees(ids)
    mean_degree = statistics.mean([degrees.get(i, 0) for i in ids]) if ids else 0.0

    memory.close()
    return {
        "hit_rate": hits / len(TRIPLETS),
        "mean_degree": mean_degree,
        "nodes": len(ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Помогают ли ассоциации, когда вспоминание и запись чередуются"
    )
    parser.add_argument("--seeds", default="1,7,13,42,99")
    parser.add_argument("--silence-days", type=float, default=7.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--logs", action="store_true")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]

    print("=" * 78)
    print(" МНОГОШАГОВОЕ ИЗВЛЕЧЕНИЕ: ФАКТ ЧЕРЕЗ СВЯЗАННУЮ РЕПЛИКУ")
    print("=" * 78)
    print(f" Вопрос не делит слов с фактом — взять его можно только по связи.")
    print(f" Троек: {len(TRIPLETS)}, сидов: {len(seeds)}, молчание {args.silence_days} сут.")
    print("-" * 78)
    print(f" {'связей на запись':22} {'узлов':>7} {'средняя степень':>17} {'факт найден':>13}")

    results = {}
    for associate in (0, 3):
        rows = [run_once(seed, associate, args.silence_days, args.top_k) for seed in seeds]
        hit = statistics.mean(r["hit_rate"] for r in rows)
        deg = statistics.mean(r["mean_degree"] for r in rows)
        nodes = statistics.mean(r["nodes"] for r in rows)
        results[associate] = hit
        label = "выключено" if associate == 0 else f"до {associate}"
        print(f" {label:22} {nodes:7.0f} {deg:17.2f} {hit*100:12.1f}%")

    print("=" * 78)
    delta = (results[3] - results[0]) * 100
    print(f" РАЗНИЦА: {delta:+.1f} пунктов")
    if delta > 5:
        print(" Ассоциации работают: факт достаётся через связанную реплику.")
    elif delta < -5:
        print(" Ассоциации МЕШАЮТ: лишние рёбра тянут в выдачу постороннее.")
    else:
        print(" Разницы нет. Либо связи не образуются, либо растекание до них")
        print(" не доходит — смотреть на среднюю степень выше.")


if __name__ == "__main__":
    main()
