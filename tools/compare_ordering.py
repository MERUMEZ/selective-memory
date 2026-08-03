"""
================================================================================
 TOOLS/COMPARE_ORDERING.PY — Приходит ли важное ПЕРВЫМ
================================================================================
Третий заход. Первые два мерили не то, и это выяснилось замером.

    compare_memory.py    равномерное покрытие — организм не лучше
                         случайной выборки. Отрицательный результат, живёт
                         в репозитории.
    compare_retention.py что переживает молчание — разрыв +40 пунктов.

Разбор knowledge-update показал, ЧТО ИМЕННО меряет второй стенд, и это
оказалось неприятной новостью: разрыв +40 измеряет УДАЛЕНИЕ. Организм не
ставит похвалённое выше обычного в выдаче — он обычное стирает. Как
только удаление предотвращено любым способом, рутина находится в 97%
случаев и разрыв схлопывается до нуля.

Значит вопрос "переживёт ли тема молчание" больше задавать нельзя: мы
уходим от удаления неважного к оптимизации его хранения. Нужен другой:

    КОГДА ОБЕ ТЕМЫ В ПАМЯТИ И ОБЕ НАХОДИМЫ — ЧТО ПРИДЁТ ПЕРВЫМ?

ЧТО СЧИТАЕТСЯ. Две величины, и обе про порядок, а не про наличие.

1. MRR (mean reciprocal rank) отдельно по похвалённым и по обычным темам.
   Спрашивается каждая тема, ответ ищется в первой десятке, берётся
   1/позиция. Единица значит "всегда первым", 0.5 — "обычно вторым".
   Если подкрепление влияет на порядок, у похвалённых MRR должен быть
   ВЫШЕ: их узлы тяжелее и обязаны теснить чужие ответы вниз.

2. Доля похвалённого в первой пятёрке на ОБЩИХ запросах — то, что память
   выдаёт, когда её не спрашивают ни о чём конкретном. Это и есть
   продуктовая поверхность: чем ассистент наполняет контекст сам.
   Случайный порядок дал бы 50%, потому что тем поровну.

УДАЛЕНИЕ ЗДЕСЬ ВЫКЛЮЧЕНО НАМЕРЕННО (--floor-base, по умолчанию 0.06).
Иначе стенд снова померил бы выживание: у стёртого узла нет позиции.
Угасание при этом работает и продолжает различать узлы по весу — именно
эта разница и должна создавать порядок.

ЧТО МОЖЕТ ПРОВАЛИТЬСЯ, и в этом весь смысл. Если MRR и доля окажутся
одинаковыми у обеих групп, значит избирательность у нас жила ТОЛЬКО в
удалении, и переносить в ранжирование пока нечего — придётся строить.
Такой исход надо получить до правок, а не после.

Запуск:
    python tools/compare_ordering.py
    python tools/compare_ordering.py --silence-days 30 --floor-base 0.06
================================================================================
"""

import argparse
import os
import random
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if "--logs" not in sys.argv:
    os.environ["LOG_LEVEL"] = "ERROR"


# is_hit намеренно НЕ используется: его мягкость (половина значимых слов)
# засчитывала один узел сразу трём темам. См. _is_topic ниже.
from tools.compare_retention import (  # noqa: E402
    NEUTRAL, PLAIN_TOPICS, PRAISE, PRAISED_TOPICS, build_stream,
)


# СПРАШИВАТЬ НАДО ВОПРОСОМ, А НЕ ТЕКСТОМ ТЕМЫ. Первая версия стенда
# запрашивала дословную формулировку, узел находил сам себя первым, и MRR
# выходил ровно 1.000 у обеих групп — конкуренции не возникало вовсе.
# Здесь у каждой темы свой вопрос, как его задал бы человек; проверка
# попадания идёт по ТЕМЕ, а не по вопросу.
PRAISED_QUESTIONS = [
    ("как меня зовут", PRAISED_TOPICS[0]),
    ("какой цвет мне нравится", PRAISED_TOPICS[1]),
    ("кем я работаю", PRAISED_TOPICS[2]),
    ("как зовут мою собаку", PRAISED_TOPICS[3]),
    ("где я живу", PRAISED_TOPICS[4]),
    ("что я делаю по выходным", PRAISED_TOPICS[5]),
]

PLAIN_QUESTIONS = [
    ("какая была погода", PLAIN_TOPICS[0]),
    ("что продают в магазине", PLAIN_TOPICS[1]),
    ("когда приходит автобус", PLAIN_TOPICS[2]),
    ("сколько кипит чайник", PLAIN_TOPICS[3]),
    ("что стоит на полке", PLAIN_TOPICS[4]),
    ("что растёт во дворе", PLAIN_TOPICS[5]),
]

# Запросы, не относящиеся ни к одной теме в отдельности. Ими проверяется,
# чем память наполняет контекст, когда её не спрашивают о конкретном, —
# ровно то, что делает ассистент перед ответом.
GENERAL_QUERIES = [
    "расскажи что ты обо мне помнишь",
    "что ты знаешь",
    "напомни важное",
    "что было раньше",
]

TOP_K = 10

# Переопределения из командной строки идут в MemorySettings, а не в
# config: после перевода на библиотеку правка config ни на что не влияет.
OVERRIDES: Dict[str, object] = {}

# СТОГ. Без него стенд бессмыслен: на двадцати узлах каждый вопрос находит
# свой ответ первым просто потому, что соперников нет, и MRR выходит
# 1.000 у обеих групп. Наполнитель из compare_retention — шесть
# повторяющихся фраз, они схлопываются в пару узлов и конкуренции не
# создают. Здесь фраз сотни и все разные.
_SUBJECTS = ["сосед", "продавец", "водитель", "врач", "почтальон", "сторож",
             "прохожий", "мастер", "сантехник", "курьер"]
_VERBS = ["говорил про", "спрашивал про", "жаловался на", "рассказывал про",
          "вспоминал про", "думал про"]
_OBJECTS = ["ремонт крыши", "новую дорогу", "цены на бензин", "старый мост",
            "прогноз погоды", "расписание поездов", "очередь в поликлинике",
            "стройку за домом", "отключение воды", "уборку двора",
            "фонарь у ворот", "объявление на двери"]


def build_haystack(rng: random.Random, count: int) -> List[str]:
    """Разные нейтральные фразы, среди которых темам придётся конкурировать."""
    pool = [f"{s} {v} {o}" for s in _SUBJECTS for v in _VERBS for o in _OBJECTS]
    rng.shuffle(pool)
    return pool[:count]


def _is_topic(topic: str, node_text: str) -> bool:
    """
    Тот ли это узел. ТОЧНОЕ ВХОЖДЕНИЕ, а не доля общих слов.

    Первая версия опознавала темы через is_hit (половина значимых слов), и
    это оказалось негодным для стенда о ПОРЯДКЕ: один узел "я живу в
    городе у самой реки" засчитывался сразу трём темам — двум
    похвалённым и одной обычной. При такой классификации любые числа
    осмысленны лишь на вид.

    Здесь мягкость не нужна: темы подаются в поток дословно, поэтому
    контекст узла содержит формулировку темы целиком.
    """
    return topic.lower() in node_text.lower()


def _rank_of_topic(graph, question: str, topic: str, now: float) -> int:
    """
    Позиция правильного ответа в выдаче, начиная с 1. Ноль означает
    "не найден в первой десятке" — и это НЕ то же самое, что стёрт:
    удаление здесь выключено, значит узел есть, просто проиграл другим.
    """
    found = graph.search(question, top_k=TOP_K, timestamp=now, with_associations=False)
    for position, match in enumerate(found, start=1):
        if _is_topic(topic, f"{match.context} {match.response}"):
            return position
    return 0


def _mrr(graph, pairs: Sequence, now: float) -> float:
    scores = []
    for question, topic in pairs:
        rank = _rank_of_topic(graph, question, topic, now)
        scores.append(1.0 / rank if rank else 0.0)
    return statistics.mean(scores) if scores else 0.0


def _praised_share_in_top(graph, now: float) -> float:
    """
    Какую долю первой пятёрки на общих запросах занимает похвалённое.
    Тем поровну, поэтому случайный порядок дал бы 0.5.
    """
    praised_hits = 0
    counted = 0
    for query in GENERAL_QUERIES:
        found = graph.search(query, top_k=5, timestamp=now, with_associations=False)
        for match in found:
            text = f"{match.context} {match.response}"
            # Наполнитель ("расскажи что-нибудь ещё") не относится ни к
            # одной группе и в знаменатель не идёт: вопрос стенда — какую
            # из ДВУХ групп память ставит выше, а не сколько мусора наверху.
            if any(_is_topic(t, text) for t in PRAISED_TOPICS):
                praised_hits += 1
                counted += 1
            elif any(_is_topic(t, text) for t in PLAIN_TOPICS):
                counted += 1
    # Знаменатель возвращается ВМЕСТЕ с долей: 100% на двух наблюдениях и
    # 100% на сорока — разные утверждения, и без n второе неотличимо от
    # первого. Этот стенд уже дважды показывал красивую долю, которая
    # держалась на одном-единственном узле.
    return (praised_hits / counted if counted else 0.0), counted


def run_once(seed: int, rounds: int, silence_days: float, balanced: bool,
             haystack: int = 200) -> Dict[str, float]:
    random.seed(seed)
    rng = random.Random(seed)

    # ЧЕРЕЗ БИБЛИОТЕКУ, А НЕ ЧЕРЕЗ ВИТРИНУ. Витрина считает эмоцию сама и
    # зовёт save_connection напрямую, минуя Memory.observe(), поэтому
    # прежние числа описывали конфигурацию, которой у покупателя нет.
    # Оценку важности здесь даёт feedback(+1.0) — ровно так, как это
    # делало бы приложение.
    from selectivemem import Memory, MemorySettings

    now = [1_700_000_000.0]
    memory = Memory(":memory:", settings=MemorySettings(**OVERRIDES),
                    clock=lambda: now[0])

    # Темы перемешаны со стогом: иначе конкурировать не с чем.
    stream = [(text, praise) for text, praise in build_stream(rng, rounds)]
    stream += [(phrase, False) for phrase in build_haystack(rng, haystack)]
    rng.shuffle(stream)

    for index, (text, praise) in enumerate(stream, start=1):
        memory.observe(text, response="понятно")
        if praise:
            memory.feedback(+1.0)
        elif balanced:
            memory.observe(rng.choice(NEUTRAL), response="понятно")
        now[0] += 300.0
        if index % 24 == 0:
            now[0] += 8 * 3600.0
            memory.forget(now=now[0])

    now[0] += silence_days * 86400.0
    memory.forget(now=now[0])
    now_ts = now[0]

    graph = memory.graph

    # Веса по группам — это ОСНОВА, из которой порядок обязан получаться.
    # Если веса различаются, а MRR нет, значит вес просто не доходит до
    # ранжирования, и чинить надо формулу оценки, а не подкрепление.
    def mean_weight(topics: Sequence[str]) -> float:
        values = [
            row["weight"] for row in graph.db.fetch_all_nodes()
            if row["node_type"] == "episodic"
            and any(_is_topic(t, row["context"] or "") for t in topics)
        ]
        return statistics.mean(values) if values else 0.0

    result = {
        "mrr_praised": _mrr(graph, PRAISED_QUESTIONS, now_ts),
        "mrr_plain": _mrr(graph, PLAIN_QUESTIONS, now_ts),
        "praised_share": _praised_share_in_top(graph, now_ts)[0],
        "share_n": _praised_share_in_top(graph, now_ts)[1],
        "weight_praised": mean_weight(PRAISED_TOPICS),
        "weight_plain": mean_weight(PLAIN_TOPICS),
        "nodes": graph.db.count_nodes_by_type("episodic"),
    }
    memory.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Приходит ли важное первым, когда всё на месте"
    )
    parser.add_argument("--seeds", default="1,7,13,42,99")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--silence-days", type=float, default=14.0)
    parser.add_argument("--balanced", action="store_true", default=True,
                        help="обе группы дают по два сообщения (контроль на объём)")
    parser.add_argument("--floor-base", type=float, default=0.06,
                        help="пол угасания. НЕ НОЛЬ по умолчанию: иначе неважное "
                             "стирается и стенд снова меряет выживание, а не порядок")
    parser.add_argument("--rerank-band", type=float, default=None,
                        help="полоса переупорядочивания по важности; 0 выключает")
    parser.add_argument("--weight-influence", type=float, default=None,
                        help="вклад веса узла в оценку поиска (умолчание 0.15). "
                             "Это и есть ручка, которой избирательность "
                             "переносится из удаления в ранжирование")
    parser.add_argument("--haystack", type=int, default=200,
                        help="сколько посторонних фраз подмешать. Без стога\n                             стенд меряет пустоту: на двадцати узлах каждый\n                             вопрос находит ответ первым за неимением соперников")
    parser.add_argument("--interference", action="store_true",
                        help="модель интерференции: важность = доля силы")
    parser.add_argument("--logs", action="store_true")
    args = parser.parse_args()

    OVERRIDES["memory_floor_base"] = args.floor_base
    if args.rerank_band is not None:
        OVERRIDES["rerank_band"] = args.rerank_band
        print(f" Полоса переупорядочивания: {args.rerank_band}")
    if args.weight_influence is not None:
        OVERRIDES["memory_weight_influence"] = args.weight_influence
        print(f" Вклад веса в оценку поиска: {args.weight_influence}")

    print("=" * 78)
    print(" ПРИХОДИТ ЛИ ВАЖНОЕ ПЕРВЫМ")
    print("=" * 78)
    print(f" Пол угасания {args.floor_base} — удаление выключено, все темы в памяти.")
    print(f" Молчание {args.silence_days} суток, сидов {len(args.seeds.split(','))}.")
    print("-" * 78)

    rows: List[Dict[str, float]] = []
    for seed in (int(s) for s in args.seeds.split(",")):
        rows.append(run_once(seed, args.rounds, args.silence_days, args.balanced,
                             args.haystack))

    def avg(key: str) -> float:
        return statistics.mean(r[key] for r in rows)

    mrr_p, mrr_o = avg("mrr_praised"), avg("mrr_plain")
    print(f" {'узлов в памяти':32} {avg('nodes'):8.0f}")
    print(f" {'средний вес похвалённого':32} {avg('weight_praised'):8.4f}")
    print(f" {'средний вес обычного':32} {avg('weight_plain'):8.4f}")
    print("-" * 78)
    print(f" {'MRR похвалённого':32} {mrr_p:8.3f}")
    print(f" {'MRR обычного':32} {mrr_o:8.3f}")
    print(f" {'разрыв MRR':32} {mrr_p - mrr_o:+8.3f}")
    print("-" * 78)
    print(f" {'доля похвалённого в топ-5':32} {avg('praised_share')*100:7.1f}%"
          f"   (тематических узлов в выдаче: {avg('share_n'):.1f})")
    print(f" {'случайный порядок дал бы':32} {50.0:7.1f}%")
    spread = ", ".join(f"{r['mrr_praised']-r['mrr_plain']:+.2f}" for r in rows)
    print(f" {'разрыв MRR по сидам':32} {spread}")
    print("=" * 78)

    # ЗНАК ВАЖЕН. Первая версия вывода смотрела на |разрыв| и объявляла
    # "зависимость есть" даже когда важное приходило ПОЗЖЕ обычного.
    # ЗНАМЕНАТЕЛЬ ОБЯЗАТЕЛЕН. Доля на 0.8 узла — это не 40%, это шум, и
    # вывод по ней уже один раз объявил "важное приходит позже" при
    # разрыве MRR ровно 0.000.
    share_gap = (avg("praised_share") - 0.5) if avg("share_n") >= 5 else 0.0
    if mrr_p - mrr_o > 0.05 and share_gap > 0.05:
        print(" ВЫВОД: важное приходит раньше обычного — есть что растить.")
    elif mrr_p - mrr_o < -0.05 or share_gap < -0.05:
        print(" ВЫВОД: важное приходит ПОЗЖЕ обычного. Подкрепление доходит до")
        print(" веса, но вес не доходит до порядка выдачи: в оценке поиска у")
        print(" него доля 0.15 против 0.5 у смысла. Механизм надо строить.")
    else:
        print(" ВЫВОД: порядок выдачи НЕ ЗАВИСИТ от важности. Избирательность")
        print(" жила только в удалении — переносить в ранжирование нечего.")


if __name__ == "__main__":
    main()
