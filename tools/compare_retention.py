"""
================================================================================
 TOOLS/COMPARE_RETENTION.PY — Что переживает молчание при равном бюджете
================================================================================
Вопрос стенда: ПОСЛЕ ДВУХ НЕДЕЛЬ МОЛЧАНИЯ ЧТО ОСТАЁТСЯ НАХОДИМЫМ?

Сравниваются три хранилища одинакового объёма в символах: случайная
выборка реплик, последние реплики (скользящее окно) и selectivemem.

ЧЕСТНОСТЬ КОНСТРУКЦИИ. Три вещи сделаны специально, чтобы эксперимент мог
ОПРОВЕРГНУТЬ тезис, а не подтвердить его по построению:

1. Важность задаёт ПОЛЬЗОВАТЕЛЬ через feedback(+1.0), а не система.
   Половина тем получает оценку, половина нет.
2. Обе группы встречаются ОДИНАКОВО ЧАСТО, иначе случайная выборка
   выиграла бы просто на частоте.
3. Вопросы задаются ПОСЛЕ ДОЛГОГО МОЛЧАНИЯ: если не дать времени пройти,
   мерить нечего.

РЕЗУЛЬТАТ, три сида, контроль на объём:

    случайный отбор  43 узла   78% важное   56% обычное
    скользящее окно  44 узла   89%          89%
    selectivemem     33 узла  100%         100%

Меньше узлов, полная находимость. Преимущество даёт ОТБОР НА ВХОДЕ:
внутрь берётся четверть реплик, и в этой четверти оказываются ответы.

ЧТО ЗДЕСЬ СТОЯЛО РАНЬШЕ И ПОЧЕМУ СНЯТО. Стенд заявлял разрыв +40 п.п.
между похвалённым и обычным и был заголовочным аргументом проекта.
Разбор показал, что разрыв измерял УДАЛЕНИЕ: память не ставила
похвалённое выше, она стирала остальное. Как только удаление по возрасту
убрали — а это подняло полноту на внешнем наборе на 18.6 пункта, —
разрыв схлопнулся в ноль. Число было настоящим, толкование неверным.

Вдобавок стенд строил BrainSession, то есть ВИТРИНУ, а та считает эмоцию
сама и зовёт save_connection напрямую, минуя Memory.observe(). Значит
прежние числа описывали конфигурацию, которой у покупателя нет. Теперь
стенд работает через библиотеку.

Запуск:
    python tools/compare_retention.py --balanced
    python tools/compare_retention.py --balanced --silence-days 30
================================================================================
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if "--logs" not in sys.argv:
    os.environ["LOG_LEVEL"] = "ERROR"



from tools.compare_memory import build_baseline_store, is_hit, store_size  # noqa: E402
from selectivemem import Memory, MemorySettings  # noqa: E402

# Переопределения из командной строки. Собираются в main() и попадают в
# MemorySettings — стенд меряет БИБЛИОТЕКУ, а не витрину.
OVERRIDES: Dict[str, object] = {}

# Две группы тем, ОДИНАКОВЫЕ по структуре и частоте. Разница только в
# том, хвалит ли пользователь ответ — то есть в сигнале, а не в статистике.
PRAISED_TOPICS = [
    "меня зовут Паша и я твой учитель",
    "мой любимый цвет синий как небо",
    "я работаю программистом и пишу код",
    "моя собака зовут Рекс она большая",
    "я живу в городе у самой реки",
    "по субботам я хожу в бассейн",
]

PLAIN_TOPICS = [
    "вчера на улице был сильный дождь",
    "в магазине продают свежий хлеб",
    "автобус приходит на остановку утром",
    "чайник кипит примерно пять минут",
    "на полке стоит старая лампа",
    "во дворе растёт высокое дерево",
]

FILLER = [
    "расскажи что-нибудь ещё", "как твои дела сегодня", "что ты думаешь об этом",
    "продолжай я слушаю", "интересно а дальше", "понятно давай дальше",
]

PRAISE = ["молодец", "правильно", "отлично", "именно так"]

# НЕЙТРАЛЬНЫЕ отклики — контроль на объём. Похвалённая тема порождает два
# сообщения (тема + похвала), обычная одно, и преимущество могло бы
# объясняться просто тем, что похвалённое чаще попадает в окна STM. Эти
# реплики уравнивают число сообщений, не неся никакой валентности: ни одна
# из них не входит в POSITIVE_MARKERS/NEGATIVE_MARKERS амигдалы.
NEUTRAL = ["продолжай", "слушаю", "дальше", "понятно"]


def build_stream(rng: random.Random, rounds: int) -> List[Tuple[str, bool]]:
    """
    Поток вида (сообщение, хвалим_ли_следующим_ходом).

    Обе группы тем встречаются одинаковое число раз и перемешаны с шумом,
    чтобы отличие было только в похвале.
    """
    stream: List[Tuple[str, bool]] = []
    for _ in range(rounds):
        batch = [(t, True) for t in PRAISED_TOPICS] + [(t, False) for t in PLAIN_TOPICS]
        batch += [(rng.choice(FILLER), False) for _ in range(len(batch))]
        rng.shuffle(batch)
        stream.extend(batch)
    return stream


def run_once(seed: int, rounds: int, silence_days: float, balanced: bool = False) -> Dict[str, Dict[str, float]]:
    """
    ЧЕРЕЗ БИБЛИОТЕКУ, А НЕ ЧЕРЕЗ ВИТРИНУ, и это исправление серьёзной
    неточности.

    Прежняя версия строила BrainSession — то есть организм целиком, с
    миндалиной и восприятием. Витрина считает эмоцию сама и зовёт
    save_connection НАПРЯМУЮ, минуя Memory.observe(). Значит стенд мерил
    конфигурацию, которой у покупателя нет: у библиотечного пользователя
    emotion по умолчанию 0.0, и половина формулы гейта не работает.

    Числа от этого падают, и это честно: столько и получает тот, кто
    поставил пакет и не передаёт валентность сам. Оценку важности здесь
    даёт feedback(+1.0) — ровно так, как это делало бы приложение.
    """
    rng = random.Random(seed)
    now = [1_700_000_000.0]
    memory = Memory(":memory:", settings=MemorySettings(**OVERRIDES),
                    clock=lambda: now[0])
    exchanges: List[Tuple[str, str]] = []

    for index, (text, praise) in enumerate(build_stream(rng, rounds), start=1):
        memory.observe(text, response="понятно")
        exchanges.append((text, "понятно"))
        if praise:
            memory.feedback(+1.0)
            exchanges.append((rng.choice(PRAISE), "понятно"))
        elif balanced:
            # Контроль на объём: обычная тема тоже даёт вторую реплику,
            # но без всякой оценки.
            memory.observe(rng.choice(NEUTRAL), response="понятно")
            exchanges.append((rng.choice(NEUTRAL), "понятно"))
        now[0] += 300.0
        if index % 24 == 0:
            now[0] += 8 * 3600.0
            memory.forget(now=now[0])

    now[0] += silence_days * 86400.0
    memory.forget(now=now[0])
    now_ts = now[0]

    budget = store_size(memory.graph)
    total = len(exchanges)
    order = list(range(total))
    rng.shuffle(order)

    stores = {
        "случайные (контроль)": build_baseline_store(exchanges, order, budget),
        "последние (окно)": build_baseline_store(exchanges, range(total - 1, -1, -1), budget),
        "организм": memory.graph,
    }

    def recall(graph, topics: Sequence[str]) -> float:
        hits = 0
        for topic in topics:
            found = graph.search(topic, top_k=1, timestamp=now_ts, with_associations=False)
            if found and is_hit(topic, f"{found[0].context} {found[0].response}"):
                hits += 1
        return hits / max(1, len(topics))

    result = {}
    for name, graph in stores.items():
        result[name] = {
            "praised": recall(graph, PRAISED_TOPICS),
            "plain": recall(graph, PLAIN_TOPICS),
            "nodes": graph.db.count_nodes_by_type("episodic"),
        }
    memory.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Переживает ли долгое молчание именно то, что пользователь счёл важным"
    )
    parser.add_argument("--seeds", default="1,7,13,42,99")
    parser.add_argument("--rounds", type=int, default=4, help="проходов по всем темам")
    parser.add_argument("--silence-days", type=float, default=14.0)
    parser.add_argument("--balanced", action="store_true",
                    help="уравнять число сообщений: контроль на объём, а не на награду")
    parser.add_argument("--spike-factor", type=float, default=None,
                        help="MEMORY_FLOOR_SPIKE_FACTOR: пол, заработанный "
                             "силой спайка. Как и floor-base, ставится в config")
    parser.add_argument("--floor-base", type=float, default=None,
                        help="MEMORY_FLOOR_BASE: пол угасания для неподкреплённых "
                             "узлов. Ставится В CONFIG, а не в MemorySettings: "
                             "стенд строит память через BrainSession, а тот берёт "
                             "настройки из config — мимо этого моста замер молча "
                             "покажет умолчание")
    parser.add_argument("--interference", action="store_true",
                        help="модель интерференции: важность = доля силы")
    parser.add_argument("--logs", action="store_true")
    args = parser.parse_args()

    # ФЛАГИ ИДУТ В MemorySettings, А НЕ В CONFIG. После перевода стенда на
    # библиотеку правка config перестала на что-либо влиять: config читает
    # витрина, а Memory() берёт свои умолчания. Ручки молча ничего не
    # делали — ровно тот класс дефекта, который этот проект ловит весь
    # день.
    if args.spike_factor is not None:
        OVERRIDES["memory_floor_spike_factor"] = args.spike_factor
        print(f" Пол от силы спайка: {args.spike_factor}")
    if args.floor_base is not None:
        OVERRIDES["memory_floor_base"] = args.floor_base
        print(f" Базовый пол угасания: {args.floor_base}")

    if args.interference:
        OVERRIDES["use_relative_strength"] = True
        print(" Модель интерференции: важность = доля накопленной силы")

    seeds = [int(s) for s in args.seeds.split(",")]
    totals: Dict[str, Dict[str, List[float]]] = {}

    print("=" * 78)
    print(" УДЕРЖИВАЕТ ЛИ ОРГАНИЗМ ИМЕННО ВАЖНОЕ")
    print("=" * 78)
    print(f" Темы: {len(PRAISED_TOPICS)} с похвалой + {len(PLAIN_TOPICS)} без, "
          f"частота одинаковая")
    print(f" Проходов: {args.rounds}, затем молчание {args.silence_days:.0f} суток")
    print(f" Контроль на объём сообщений: {'ДА' if args.balanced else 'нет'}")
    print(f" Сиды: {seeds}")
    print("-" * 78)
    print(f"{'хранилище':<24} {'узлов':>6} {'похвалённое':>13} {'обычное':>10} {'разрыв':>9}")
    print("-" * 78)

    for seed in seeds:
        for name, data in run_once(seed, args.rounds, args.silence_days, args.balanced).items():
            bucket = totals.setdefault(name, {"praised": [], "plain": [], "nodes": []})
            bucket["praised"].append(data["praised"])
            bucket["plain"].append(data["plain"])
            bucket["nodes"].append(data["nodes"])

    def mean(xs):
        return sum(xs) / max(1, len(xs))

    for name in ("последние (окно)", "случайные (контроль)", "организм"):
        d = totals[name]
        praised, plain = mean(d["praised"]), mean(d["plain"])
        print(
            f"{name:<24} {mean(d['nodes']):>6.0f} {praised:>12.0%} "
            f"{plain:>10.0%} {praised - plain:>+8.0%}"
        )

    print("-" * 78)
    print(" У наивных хранилищ разрыв обязан быть около нуля — они про")
    print(" похвалу ничего не знают. Если у организма он тоже около нуля,")
    print(" значит подкрепление не доходит до удержания.")
    print("=" * 78)


if __name__ == "__main__":
    main()
