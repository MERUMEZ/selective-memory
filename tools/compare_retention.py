"""
================================================================================
 TOOLS/COMPARE_RETENTION.PY — Удерживает ли организм именно ВАЖНОЕ
================================================================================
Второй заход после отрицательного результата в tools/compare_memory.py.

Там мерилось равномерное покрытие — "сколько из всего сказанного можно
найти", — и организм не обыграл случайную выборку (89.2% против 90.0% на
пяти сидах). Это ожидаемо: против равномерных вопросов случайный отбор
непобедим по построению, потому что он несмещённый, а любой осмысленный
отбор смещён. Организм проигрывал ровно там, где его смещение и есть
смысл.

Здесь задаётся правильный вопрос:

    ПЕРЕЖИВАЕТ ЛИ ВРЕМЯ ИМЕННО ТО, ЧТО ПОЛЬЗОВАТЕЛЬ СЧЁЛ ВАЖНЫМ?

ЧЕСТНОСТЬ КОНСТРУКЦИИ. Три вещи сделаны специально, чтобы эксперимент мог
ОПРОВЕРГНУТЬ тезис, а не подтвердить его по построению:

1. Важность задаёт ПОЛЬЗОВАТЕЛЬ, а не система. Половина тем сопровождается
   похвалой ("молодец", "правильно"), половина — нет. Организм про это
   деление ничего не знает заранее, он только получает сигнал валентности.

2. Обе группы встречаются ОДИНАКОВО ЧАСТО. Иначе случайная выборка
   выиграла бы просто на частоте, и мы бы мерили не тот механизм.

3. Вопросы задаются ПОСЛЕ ДОЛГОГО МОЛЧАНИЯ. Заявленное преимущество —
   забывание; если не дать ему сработать, мерить нечего. В первом заходе
   разрыв был 8 часов на 20 сообщений, и угасание не успевало ничего
   отсеять.

Ожидание, которое можно провалить: у организма разрыв между похваленным и
непохваленным должен быть ЗАМЕТНО больше, чем у наивных хранилищ (у
которых он обязан быть около нуля — они про похвалу не знают). Если
разрыва нет, значит подкрепление не доходит до удержания, и это надо
знать.

РЕЗУЛЬТАТ на пяти сидах (1, 7, 13, 42, 99), молчание 14 суток:

                          узлов  похвалённое  обычное   разрыв
    последние (окно)         23          70%      73%      -3%
    случайные (контроль)     23          57%      67%     -10%
    организм                 14         100%      70%     +30%

С КОНТРОЛЕМ НА ОБЪЁМ (--balanced, обе группы дают по два сообщения):
    последние (окно)         36          77%      70%      +7%
    случайные (контроль)     37          43%      50%      -7%
    организм                 16         100%      60%     +40%

Числа менялись трижды, и каждый раз от осознанной правки, а не сами.
Последнее изменение: порог записи снижен 0.35 -> 0.25 по внешнему
бенчмарку (полнота выросла втрое), разрыв при этом упал с +50 до +40.
Обмен признан выгодным: +40 против нуля у наивных хранилищ по-прежнему
решающий, а полнота была нашим слабым местом.

Разрыв у наивных хранилищ около нуля, как и должно быть — они про похвалу
не знают; +13 у окна под контролем на объём — предел разрешения стенда
(вопросов около тридцати, один вопрос это ~3 п.п.). У организма разрыв
+53..+67 п.п. и ПЕРЕЖИВАЕТ контроль на объём,
значит эффект даёт подкрепление, а не то, что похвалённая тема порождает
лишнее сообщение и чаще попадает в окна STM.

Это и есть то, чего не показал первый заход: при равном бюджете организм
не лучше случайной выборки ВООБЩЕ, но заметно лучше НА ТОМ, ЧТО
ПОЛЬЗОВАТЕЛЬ ОТМЕТИЛ КАК ВАЖНОЕ. Ставка на забывание оправдывается именно
избирательностью, а не объёмом.

Запуск:
    python tools/compare_retention.py
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

# СТЕНД ВСЕГДА МЕРИТ ИССЛЕДОВАТЕЛЬСКУЮ КОНФИГУРАЦИЮ.
#
# SPEECH_DEMO_PACE ускоряет речевые стадии для показа, но при этом
# ИСКАЖАЕТ измерение: организм раньше выходит на генерацию, чаще
# обращается к памяти, всё подряд получает стабильность — и
# избирательность удержания исчезает. Замер показал 100% против 100%
# (разрыв +0) вместо честных 97% против 43% — разрыва в полсотни
# пунктов.
#
# Ускорять демонстрацию можно, мерить ускоренную — нельзя: это другая
# система. Поэтому темп прибит здесь, до импорта config.
os.environ.setdefault("SPEECH_DEMO_PACE", "false")


import config  # noqa: E402
from tools.compare_memory import build_baseline_store, is_hit, store_size  # noqa: E402
from tools.simulate_learning import install_llm_stub  # noqa: E402

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
    install_llm_stub()
    random.seed(seed)
    rng = random.Random(seed)

    from core.brain_session import BrainSession, ManualWallClock

    # Часы стенда НЕ идут сами. Раньше источником настенного времени был
    # time.time(), а он входит в субъективное с множителем
    # TIME_ACCELERATION, поэтому в возраст узлов подмешивалась реальная
    # длительность прогона — и два запуска одного и того же кода
    # расходились. Паузы между сессиями стенд задаёт явно, ниже.
    #
    # Шаг НУЛЕВОЙ намеренно: так замер сохраняет прежний смысл, а правка
    # убирает только случайность. Разговор с паузами в полминуты между
    # репликами — отдельный, более реалистичный режим, и включать его
    # надо осознанно, перемеряв все числа заново.
    #
    # Точка отсчёта ФИКСИРОВАННАЯ, а не "сейчас": из неё же берётся эпоха
    # мозга, а эпоха входит в каждую метку времени в базе. Пока она была
    # настоящим временем запуска, два прогона расходились уже на первом
    # сообщении — при побитово одинаковом графе и одинаковом состоянии
    # генератора случайных чисел. Паузы задаются wall.advance() ниже.
    wall = ManualWallClock(start=1_700_000_000.0, seconds_per_call=0.0)
    session = BrainSession(db_path=":memory:", wall_clock=wall)
    exchanges: List[Tuple[str, str]] = []

    stream = build_stream(rng, rounds)
    for index, (text, praise) in enumerate(stream, start=1):
        response = session.process_message(text)
        exchanges.append((text, response.text))
        if praise:
            # Похвала идёт СЛЕДУЮЩЕЙ репликой — так её и даёт живой человек
            marker = rng.choice(PRAISE)
            reply = session.process_message(marker)
            exchanges.append((marker, reply.text))
        elif balanced:
            # Контроль на объём: обычная тема тоже получает вторую реплику,
            # но без всякой валентности
            marker = rng.choice(NEUTRAL)
            reply = session.process_message(marker)
            exchanges.append((marker, reply.text))
        if index % 24 == 0:
            wall.advance(8 * 3600.0)
            session.memory.apply_decay(now=session.clock.get_brain_time())

    # ДОЛГОЕ МОЛЧАНИЕ — то, ради чего всё затевалось
    wall.advance(silence_days * 86400.0)
    now = session.clock.get_brain_time()
    session.memory.apply_decay(now=now)

    budget = store_size(session.memory)
    total = len(exchanges)
    order = list(range(total))
    rng.shuffle(order)

    stores = {
        "случайные (контроль)": build_baseline_store(exchanges, order, budget),
        "последние (окно)": build_baseline_store(exchanges, range(total - 1, -1, -1), budget),
        "организм": session.memory,
    }

    def recall(graph, topics: Sequence[str]) -> float:
        hits = 0
        for topic in topics:
            found = graph.search(topic, top_k=1, timestamp=now, with_associations=False)
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
    session.close()
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

    if args.spike_factor is not None:
        config.MEMORY_FLOOR_SPIKE_FACTOR = args.spike_factor
        print(f" Пол от силы спайка: {args.spike_factor}")
    if args.floor_base is not None:
        config.MEMORY_FLOOR_BASE = args.floor_base
        print(f" Базовый пол угасания: {args.floor_base}")

    if args.interference:
        config.USE_RELATIVE_STRENGTH = True
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
