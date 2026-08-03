"""
================================================================================
 TOOLS/COMPARE_MEMORY.PY — Экономика памяти против «сохранить всё»
================================================================================
Отвечает на вопрос, ради которого построен весь проект:

    ПРИ РАВНОМ БЮДЖЕТЕ ХРАНЕНИЯ отбор организма лучше наивного?

Индустрия агентов делает "append forever + top-k": контекст растёт,
векторная база копится, ничего не забывается. Здесь сделана обратная
ставка — забывание это функция, а важность не объявляют, её зарабатывают
употреблением. Стенд проверяет, есть ли за этой ставкой что-то, кроме
красивой метафоры.

Сравниваются четыре хранилища на ОДНОМ И ТОМ ЖЕ потоке сообщений:

    всё         — 120 записей, ничего не забыто (верхняя граница)
    последние-N — скользящее окно, самый частый приём в проде
    случайные-N — контроль: отбор без всякой логики
    организм    — N узлов, отобранных спайком и укреплённых употреблением

N берётся из того, сколько сохранил САМ организм, поэтому три нижние
строки сравниваются при равном размере.

ЧТО СЧИТАЕТСЯ ПОПАДАНИЕМ: по каждой фразе, которая реально звучала в
разговоре, задаётся вопрос, и проверяется, вернёт ли поиск узел с этим
содержанием.

РЕЗУЛЬТАТ НА ПЯТИ СИДАХ (1, 7, 13, 42, 99) — ОТРИЦАТЕЛЬНЫЙ ДЛЯ СИЛЬНОЙ
ВЕРСИИ ТЕЗИСА:

    скользящее окно  90.0%
    случайный отбор  90.8%
    организм         92.4%

ЭТОТ СТЕНД БОЛЬШЕ НЕ ПОКАЗЫВАЕТ ОТРИЦАТЕЛЬНЫЙ РЕЗУЛЬТАТ, и шапку
пришлось переписать. Раньше здесь стояло 87.2% против 88.8% — организм
отставал от случайного отбора. После снижения порога записи (0.35 ->
0.25, сделано по внешнему бенчмарку) отставание исчезло.

Победу объявлять рано: разброс по сидам 82-98%, значит 92.4 против 90.8
это «сравнялись». Но прежнее утверждение замеру больше не соответствует,
и оставлять его было бы враньём в свою пользу наоборот.

Ждать здесь большего и не стоит: против равномерных вопросов
несмещённый отбор непобедим по построению.

ПОЧЕМУ МЕТРИКА НЕ ТА (ошибка дизайна, а не системы): здесь меряется
РАВНОМЕРНОЕ ПОКРЫТИЕ — "сколько из всего сказанного можно найти". Против
равномерных вопросов случайная выборка непобедима по построению: она
несмещённая, а любой осмысленный отбор смещён. Организм проигрывает ровно
там, где его смещение и есть смысл.

Правильный вопрос — не "что вообще было сказано", а "что оказалось
ВАЖНЫМ": см. tools/compare_retention.py, где важность задаётся
пользователем (похвала), а не системой, и разнесена во времени, чтобы
забывание успело сработать.

Запуск:
    python tools/compare_memory.py
    python tools/compare_memory.py --messages 200 --seed 7
================================================================================
"""

import argparse
import os
import random
import sys
import time
from collections import Counter
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
from tools.simulate_learning import (  # noqa: E402
    CORPUS,
    build_message_stream,
    install_llm_stub,
)

STOP = {
    "и", "в", "на", "с", "по", "к", "у", "из", "за", "от", "до", "для", "что",
    "как", "это", "то", "я", "ты", "он", "она", "мы", "вы", "они", "не", "но",
    "а", "же", "бы", "ли", "или", "его", "её", "ее", "их", "есть", "мне",
    "меня", "мой", "про", "о",
}


def content_words(text: str) -> List[str]:
    return [
        w.strip(".,!?;:").lower()
        for w in text.split()
        if w.strip(".,!?;:").lower() not in STOP and len(w) > 2
    ]


def is_hit(probe: str, retrieved_text: str) -> bool:
    """
    Попадание — если найденный узел содержит хотя бы половину
    содержательных слов вопроса. Порог намеренно мягкий: проверяется
    "нашлось ли нужное воспоминание", а не дословное совпадение.
    """
    words = content_words(probe)
    if not words:
        return False
    found = sum(1 for w in words if w in retrieved_text.lower())
    return found >= max(1, len(words) // 2)


# --------------------------------------------------------------------------
# Хранилища
# --------------------------------------------------------------------------

def store_size(graph) -> int:
    """
    Бюджет считается в СИМВОЛАХ, а не в узлах.

    Сравнение по числу узлов было бы нечестным: consolidate_from_stm
    склеивает окно STM в один узел, поэтому один узел организма несёт
    содержимое нескольких сообщений. По узлам получалось сжатие в 9 раз,
    по тексту — только в 3.1. Наивные хранилища обязаны получить столько
    же символов, иначе мы сравниваем разные объёмы памяти.
    """
    rows = graph.db._conn.execute(
        "SELECT context, response FROM nodes WHERE node_type = 'episodic'"
    ).fetchall()
    return sum(len(r["context"] or "") + len(r["response"] or "") for r in rows)


def build_baseline_store(
    exchanges: Sequence[Tuple[str, str]],
    order: Sequence[int],
    char_budget: int = None,
):
    """
    Наивное хранилище: каждый обмен — узел с одинаковым весом, никакого
    забывания. order задаёт приоритет отбора, char_budget — сколько
    символов разрешено (None = без ограничения).
    """
    from selectivemem.database import Database
    from selectivemem.graph_memory import MemoryGraph

    graph = MemoryGraph(db=Database(db_path=":memory:"))
    used = 0
    for i in order:
        user_text, bot_text = exchanges[i]
        cost = len(user_text) + len(bot_text)
        if char_budget is not None and used + cost > char_budget:
            continue
        graph.db.insert_node(
            context=user_text, response=bot_text,
            weight=0.6, timestamp=float(i), node_type="episodic",
        )
        used += cost
    return graph


def evaluate(graph, probes: Sequence[str]) -> Dict[str, float]:
    """Доля вопросов, на которые поиск вернул содержательно верный узел."""
    hits = 0
    for probe in probes:
        found = graph.search(probe, top_k=1, timestamp=1e9, with_associations=False)
        if found and is_hit(probe, f"{found[0].context} {found[0].response}"):
            hits += 1
    return hits / max(1, len(probes))


# --------------------------------------------------------------------------
# Прогон
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Сравнивает экономику памяти с наивным «сохранить всё»"
    )
    parser.add_argument("--messages", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--session-length", type=int, default=20)
    parser.add_argument("--gap-hours", type=float, default=8.0)
    parser.add_argument("--interference", action="store_true",
                        help="модель интерференции: важность = доля силы, "
                             "вытеснение по накопленной силе, без удаления по возрасту")
    parser.add_argument("--logs", action="store_true")
    args = parser.parse_args()

    if args.interference:
        # Решающий замер для всего замысла: единственный стенд, который
        # ставит вопрос при РАВНОМ БЮДЖЕТЕ. Без ограничения хранилища
        # несмещённый отбор непобедим по построению, и любая
        # избирательность выглядит помехой — что все прочие стенды и
        # показывали.
        config.USE_RELATIVE_STRENGTH = True
        config.DELETE_ON_DECAY = False
        print(" Модель интерференции: важность = доля накопленной силы")

    install_llm_stub()
    random.seed(args.seed)
    rng = random.Random(args.seed)

    from core.brain_session import BrainSession, ManualWallClock

    stream = build_message_stream(args.messages, 7, rng, tail_ratio=0.3)

    # --- Прогон через настоящий организм ---
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
    for i, text in enumerate(stream, start=1):
        response = session.process_message(text)
        exchanges.append((text, response.text))
        if args.session_length and args.gap_hours and i % args.session_length == 0:
            wall.advance(args.gap_hours * 3600.0)
            session.memory.apply_decay(now=session.clock.get_brain_time())

    kept = session.memory.db.count_nodes_by_type("episodic")

    # --- Вопросы: фразы, которые РЕАЛЬНО звучали ---
    said = Counter()
    for message in stream:
        for phrase in CORPUS:
            if phrase in message:
                said[phrase] += 1
    probes = sorted(said)
    rare = [p for p in probes if said[p] <= 2]
    common = [p for p in probes if said[p] >= 6]

    # --- Наивные хранилища ТОГО ЖЕ ОБЪЁМА В СИМВОЛАХ ---
    total = len(exchanges)
    budget = store_size(session.memory)

    everything = build_baseline_store(exchanges, range(total))
    recent = build_baseline_store(exchanges, range(total - 1, -1, -1), budget)
    shuffled = list(range(total))
    rng.shuffle(shuffled)
    sampled = build_baseline_store(exchanges, shuffled, budget)

    stores = [
        ("всё (верхняя граница)", everything),
        ("последние (окно)", recent),
        ("случайные (контроль)", sampled),
        ("организм", session.memory),
    ]

    print("=" * 78)
    print(" ЭКОНОМИКА ПАМЯТИ ПРОТИВ «СОХРАНИТЬ ВСЁ»")
    print("=" * 78)
    print(f" Поток: {args.messages} сообщений, паузы {args.gap_hours}ч каждые "
          f"{args.session_length}, seed={args.seed}")
    stream_chars = sum(len(u) + len(b) for u, b in exchanges)
    print(f" Организм сохранил {kept} эпизодов из {total} обменов")
    print(f" Бюджет: {budget} символов из {stream_chars} -> сжатие в "
          f"{stream_chars / max(1, budget):.1f} раз (СЧИТАЕТСЯ ПО ТЕКСТУ, не по узлам:")
    print(f" один узел организма — склейка окна STM, поэтому счёт узлов обманывал)")
    print(f" Вопросов: {len(probes)} (редких {len(rare)}, частых {len(common)})")
    print("-" * 78)
    print(f"{'хранилище':<24} {'узлов':>6} {'символов':>9} {'все':>7} {'редкое':>8} {'частое':>8}")
    print("-" * 78)

    for name, graph in stores:
        nodes = graph.db.count_nodes_by_type("episodic")
        print(
            f"{name:<24} {nodes:>6} {store_size(graph):>9} "
            f"{evaluate(graph, probes):>6.0%} "
            f"{evaluate(graph, rare):>7.0%} {evaluate(graph, common):>7.0%}"
        )

    print("-" * 78)
    print(" Ожидание: организм ЛУЧШЕ наивных на частом и ХУЖЕ на редком —")
    print(" он для того и построен. Лучше везде -> метрика врёт.")
    print(" Хуже везде -> ставка на забывание не работает.")
    print("=" * 78)

    session.close()


if __name__ == "__main__":
    main()
