"""
================================================================================
 TOOLS/COMPARE_GATE.PY — Складывать эмоцию с новизной или умножать
================================================================================
Вопрос стенда: ПРАВИЛЬНО ЛИ ГЕЙТ РАЗЛИЧАЕТ ЧЕТЫРЕ РОДА СОБЫТИЙ.

Плотность записи считается как среднее эмоции и новизны:

    плотность = 0.5·эмоция + 0.5·новизна

Такой арифметики в биологии нет. Норадреналин не складывается с новизной —
он УМНОЖАЕТ пластичность, которую новизна уже открыла. Из-за среднего
событие страшное-но-привычное и безразличное-но-небывалое получают почти
одинаковую оценку:

    эмоция 0.9, новизна 0.2  ->  среднее 0.550, умножение 0.380
    эмоция 0.1, новизна 0.9  ->  среднее 0.500, умножение 0.990

Среднее считает их одинаковыми, умножение разводит втрое.

ПОЧЕМУ ЭТОТ СТЕНД ПРИШЛОСЬ ПОСТРОИТЬ. Ни один существующий не мог
ответить: все они передают ПОСТОЯННУЮ эмоцию (0.6 или 0.9). А при
постоянной эмоции обе формы — монотонные функции новизны, то есть одно и
то же решающее правило с разным порогом. Различить их нечем по построению,
и любой замер показал бы разницу, которой нет.

ЧЕСТНОСТЬ КОНСТРУКЦИИ. Пороги КАЛИБРУЮТСЯ так, чтобы обе формы писали
одинаковую долю реплик. Иначе сравнивались бы разные бюджеты записи, а не
разные правила отбора — ровно та ошибка, из-за которой в этом проекте уже
дважды снимали заявленные числа.

Запуск:
    python tools/compare_gate.py
    python tools/compare_gate.py --gain 2.0 --seeds 1,7,13,42,99
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

# Четыре рода событий. Эмоцию задаёт приложение, новизну — сам организм,
# поэтому «привычное» здесь достигается ПОВТОРЕНИЕМ до замера, а не
# подкруткой числа.
CHARGED_FAMILIAR = [
    "у меня аллергия на пенициллин",
    "моя дочь боится собак",
    "я не переношу молочное",
]
PLAIN_NOVEL = [
    "вчера в парке открыли новую карусель",
    "сосед купил синий велосипед",
    "в магазине завезли гречку по акции",
    "автобус теперь ходит через мост",
    "во дворе поставили скамейку",
    "на углу открыли кофейню",
]
CHARGED_NOVEL = [
    "меня укусила собака в подъезде",
    "я потерял кошелёк с документами",
]
PLAIN_FAMILIAR = [
    "сегодня обычная погода",
    "всё как всегда",
]

FILLER = ["понятно", "ага", "ясно", "хорошо", "ладно"]


def _emotion(text: str) -> float:
    if text in CHARGED_FAMILIAR or text in CHARGED_NOVEL:
        return 0.9
    return 0.05


def run_once(seed: int, gain: float, threshold: float) -> Dict[str, object]:
    rng = random.Random(seed)
    now = [1_700_000_000.0]
    memory = Memory(
        ":memory:",
        settings=MemorySettings(
            gate_emotion_gain=gain,
            base_plasticity_threshold=threshold,
            delete_on_decay=False,
        ),
        clock=lambda: now[0],
    )

    # ВЗРОСЛЕНИЕ. Заряженное-привычное и пустое-привычное повторяются, чтобы
    # СОБСТВЕННАЯ новизна у них упала. Это единственный честный способ
    # сделать событие привычным: подкрутить удивление снаружи нельзя.
    for _ in range(6):
        for text in CHARGED_FAMILIAR + PLAIN_FAMILIAR:
            memory.observe(text, response=rng.choice(FILLER),
                           emotion=_emotion(text), timestamp=now[0])
            now[0] += 300.0

    written: Dict[str, int] = {}
    total = 0
    for group, texts in (
        ("заряженное привычное", CHARGED_FAMILIAR),
        ("пустое небывалое", PLAIN_NOVEL),
        ("заряженное небывалое", CHARGED_NOVEL),
        ("пустое привычное", PLAIN_FAMILIAR),
    ):
        hits = 0
        for text in texts:
            obs = memory.observe(text, response=rng.choice(FILLER),
                                 emotion=_emotion(text), timestamp=now[0])
            now[0] += 300.0
            hits += obs.written
            total += 1
        written[group] = hits

    # НАХОДИМОСТЬ ВАЖНЕЕ ДОЛИ ЗАПИСИ, и без неё стенд ответил бы неверно.
    # Умножение отказывается писать заряженное-привычное — но оно было
    # записано в ПЕРВЫЙ раз, когда ещё было новым. Вопрос не в том, пишем
    # ли повторно, а в том, находится ли оно потом.
    found = {}
    for group, texts in (
        ("заряженное привычное", CHARGED_FAMILIAR),
        ("пустое небывалое", PLAIN_NOVEL),
        ("заряженное небывалое", CHARGED_NOVEL),
    ):
        hits = 0
        for text in texts:
            got = memory.recall(text, top_k=3, timestamp=now[0],
                                with_associations=False)
            hits += any(text.lower()[:18] in g.context.lower() for g in got)
        found[group] = hits

    stored = memory.graph.gate.episodic.count()
    memory.close()
    return {"written": written, "found": found, "stored": stored, "total": total}


def _write_rate(seeds: List[int], gain: float, threshold: float) -> float:
    rows = [run_once(s, gain, threshold) for s in seeds]
    return statistics.mean(
        sum(r["written"].values()) / r["total"] for r in rows
    )


def calibrate(seeds: List[int], gain: float, target: float) -> float:
    """
    Подобрать порог так, чтобы доля записи совпала с целевой.

    БЕЗ ЭТОГО СРАВНЕНИЕ ВРЁТ. Умножение при нулевой эмоции даёт вдвое
    большую плотность, чем среднее, поэтому при одном пороге оно просто
    пишет больше — и «выигрыш» оказался бы выигрышем бюджета, а не
    правила отбора.
    """
    lo, hi = 0.01, 0.99
    for _ in range(12):
        mid = (lo + hi) / 2
        rate = _write_rate(seeds, gain, mid)
        if rate > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Складывать эмоцию с новизной или умножать"
    )
    parser.add_argument("--seeds", default="1,7,13")
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--logs", action="store_true")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    base_threshold = MemorySettings().base_plasticity_threshold

    print("=" * 76)
    print(" СКЛАДЫВАТЬ ЭМОЦИЮ С НОВИЗНОЙ ИЛИ УМНОЖАТЬ")
    print("=" * 76)
    print(" Привычное становится привычным ПОВТОРЕНИЕМ — шесть кругов до замера,")
    print(" чтобы у организма упала собственная новизна. Подкрутить удивление")
    print(" снаружи нельзя, и это единственный честный способ.")
    print("-" * 76)

    target = _write_rate(seeds, 0.0, base_threshold)
    print(f" Доля записи у среднего при пороге {base_threshold:.2f}: {target:.1%}")
    tuned = calibrate(seeds, args.gain, target)
    tuned_rate = _write_rate(seeds, args.gain, tuned)
    print(f" Порог для умножения, подобранный под ту же долю: {tuned:.3f} "
          f"(вышло {tuned_rate:.1%})")
    print("-" * 76)
    print(f" {'род события':<24}{'записано':>19}{'НАЙДЕНО ПОТОМ':>22}")
    print(f" {'':<24}{'среднее':>10}{'умножение':>9}{'среднее':>11}{'умножение':>11}")

    rows_avg = [run_once(s, 0.0, base_threshold) for s in seeds]
    rows_mul = [run_once(s, args.gain, tuned) for s in seeds]
    for group in ("заряженное привычное", "пустое небывалое",
                  "заряженное небывалое", "пустое привычное"):
        a = statistics.mean(r["written"][group] for r in rows_avg)
        b = statistics.mean(r["written"][group] for r in rows_mul)
        fa = statistics.mean(r["found"].get(group, 0) for r in rows_avg)
        fb = statistics.mean(r["found"].get(group, 0) for r in rows_mul)
        cell = f"{fa:>11.1f}{fb:>11.1f}" if group in rows_avg[0]["found"] else f"{'—':>11}{'—':>11}"
        print(f" {group:<24}{a:>10.1f}{b:>9.1f}{cell}")
    print()
    print(f" узлов в памяти: среднее {statistics.mean(r['stored'] for r in rows_avg):.1f}, "
          f"умножение {statistics.mean(r['stored'] for r in rows_mul):.1f}")

    print("=" * 76)
    print(" ЧЕГО ЖДЁМ ОТ ПРАВИЛЬНОЙ ФОРМЫ: заряженное-небывалое пишется всегда,")
    print(" пустое-привычное не пишется никогда, а между ними — предпочтение")
    print(" НЕБЫВАЛОМУ, потому что привычное уже лежит в памяти и второй след")
    print(" ему не нужен, сколько бы эмоции к нему ни прилагалось.")
    print("=" * 76)


if __name__ == "__main__":
    main()
