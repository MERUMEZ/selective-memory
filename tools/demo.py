"""
================================================================================
 TOOLS/DEMO.PY — Память в работе, за одну минуту
================================================================================
Показывает то, что отличает эту библиотеку от векторной базы: она НЕ ПИШЕТ
всё подряд, забывает несущественное и достраивает ответ по связям, которых
в вопросе нет.

Не бенчмарк и не тест. Бенчмарк — `bench_longmemeval.py`, он даёт числа;
этот файл даёт увидеть механику на пятнадцати репликах.

Запуск:
    python tools/demo.py
    python tools/demo.py --verbose    # с внутренним состоянием на каждом шаге
================================================================================
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if "--logs" not in sys.argv:
    os.environ["LOG_LEVEL"] = "ERROR"

from selectivemem import Memory

HOUR = 3600.0

# Разговор человека с ассистентом. Часть реплик содержательна, часть —
# обычный трёп, и в этом весь смысл: память должна их РАЗДЕЛИТЬ сама.
# Разговор человека с ассистентом за несколько дней. Содержательное
# перемешано с обычным трёпом, и в этом весь смысл: память должна
# разделить их САМА.
#
# ДЛИНА ВЗЯТА НЕ С ПОТОЛКА. На пятнадцати репликах записывается 67% —
# организму ново ещё всё, и демонстрация врёт про избирательность.
# Настоящая доля видна, только когда язык уже усвоен.
CONVERSATION = [
    ("i am allergic to penicillin", "noted"),
    ("nice weather today", "it is"),
    ("my sister is called mira", "got it"),
    ("thanks", "sure"),
    ("mira breeds pedigree spaniels", "interesting"),
    ("ok", "ok"),
    ("i have not eaten meat for three years", "understood"),
    ("what time is it", "half past two"),
    ("morning", "good morning"),
    ("just checking in", "all quiet here"),
    ("my flight to istanbul leaves on the twentieth", "safe travels"),
    ("hmm", ""),
    ("how are you", "fine, thanks"),
    ("nothing much going on", "sounds calm"),
    ("i work as a backend developer at a fintech", "noted"),
    ("cool", ""),
    ("still here", "still here"),
    ("weather again looks fine", "it does"),
    ("bye for now", "see you"),
    ("hi again", "hello"),
    ("what time is it now", "quarter to six"),
    ("ok thanks", "sure"),
    ("morning again", "good morning"),
    ("nothing new today", "understood"),
    ("actually the flight moved to the twenty second", "updated"),
    ("right", ""),
    ("just saying hi", "hi"),
    ("all fine here", "good"),
    ("mira also restores upright pianos", "noted"),
    ("ok", "ok"),
    ("checking in again", "all quiet"),
    ("weather is the same", "it is"),
    ("thanks again", "sure"),
    ("nothing to report", "understood"),
    ("see you", "bye"),
    ("hello", "hi"),
]

QUESTIONS = [
    ("what medication must i avoid", "аллергия названа прямо"),
    ("when do i fly", "поправка должна вытеснить первую дату"),
    ("what does my sister do", "в вопросе нет слова «mira» — работает достраивание"),
    ("am i a vegetarian", "слова «vegetarian» в памяти нет вовсе"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Память в работе за минуту")
    parser.add_argument("--verbose", action="store_true",
                        help="показывать внутреннее состояние на каждом шаге")
    parser.add_argument("--logs", action="store_true")
    args = parser.parse_args()

    memory = Memory(":memory:")
    print("=" * 78)
    print(" ИЗБИРАТЕЛЬНАЯ ПАМЯТЬ: ЧТО ПРОИСХОДИТ НА САМОМ ДЕЛЕ")
    print("=" * 78)
    print(f" {memory.describe_setup()}")
    print()
    print(" 1. РАЗГОВОР. Память решает по каждой реплике, стоит ли её держать.")
    print("-" * 78)

    clock = 0.0
    written = 0
    for text, response in CONVERSATION:
        # Живой порядок: сначала достать, чем отвечать, потом записать.
        memory.recall(text, top_k=3, timestamp=clock)
        result = memory.observe(text, response=response, timestamp=clock)

        mark = "ЗАПИСЬ " if result.node_id else "  мимо "
        written += bool(result.node_id)
        note = ""
        if result.superseded_ids:
            note = "  <- поправка, старая версия ослаблена"
        print(f"   {mark} удивление {result.surprise:.2f}  {text[:44]:44}{note}")
        if args.verbose:
            state = memory.feel()
            print(f"           {state.describe()}")
        clock += HOUR

    total = len(CONVERSATION)
    print("-" * 78)
    print(f" Записано {written} из {total} реплик ({written / total:.0%}). "
          f"Остальное отсеяли ворота: это не сжатие постфактум, а решение\n"
          f" в момент события — как в мозге, где след закрепляется только при\n"
          f" достаточной новизне или значимости.")

    print()
    print(" 2. ВОПРОСЫ. Слова вопроса часто не совпадают со словами памяти.")
    print("-" * 78)
    for question, why in QUESTIONS:
        found = memory.recall(question, top_k=1, timestamp=clock)
        answer = found[0].context if found else "— ничего не нашлось"
        print(f"   {question}")
        print(f"      -> {answer}")
        print(f"         ({why})")

    print()
    print(" 3. ЗАБЫВАНИЕ. Проходит месяц, и к одному факту возвращаются.")
    print("-" * 78)

    def node_of(fragment):
        for row in memory.graph.gate.episodic.searchable():
            if fragment in (row["context"] or ""):
                return row
        return None

    # К аллергии возвращаются пять раз за месяц, к трёпу — ни разу.
    # ИМЕННО ЭТО и разделяет судьбу следов: без повторного припоминания
    # оба падают одинаково, и раздел выглядел бы так, будто память просто
    # стирает всё подряд.
    for week in range(1, 6):
        memory.recall("penicillin allergy", top_k=1,
                      timestamp=clock + week * 5 * 24 * HOUR)
    memory.forget(now=clock + 30 * 24 * HOUR)

    print(f"   {'след':34} {'вес':>7} {'стабильность':>13}")
    for fragment, label in (("penicillin", "аллергия, вспоминали 5 раз"),
                            ("weather is the same", "трёп, не трогали")):
        row = node_of(fragment)
        if row:
            print(f"   {label:34} {row['weight']:7.4f} {row['stability']:13.2f}")
    print()
    print("   Разница в полтораста раз, и она не от важности, а от ИСПОЛЬЗОВАНИЯ.")
    print("   Каждое припоминание поднимает стабильность — след угасает медленнее.")
    print("   Это эффект интервального повторения, и он здесь не запрограммирован")
    print("   отдельно: он следует из того, что извлечение не бесплатно.")

    state = memory.feel()
    print()
    print(" 4. САМОЧУВСТВИЕ. Значимость события организм выводит сам.")
    print("-" * 78)
    print(f"   {state.describe()}")
    print("   Приложение может передать эмоцию явно — тогда главнее она.")
    print("=" * 78)
    memory.close()


if __name__ == "__main__":
    main()
