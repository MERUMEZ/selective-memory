"""
================================================================================
 TOOLS/CHECK_LIVENESS.PY — Срабатывает ли механизм ВООБЩЕ
================================================================================
Стенд родился из закономерности, а не из идеи. За двое суток нашлись
ЧЕТЫРЕ механизма, которые заявлены в описании, написаны, покрыты тестами
— и не включаются ни разу в живой работе:

  1. Связей между эпизодами не создавалось вовсе. 201 узел, 0 рёбер.
     Растекающейся активации, заявленной как multi-hop, было не по чему
     идти.
  2. Свежее ребро рождалось с весом 0.150 при пороге активации 0.3.
     Выше порога — ноль рёбер из двадцати семи.
  3. Пережившее это ребро опускалось ниже порога через пять суток,
     оставаясь в базе: связь есть, в поиске не участвует.
  4. Удаление по возрасту стирало улики раньше, чем их спрашивали:
     12 узлов из 12 в каждом разобранном случае.

Ни один из четырёх не ловится обычными тестами, и это не упрёк тестам.
Тест проверяет, что механизм РАБОТАЕТ ПРАВИЛЬНО, когда его вызвали с
подходящими данными. Он не проверяет, что в живом режиме такие данные
вообще возникают.

Здесь проверяется именно это: прогнать разговор, похожий на настоящий, и
посчитать, сколько раз сработал каждый механизм. Ноль — приговор.

Это НЕ замер качества. Механизм может срабатывать и приносить вред —
на такой вопрос отвечают compare_*.py. Здесь только "жив или мёртв".

Запуск:
    python tools/check_liveness.py
    python tools/check_liveness.py --messages 60
================================================================================
"""

import argparse
import collections
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("LOG_LEVEL", "INFO")

from selectivemem import Memory, MemorySettings  # noqa: E402

DAY = 86400.0

# Метки в логах, по которым видно срабатывание. Ключ — читаемое имя
# механизма, значение — фрагмент, который он печатает.
MARKERS = {
    "запись (спайк-гейт)": "[MEMORY SAVED]",
    "угасание": "[DECAY APPLIED]",
    "вытеснение устаревшего": "[SUPERSEDED]",
    "растекание активации": "[ASSOCIATION]",
    "переупорядочивание": "[RERANK]",
    "вытеснение по ёмкости": "[CAPACITY]",
    "консолидация": "[CONSOLIDATION]",
    "подрезка связей": "[EDGE DECAY APPLIED]",
    # СОН. Добавлены после того, как первая версия стенда объявила "все
    # механизмы живы", проверив только те восемь, что я вспомнил при
    # написании. Инструмент, созданный ловить слепые зоны, имел
    # собственную: подрезка, поиск хабов и абстрактные узлы в список не
    # попали. Отсюда правило — список механизмов сверяется с кодом, а не
    # с памятью.
    "синаптическая подрезка": "[SLEEP PRUNING]",
    "поиск хаб-кластеров": "[SLEEP CLUSTER]",
    "абстрактные узлы": "[SLEEP CONSOLIDATION]",
}

TOPICS = [
    "меня зовут Паша и я работаю программистом",
    "у меня аллергия на пенициллин",
    "моя собака зовут Рекс",
    "я живу в городе у реки",
    "мой отпуск начинается в июле",
    "я не ем мясо уже пять лет",
]

# Противоречия — чтобы проверить вытеснение устаревшего.
UPDATES = [
    "мою собаку теперь зовут Бобик",
    "я переехал в другой город",
]

FILLER = [
    "сосед шумит по вечерам", "погода обещает быть ясной",
    "надо разобрать кладовку", "цены в магазине растут",
    "во дворе новая лавочка", "лифт починили к среде",
]


class MarkerCounter(logging.Handler):
    """Считает, сколько раз в логах мелькнула каждая метка."""

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.counts: Dict[str, int] = collections.Counter()

    def emit(self, record):
        try:
            text = record.getMessage()
        except Exception:
            return
        for name, marker in MARKERS.items():
            if marker in text:
                self.counts[name] += 1


def run(messages: int, seed: int) -> Dict[str, int]:
    rng = random.Random(seed)
    counter = MarkerCounter()
    root = logging.getLogger("selectivemem")
    root.setLevel(logging.DEBUG)
    root.addHandler(counter)

    now = [1_700_000_000.0]
    settings = MemorySettings(
        # Всё, что мы починили за эти сутки, ВКЛЮЧЕНО: стенд отвечает на
        # вопрос "жив ли механизм, когда его не выключили", а не "жив ли
        # он при умолчаниях".
        associate_recalled_limit=3,
        rerank_band=0.05,
        memory_capacity=40,
        delete_on_decay=False,
        contradiction_search_threshold=0.5,
        consolidate_from_stm=True,
    )
    memory = Memory(":memory:", settings=settings, clock=lambda: now[0])

    stream = TOPICS + FILLER * 3 + UPDATES + FILLER * 3
    rng.shuffle(stream)
    stream = (stream * 10)[:messages]

    for index, text in enumerate(stream, start=1):
        memory.context_for(text, top_k=3)
        memory.observe(text, response="понятно", emotion=0.6)
        if index % 5 == 0:
            memory.feedback(+1.0)
        now[0] += 300.0
        if index % 10 == 0:
            now[0] += 6 * 3600.0
            memory.forget(now=now[0])

    now[0] += 3 * DAY
    memory.forget(now=now[0])
    # Сон вызывается ЯВНО, как forget: у библиотеки нет планировщика и она
    # не заводит потоков. Приложение само решает, когда простаивает.
    memory.sleep(timestamp=now[0])
    memory.recall("что ты обо мне помнишь", top_k=5)

    memory.close()
    root.removeHandler(counter)
    return counter.counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Срабатывает ли каждый механизм")
    parser.add_argument("--messages", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    counts = run(args.messages, args.seed)

    print("=" * 70)
    print(" ЖИВУЧЕСТЬ МЕХАНИЗМОВ")
    print("=" * 70)
    print(f" Разговор из {args.messages} сообщений, затем 3 суток молчания.")
    print("-" * 70)

    dead = []
    for name in MARKERS:
        hits = counts.get(name, 0)
        mark = "мёртв" if hits == 0 else f"{hits}"
        if hits == 0:
            dead.append(name)
        print(f" {name:28} {mark:>10}")

    print("=" * 70)
    if dead:
        print(f" МЁРТВЫХ МЕХАНИЗМОВ: {len(dead)}")
        for name in dead:
            print(f"   - {name}")
        sys.exit(1)
    print(" Все механизмы срабатывают хотя бы раз.")


if __name__ == "__main__":
    main()
