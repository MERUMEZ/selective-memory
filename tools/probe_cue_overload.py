"""
================================================================================
 TOOLS/PROBE_CUE_OVERLOAD.PY — Сколько записей может делить одну подсказку
================================================================================
Связи между воспоминаниями возникают, только если поиск ПЕРЕД записью уже
нашёл родственное. Стенд многошаговых цепочек показал, что находит он всё
хуже, и вопрос был — от чего.

ОТВЕТ: НЕ ОТ ОБЪЁМА ПАМЯТИ. При пятикратном складе (180 -> 840 узлов) связи
образуются ЧАЩЕ: 13/60 -> 24/60. Дело в том, сколько записей делят одну
подсказку:

    цепочек на одно имя    якорь найден
              1               23/60
              2               16/60
              4                3/60
              8                4/60

Дозовая зависимость налицо, разница восьмикратная. Это перегрузка ключа —
явление, описанное в психологии памяти ровно так же: извлечение ухудшается
с числом следов, связанных С ДАННОЙ подсказкой, и почти не зависит от
общего объёма хранимого.

ДЛЯ ПРИМЕНЕНИЯ это значит вот что: если пользователь упоминает «Анну» в
восьми разных обстоятельствах, связи между этими воспоминаниями почти не
завяжутся. Проверьте на своих данных, сколько записей у вас делит типичное
имя или термин.

Запуск:
    python tools/probe_cue_overload.py
================================================================================
"""
import os, sys, random; os.environ["LOG_LEVEL"]="ERROR"
sys.path.insert(0, "/var/www/mindnumbness")
from selectivemem import Memory
CONS, VOW = "bdfgklmnprstvz", "aeiou"

def run(share, count=60, seed=2):
    """share — сколько цепочек делят одно имя. Всё остальное уникально."""
    rng = random.Random(seed); seen = set()
    def token(n=3):
        while True:
            w = "".join(rng.choice(CONS) + rng.choice(VOW) for _ in range(n))
            if w not in seen:
                seen.add(w); return w
    names = [token() for _ in range((count + share - 1) // share)]
    chains = []
    for i in range(count):
        rel, verb, obj = token(), token(), token()
        name = names[i // share]
        chains.append(([f"my {rel} is called {name}", f"{name} {verb} the {obj}"],
                       obj))
    m = Memory(":memory:"); clock = 0.0; found = 0
    for chain, ans in chains:
        m.recall(chain[0], top_k=3, timestamp=clock)
        m.observe(chain[0], emotion=1.0, timestamp=clock); clock += 3600
        n = " ".join(token() for _ in range(5))
        m.recall(n, top_k=3, timestamp=clock); m.observe(n, timestamp=clock); clock += 3600
        got = m.recall(chain[1], top_k=3, timestamp=clock)
        found += any(chain[0] == (x.context or "") for x in got)
        m.observe(chain[1], emotion=1.0, timestamp=clock); clock += 3600
    m.close()
    return found

print(f"  {'цепочек на одно имя':24} {'якорь найден':>14}")
print("  " + "-" * 40)
for share in (1, 2, 4, 8):
    print(f"  {share:<24} {run(share):>10}/60")
