"""
================================================================================
 TOOLS/COMPARE_INTERFERENCE.PY — Есть ли у хранения цена
================================================================================
Замер, от которого зависит, чем является продукт.

В мозге забывание существует НЕ ради места. Долговременная память не
переполняется; забывание нужно, чтобы извлечение оставалось возможным.
Чем больше похожих следов отзывается на один и тот же ключ, тем труднее
достать любой — это называется перегрузкой признака.

У нас поиск оценивает каждого кандидата НЕЗАВИСИМО и берёт лучших.
Значит лишний узел, возможно, ничего не стоит. А если хранение
бесплатно, то забывание может только вредить — что все наши замеры и
показывали: минус 18.6 пункта, как только перестали удалять.

    ЕСЛИ ХРАНЕНИЕ БЕСПЛАТНО, ПРОДАВАТЬ ЗАБЫВАНИЕ НЕЛЬЗЯ.

Здесь это проверяется прямо: одни и те же вопросы, растущий стог.

ОГОВОРКА, БЕЗ КОТОРОЙ ЗАМЕР СОВРЁТ. Перегрузка признака — про ПОХОЖИХ
конкурентов, а не про любых. Тысяча реплик про погоду не мешает
вспомнить про аллергию; мешают двадцать реплик про лекарства. Поэтому
отвлекающие узлы здесь строятся ИЗ СЛОВ САМИХ ФАКТОВ и делят с вопросом
ключи, но ответа не содержат.

Прогон со случайным шумом показал бы ровно ничего и был бы пятым пустым
замером за две сессии.

ВТОРОЙ ВОПРОС, И ОН ЧУТЬ НЕ ОСТАЛСЯ БЕЗ ОТВЕТА: что решает выдачу —
вес узла или накопленная сила. Долгое время стенд отвечал на него
неверно, и разница «плюс 33 пункта» попала в аудит, ничем не будучи
измерена: включение и выключение ранжирования по силе давало ПОБАЙТОВО
одинаковые числа.

Причина не в том, что силы у узлов равны между собой — они как раз
разные, двойники давят друг друга штрафом за устаревание, и шесть фактов
приходят к 0.000 против 0.029 у отвлекающих. Причина в том, что СИЛА И
ВЕС ДЕРЖАТ ОДНО И ТО ЖЕ ЗНАЧЕНИЕ: штраф бьёт по обеим одинаково, а
больше их здесь ничто не двигало. Выбор между двумя равными числами не
меняет ничего.

Расходятся они только от подкрепления, извлечения и затухания. Теперь
это делают два флага:

    --compare-strength   шесть фактов подкреплены, как их отметило бы
                         приложение:
                             из веса узла        44.4%  44.4%  61.1%
                             из накопленной силы 88.9%  94.4% 100.0%

    --sweep-use-step     похвалы нет вовсе, сила растёт только оттого,
                         что к теме возвращаются. Отвечает на вопрос,
                         сколько таких возвращений нужно, и показывает,
                         что эффект включается ПОРОГОМ около +0.6.

Запуск:
    python tools/compare_interference.py
    python tools/compare_interference.py --compare-strength
    python tools/compare_interference.py --sweep-use-step
    python tools/compare_interference.py --levels 0,100,400,1600
================================================================================
"""

import argparse
import os
import random
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if "--logs" not in sys.argv:
    os.environ["LOG_LEVEL"] = "ERROR"

from selectivemem import Memory, MemorySettings  # noqa: E402

# Факт -> вопрос о нём. Вопрос делит с фактом часть слов, как в жизни.
TARGETS: List[Tuple[str, str]] = [
    ("у меня аллергия на пенициллин", "на какое лекарство у меня аллергия"),
    ("моя собака рекс боится грозы", "чего боится моя собака"),
    ("я работаю программистом в банке", "кем я работаю"),
    ("мой отпуск начинается в июле", "когда у меня отпуск"),
    ("я живу на улице пушкина", "на какой улице я живу"),
    ("моя дочь учится в третьем классе", "в каком классе моя дочь"),
]

# Слова, из которых строится БЛИЗКИЙ стог: они взяты из самих фактов и
# вопросов, поэтому отвлекающие узлы отзываются на те же ключи.
# ПОЧТИ ДВОЙНИКИ. Первая версия стенда делала отвлекающие узлы из
# одного-двух общих слов, и замер упёрся в потолок: 100% уже без стога.
# Настоящая перегрузка признака — это когда конкурент делит с фактом
# ПОЧТИ ВСЕ слова и отличается только тем, к кому относится.
_WHO = ["у соседа", "у коллеги", "у знакомого", "у брата", "у врача",
        "у продавца", "у водителя", "у племянника", "у сослуживца",
        "у попутчика", "у соседки", "у товарища"]

_SHADOWS = [
    "{who} аллергия на пенициллин",
    "{who} собака боится грозы",
    "{who} работа программистом в банке",
    "{who} отпуск начинается в июле",
    "{who} квартира на улице пушкина",
    "{who} дочь учится в третьем классе",
]


def build_distractors(rng: random.Random, count: int) -> List[str]:
    """
    Двойники фактов: те же слова, другой носитель. Именно они и создают
    перегрузку признака — на ключ "аллергия пенициллин" отзывается
    двадцать узлов, и нужный тонет среди них.
    """
    pool = [tpl.format(who=who) for tpl in _SHADOWS for who in _WHO]
    extra = [f"{who} тоже {tpl.format(who='').strip()}"
             for tpl in _SHADOWS for who in _WHO]
    pool += extra
    rng.shuffle(pool)
    out = []
    while len(out) < count:
        out.extend(pool)
    return out[:count]


def run_once(seed: int, distractors: int, top_k: int,
             suppression: float = 0.0, rehearsals: int = 0,
             reinforce: float = 0.0, use_strength: bool = True,
             use_step: Optional[float] = None) -> Dict[str, float]:
    rng = random.Random(seed)
    now = [1_700_000_000.0]
    memory = Memory(
        ":memory:",
        settings=MemorySettings(
            delete_on_decay=False,
            use_relative_strength=use_strength,
            retrieval_suppression=suppression,
            **({} if use_step is None else {"strength_use_step": use_step}),
        ),
        clock=lambda: now[0],
    )

    # Факты записываются ПЕРВЫМИ, дальше их заваливает похожим.
    target_ids = []
    for fact, _ in TARGETS:
        target_ids.append(
            memory.graph.save_connection(fact, "понятно", weight=0.7, timestamp=now[0])
        )
        now[0] += 60.0
    for text in build_distractors(rng, distractors):
        memory.graph.save_connection(text, "понятно", weight=0.7, timestamp=now[0])
        now[0] += 60.0

    # ПОДКРЕПЛЕНИЕ. Без него все узлы имеют одну и ту же силу, и
    # ранжирование по силе не может ничего переставить: слагаемое одинаково
    # у всех и лишь сдвигает счёт целиком. Прежняя версия стенда этого не
    # делала — и её сравнение «вес против силы» давало ПОБАЙТОВО одинаковые
    # числа, которые попали в аудит как «плюс 33 пункта».
    #
    # Здесь важное отмечается ровно так, как это сделало бы приложение:
    # пользователь вернулся к теме, и она заработала себе силу.
    if reinforce > 0.0:
        for node_id in target_ids:
            if node_id is not None:
                memory.graph.db.add_strength(node_id, reinforce,
                                             memory.settings.strength_max)

    # Подавление КОПИТСЯ: одно извлечение ничего не решает, как и у людей.
    for _ in range(rehearsals):
        for _fact, question in TARGETS:
            memory.recall(question, top_k=top_k, timestamp=now[0],
                          with_associations=False)

    hits_1 = hits_k = 0
    for fact, question in TARGETS:
        found = memory.recall(question, top_k=top_k, timestamp=now[0],
                              with_associations=False)
        texts = [f"{m.context}".lower() for m in found]
        if texts and fact.lower() in texts[0]:
            hits_1 += 1
        if any(fact.lower() in t for t in texts):
            hits_k += 1

    nodes = memory.graph.db.count_nodes_by_type("episodic")
    memory.close()
    return {"r1": hits_1 / len(TARGETS), "rk": hits_k / len(TARGETS), "nodes": nodes}


def main() -> None:
    parser = argparse.ArgumentParser(description="Дорожает ли извлечение с ростом памяти")
    parser.add_argument("--levels", default="0,50,200,800")
    parser.add_argument("--seeds", default="1,7,13")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--suppression", type=float, default=0.0,
                        help="подавление проигравших конкурентов при извлечении")
    parser.add_argument("--rehearsals", type=int, default=0,
                        help="сколько раз спросить ДО замера: подавление копится")
    parser.add_argument("--reinforce", type=float, default=0.0,
                        help="добавить силы шести фактам: без этого силы у всех "
                             "узлов равны и ранжировать по ней нечего")
    parser.add_argument("--compare-strength", action="store_true",
                        help="таблица «вес против силы» при подкреплённых фактах")
    parser.add_argument("--sweep-use-step", action="store_true",
                        help="сколько возвращений к теме нужно, чтобы она "
                             "поднялась: развёртка по strength_use_step")
    parser.add_argument("--logs", action="store_true")
    args = parser.parse_args()

    levels = [int(x) for x in args.levels.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    print("=" * 70)
    print(" ЕСТЬ ЛИ У ХРАНЕНИЯ ЦЕНА")
    print("=" * 70)
    print(f" {len(TARGETS)} фактов, стог из БЛИЗКИХ по словам узлов, сидов {len(seeds)}.")
    print(" Если извлечение не проседает — хранение бесплатно, и забывание")
    print(" в нынешней архитектуре продавать нельзя.")
    print("-" * 70)
    print(f" {'отвлекающих':>12} {'узлов':>8} {'R@1':>8} {f'R@{args.top_k}':>8}")

    first = None
    last = None
    for level in levels:
        rows = [run_once(seed, level, args.top_k, args.suppression, args.rehearsals,
                         args.reinforce)
                for seed in seeds]
        r1 = statistics.mean(r["r1"] for r in rows)
        rk = statistics.mean(r["rk"] for r in rows)
        nodes = statistics.mean(r["nodes"] for r in rows)
        if first is None:
            first = (r1, rk)
        last = (r1, rk)
        print(f" {level:>12} {nodes:8.0f} {r1*100:7.1f}% {rk*100:7.1f}%")

    print("=" * 70)
    drop_1 = (first[0] - last[0]) * 100
    drop_k = (first[1] - last[1]) * 100
    print(f" ПРОСАДКА: R@1 {drop_1:+.1f} пунктов, R@{args.top_k} {drop_k:+.1f} пунктов")
    if drop_k < 5 and drop_1 < 5:
        print(" Хранение БЕСПЛАТНО: конкуренции при извлечении нет, её надо")
        print(" строить с нуля — либо честно продавать дешёвую запись.")
    else:
        print(" Конкуренция при извлечении ЕСТЬ. Избирательность окупается на")
        print(" объёмах, и её надо усиливать, а не изобретать.")

    if args.compare_strength:
        # ОТДЕЛЬНАЯ ТАБЛИЦА, И ТОЛЬКО С ПОДКРЕПЛЕНИЕМ. Сравнивать вес с
        # силой на узлах равной силы бессмысленно: обе ветки дают
        # одинаковый ответ, и это уже однажды приняли за результат.
        print()
        print("=" * 70)
        print(" ЧТО РЕШАЕТ ВЫДАЧУ: ВЕС УЗЛА ИЛИ НАКОПЛЕННАЯ СИЛА")
        print("=" * 70)
        print(" Шесть фактов подкреплены, отвлекающие — нет. Проверяется, умеет")
        print(" ли память поднять важное над однословными двойниками.")
        print("-" * 70)
        print(f" {'важность из':<22} " + " ".join(f"{f'R@1 при {l}':>13}" for l in levels))
        for label, use_strength in (("веса узла", False), ("накопленной силы", True)):
            cells = []
            for level in levels:
                vals = [run_once(seed, level, args.top_k, args.suppression,
                                 args.rehearsals, reinforce=0.8,
                                 use_strength=use_strength)["r1"]
                        for seed in seeds]
                cells.append(f"{statistics.mean(vals)*100:12.1f}%")
            print(f" {label:<22} " + " ".join(cells))
        print("=" * 70)

    if args.sweep_use_step:
        # ПОДКРЕПЛЕНИЕ БЕЗ ЕДИНОГО ВЫЗОВА feedback. Здесь никто не хвалит:
        # сила растёт только оттого, что к теме возвращаются. Это главный
        # для продукта случай — приложения почти никогда не сообщают
        # важность сами, а пользователь возвращается к своему всегда.
        print()
        print("=" * 70)
        print(" СКОЛЬКО ВОЗВРАЩЕНИЙ К ТЕМЕ НУЖНО, ЧТОБЫ ОНА ПОДНЯЛАСЬ")
        print("=" * 70)
        print(" Похвалы нет. Растёт только то, что извлекали.")
        print("-" * 70)
        print(f" {'шаг':>6} {'повторов':>9} {'прирост':>9} "
              + " ".join(f"{f'R@1 при {l}':>13}" for l in levels))
        for step in (0.05, 0.15, 0.30):
            for reh in (3, 6, 12):
                cells = []
                for level in levels:
                    vals = [run_once(seed, level, args.top_k, args.suppression,
                                     rehearsals=reh, reinforce=0.0,
                                     use_strength=True, use_step=step)["r1"]
                            for seed in seeds]
                    cells.append(f"{statistics.mean(vals)*100:12.1f}%")
                print(f" {step:>6.2f} {reh:>9} {step*reh:>9.2f} " + " ".join(cells))
        print("=" * 70)
        print(" Эффект включается ПОРОГОМ около +0.6, а не плавно. Отсюда")
        print(" и умолчание 0.15: при 0.05 порог берётся за 12 возвращений,")
        print(" чего в живом разговоре не бывает.")


if __name__ == "__main__":
    main()
