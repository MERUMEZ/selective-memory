"""
================================================================================
 TOOLS/COMPARE_ASSOCIATION.PY — Даёт ли что-нибудь растекание активации
================================================================================
Растекание — самый частый механизм в системе: 2787 срабатываний за разговор
из восьмидесяти сообщений (`check_liveness`). И до сих пор его влияние на
качество не мерил НИ ОДИН стенд из пяти.

ПОЧЕМУ НЕ МЕРИЛ — И ЭТО ГЛАВНОЕ, ЧТО ВЫЯСНИЛОСЬ. LongMemEval с флагом
`--associations` даёт числа, СОВПАДАЮЩИЕ ПОБАЙТОВО с прогоном без него.
Причина не в том, что механизм бесполезен, а в том, что в бенчмарке ему
не по чему течь:

    рёбер эпизод-эпизод после заливки стога:      0
    рёбер слово-слово (граф языка):             640

Связи между воспоминаниями рождаются из `_associate_with_recalled` — новое
цепляется за то, что было ВЫНУТО ИЗ ПАМЯТИ незадолго до. Бенчмарк заливает
стог и только потом спрашивает, вспоминания по ходу не происходит, и
ассоциативной сети не возникает вовсе. То же верно для всех пяти стендов
проекта: все они устроены как «записать всё, потом спросить».

А живое приложение работает иначе: на каждую реплику оно СНАЧАЛА достаёт
из памяти нужное, и лишь потом сохраняет новое. Тогда связи и появляются.
Этот стенд воспроизводит именно такой порядок.

ЧТО ПРОВЕРЯЕТСЯ. Достраивание по части: вопрос делит слова с ОДНИМ узлом,
а ответ лежит в ДРУГОМ, связанном с ним по совместному припоминанию.

    записано:  «мою сестру зовут аня»       <- делит слово с вопросом
    записано:  «аня работает ветеринаром»   <- здесь ответ
    вопрос:    «кем работает моя сестра»    <- слова «аня» в вопросе нет

Без растекания находится первый узел и на этом всё. С растеканием ко
второму ведёт ребро, и ответ достаётся целиком. Это ровно то, чем занята
CA3: по обрывку восстановить образ целиком.

ИЗМЕРЕНО НА ШЕСТИДЕСЯТИ ЦЕПОЧКАХ. Прямой поиск не находит ответ НИКОГДА —
в узле-ответе нет ни одного слова из вопроса, — поэтому левая колонка
всюду ноль, и это не поломка, а условие задачи.

                            k=3     k=5
    дописывание в хвост    0/60   20/60   (как было)
    соревнование за места 10/60   44/60
    ДОСТРАИВАНИЕ          23/60   36/60

На малых k — а их и берёт приложение — достраивание выигрывает вдвое.
Включено по умолчанию: цена на LongMemEval 0.2 пункта, один вопрос из
пятисот, а целый класс вопросов переходит из «нерешаемо» в «решается».

УЗКОЕ МЕСТО ОКАЗАЛОСЬ НЕ В СЧЁТЕ, А В ОБРАЗОВАНИИ СВЯЗЕЙ. Связь возникает,
только если поиск ПЕРЕД записью уже нашёл родственное воспоминание. А он
находит всё хуже по мере наполнения:

    первые 20 цепочек:   якорь найден в 17/20
    средние 20:                          2/20
    последние 20:                        0/20

То есть ассоциации завязываются, пока память мала, и перестают, когда она
наполняется. Часть этого — настоящее явление (перегрузка ключа: чем больше
записей делят подсказку, тем труднее достать любую), часть — повторное
использование словаря в самом стенде. Разделить их — отдельная работа.

ПРЕЖНИЙ ВЫВОД БЫЛ «НОЛЬ ВЛИЯНИЯ ВО ВСЕХ УСЛОВИЯХ», и он держался на шести
цепочках. Ниже — что именно с ним было не так.

    условие                          рёбер   k=1      k=3      k=5
    залить и спросить потом              0   3->3     4->4     4->4
    вспоминать по ходу                  45   1->1     1->1     4->4
    вспоминать по ходу + соревнование   45   1->1     1->1     4->4

Стрелка — «без растекания -> с растеканием». Не изменилось НИ ОДНО число.

ПРИЧИНА ДВОЙНАЯ, И ОБЕ ЧАСТИ УСТРОЙСТВЕННЫЕ.

  1. Подтянутые узлы ДОПИСЫВАЮТСЯ В ХВОСТ, за пределы top_k. Замер: при
     top_k=3 дописанный узел стабильно оказывался на позиции 3, то есть
     четвёртым. Пока ранжированных совпадений набирается k, растекание в
     окно не попадает никогда.

  2. Соревнование за места (`associations_compete`) тоже не помогло, и
     это глубже первого. Счёт ассоциации равен счёту ЕЁ ИСТОЧНИКА,
     умноженному на затухание ребра, — то есть всегда НИЖЕ источника.
     Подтянуть она может только то, что и так стояло ниже.

ОСТОРОЖНО С ЧИСЛАМИ ЭТОГО СТЕНДА: ИХ ШЕСТЬ.

Здесь стоял вывод «живой порядок УХУДШАЕТ позднейший поиск, 3/6 -> 1/6».
Он не подтвердился. Замер на полном наборе (LongMemEval, 500 вопросов,
`--live`) показал обратное:

    залить и спросить потом   записано 26.3%   R@1 96.2%
    вспоминать по ходу        записано 26.0%   R@1 97.4%

Живой порядок ЛУЧШЕ на 1.2 пункта при той же избирательности — и это
объяснимо: припоминание поднимает стабильность вспомненного, то есть
нужное перестаёт забываться (эффект интервального повторения).

Шесть вопросов не отличают сигнал от шума: разница «3/6 против 1/6» — это
два вопроса, и те же числа поехали от посторонней правки кодировщика.
Стенд годится, чтобы показать НАЛИЧИЕ рёбер и отсутствие влияния
растекания; выводить по нему качество поиска нельзя.

Запуск:
    python tools/compare_association.py
================================================================================
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if "--logs" not in sys.argv:
    os.environ["LOG_LEVEL"] = "ERROR"

from selectivemem import Memory
from selectivemem.settings import MemorySettings

# ----------------------------------------------------------------------
# ЦЕПОЧКИ. В каждой: узел-зацепка (делит слова с вопросом), узел-ответ
# (слов вопроса не содержит) и сам вопрос с искомым словом.
# ----------------------------------------------------------------------

# ЦЕПОЧКИ ПОРОЖДАЮТСЯ, А НЕ ПЕРЕЧИСЛЯЮТСЯ.
#
# Первая версия держала шесть штук вручную, и по ним был сделан вывод,
# который потом опроверг полный набор: разница «3/6 против 1/6» — это два
# вопроса, и те же числа поехали от посторонней правки кодировщика.
# Шестьдесят цепочек отличают сигнал от шума.
#
# Строение цепочки: узел-зацепка делит слово с вопросом, узел-ответ не
# делит с вопросом НИЧЕГО. Связь между ними возникает только от
# совместного припоминания — то есть проверяется именно достраивание.

_RELATIONS = ["sister", "brother", "neighbour", "dentist", "landlord",
              "trainer", "tutor", "plumber", "barber", "accountant",
              "nephew", "godmother", "roommate", "supervisor", "florist"]
_NAMES = ["mira", "kovalenko", "brendt", "ashworth", "delgado", "okonkwo",
          "vasilyev", "haruki", "lindqvist", "moreau", "tanaka", "novak",
          "reyes", "fitzgerald", "abubakar"]
_PREDICATES = [
    ("breeds pedigree spaniels", "spaniels"),
    ("collects prewar postage stamps", "stamps"),
    ("restores upright pianos", "pianos"),
    ("teaches evening pottery", "pottery"),
    ("races vintage motorcycles", "motorcycles"),
    ("grows heirloom tomatoes", "tomatoes"),
    ("repairs mechanical watches", "watches"),
    ("photographs migrating cranes", "cranes"),
    ("brews unfiltered cider", "cider"),
    ("carves alabaster figurines", "figurines"),
    ("studies medieval calligraphy", "calligraphy"),
    ("sails a wooden ketch", "ketch"),
    ("bakes sourdough overnight", "sourdough"),
    ("tunes concert harpsichords", "harpsichords"),
    ("paints watercolour marshes", "marshes"),
]


# Разные обороты для зацепки и для ответа. ОБЯЗАТЕЛЬНО: первая версия
# генератора лепила все шестьдесят цепочек по одному шаблону, и гейт
# записал ДВЕ из шестидесяти — после третьего повторения формы фраза
# перестаёт удивлять. Восьмой случай вырожденных данных в этом проекте, и
# на этот раз в стенде, который сам же и проверяет память.
_ANCHOR_FORMS = [
    "my {rel} is called {name}",
    "we finally met my {rel}, {name}",
    "{name} turned out to be my {rel} after all",
    "everyone in the family calls my {rel} just {name}",
    "i introduced my {rel} {name} to the neighbours",
    "my {rel} signs everything as {name}",
]
_FACT_FORMS = [
    "{name} {pred}",
    "apparently {name} {pred} on weekends",
    "nobody knew that {name} {pred}",
    "{name} has {pred} since childhood",
    "it turns out {name} {pred} rather seriously",
    "{name} quietly {pred} and never mentions it",
]


def make_chains(count: int):
    """Собрать цепочки, не повторяя ни словаря, ни оборота подряд."""
    chains = []
    for index in range(count):
        relation = _RELATIONS[index % len(_RELATIONS)]
        name = _NAMES[(index * 7) % len(_NAMES)]
        predicate, answer = _PREDICATES[(index * 11) % len(_PREDICATES)]
        anchor = _ANCHOR_FORMS[index % len(_ANCHOR_FORMS)]
        fact = _FACT_FORMS[(index * 5) % len(_FACT_FORMS)]
        round_index = index // len(_RELATIONS)
        tag = "" if round_index == 0 else f" on the {_NAMES[round_index]} side"
        chains.append((
            [anchor.format(rel=relation + tag, name=name),
             fact.format(name=name, pred=predicate)],
            f"what does my {relation} do",
            answer,
        ))
    return chains


CHAINS = make_chains(60)

# Посторонние реплики между звеньями: без них цепочки шли бы подряд и
# связывались бы просто потому, что рядом во времени.
NOISE = [
    "сегодня опять забыл зонт дома",
    "в магазине снова переставили полки",
    "погода испортилась к вечеру",
    "надо бы поменять резину на зимнюю",
    "почта работает до восьми",
]


def build(with_recall: bool, compete: bool = False,
          completion: bool = False) -> Memory:
    """
    Прожить разговор.

    with_recall=True воспроизводит порядок живого приложения: СНАЧАЛА
    достать из памяти нужное для ответа, ПОТОМ сохранить сказанное. Именно
    в этом порядке возникают связи между воспоминаниями — новое цепляется
    за то, что было активно.

    with_recall=False — порядок всех наших стендов: залить и спросить
    потом. Здесь ассоциативной сети не возникает вовсе, и это измерено:
    ноль рёбер эпизод-эпизод после целого стога.
    """
    settings = MemorySettings()
    settings.associations_compete = compete
    settings.pattern_completion = completion
    memory = Memory(":memory:", settings=settings)
    clock = 0.0
    lines = []
    for index, (chain, _, _) in enumerate(CHAINS):
        lines.append((chain[0], True))
        lines.append((NOISE[index % len(NOISE)], False))
        lines.append((chain[1], True))

    for line, is_chain in lines:
        if with_recall:
            # Ассистент сначала ищет, чем ответить, и только потом пишет.
            memory.recall(line, top_k=3, timestamp=clock)
        # ЗВЕНЬЯ ЦЕПОЧЕК ПИШУТСЯ ПРИНУДИТЕЛЬНО (emotion=1.0 — документный
        # путь «пользователь сказал запомни»). Стенд меряет ПОИСК, а не
        # гейт, а гейт при шестидесяти однотипных цепочках записывал семь
        # из шестидесяти — и мерить было бы нечего. Посторонний поток идёт
        # обычным путём и тесноту создаёт честно.
        memory.observe(line, emotion=1.0 if is_chain else 0.0, timestamp=clock)
        clock += 3600.0
    return memory


def count_episode_edges(memory) -> int:
    cursor = memory.graph.db._conn.cursor()
    return cursor.execute(
        """SELECT COUNT(*) FROM edges
           WHERE node_from IN (SELECT id FROM episodes)
             AND node_to IN (SELECT id FROM episodes)"""
    ).fetchone()[0]


def probe(memory, with_associations: bool, top_k: int) -> int:
    hits = 0
    for _, question, answer_word in CHAINS:
        found = memory.recall(question, top_k=top_k, timestamp=1e6,
                              with_associations=with_associations)
        if any(answer_word in (m.context or "").lower() for m in found[:top_k]):
            hits += 1
    return hits


def main() -> None:
    total = len(CHAINS)
    print("=" * 78)
    print(" ДАЁТ ЛИ ЧТО-НИБУДЬ РАСТЕКАНИЕ АКТИВАЦИИ")
    print("=" * 78)

    for label, with_recall, compete, completion in (
            ("залить и спросить потом", False, False, False),
            ("вспоминать по ходу разговора", True, False, False),
            ("вспоминать по ходу + соревнование", True, True, False),
            ("вспоминать по ходу + ДОСТРАИВАНИЕ", True, False, True)):
        memory = build(with_recall, compete, completion)
        edges = count_episode_edges(memory)
        print(f"\n  {label}  (рёбер эпизод-эпизод: {edges})")
        print(f"    {'k':>3} {'без растекания':>16} {'с растеканием':>15}")
        for k in (1, 3, 5):
            off = probe(memory, False, k)
            on = probe(memory, True, k)
            mark = "  <--" if on > off else ""
            print(f"    {k:>3} {off:>12}/{total} {on:>11}/{total}{mark}")
        memory.close()

    print()
    print(" Верхняя половина — устройство всех пяти наших стендов. Если в ней")
    print(" рёбер ноль, растекание там не измеряется в принципе.")
    print("=" * 78)


if __name__ == "__main__":
    main()
