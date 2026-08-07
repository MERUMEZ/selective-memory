"""
================================================================================
 TEST_ASSOCIATION.PY — Библиотека обязана связывать воспоминания между собой
================================================================================
Замер, из которого родился этот файл: после 200 вызовов observe() в базе
оказался 201 эпизодический узел и НОЛЬ рёбер между ними. Не мало — ни
одного. Библиотека не связывала воспоминания вообще; рёбра, которые видно
в демонстрации, создаёт витрина (core/brain_session.py), у которой под
рукой сразу оба конца — и вспомненный узел, и записываемый.

Для человека, поставившего пакет, это значило две вещи сразу:

  1. связность не могла работать сигналом важности — нечего измерять;
  2. растекающейся активации, заявленной в README как multi-hop и
     занимающей заметную часть кода поиска, было НЕ ПО ЧЕМУ растекаться.

Второе — расхождение витрины с пакетом, то есть ровно тот разряд ошибок,
который вскрывается на демонстрации у покупателя.

Правило связывания хеббовское: что было возбуждено вместе, то и
связывается. Приложение только что достало из памяти какие-то узлы — это
и есть контекст, в котором появилось новое воспоминание.
================================================================================
"""

import pytest

from selectivemem import Memory, MemorySettings
from selectivemem import embeddings

# Проверка опирается на СЕМАНТИКУ: запрос сформулирован другими словами,
# чем сохранённый текст. Без модели поиск честно отвечает пустотой и сам
# об этом предупреждает в логе — это заявленная деградация, а не поломка.
requires_model = pytest.mark.skipif(
    not embeddings.is_available(),
    reason="запрос сформулирован иначе, чем запись: нужна семантическая модель",
)



@pytest.fixture
# ВТОРОЙ ИСТОЧНИК СВЯЗЕЙ ВЫКЛЮЧАЕТСЯ ЗДЕСЬ ЯВНО.
#
# Кроме связывания по припоминанию у памяти есть временная смежность:
# новое цепляется за записанное рядом по времени независимо от
# содержания. Проверки в этом файле про ПЕРВЫЙ механизм, и без явного
# temporal_link_window=0 они падали на рёбрах, созданных вторым.

def memory():
    m = Memory(":memory:", settings=MemorySettings(associate_recalled_limit=3, temporal_link_window=0))
    yield m
    m.close()


def _episodic_ids(m):
    return [r["id"] for r in m.graph.db.fetch_all_nodes() if r["node_type"] == "episodic"]


def test_recalled_node_links_to_the_next_memory(memory):
    """Вспомнили — записали. Между ними должно появиться ребро."""
    first = memory.observe("у меня аллергия на пенициллин", emotion=0.9).node_id
    assert first is not None

    memory.recall("аллергия")
    second = memory.observe("врач выписал другой антибиотик", emotion=0.8).node_id
    assert second is not None

    edges = memory.graph.db.get_edges_between([first, second])
    assert edges, "новая запись не связалась с тем, что было вспомнено"


def test_without_recall_nothing_is_linked(memory):
    """
    Связь возникает от СОВМЕСТНОЙ АКТИВНОСТИ, а не от соседства во времени.
    Две записи подряд без единого вспоминания связывать не за что.
    """
    first = memory.observe("у меня аллергия на пенициллин", emotion=0.9).node_id
    second = memory.observe("во дворе растёт высокое дерево", emotion=0.9).node_id

    assert memory.graph.db.get_edges_between([first, second]) == []


def test_zero_limit_restores_previous_behaviour():
    """Ноль обязан полностью выключать механизм: это путь отката."""
    m = Memory(":memory:", settings=MemorySettings(associate_recalled_limit=0, temporal_link_window=0))
    first = m.observe("у меня аллергия на пенициллин", emotion=0.9).node_id
    m.recall("аллергия")
    second = m.observe("врач выписал другой антибиотик", emotion=0.8).node_id

    assert m.graph.db.get_edges_between([first, second]) == []
    m.close()


def test_connectivity_stops_being_empty(memory):
    """
    Главное, ради чего всё затевалось: степень узлов перестаёт быть нулевой.
    Без этого связность как сигнал важности измеряет ровно ничто — что и
    показал замер на LongMemEval: 201 узел, у всех 201 степень ноль.
    """
    # Запрос совпадает со СЛОВАМИ записей дословно: тест не должен
    # зависеть от наличия семантической модели. Прежняя версия спрашивала
    # "кот" про записи со словом "кота" и проходила только там, где
    # доступен navec, — то есть на машине автора.
    # ТЕКСТЫ РАЗНЫЕ, и это не косметика. Прежде здесь стояло
    # f"событие номер {index} про кота и собаку" — с точки зрения организма
    # шесть ОДИНАКОВЫХ текстов, потому что цифры в словарь не попадают.
    # После смены формы гейта (нормированное произведение вместо среднего)
    # из шести записывались два: новизна падала до нуля, а эмоция больше не
    # перетаскивала через порог сама.
    #
    # Тест проверяет СВЯЗЫВАНИЕ, а не гейт, поэтому поток должен состоять
    # из разных событий, делящих общую тему.
    _EVENTS = [
        "кот уронил чашку а собака залаяла",
        "собака утащила носок а кот смотрел",
        "кот забрался на шкаф собака внизу",
        "собака гоняла кота по коридору",
        "кот и собака делили одну миску",
        "собака охраняла кота на прогулке",
    ]
    for text in _EVENTS:
        memory.observe(text, emotion=0.9)
        memory.recall("кот и собака")

    ids = _episodic_ids(memory)
    degrees = memory.graph.db.get_degrees(ids)
    connected = [node_id for node_id in ids if degrees.get(node_id, 0) > 0]

    assert connected, "граф эпизодов остался пустым"


def test_a_memory_is_never_linked_to_itself(memory):
    """
    Узел, только что найденный поиском, может оказаться и тем, который
    сейчас пишется, — петля в графе никому не нужна.
    """
    first = memory.observe("у меня аллергия на пенициллин", emotion=0.9).node_id
    memory.recall("аллергия на пенициллин")
    second = memory.observe("у меня аллергия на пенициллин совсем", emotion=0.9).node_id

    for node_id in {first, second} - {None}:
        edges = memory.graph.db.get_edges_between([node_id])
        assert all(e["node_from"] != e["node_to"] for e in edges)


# ---------------------------------------------------------------------------
# ВРЕМЕННАЯ СМЕЖНОСТЬ — второй, независимый источник связей
# ---------------------------------------------------------------------------
def test_temporal_linking_does_not_need_search_to_succeed():
    """
    Главное свойство механизма: он работает ТАМ, ГДЕ ПОИСК НЕ РАБОТАЕТ.

    Связывание по припоминанию опирается на поиск и потому глохнет при
    перегрузке ключа — измерено дозой: когда одну подсказку делят четыре
    записи вместо одной, связь завязывается в 3 случаях из 60 вместо 23.
    Временная смежность от поиска не зависит вовсе, и в свободном
    припоминании человек ведёт себя так же: вспомнив эпизод, чаще называет
    следующим соседа ПО ВРЕМЕНИ, а не по смыслу.

    Здесь две записи не делят НИ ОДНОГО значимого слова — поиск связать их
    не может в принципе.

    ПРОМЕЖУТОК МАЛЕНЬКИЙ, И ЭТО СУЩЕСТВЕННО. Раньше связь ставилась окном
    из двух последних записей — через любой промежуток, хоть через месяц.
    Теперь решает временной контекст: сказанное подряд ложится на общий
    фон, а после долгой паузы фон успевает смениться. Проверено:

        полминуты, пять минут, полчаса -> связь есть
        час, сутки                     -> связи нет

    При содержательно НЕСВЯЗАННЫХ фразах — то есть достоинство механизма
    (работать там, где поиск бессилен) сохранено, а прежняя слепота к
    паузе убрана.
    """
    memory = Memory(":memory:", settings=MemorySettings(
        associate_recalled_limit=0, temporal_link_window=2))
    first = memory.observe("mira breeds pedigree spaniels", emotion=1.0,
                           timestamp=0.0).node_id
    second = memory.observe("the boiler needs descaling again", emotion=1.0,
                            timestamp=300.0).node_id
    assert first and second

    cursor = memory.graph.db._conn.cursor()
    linked = cursor.execute(
        "SELECT COUNT(*) FROM edges WHERE (node_from=? AND node_to=?) "
        "OR (node_from=? AND node_to=?)",
        (first, second, second, first),
    ).fetchone()[0]
    assert linked == 1, "соседние по времени записи не связались"
    memory.close()


def test_temporal_window_bounds_how_far_back_it_reaches():
    """
    Окно — это окно, а не «всё подряд».

    Замер показал, что большие окна ХУЖЕ: 108/120 при окне 2 против 99/120
    при окне 5. Дальние соседи по времени дают уже случайные связи и
    размывают растекание, поэтому предел обязан соблюдаться.
    """
    memory = Memory(":memory:", settings=MemorySettings(
        associate_recalled_limit=0, temporal_link_window=1))
    ids = [memory.observe(text, emotion=1.0, timestamp=i * 3600.0).node_id
           for i, text in enumerate([
               "mira breeds pedigree spaniels",
               "the boiler needs descaling again",
               "tuesday parking permit expires",
           ])]
    cursor = memory.graph.db._conn.cursor()
    far = cursor.execute(
        "SELECT COUNT(*) FROM edges WHERE (node_from=? AND node_to=?) "
        "OR (node_from=? AND node_to=?)",
        (ids[0], ids[2], ids[2], ids[0]),
    ).fetchone()[0]
    assert far == 0, "связь ушла дальше окна"
    memory.close()


def test_temporal_linking_is_on_by_default():
    """
    Умолчание изменено ПОСЛЕ замера, и обратное переключение тоже должно
    его потребовать.

    120 многошаговых цепочек с уникальными подсказками: 10/120 при
    выключенном механизме против 108/120 при окне 2, k=3. Цена на
    LongMemEval — ноль, R@1 97.2% в обоих случаях.
    """
    assert MemorySettings().temporal_link_window == 2


@requires_model
def test_a_long_pause_breaks_the_temporal_link():
    """
    После долгой паузы соседство во времени перестаёт быть соседством.

    Прежнее окно связывало две последние записи независимо от того, прошла
    между ними минута или месяц, — то есть память не отличала продолжение
    разговора от нового. Теперь фон дрейфует по паузе: ослабленный
    предыдущий контекст перебивается новой репликой, и связь не ставится.
    """
    memory = Memory(":memory:", settings=MemorySettings(
        associate_recalled_limit=0, temporal_link_window=2))
    memory.observe("mira breeds pedigree spaniels", emotion=1.0, timestamp=0.0)
    memory.observe("the boiler needs descaling again", emotion=1.0,
                   timestamp=86400.0)

    linked = memory.graph.db._conn.execute(
        "SELECT COUNT(*) FROM edges WHERE edge_type = 'temporal'"
    ).fetchone()[0]
    assert linked == 0, "через сутки это уже не соседство во времени"
    memory.close()
