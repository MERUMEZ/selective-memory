"""
================================================================================
 TEST_ENCODER.PY — Язык задаётся кодировщиком, а не библиотекой
================================================================================
Встроенная семантика — navec, русские статические векторы. Это выбор под
железо (51 МБ, без torch), а не позиция: библиотека, работающая только с
русским, не годится ни для геймдева, ни для B2B за пределами рунета.

Поэтому кодировщик подключаемый: любая функция text -> вектор или None.
Здесь проверяется, что подключается действительно ЛЮБАЯ (в тестах —
игрушечный мешок слов, чтобы не тащить модель), что она реально влияет на
поиск, и что смена кодировщика на существующей базе не портит выдачу
молча.

Последнее — главное. BLOB читается как массив float32 без разметки,
поэтому вектор чужой модели не вызовет ошибку: он даст бессмысленную
близость. Такое ловится только замером, и лучше, чтобы не ловилось вовсе.
================================================================================
"""

import pytest

from selectivemem import Memory
from selectivemem.database import Database
from selectivemem.graph_memory import MemoryGraph

EN_VOCAB = "cat dog pet animal house home city sea food music friend work book".split()
RU_VOCAB = "кот собака дом город море еда музыка друг работа книга".split()


def _bag_encoder(vocab):
    """Игрушечный кодировщик: мешок слов по словарю. Размерность = len(vocab)."""
    def encode(text):
        words = {w.strip(".,!?").lower() for w in (text or "").split()}
        vector = [1.0 if w in words else 0.0 for w in vocab]
        return vector if any(vector) else None
    return encode


def test_english_works_through_a_custom_encoder():
    """Английский — обычный случай, а не исключение."""
    memory = Memory(":memory:", encoder=_bag_encoder(EN_VOCAB))
    memory.observe("I have a cat at home", "tell me about it", emotion=0.6)
    memory.observe("my favourite music is jazz", "nice", emotion=0.6)

    found = memory.recall("tell me about the pet cat", top_k=1)
    assert found and "cat" in found[0].context
    memory.close()


def test_encoder_is_actually_used():
    """
    Кодировщик должен реально зваться, а не лежать декорацией: иначе
    пользователь думает, что настроил язык, а поиск работает по-старому.
    """
    calls = []

    def counting_encoder(text):
        calls.append(text)
        return [1.0, 0.0]

    memory = Memory(":memory:", encoder=counting_encoder)
    memory.observe("любой текст", emotion=0.6)
    memory.recall("любой текст")

    assert calls, "кодировщик ни разу не вызвался"
    memory.close()


def test_encoder_returning_none_degrades_softly():
    """
    None — законный ответ кодировщика ("для этого текста вектора нет").
    Поиск обязан продолжить работать на строковом сходстве.
    """
    memory = Memory(":memory:", encoder=lambda text: None)
    memory.observe("у меня есть кот", "какой он?", emotion=0.6)

    found = memory.recall("у меня есть кот", top_k=1)
    assert found, "без семантики поиск обязан работать строкой"
    memory.close()


def test_switching_encoder_does_not_poison_search():
    """
    ГЛАВНОЕ. База, набитая векторами одной модели, однажды откроется с
    другой. Старые BLOB'ы читаются без ошибки, поэтому расхождение
    размерности проявилось бы не падением, а тихо неверной выдачей.

    Проверяется, что после смены кодировщика поиск находит то же, что и
    на чистой базе с этим кодировщиком.
    """
    path = ":memory:"

    # Первая жизнь: короткие векторы
    short = MemoryGraph(db=Database(db_path=path), settings=None,
                        encoder=_bag_encoder(RU_VOCAB[:4]))
    for text in ("у меня есть кот", "я люблю музыку", "работа в городе"):
        short.save_connection(context=text, response="ага", weight=0.6, timestamp=1.0)
    # Запрос словами ИЗ словаря: игрушечный кодировщик сравнивает точно,
    # и "кота" для него не то же самое, что "кот".
    short.search("кот", top_k=1, timestamp=2.0, with_associations=False)

    # Вторая жизнь: та же база, кодировщик другой размерности
    long = MemoryGraph(db=short.db, settings=None, encoder=_bag_encoder(RU_VOCAB))
    found = long.search("кот", top_k=1, timestamp=3.0, with_associations=False)

    assert found, "после смены кодировщика поиск не должен слепнуть"
    assert "кот" in found[0].context, f"нашлось не то: {found[0].context!r}"
    short.close()


def test_stale_vectors_are_recomputed_not_reused():
    """
    Пересчёт должен быть фактическим: старый BLOB другой длины обязан
    замениться, а не игнорироваться при каждом поиске заново.
    """
    graph = MemoryGraph(db=Database(db_path=":memory:"), settings=None,
                        encoder=_bag_encoder(RU_VOCAB[:4]))
    node_id = graph.save_connection(context="у меня есть кот", response="ага",
                                    weight=0.6, timestamp=1.0)
    graph.search("кот", top_k=1, timestamp=2.0, with_associations=False)
    before = graph.db.get_node(node_id)["embedding"]

    graph.encoder = _bag_encoder(RU_VOCAB)
    graph._vector_dim = None
    graph.search("кот", top_k=1, timestamp=3.0, with_associations=False)
    after = graph.db.get_node(node_id)["embedding"]

    assert before is not None and after is not None
    assert len(after) > len(before), "вектор должен был пересчитаться под новую модель"
    graph.close()


def test_module_level_patching_still_works(monkeypatch):
    """
    Кодировщик по умолчанию резолвится НА ВЫЗОВЕ, а не связывается в
    конструкторе. Иначе подмена selectivemem.embeddings.encode снаружи
    (тесты, стенды, отключение семантики на ходу) молча перестаёт
    действовать — на это уже наступали с заглушкой LLM.
    """
    from selectivemem import embeddings

    graph = MemoryGraph(db=Database(db_path=":memory:"))
    monkeypatch.setattr(embeddings, "encode", lambda text: None)

    assert graph._encode("любой текст") is None
    graph.close()


def test_missing_semantics_is_announced_once():
    """
    Поиск без кодировщика находит только то, что делит с запросом слова.
    Молчать об этом нельзя: человек получает пустоту без намёка на
    причину — так был потерян день на внешнем бенчмарке, пока не стало
    понятно, что поиск не сломан, а слеп.

    Ровно один раз за жизнь графа: предупреждение на каждый запрос
    превратилось бы в шум, который перестают читать.
    """
    import logging

    memory = Memory(":memory:", encoder=lambda text: None)
    memory.observe("у меня аллергия на пенициллин", emotion=0.9)

    logger = logging.getLogger("selectivemem.graph_memory")
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    logger.addHandler(handler)
    try:
        memory.recall("что мне нельзя принимать")
        memory.recall("ещё один запрос")
        memory.recall("и ещё")
    finally:
        logger.removeHandler(handler)

    warnings = [r for r in records if r.levelno >= logging.WARNING
                and "No semantics available" in r.getMessage()]
    assert len(warnings) == 1, f"ожидалось одно предупреждение, вышло {len(warnings)}"
    memory.close()


def test_stats_report_whether_semantics_works():
    """
    Признак должен быть доступен программе, а не только в логах: интегратор
    вправе проверить его при старте и отказаться работать вслепую.
    """
    blind = Memory(":memory:", encoder=lambda text: None)
    assert blind.stats().semantic is False
    blind.close()

    seeing = Memory(":memory:", encoder=lambda text: [1.0, 0.0])
    assert seeing.stats().semantic is True
    seeing.close()
