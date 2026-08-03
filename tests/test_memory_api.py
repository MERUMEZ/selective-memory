"""
================================================================================
 TEST_MEMORY_API.PY — Публичный фасад: то, что увидит чужой человек
================================================================================
Остальные тесты проверяют внутренности. Здесь проверяется контракт,
который мы обещаем наружу, — и он должен держаться, даже когда
внутренности перестраиваются.

Отдельно проверяется, что пакет НЕ требует ни config, ни навешенной
модели эмбеддингов: обещание "ставится куда угодно и деградирует мягко"
это часть контракта, а не пожелание.
================================================================================
"""

import pytest

from selectivemem import Memory
from selectivemem import embeddings
from selectivemem.settings import MemorySettings

# Проверка опирается на СЕМАНТИКУ: запрос сформулирован другими словами,
# чем сохранённый текст. Без модели поиск честно отвечает пустотой и сам
# об этом предупреждает в логе — это заявленная деградация, а не поломка.
# Без этой отметки чистая установка без extras давала девять красных
# тестов и создавала впечатление сломанного пакета.
requires_model = pytest.mark.skipif(
    not embeddings.is_available(),
    reason="запрос сформулирован иначе, чем запись: нужна семантическая модель",
)


@pytest.fixture
def memory():
    m = Memory(":memory:")
    yield m
    m.close()


# ---------------------------------------------------------------------------
# observe: решение о записи
# ---------------------------------------------------------------------------

def test_first_encounter_is_written(memory):
    """Новорождённой памяти ново всё — первое событие обязано записаться."""
    obs = memory.observe("меня зовут Паша", "приятно познакомиться", emotion=0.4)
    assert obs.written
    assert obs.surprise == pytest.approx(1.0)
    assert "spike" in obs.reason


def test_repetition_stops_being_surprising(memory):
    """
    Главное обещание: организм перестаёт удивляться привычному. Если
    удивление не падает, порог записи ничем не управляет и вся экономия
    памяти — фикция.
    """
    first = memory.observe("сегодня хорошая погода", emotion=0.1)
    for _ in range(3):
        memory.observe("сегодня хорошая погода", emotion=0.1)
    last = memory.observe("сегодня хорошая погода", emotion=0.1)

    assert last.surprise < first.surprise
    assert not last.written, "рутина не должна попадать в память"
    assert "short of the threshold" in last.reason


def test_emotion_can_push_routine_over_the_threshold(memory):
    """Два независимых повода запомнить: неожиданность И заряд."""
    for _ in range(4):
        memory.observe("обычная фраза про погоду", emotion=0.0)

    calm = memory.observe("обычная фраза про погоду", emotion=0.0)
    charged = memory.observe("обычная фраза про погоду", emotion=1.0)

    assert not calm.written
    assert charged.written


def test_load_raises_the_threshold(memory):
    """Под перегрузкой организм хуже усваивает новое — это самосохранение."""
    calm = memory.gate.effective_threshold(load=0.0)
    stressed = memory.gate.effective_threshold(load=1.0)
    assert stressed > calm


# ---------------------------------------------------------------------------
# recall / context_for
# ---------------------------------------------------------------------------

@requires_model
def test_recall_finds_what_was_stored(memory):
    memory.observe("у меня есть кот Мурзик", "какой он?", emotion=0.6)
    found = memory.recall("расскажи про кота")
    assert found
    assert "Мурзик" in found[0].context


def test_context_for_is_empty_when_nothing_relevant(memory):
    """
    Пустая строка — законный ответ. Память, которая всегда что-то
    подмешивает в промпт, подмешивает шум.
    """
    memory.observe("у меня есть кот", emotion=0.6)
    assert memory.context_for("квантовая хромодинамика") == ""


# ---------------------------------------------------------------------------
# Время и забывание
# ---------------------------------------------------------------------------

def test_time_comes_from_outside():
    """
    Часы подменяемы: без этого нельзя ни ускорить демонстрацию, ни
    воспроизвести замер. Библиотека, которая сама зовёт time.time,
    навязывает свою шкалу времени всему приложению.
    """
    now = [1000.0]
    m = Memory(":memory:", clock=lambda: now[0])

    m.observe("важное событие", emotion=0.9)
    now[0] += 30 * 86400.0          # месяц молчания
    forgotten = m.forget()

    assert forgotten > 0, "за месяц должно было угаснуть хоть что-то"
    m.close()


def test_recall_resists_forgetting():
    """
    Вспоминание — не бесплатное чтение: оно повышает стабильность узла.
    Это эффект интервального повторения, ради него всё и затевалось.
    """
    now = [1000.0]
    m = Memory(":memory:", clock=lambda: now[0])

    used = m.observe("телефон восемь девятьсот", emotion=0.9)
    unused = m.observe("случайная реплика про дождь", emotion=0.9)
    assert used.written and unused.written

    for _ in range(5):
        now[0] += 3600.0
        m.recall("телефон")

    stability_used = m.graph.db.get_node(used.node_id)["stability"]
    stability_unused = m.graph.db.get_node(unused.node_id)["stability"]
    assert stability_used > stability_unused
    m.close()


# ---------------------------------------------------------------------------
# Подкрепление
# ---------------------------------------------------------------------------

def test_feedback_applies_to_the_last_action(memory):
    """Оценка приходит следующей репликой и относится к предыдущей."""
    obs = memory.observe("столица Франции Париж", "верно", emotion=0.5)
    before = memory.graph.db.get_node(obs.node_id)["weight"]

    memory.feedback(+1.0)

    after = memory.graph.db.get_node(obs.node_id)["weight"]
    assert after > before


def test_feedback_needs_no_language(memory):
    """
    Ядро не знает ни одного слова оценки: valence это число. Откуда оно
    (кнопка, эмодзи, классификатор, разбор реплики) — дело приложения.
    Иначе пакет годился бы только для русского языка.
    """
    memory.observe("что-то произошло", emotion=0.5)
    outcome = memory.feedback(-0.8)
    assert outcome is not None


# ---------------------------------------------------------------------------
# Настройки и состояние
# ---------------------------------------------------------------------------

def test_settings_change_behaviour():
    """Параметры реально доезжают до поведения, а не лежат мёртвым грузом."""
    strict = Memory(":memory:", settings=MemorySettings(base_plasticity_threshold=0.95))
    obs = strict.observe("новое событие", emotion=0.3)
    assert not obs.written, "при пороге 0.95 записываться почти ничего не должно"
    strict.close()


def test_stats_reports_state(memory):
    memory.observe("первое", emotion=0.9)
    memory.observe("второе", emotion=0.9)
    stats = memory.stats()
    assert stats.episodes == 2
    assert stats.nodes >= 2


def test_context_manager_closes():
    with Memory(":memory:") as m:
        m.observe("проверка", emotion=0.9)
    # Закрытие не должно бросать; повторное — тоже
    m.close()
