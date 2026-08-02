"""
================================================================================
 TEST_README_EXAMPLES.PY — README не должен врать
================================================================================
Первое, что делает человек с библиотекой, — копирует пример из README.
Если пример не работает, он уходит и не возвращается, и никакие
измеренные +50 п.п. его уже не вернут.

Проверка не абстрактная: при написании README я написал в быстром старте
запрос "что мне нельзя принимать" — и он возвращал ПУСТУЮ строку.
Встроенная модель обучена на художественной литературе и связи
"нельзя принимать" -> "аллергия" не знает. Пример пришлось заменить на
работающий, а не подогнать текст под желаемое.

Здесь дословно выполняются все примеры README. Меняете README — этот
тест обязан упасть.
================================================================================
"""

from selectivemem import Memory, MemorySettings


def test_quickstart_block_russian():
    """Первый блок README.ru: наблюдение, оценка, рутина, контекст."""
    memory = Memory(":memory:")

    important = memory.observe("у меня аллергия на пенициллин")
    memory.feedback(+1.0)
    routine = memory.observe("спасибо")

    assert important.written, "факт обязан записаться"
    assert not routine.written, "междометие записываться не должно"
    # Запрос словом ИЗ записи. README честно предупреждает, что базовая
    # установка ищет по словам: без кодировщика "какие у меня аллергии"
    # не находит ничего, и первая версия README это обещала.
    assert memory.context_for("аллергия") == "- у меня аллергия на пенициллин"
    memory.close()


def test_quickstart_block_english():
    """
    Тот же блок из английского README, и он проверяется отдельно не для
    симметрии.

    Встроенная модель РУССКАЯ, поэтому на английском семантики нет вовсе
    и работает только совпадение слов. Первая версия английского примера
    спрашивала "allergy" про запись "I am allergic to penicillin" — и
    возвращала пустоту, потому что "allergy" и "allergic" для строкового
    сходства разные слова. Пример исправлен, а README получил
    предупреждение в самое начало.
    """
    memory = Memory(":memory:")

    important = memory.observe("I am allergic to penicillin")
    memory.feedback(+1.0)
    routine = memory.observe("thanks")

    assert important.written
    assert not routine.written
    assert "short of the threshold" in routine.reason
    assert memory.context_for("what am I allergic to") == (
        "- I am allergic to penicillin"
    )
    memory.close()


def test_observation_reports_a_reason():
    """README обещает, что на «почему не запомнил» есть ответ числом."""
    memory = Memory(":memory:")
    memory.observe("сегодня совершенно обычная погода за окном")
    obs = memory.observe("сегодня совершенно обычная погода за окном")

    assert not obs.written
    assert "short of the threshold" in obs.reason
    assert 0.0 <= obs.surprise <= 1.0
    memory.close()


def test_clock_block():
    """Блок про подменяемые часы."""
    now = [1_700_000_000.0]
    memory = Memory(":memory:", clock=lambda: now[0])

    memory.observe("важное событие", emotion=0.9)
    now[0] += 30 * 86400
    assert memory.forget() > 0
    memory.close()


def test_settings_block():
    """Блок про настройки датаклассом."""
    memory = Memory(":memory:", settings=MemorySettings(decay_rate=0.02, age_t0=3600))
    assert memory.settings.decay_rate == 0.02
    assert memory.settings.age_t0 == 3600
    memory.close()


def test_five_actions_all_exist():
    """
    README перечисляет ровно пять действий плюс stats. Если что-то из
    этого переименуют, текст обязан упасть вместе с кодом.
    """
    memory = Memory(":memory:")
    for name in ("observe", "feedback", "recall", "context_for", "forget", "stats"):
        assert callable(getattr(memory, name)), f"README обещает {name}()"
    memory.close()


def test_encoder_block_signature():
    """
    README показывает Memory(..., encoder=lambda text: model.encode(text)).
    Проверяется, что такой вызов вообще принимается.
    """
    memory = Memory(":memory:", encoder=lambda text: [1.0, 0.0, 0.0])
    memory.observe("проверка кодировщика", emotion=0.9)
    assert memory.stats().nodes > 0
    memory.close()
