# Copyright (C) 2026 MERUMEZ <selectivemem@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License version 3 as
# published by the Free Software Foundation. See LICENSE for the full text.
"""
Профиль: что память знает О ЧЕЛОВЕКЕ, независимо от запроса.

ЗАЧЕМ ОТДЕЛЬНО ОТ recall. Живой разговор показал дыру: на «может ещё
что-то?» ассистент ответил, что больше ничего не знает — при двадцати
записях в памяти. Запрос не совпал ни с чем, и память промолчала, хотя
знала. Ассистенту нужно держать при себе то, что о человеке известно
ВСЕГДА, а не только когда спросили впрямую.
"""

from selectivemem import Memory
from selectivemem.settings import MemorySettings


def test_profile_keeps_facts_about_the_user():
    memory = Memory(":memory:")
    for text in ("я люблю кофе", "у меня аллергия на пенициллин",
                 "космонавтов звали Армстронг и Олдрин",
                 "погода сегодня нормальная"):
        memory.observe(text, response="ок")

    profile = [m.context for m in memory.profile()]
    assert any("кофе" in t for t in profile)
    assert any("пенициллин" in t for t in profile)
    assert not any("космонавт" in t for t in profile), profile
    assert not any("погода" in t for t in profile), profile
    memory.close()


def test_profile_ignores_questions():
    """
    Вопрос содержит «мне» и проходил отбор наравне с фактом: профиль
    засорялся тем, что человек СПРАШИВАЛ. Поймано на живом запуске —
    «что мне заказать на ужин» стояло в профиле рядом с аллергией.
    """
    memory = Memory(":memory:")
    memory.observe("я не ем мясо", response="понял")
    memory.observe("что мне заказать на ужин?", response="попробуйте салат")

    profile = [m.context for m in memory.profile()]
    assert any("мясо" in t for t in profile)
    assert not any("заказать" in t for t in profile), profile
    memory.close()


def test_profile_is_empty_on_a_fresh_memory():
    """Пустой профиль — законный ответ, и вызывающий обязан его пережить."""
    memory = Memory(":memory:")
    assert memory.profile() == []
    assert memory.profile_text() == ""
    memory.close()


def test_profile_text_is_ready_for_a_prompt():
    memory = Memory(":memory:")
    memory.observe("я работаю бэкенд-разработчиком", response="ясно")
    text = memory.profile_text()
    assert text.startswith("- ")
    assert "бэкенд" in text
    memory.close()


def test_derived_facts_come_before_guesses():
    """
    То, что приложение ВЫВЕЛО, стоит дороже того, что память угадала.

    Замер цены догадки: отбор профиля по речи о себе находит предпочтение
    в 3 случаях из 30 против 15 у обычного поиска — потому что в разговоре
    с ассистентом о себе говорит КАЖДАЯ реплика, и признак не различает
    ничего.
    """
    memory = Memory(":memory:")
    memory.observe("я хочу приготовить завтрак", response="ок")
    memory.observe("у меня аллергия на пенициллин", response="понял")
    memory.remember_about_user("не ест мясо")

    profile = [m.context for m in memory.profile(limit=5)]
    assert profile[0] == "не ест мясо", profile
    memory.close()


def test_derived_facts_survive_capacity_pressure():
    """
    Персонализация не должна вытесняться наравне с трёпом.

    Выведенные факты ложатся в КОРУ, а ёмкость ограничивает только
    гиппокамп. Аллергия переживёт тесноту, разговор о погоде — нет.
    """
    settings = MemorySettings()
    settings.memory_capacity = 3
    memory = Memory(":memory:", settings=settings)
    memory.remember_about_user("не ест мясо")

    clock = 0.0
    for index in range(20):
        memory.observe(f"посторонняя реплика номер {index} про разное",
                       emotion=1.0, timestamp=clock)
        clock += 3600.0
    memory.forget(now=clock)

    profile = [m.context for m in memory.profile(limit=5)]
    assert any("мясо" in text for text in profile), profile
    memory.close()


def test_empty_fact_is_ignored():
    """Пустая строка — не факт, и записывать её нечего."""
    memory = Memory(":memory:")
    memory.remember_about_user("   ")
    assert memory.profile() == []
    memory.close()
