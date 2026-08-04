# Copyright (C) 2026 MERUMEZ <selectivemem@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License version 3 as
# published by the Free Software Foundation. See LICENSE for the full text.
"""
Библиотека обязана работать в любой из заявленных конфигураций.

Проверяется то, что однажды тихо сломалось: порог отсечения был калиброван
только под режим с семантикой, и БЕЗ кодировщика поиск переставал отдавать
найденное. На полном наборе LongMemEval это давало R@1 9.2% против 88.2%
после починки — библиотека выглядела сломанной, хотя искать умела.

Такое не ловится обычными проверками: они все идут с моделью.
"""

import pytest

from selectivemem import Memory
from selectivemem.settings import MemorySettings

FACTS = [
    "the quarterly report is due on friday",
    "my daughter liza turns seven in march",
    "we moved the standup to eleven",
    "the office printer jams on thick paper",
    "sofia handles the vendor contracts",
]


def store(memory):
    clock = 0.0
    for fact in FACTS:
        memory.observe(fact, emotion=1.0, timestamp=clock)
        clock += 3600.0
    return clock


def test_search_returns_something_without_any_encoder():
    """
    Голая установка без семантики обязана НАХОДИТЬ.

    Здесь ломалось молча: счёт без смыслового слагаемого живёт на другой
    шкале, а порог остался прежним — выдача становилась пустой, и ни одна
    проверка этого не видела.
    """
    memory = Memory(":memory:", encoder=lambda text: None)
    clock = store(memory)
    found = memory.recall("when is the report due", top_k=3, timestamp=clock)
    assert found, "без кодировщика поиск не вернул НИЧЕГО"
    assert any("report" in (m.context or "") for m in found)
    memory.close()


def test_lexical_threshold_is_lower_than_the_semantic_one():
    """
    Два порога, и это не дублирование.

    Один калиброван по шкале «слова + близость векторов», другой — по
    шкале «только слова». Замер: при общем пороге 0.20 бессемантический
    режим давал R@1 0.0%, при 0.06 — 91.2%.
    """
    settings = MemorySettings()
    assert settings.memory_search_threshold_lexical < settings.memory_search_threshold


def test_setup_reports_itself():
    """
    Семантика деградирует ТИХО, поэтому конфигурация обязана называться.

    Разница между режимами — до десяти раз по R@1, а узнать, в каком ты
    режиме, можно было только по логам при подходящем уровне.
    """
    memory = Memory(":memory:", encoder=lambda text: None)
    line = memory.describe_setup()
    assert "смысл:" in line and "значимость:" in line
    memory.close()


@pytest.mark.parametrize("query", [
    "who deals with vendors",
    "what time is standup",
    "how old is my daughter",
])
def test_bare_install_answers_plain_questions(query):
    """
    Тот же набор вопросов на голой установке: своё восприятие вместо
    модели. Проверяется не точность, а то, что путь вообще живой.
    """
    settings = MemorySettings()
    settings.grow_perception = True
    memory = Memory(":memory:", settings=settings)
    clock = store(memory)
    found = memory.recall(query, top_k=5, timestamp=clock)
    assert found, f"голая установка не ответила ничем на {query!r}"
    memory.close()
