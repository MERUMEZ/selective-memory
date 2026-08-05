# Copyright (C) 2026 MERUMEZ <selectivemem@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License version 3 as
# published by the Free Software Foundation. See LICENSE for the full text.
"""
Ответ на заявленную нужду записывается, даже если не удивляет.

ПОЧЕМУ ЭТО ГОВОРИТ ПРИЛОЖЕНИЕ, А НЕ РЕШАЕТ ПАМЯТЬ. Внутрь библиотеки был
собран механизм, замечающий брешь самостоятельно: поиск вернул мало —
следующее событие получает надбавку. Он не сработал НИ РАЗУ на разговоре
из тридцати шести реплик.

Разбор показал принципиальную причину: на запрос «как зовут пса» поиск
вернул 0.776 за «у меня есть пёс». Память НАШЛА — уверенно и не то. Она
меряет похожесть, а не удовлетворённость нужды, и порогом уверенности
одно от другого не отделить.

Значит сигнал доступен только тому, кто спрашивал.
"""

from selectivemem import Memory
from selectivemem.settings import MemorySettings


def _ask_about_the_dog(memory):
    """Разговор из живого запуска, где кличка терялась."""
    for text in ("у меня есть пёс", "почему он такой крикливый"):
        memory.recall(text, top_k=3)
        memory.observe(text, response="а как его зовут?")
    memory.recall("как зовут пса", top_k=3)


def test_short_answer_is_lost_without_the_signal():
    """
    Без сигнала кличка теряется — и это НЕ ошибка гейта.

    «Леви» — одно знакомое слово, новизна 0.33, порог не взят. По своему
    критерию память права: удивления нет. Проверка закрепляет, что цена
    отказа от сигнала именно такова.
    """
    memory = Memory(":memory:")
    _ask_about_the_dog(memory)
    result = memory.observe("Леви", response="приятно познакомиться")
    assert result.node_id is None, "ожидалось, что без сигнала кличка потеряется"
    memory.close()


def test_short_answer_is_kept_when_the_caller_declares_it():
    memory = Memory(":memory:")
    _ask_about_the_dog(memory)
    result = memory.observe("Леви", response="приятно познакомиться",
                            fills_gap=True)
    assert result.node_id is not None
    found = [m.context for m in memory.recall("кличка пса", top_k=5)]
    assert any("Леви" in text for text in found), found
    memory.close()


def test_explicit_emotion_still_wins():
    """Приложение, знающее больше, не должно быть перебито этим входом."""
    memory = Memory(":memory:")
    _ask_about_the_dog(memory)
    result = memory.observe("Леви", response="ок", emotion=0.0, fills_gap=True)
    assert result.node_id is None, "явный ноль обязан остаться нулём"
    memory.close()


def test_significance_is_a_stated_setting():
    """
    Единица, а не меньше, и это арифметика.

    Плотность равна новизна * (1 + значимость) / 2 и при новизне 0.33 не
    превысит 0.33 ни при какой значимости ниже единицы, тогда как порог
    под напряжением поднимается к 0.40. Проверено: с 0.8 кличка терялась.
    """
    assert MemorySettings().gap_fill_significance == 1.0
