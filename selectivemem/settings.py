# Copyright (C) 2026 MERUMEZ <selectivemem@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License version 3 as
# published by the Free Software Foundation. See LICENSE for the full text.
#
# It is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.
#
# A commercial licence is available for use in closed products — see
# COMMERCIAL.md.
"""
================================================================================
 SETTINGS.PY — Settings of the memory core
================================================================================
The third step of extracting the package: the core NO LONGER READS the
application's global config.

memory/* used to import config.py — a thousand-line module holding a
Telegram token, prompts and persona settings. Unacceptable for a library
installed via pip: it must carry its own configuration rather than depend
on one particular application's file.

The defaults are the values calibrated by measurement (see AUDIT.md). An
application may pass its own:

    graph = MemoryGraph(db=db, settings=MemorySettings(decay_rate=0.02))

or build them from its config wholesale:

    graph = MemoryGraph(db=db, settings=MemorySettings.from_module(config))

This file was GENERATED from config.py so the values are guaranteed to
match: retyping seventy constants by hand is a reliable way to change
something without noticing. test_settings.py checks them on every run.
================================================================================
"""

from dataclasses import dataclass, fields
from typing import Any


@dataclass
class MemorySettings:
    """Memory core parameters. Names mirror config.py, lower-cased."""

    age_t0: float = 25200.0
    babbling_syllable_pool_size: int = 30
    base_plasticity_threshold: float = 0.25
    # Насколько эмоция УСИЛИВАЕТ новизну в решении о записи.
    #
    # Прежняя формула — среднее: (эмоция + удивление) / 2. Такой арифметики
    # в биологии нет: норадреналин умножает пластичность, которую новизна
    # уже открыла, а не складывается с ней. Из-за среднего событие
    # страшное-но-привычное и безразличное-но-небывалое оценивались
    # одинаково.
    #
    # Форма выбрана так, чтобы новизна оставалась ОСНОВАНИЕМ:
    #     плотность = удивление * (1 + эмоция * этот множитель)
    # Чистое произведение обнулило бы гейт у всех, кто не передаёт эмоцию,
    # а по умолчанию она 0.0 — то есть у обычного библиотечного
    # пользователя память перестала бы писать вообще.
    #
    # Ноль возвращает прежнее среднее.
    gate_emotion_gain: float = 0.0
    concept_max_similar_links: int = 2
    concept_node_weight: float = 0.7
    concept_similarity_edge_weight: float = 0.25
    concept_similarity_link_threshold: float = 0.35
    concept_user_edge_weight: float = 0.4
    contradiction_correction_relief: float = 0.15
    contradiction_repeat_threshold: float = 0.85
    contradiction_stability_factor: float = 0.25
    contradiction_topic_threshold: float = 0.8
    # Ненулевое значение включает НОВЫЙ путь поиска устаревшего: кандидаты
    # берутся из обычного поиска, а не перебором всех узлов с высоким
    # порогом косинуса. Смысл замены — в графе, а не в числе.
    #
    # Прежний путь сравнивал предложения целиком и требовал 0.8. Но
    # противоречие — это "та же тема, ДРУГОЕ значение", поэтому чем сильнее
    # изменение, тем меньше сходство предложений и тем вернее обновление
    # проскочит. Доказательство противоречия вычиталось из доказательства
    # связанности.
    #
    # Замер против порога 0.8:
    #   "мою собаку зовут Рекс" -> "... Бобик"            0.923  ловит
    #   "моя собака зовут Рекс" -> "мою собаку ТЕПЕРЬ..." 0.722  мимо
    #   "я живу в Москве" -> "я ПЕРЕЕХАЛ в Питер"         0.369  мимо
    # Первая пара — пример из докстринга самого механизма.
    #
    # Через поиск: пять обновлений из шести отделяются с запасом (худшее
    # настоящее 0.642 против лучшего постороннего 0.433). Шестое —
    # "Москва/Питер" — не отделяется ничем, там нет общих слов вовсе.
    #
    # Ноль возвращает прежний путь.
    # ОСТАЁТСЯ ВЫКЛЮЧЕННЫМ, и это отмена собственной рекомендации.
    #
    # Путь через поиск ловит 5 обновлений из 6 против 3 у прежнего, и
    # первая проверка не нашла ложных срабатываний. Проверка была
    # НЕПОЛНОЙ: мерились истинные обновления и далёкий шум, но не близкие
    # СОВМЕСТИМЫЕ факты — а ломается он именно там.
    #
    #   ОБНОВЛЕНИЕ  0.818  "мою собаку зовут Бобик"
    #   ОБНОВЛЕНИЕ  0.547  "мою собаку теперь зовут Бобик"
    #   СОВМЕСТИМО  0.710  "я люблю кофе"      (после "я люблю чай")
    #   СОВМЕСТИМО  0.669  "у меня есть собака" (после "у меня есть кошка")
    #   СОВМЕСТИМО  0.603  "у меня есть сестра" (после "у меня есть брат")
    #
    # Порога, разделяющего эти группы, НЕ СУЩЕСТВУЕТ: обновление на 0.547
    # ниже совместимого на 0.603. Пересечение слов тоже не разделяет —
    # обе группы в диапазоне 0.5-0.75.
    #
    # Различие семантическое: "мою собаку зовут Бобик" ЗАМЕНЯЕТ, "у меня
    # есть собака" ДОБАВЛЯЕТ. Зависит от того, однозначно ли поле, а этого
    # знания у библиотеки нет.
    #
    # Цена ошибок несимметрична — пропустить противоречие дешевле, чем
    # ослабить верный факт, — поэтому по умолчанию оставлен прежний,
    # осторожный путь. Приложение, готовое платить порчей за полноту,
    # включает 0.5 осознанно.
    contradiction_search_threshold: float = 0.0
    # Сколько кандидатов брать из поиска при проверке на устаревание.
    contradiction_candidates: int = 5
    # Минимальная доля общих значимых слов, без которой вытеснение не
    # срабатывает. ЗАЩИТА ОТ ЧУЖОГО КОДИРОВЩИКА, и она измерена.
    #
    # Замер: русская модель (та, что в пакете) на английском тексте даёт
    # "my dog is called Rex" против "the price of bread went up" косинус
    # 0.808 при пороге вытеснения 0.8. Порог сидит внутри шума, и память
    # начинает ослаблять факт про собаку, потому что подорожал хлеб. На
    # тексте LongMemEval это давало 3080 срабатываний на 79 записей —
    # сорок ослаблений на каждую запись, молча.
    #
    # Проверки "хотя бы одно общее слово" НЕ ХВАТИЛО: _extract_keywords
    # отсекает РУССКИЕ стоп-слова, поэтому на английском "the", "is",
    # "my" остаются значимыми и пересечение всегда больше нуля.
    #
    # Откуда 0.25. Посторонние английские пары под русской моделью:
    # медиана 0.080, выше 0.25 только 14%. Настоящие обновления по-русски:
    # 0.25, 0.33, 0.50, 0.75, 0.75 — то есть порог сохраняет пять из шести
    # (шестое, "Москва/Питер", не берётся ничем и записано отдельным
    # тестом как известный предел).
    contradiction_min_overlap: float = 0.25
    # Сколько кандидатов брать из полнотекстового индекса при проверке на
    # устаревание. Раньше проверка сканировала ВСЮ базу и считала косинус
    # для каждого узла — при каждой записи.
    supersede_scan_limit: int = 200
    # Включает КОНСОЛИДАЦИЮ в фасаде Memory: накапливать кратковременный
    # буфер и по его заполнении сворачивать эпизод.
    #
    # Механизм был написан в пакете и не использовался им — consolidate_from_stm
    # звала только витрина (core/brain_session.py). Библиотечный пользователь
    # не получал ни абстрактных узлов, ни свёртки эпизодов, хотя они
    # описаны как часть памяти. Найдено tools/check_liveness.py, который
    # считает срабатывания механизмов: у консолидации был ноль.
    #
    # Тот же разряд расхождения витрины с пакетом, что и с ассоциациями.
    consolidate_from_stm: bool = False
    contradiction_weight_penalty: float = 0.25
    decay_rate: float = 0.05
    # Fallback text for the "user model" meta-node. This is PERSONA
    # content, not a memory parameter — it only exists because
    # get_user_model_content needs something to return before the
    # application has written its own. Anything meaningful here comes from
    # the application; the default is deliberately neutral.
    default_user_model: str = "The person I am talking to. I know nothing about them yet."
    edge_activation_decay: float = 0.6
    edge_activation_threshold: float = 0.3
    edge_boost_step: float = 0.15
    edge_decay_rate: float = 0.08
    edge_forget_threshold: float = 0.03
    edge_initial_weight: float = 0.2
    edge_max_hop_nodes: int = 3
    embeddings_enabled: bool = True
    embedding_model_path: str = '/var/www/mindnumbness/storage/models/navec_hudlit_v1_12B_500K_300d_100q.tar'
    forget_threshold: float = 0.05
    lexical_acquisition_enabled: bool = True
    lexical_age_t0: float = 2592000.0
    lexical_max_tokens_per_input: int = 20
    lexical_min_token_length: int = 2
    memory_fuzzy_weight: float = 0.1
    # Ceiling of the decay floor: the weight a node with fully earned
    # approval (reward_expectation = 1.0) settles at instead of vanishing.
    # Zero disables the floor and restores the previous behaviour.
    memory_floor_max: float = 0.25
    # Пол для узла, которого НИКОГДА не подкрепляли и не вспоминали.
    # Замер, ради которого он появился: в LongMemEval улики к вопросам
    # категории knowledge-update старше вопроса в среднем на 16 дней, и
    # угасание СТИРАЛО их все до единой — 12 узлов из 12 в каждом из пяти
    # разобранных случаев, при том что всего удалялась лишь десятая часть
    # памяти. Ранжирование значения не имело: данных уже не было.
    #
    # Ноль возвращает прежнее поведение, когда пол давала только явная
    # похвала. Смысл ненулевого: прохождение спайк-гейта само по себе
    # признак важности — три четверти реплик до памяти не доходят вовсе.
    #
    # ПЛОСКИЙ ПОЛ ЗАМЕРЕН И ОТВЕРГНУТ как умолчание: 0.06 поднимает
    # knowledge-update с 18% до 85%, но разрыв "важное против рутины"
    # падает с +40 пунктов до нуля. Не забывается вообще ничего, а на
    # избирательности держится весь смысл библиотеки. Ручка оставлена
    # тем, кому нужна полнота любой ценой.
    memory_floor_base: float = 0.0
    # Пол, ЗАРАБОТАННЫЙ СИЛОЙ СПАЙКА: floor = spike_strength * этот
    # множитель. Плотность события известна при записи и хранится в
    # nodes.spike_strength, потому что вес с тех пор растает.
    #
    # ЗАМЫСЕЛ ПРОВЕРЕН И ПРОВАЛИЛСЯ — оставлено ручкой и уроком.
    #
    # Расчёт был такой: узел стирается ниже forget_threshold = 0.05, порог
    # записи = 0.25, значит множитель внутри вилки 0.10-0.20 обязан спасать
    # сильные спайки и оставлять рутину смертной. Замер (compare_retention
    # --balanced, разрыв "похвалённое против обычного"):
    #
    #     множитель   важное   рутина   разрыв
    #        0.00      100%      60%     +40
    #        0.10      100%      97%      +3
    #        0.15      100%     100%      +0
    #
    # Разрыв схлопывается уже на нижнем краю вилки. Причина не в числах:
    # сила спайка меряет НОВИЗНУ, а не важность. В стенде "меня зовут Паша"
    # и "вчера был дождь" одинаково новы, отличает их только похвала — а
    # она уже учтена слагаемым reward_expectation выше. Пол от спайка
    # защищает обе группы поровну, потому разрыв и исчезает.
    #
    # Более общий вывод, ради которого это стоит хранить: разрыв +40
    # ИЗМЕРЯЕТ УДАЛЕНИЕ. Организм не ставит похвалённое выше обычного в
    # выдаче — он обычное стирает. Пока избирательность живёт в удалении,
    # любой способ перестать терять данные её убивает.
    memory_floor_spike_factor: float = 0.0
    memory_keyword_weight: float = 0.3
    memory_min_keyword_length: int = 3
    memory_search_threshold: float = 0.3
    memory_semantic_weight: float = 0.5
    memory_weight_influence: float = 0.15
    # Полоса переупорядочивания. Кандидаты, чья релевантность отстаёт от
    # лучшей не больше чем на эту величину, считаются РАВНО ПОДХОДЯЩИМИ, и
    # порядок между ними решает важность, а не смысл.
    #
    # Зачем отдельная ступень вместо ещё одного слагаемого. Важность как
    # слагаемое конкурирует с релевантностью, и замер показал цену: подъём
    # доли веса с 0.15 до 0.70 уронил MRR похвалённого с 0.950 до 0.649, а
    # обычного с 0.877 до 0.751. Тяжёлый узел всплывает наверх КАЖДОГО
    # запроса, включая те, где он неверный ответ.
    #
    # Ноль выключает ступень и возвращает прежнее поведение.
    rerank_band: float = 0.0
    # ------------------------------------------------------------------
    # СИГНАЛЫ ВАЖНОСТИ — из чего складывается "насколько эта память дорога".
    #
    # Каждый входит отдельным весом и по умолчанию ВЫКЛЮЧЕН, кроме веса
    # узла. Пакетом их включать нельзя: сегодня уже выяснилось, что сигнал,
    # звучащий убедительно, и сигнал, который работает, — разные вещи (сила
    # спайка оказалась мерой новизны, а не важности). Каждый меряется
    # поодиночке на tools/compare_ordering.py и на внешнем наборе.
    # ------------------------------------------------------------------
    # Вес узла: до него подкрепление доказанно доходит — похвалённые узлы
    # тяжелее обычных в 2.7 раза (0.2806 против 0.1044).
    # Считать важность ДОЛЕЙ накопленной силы, а не весом узла.
    #
    # Вес смешивает три смысла: свежесть (он затухает), важность (растёт от
    # подкрепления) и силу извлечения (входит в оценку поиска). Почти все
    # измеренные за две сессии дефекты выросли из этого смешения —
    # переупорядочивание "по важности" оказывалось переупорядочиванием по
    # возрасту и роняло R@1 с 32% до 18%.
    #
    # strength копится от подкрепления и обращений и НЕ ПАДАЕТ ОТ ЧАСОВ.
    # Ценность при извлечении — доля от суммы по кандидатам. Тогда
    # забывание становится проигрышем в конкуренции (интерференция), а не
    # функцией времени (распад) — и это лучше подтверждённая теория
    # забывания у людей.
    use_relative_strength: bool = True
    # Потолок накопленной силы и шаги её роста.
    strength_max: float = 3.0
    # Прибавка за успешное извлечение: память пригодилась.
    strength_use_step: float = 0.05
    # Прибавка за единицу одобрения: пользователь сказал, что это важно.
    strength_reward_step: float = 0.5
    # Насколько слабеет кандидат, проигравший конкуренцию при извлечении.
    #
    # Вызванное забывание: у людей извлечение одного следа ТОРМОЗИТ
    # соседние по признаку. Это и делает нужное находимым среди похожих —
    # забывание нужно не ради места, а чтобы извлечение оставалось
    # возможным.
    #
    # Замер: на 50 почти-двойниках R@1 падает со 100% до 50%, на 800
    # держится 50% при R@5 83.3%. Нужный узел лежит в выдаче, но не
    # первым, — его топят соседи с теми же словами.
    #
    # Ноль выключает. Шаг мал намеренно: одно извлечение не должно решать.
    retrieval_suppression: float = 0.0
    # Сколько проигравших подавлять за раз: подавлять весь хвост
    # бессмысленно, конкуренты — это те, кто был близок.
    retrieval_suppression_limit: int = 10
    importance_weight_signal: float = 1.0
    # Связность: сколько у узла рёбер. В исследованиях памяти это "глубина
    # обработки" — вплетённое в известное держится лучше одинокого.
    importance_connectivity: float = 0.0
    # Сколько рёбер считать полной связностью при нормировке.
    importance_degree_full: int = 8
    # Самореференция: говорит ли человек О СЕБЕ. Эффект самореференции —
    # одна из самых устойчивых находок в психологии памяти, и для памяти
    # ассистента он попадает точно в цель: помнить надо факты о пользователе.
    importance_self_reference: float = 0.0
    # Использование: стабильность растёт в 1.5 раза при каждом обращении,
    # то есть уже является счётчиком успешных вспоминаний. Важность ПО
    # ПОСЛЕДСТВИЯМ, а не по догадке в момент записи.
    importance_use: float = 0.0
    # Со сколькими последними вспомненными узлами связывать новую запись.
    #
    # Замер, ради которого поле появилось: после 200 вызовов observe() в
    # базе оказался 201 эпизодический узел и НОЛЬ рёбер между ними. Не
    # мало — ни одного. Библиотека вообще не связывала воспоминания друг с
    # другом; рёбра, которые видно в демонстрации, создаёт витрина
    # (core/brain_session.py), у которой под рукой оба конца.
    #
    # Для библиотечного пользователя это значило сразу две вещи: связность
    # не могла работать сигналом важности, а растекающейся активации —
    # заявленной как multi-hop и занимающей заметную часть кода поиска —
    # было не по чему растекаться.
    #
    # Ноль возвращает прежнее поведение.
    associate_recalled_limit: int = 3
    # Вес, с которым рождается АССОЦИАТИВНОЕ ребро между эпизодами.
    #
    # Отдельное поле, а не edge_boost_step, по измеренной причине. Замер:
    # все 27 связей, созданных за разговор, весили 0.150 при пороге
    # активации 0.3 — выше порога НОЛЬ рёбер из двадцати семи. Связь
    # рождалась неактивной: чтобы участвовать в поиске, ей надо было
    # подкрепиться, а чтобы подкрепиться — сработать. Замкнутый круг, из-за
    # которого многошаговое извлечение давало ровно +0.0 пунктов.
    #
    # Общий edge_activation_threshold трогать НЕЛЬЗЯ: он же считает
    # знакомость словарных пар в compute_surprise, то есть управляет
    # удивлением и через него порогом записи. Нынешние 0.2/0.3 верны для
    # словаря, где ребро подкрепляется каждым повтором фразы; между
    # эпизодами повторов пары не бывает, и те же числа означают "никогда".
    associate_edge_weight: float = 0.6
    # Скорость затухания АССОЦИАТИВНЫХ рёбер. Отдельно от edge_decay_rate
    # (0.08), потому что тот откалиброван под словарь: там связь
    # подкрепляется каждым повтором фразы и обязана быстро выветриваться,
    # если слово перестали произносить.
    #
    # Между эпизодами повторов пары не бывает ни разу, поэтому на общей
    # скорости связи исчезали между четвёртыми и одиннадцатыми сутками.
    # Замер: выигрыш многошагового извлечения падал с +6.7 пункта до нуля
    # после недели молчания.
    associate_edge_decay_rate: float = 0.005
    # Прибавка ребру за то, что по нему ПРОШЛИ при извлечении.
    #
    # Растекание активации читало вес ребра и никогда в него не писало —
    # значит связь не могла отметить, что оказалась полезной. Укрепить её
    # умела только совместная встречаемость при записи.
    #
    # Из-за этого агрессивная подрезка была невозможна: резать смело
    # значило рубить рабочие дороги вместе с заброшенными, потому что
    # отличить их было нечем.
    #
    # Биология кладёт предел именно сюда, а не на объём хранилища.
    # Долговременная память не переполняется; мозг непрерывно и дорого
    # подрезает СВЯЗИ. Воспоминание теряется, становясь недостижимым, а не
    # стёртым — имя одноклассника не вспоминается, но узнаётся при встрече.
    edge_use_boost: float = 0.1
    # ------------------------------------------------------------------
    # ЁМКОСТЬ ВМЕСТО ВОЗРАСТА
    #
    # Сколько воспоминаний держать. При переполнении уходят НАИМЕНЕЕ
    # ЗАСЛУЖИВШИЕ (см. MemoryGraph._keep_score), а не самые старые.
    #
    # Зачем замена. Удаление по возрасту принимало решения само, и замер
    # показал какие: на каждом проходе стиралась десятая часть памяти, и
    # ответы на последующие вопросы всегда оказывались внутри этой
    # десятой. Улики к вопросам knowledge-update старше вопроса на 16
    # дней — стёрлись все, 12 из 12.
    #
    # Ёмкость к тому же и есть то, чем приложение хочет управлять.
    # Дизайнер игры задаёт "этот персонаж помнит двести вещей"; никто не
    # хочет задавать "этот персонаж забывает через одиннадцать дней".
    #
    # Ноль означает без ограничения.
    memory_capacity: int = 0
    # Удалять ли узлы, чей вес упал ниже forget_threshold. Прежнее
    # поведение — да; при ёмкостной политике осмысленно выключить, тогда
    # старое лишь тускнеет и уступает место в выдаче, но не пропадает.
    delete_on_decay: bool = False
    retrospective_correction_enabled: bool = True
    retrospective_reversal_strength: float = 1.6
    retrospective_time_window_seconds: float = 180.0
    retrospective_window_size: int = 4
    reward_expectation_learning_rate: float = 0.3
    reward_min_learning_scale: float = 0.15
    reward_negative_penalty: float = 0.3
    reward_positive_boost: float = 0.2
    reward_positive_freshness_bonus: float = 600.0
    reward_preference_weight: float = 0.35
    sleep_abstract_node_weight: float = 0.75
    sleep_archive_weight_multiplier: float = 0.3
    sleep_hub_min_edge_weight: float = 0.5
    sleep_max_cluster_spokes: int = 2
    sleep_min_cluster_spokes: int = 2
    sleep_orphan_weight_threshold: float = 0.3
    stability_growth_factor: float = 1.5
    stability_initial: float = 1.0
    plasticity_stress_modifier: float = 0.25
    # Candidate pool for the fuzzy comparison. Fuzzy similarity is
    # expensive (measured: 82% of search time), so it runs on the best
    # candidates from the cheap pre-filter rather than on every node.
    # Setting the minimum above the node count restores the exhaustive
    # scan — which is exactly how the test verifies this trade-off.
    search_candidate_multiplier: int = 20
    search_candidate_minimum: int = 50
    stability_max: float = 40.0
    stm_capacity: int = 16
    stm_emotional_threshold: float = 0.6
    stm_structural_threshold: float = 0.55
    # Во сколько раз слабее рождается СВЁРНУТЫЙ эпизод по сравнению с
    # обычной записью.
    #
    # Замер, ради которого поле появилось: включённая консолидация роняет
    # R@1 с 76% до 42% (буфер 16) и до 52% (буфер 4), почти не трогая R@5
    # и R@10. То есть улика остаётся в памяти, но свёртка оттесняет её
    # сверху: восемь обменов в одном узле совпадают почти с любым
    # запросом.
    #
    # Единица возвращает прежнее поведение — свёртка на равных с
    # подробностью. Меньшее значение делает её тем, чем схема является у
    # людей: она всплывает, когда подробность уже недоступна.
    consolidated_strength_factor: float = 1.0
    stm_structural_weight: float = 0.5
    # How many content words an utterance needs to surprise at full
    # strength. Below that, surprise is scaled down proportionally.
    surprise_full_content_tokens: int = 3
    surprise_lexical_weight: float = 0.6
    surprise_structural_weight: float = 0.4
    syllable_node_initial_weight: float = 0.1
    syllable_node_reinforce_step: float = 0.03
    syllable_word_edge_weight: float = 0.45
    vocabulary_mastery_min_weight: float = 0.18
    word_cooccurrence_edge_weight: float = 0.12
    word_node_initial_weight: float = 0.15
    word_node_reinforce_step: float = 0.04

    # Path to the database file. The only field here that is not a
    # behavioural parameter — but Database needs it, and dragging in a
    # global config just for that would defeat the point.
    db_path: str = "memory.db"

    # Names spelled differently in config.py. Kept deliberately short:
    # diverging names are a source of silent mismatches.
    _ALIASES = {"db_path": "BRAIN_DB_PATH"}

    @classmethod
    def from_module(cls, module: Any) -> "MemorySettings":
        """
        Builds settings from a module of UPPER_CASE constants. Missing
        fields keep their defaults — an application may override only part
        of them.
        """
        values = {}
        for f in fields(cls):
            name = cls._ALIASES.get(f.name, f.name.upper())
            if hasattr(module, name):
                values[f.name] = getattr(module, name)
        return cls(**values)
