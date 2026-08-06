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
 HIPPOCAMPUS.PY — Быстрая запись эпизода и связывание одновременного
================================================================================
Гиппокамп — не хранилище воспоминаний, а быстрый ИНДЕКС к ним. Он умеет
записать событие с одного раза, связать то, что случилось вместе, и
заметить, что новое противоречит старому. Подробности со временем уходят в
кору; гиппокамп держит указатель.

ЧТО ЗДЕСЬ ЕСТЬ:

    save_connection        запись эпизода с одного предъявления;
    find_superseded        обнаружение рассогласования: пришедшее не
                           сходится с тем, что записано. В мозге это
                           работа поля CA1 — сравнение ожидаемого с
                           действительным;
    connect_nodes,
    reinforce_coactivation связывание одновременно активного. Рекуррентная
                           сеть поля CA3, гебовское "сработало вместе —
                           связалось вместе";
    apply_reward,
    penalize_node          приложение сигнала подкрепления к узлам. САМ
                           сигнал считается не здесь: его источник —
                           дофаминовый контур в reinforcement.py. Здесь
                           только принимающая сторона, ровно как в мозге,
                           где дофамин из среднего мозга модулирует
                           пластичность гиппокампа, а не заменяет её.

ЧЕГО ЗДЕСЬ НЕТ, И ЭТО НАШ ГЛАВНЫЙ БИОЛОГИЧЕСКИЙ ПРОБЕЛ — РАЗДЕЛЕНИЯ
ОБРАЗОВ. Зубчатая извилина превращает похожие входы в непохожие ДО
записи: два почти одинаковых события получают заведомо разные коды,
именно чтобы потом не мешать друг другу при извлечении. У нас похожее
пишется как похожее, и стенд почти-двойников это прямо показывает: R@1
падает со 100% до 50% на пятидесяти двойниках. Мы боремся с последствием
на стороне извлечения, тогда как биология решает это на входе.

Класс — миксин: состояние принадлежит MemoryGraph, который собирает
участки вместе.
================================================================================
"""

import logging
import time
from typing import Any, Dict, List, Optional, Set

from selectivemem import embeddings
from selectivemem.records import MemoryMatch, RewardSignal, SupersededNode

logger = logging.getLogger(__name__)


class HippocampusMixin:
    """Быстрая запись, рассогласование, связывание, приём подкрепления."""

    def find_superseded(
        self,
        text: str,
        exclude_id: Optional[int] = None,
        explicit_correction: bool = False,
    ) -> List["SupersededNode"]:
        """
        Which existing memories a new one SUPERSEDES.

        Without this, memory piled up mutually exclusive facts and
        returned an arbitrary one: "my dog is called Rex", later "my dog is
        called Buddy" — both nodes equal, and the stale one actually
        scoring BETTER (0.906 against 0.875), because the ranking is
        decided by string similarity rather than by time.

        Supersession requires two conditions at once:
          1. high SEMANTIC similarity: they are about the same thing;
          2. INCOMPLETE word overlap: this is a different version, not a
             repetition. A plain repetition must simply reinforce the node.

        explicit_correction means the user corrected something outright
        ("no", "that's wrong"). That is strong evidence, so the topic
        threshold is lowered: without such a marker we are cautious, with
        one we trust.

        The threshold is deliberately high. Erring towards "missed a
        contradiction" is cheaper than weakening an independent memory —
        though even that is no catastrophe, because nodes are weakened
        rather than deleted (see supersede_node).
        """
        query_vector = self._encode(text)
        if query_vector is None:
            # Without semantics there is no way to tell "a different
            # version" from "a different subject": string similarity is
            # equally high for "called Rex"/"called Buddy" and for
            # "called Rex"/"called Rex".
            return []

        threshold = self.settings.contradiction_topic_threshold
        if explicit_correction:
            threshold -= self.settings.contradiction_correction_relief

        new_words = self._extract_keywords(text.lower())

        if self.settings.contradiction_search_threshold > 0.0:
            return self._superseded_via_search(
                text, new_words, exclude_id, explicit_correction
            )

        found: List[SupersededNode] = []

        # КАНДИДАТЫ ИЗ ИНДЕКСА, а не перебор всей базы. Проверка на
        # устаревание идёт при каждой записи, и полный скан с косинусом на
        # каждый узел делал запись невыносимой задолго до того, как
        # замедлялся поиск: три тысячи узлов не записывались за две минуты.
        #
        # Отбор по общим словам ничего не теряет: вытеснение всё равно
        # требует пересечения не ниже contradiction_min_overlap, то есть
        # узел без единого общего слова был бы отвергнут дальше.
        candidates = self.gate.episodic.candidates_by_text(
            sorted(new_words), self.settings.supersede_scan_limit,
        )
        for row in candidates:
            if row["id"] == exclude_id or row["is_meta"]:
                continue

            # Compare ONLY what the user said, without the bot's replies.
            # _node_vector builds its vector from the question-and-answer
            # pair, which is right for search but wrong here: a fact lives
            # in what the PERSON said, while the bot's "got it" or "okay"
            # is noise that shifts the vector and decides the comparison.
            similarity = embeddings.cosine(
                query_vector, self._encode(row["context"] or "")
            )
            if similarity < threshold:
                continue

            old_words = self._extract_keywords((row["context"] or "").lower())
            overlap = self._keyword_overlap(new_words, old_words)
            if overlap >= self.settings.contradiction_repeat_threshold:
                continue  # a repetition, not a new version

            # ОБЩЕЕ СЛОВО ОБЯЗАНО БЫТЬ РЕДКИМ — иначе это не поправка, а
            # совпадение по обороту речи.
            #
            # Живой разговор без этой проверки: за 23 реплики семь
            # вытеснений, ВСЕ СЕМЬ ошибочные. «тосты люблю, и омлет»
            # ослабило «я люблю кофе» — совпадение по слову «люблю».
            # Настоящая поправка делит ПРЕДМЕТ («собака», «рейс») и меняет
            # значение; предмет редок, оборот — нет.
            #
            # Первая попытка спрашивала только граф языка и не сработала
            # ни разу: в разговоре из двадцати трёх реплик редко ВСЁ.
            # Понадобилось априорное знание о частотности (lexicon.py) —
            # то, с чем взрослый приходит в разговор.
            ceiling = self.settings.contradiction_subject_familiarity
            if ceiling < 1.0:
                shared = new_words & old_words
                known = self._familiarity(shared)
                if not any(known.get(w, 0.0) <= ceiling for w in shared):
                    logger.info(
                        "[SUPERSEDE] Отклонено: общие слова частые, "
                        "предмета нет — %r против %r",
                        text[:40], (row["context"] or "")[:40],
                    )
                    continue
            if overlap < self.settings.contradiction_min_overlap:
                # Слишком мало общих слов — защита от чужого кодировщика,
                # см. тот же охранник в _superseded_via_search. Здесь он
                # нужнее: этот путь стоит по умолчанию, и на английском
                # тексте с русской моделью он срабатывал 3080 раз на 79
                # записей, ослабляя по сорок узлов на каждую запись.
                continue

            found.append(
                SupersededNode(
                    id=row["id"],
                    context=row["context"],
                    similarity=similarity,
                    word_overlap=overlap,
                )
            )

        return self._separate_patterns(text, found)

    def _familiarity(self, words) -> dict:
        """
        Насколько привычно каждое слово: опыт ИЛИ априорное знание.

        Берётся максимум. Граф языка — запись того, что организм слышал;
        `lexicon` — то, что он знал о языке до первого разговора. Второе
        не отменяет первого, а закрывает его слепое пятно: в коротком
        разговоре редко ВСЁ, включая «люблю» и «просто».
        """
        from selectivemem.lexicon import prior_familiarity

        words = sorted(words)
        if not words:
            return {}
        rows = self.gate.semantic.lexical_by_texts("word", words)
        seen = {row["context"]: row["weight"] for row in rows}
        return {w: max(seen.get(w, 0.0), prior_familiarity(w)) for w in words}

    def _separate_patterns(
        self, text: str, found: List["SupersededNode"]
    ) -> List["SupersededNode"]:
        """
        РАЗДЕЛЕНИЕ ОБРАЗОВ: в тесноте похожих поправка неотличима от
        соседа, и правильный ответ — не трогать никого.

        Зубчатая извилина делает похожие входы непохожими ДО записи,
        чтобы два близких события хранились раздельно и не мешали друг
        другу при извлечении. У нас такого шага нет, и последствие
        измерено: механизм вытеснения устаревшего, написанный для
        поправок, принимает соседа за исправление.

            200 почти-двойников при записи -> 6078 вытеснений,
            то есть каждая запись "исправляла" тридцать чужих узлов.

        Бьёт это в первую очередь по НАСТОЯЩИМ фактам: они похожи на всех
        своих двойников сразу, поэтому штраф достаётся им чаще всех. Замер
        на стенде: у шести фактов сила падает до 0.0000, у двойников
        остаётся 0.0292 — оригинал оказывается слабее собственных копий.
        Чистая релевантность в тех же условиях даёт 5 попаданий из 6, а
        полный поиск 3 из 6: разницу создаёт именно эта порча.

        ЧТО ЗДЕСЬ ДЕЛАЕТСЯ. Поправка по смыслу заменяет ОДНО конкретное
        воспоминание: "мою собаку зовут Рекс" -> "... Бобик". Если же на
        новый текст одинаково хорошо отзывается десяток узлов, вопрос
        "какой из них исправлен" не имеет ответа — и честный ответ
        "никакой, это новое отдельное воспоминание".

        Это перегрузка признака, применённая НА ВХОДЕ, а не при
        извлечении: там мы боремся с последствием, здесь снимаем причину.

        Ноль в pattern_separation_limit выключает проверку и возвращает
        прежнее поведение.
        """
        limit = self.settings.pattern_separation_limit
        if limit <= 0 or len(found) <= limit:
            return found

        logger.info(
            "[PATTERN SEPARATION] %d candidates for %r — too crowded to be a "
            "correction, stored as a separate memory",
            len(found), text[:40],
        )
        self._generalise(text, found)
        return []

    def _generalise(self, text: str, similar: List["SupersededNode"]) -> None:
        """
        КОРА ВЫВОДИТ УСТОЙЧИВОЕ ИЗ ПОВТОРЯЮЩЕГОСЯ.

        Сюда попадают ровно те случаи, когда на новый текст откликнулся
        десяток похожих записей. До сих пор это значило только «не
        вытеснять никого» — сигнал повторяемости выбрасывался.

        А ведь это и есть то, на чём учится кора: тема вернулась. Событие
        помнит гиппокамп, а кора копит знание, что тема вообще есть, и
        насколько часто к ней возвращаются.

        ТЕМА — ОБЩИЕ СЛОВА, а не текст очередного случая. «Люблю кофе по
        утрам» и «кофе мой любимый» делят слово «кофе», и факт у них
        должен быть один. Если общего слишком мало, темы нет: два текста
        могли совпасть случайно.
        """
        threshold = self.settings.cortex_fact_threshold
        if threshold <= 0 or len(similar) < threshold:
            return

        # СЛУЖЕБНЫЕ СЛОВА ТЕМОЙ НЕ БЫВАЮТ. Список стоп-слов для поиска
        # уже, чем нужно здесь: он оставляет предлоги вроде «про», и замер
        # это поймал — кора вывела «тему» ПРО с двадцатью встречами.
        #
        # Берём тот же список, которым кодировщик чистит фразу перед
        # усреднением: он для того и составлен, чтобы отсеять слова, не
        # несущие смысла сами по себе.
        from selectivemem.embeddings import _FUNCTION_WORDS

        shared = {
            w for w in self._extract_keywords(text.strip().lower())
            if w not in _FUNCTION_WORDS
        }
        for node in similar:
            shared &= {
                w for w in self._extract_keywords(node.context.strip().lower())
                if w not in _FUNCTION_WORDS
            }
            if not shared:
                return

        # ОДНО СЛОВО ТЕМОЙ БЫТЬ МОЖЕТ, если оно редкое. «Кофе» — тема,
        # «день» — нет, хотя оба прошли отсев стоп-слов.
        #
        # Первая версия требовала двух общих слов, и механизм не сработал
        # ни разу: у настоящих тем общее слово обычно ОДНО, остальное
        # разное — в том и смысл повторения другими словами.
        #
        # Редкость берётся оттуда же, откуда её берёт поиск: сколько
        # записей содержат это слово. Тема — то, что встречается у
        # немногих; служебное — то, что у всех.
        if not shared:
            return

        # РЕДКОСТЬ ПРОВЕРЯЕТСЯ У КАЖДОГО СЛОВА, А НЕ ТОЛЬКО У ОДИНОЧНОГО.
        #
        # Здесь стояла проверка при len(shared) < 2: одно слово обязано
        # было доказать редкость, а два и больше проходили без разговора —
        # много общих слов считалось доказательством посильнее.
        #
        # ВСЁ РОВНО НАОБОРОТ. У настоящей темы общее слово обычно ОДНО
        # («виолончель»), остальное разное — в том и смысл повторения
        # другими словами. А вот повторяющийся ОБОРОТ РЕЧИ делит слов
        # много, потому что он и есть один и тот же оборот. Замер поймал
        # ровно это: из девяти фактов настоящих тем оказалось два, а
        # остальные — канцелярит наполнителя ('рано утром', 'прислал
        # счёт', 'записку неделе оставил прошлой').
        #
        # РЕДКОСТЬ БЕРЁТСЯ ИЗ ГРАФА ЯЗЫКА, А НЕ ИЗ ЧИСЛА ЗАПИСЕЙ.
        #
        # Сначала здесь стояла доля эпизодов, содержащих слово. Она НЕ
        # РАЗДЕЛЯЕТ: «рано утром» сидит в пяти процентах записей и проходит
        # любой разумный порог, оставаясь при этом чистым канцеляритом.
        # Замер после её распространения на все слова: мусора как было
        # семь из девяти, так и осталось.
        #
        # Разделяет другое — насколько слово ПРИВЫЧНО САМОМУ ОРГАНИЗМУ.
        # Он ведёт граф языка с весами: сколько раз слово встречалось,
        # с насыщением к единице. Замер на том же прогоне:
        #
        #     виолончель 0.264   пенициллин 0.264   марафон 0.186
        #     утром 1.000        обедом 1.000       перенёс 1.000
        #
        # Разделение полное: темы не выше 0.264, канцелярит не ниже 0.999.
        # Слово, которое организм знает назубок, не может быть тем, ЧЕМ
        # этот случай отличается от других, — оно есть везде.
        #
        # ОГОВОРКА ЧЕСТНАЯ: зазор такой широкий на стенде, где наполнитель
        # собран из шести оборотов и потому насыщается до единицы. В живой
        # речи он будет уже, и порог придётся перемерить.
        # Привычность — максимум из опыта и априорного знания о языке.
        # Без второго проверка слепа в начале жизни: в первые дни
        # организму незнаком и канцелярит, поэтому «рано утром» оседало
        # темой и снималось только пересмотром во сне.
        familiarity = self._familiarity(shared)
        ceiling = self.settings.cortex_theme_max_familiarity
        shared = {w for w in shared if familiarity.get(w, 0.0) <= ceiling}
        if not shared:
            return

        # ПОРЯДОК СЛОВ — КАК В ТЕКСТЕ, а не по алфавиту. Тема собиралась
        # из множества, и в базе оседали перевёртыши вроде «обедом перед»
        # и «записку неделе оставил прошлой». Читать такое нельзя, а
        # выдаётся оно наравне с эпизодами.
        lowered = text.lower()
        theme = " ".join(sorted(shared, key=lowered.find))
        self.gate.semantic.record_fact(
            theme=theme,
            text=theme,
            meaning=text,
            strength_step=self.settings.cortex_fact_strength,
            cap=self.settings.strength_max,
        )

    def _superseded_via_search(
        self,
        text: str,
        new_words: Set[str],
        exclude_id: Optional[int],
        explicit_correction: bool,
    ) -> List["SupersededNode"]:
        """
        Finds stale versions among what ORDINARY SEARCH returns, instead of
        scanning every node with a high cosine bar.

        WHY THE OLD WAY WAS BROKEN BY DESIGN, not by a badly chosen number.
        It compared whole sentences and demanded cosine >= 0.8. But a
        contradiction is "same subject, DIFFERENT value" — so the stronger
        the change, the less similar the sentences, and the more surely the
        update slips through. The evidence of a contradiction was being
        subtracted from the evidence of relatedness.

        Measured, against a threshold of 0.8:

            "мою собаку зовут Рекс" -> "... Бобик"        0.923  caught
            "моя собака зовут Рекс" -> "мою собаку ТЕПЕРЬ
                                        зовут Бобик"      0.722  missed
            "я живу в Москве" -> "я ПЕРЕЕХАЛ в Питер"     0.369  missed

        The first pair is the example from this method's own docstring. The
        mechanism was calibrated for restatements that swap a single word
        into the same template, and blind to how people actually report a
        change.

        Search finds the old node in ALL of those cases, because it blends
        keywords, fuzzy similarity and meaning rather than trusting one
        cosine. So the roles swap: SEARCH FINDS, and the word-overlap test
        decides whether this is a new version or a repetition.

        HONEST LIMIT. Five of six real updates separate cleanly — the worst
        scores 0.642 while the best unrelated memory scores 0.433. The
        sixth does not separate at all: "я живу в Москве" and "я переехал в
        Питер" share no words, and no measure of string similarity will
        connect them. That needs knowing that "moved" cancels "live in",
        which is knowledge this library does not have.
        """
        threshold = self.settings.contradiction_search_threshold
        if explicit_correction:
            threshold -= self.settings.contradiction_correction_relief

        candidates = self.search(
            text,
            top_k=self.settings.contradiction_candidates,
            with_associations=False,
            touch=False,          # проверка, а не использование
        )

        found: List[SupersededNode] = []
        for match in candidates:
            if match.id == exclude_id or match.similarity < threshold:
                continue

            old_words = self._extract_keywords((match.context or "").lower())
            overlap = self._keyword_overlap(new_words, old_words)
            if overlap >= self.settings.contradiction_repeat_threshold:
                continue          # повтор, а не новая версия
            if overlap < self.settings.contradiction_min_overlap:
                # СЛИШКОМ МАЛО ОБЩИХ СЛОВ — защита от чужого кодировщика.
                # Замер: русская модель (та, что в пакете) на английском
                # тексте даёт "my dog is called Rex" против "the price of
                # bread went up" косинус 0.808 при пороге 0.8. Порог сидит
                # внутри шума, и память начинает ослаблять факт про собаку,
                # потому что подорожал хлеб. На том же тексте прежний путь
                # срабатывал 3080 раз на 79 записей — сорок ослаблений на
                # каждую запись.
                #
                # Общее слово подделать эмбеддингом нельзя, поэтому проверка
                #языконезависима и стоит один проход по множеству.
                continue

            found.append(
                SupersededNode(
                    id=match.id,
                    context=match.context,
                    similarity=match.similarity,
                    word_overlap=overlap,
                )
            )
        return found

    def supersede_node(self, node_id: int, timestamp: Optional[float] = None,
                       replacement_id: Optional[int] = None) -> None:
        """
        Пометить воспоминание устаревшим: связать его с тем, что пришло
        на замену, и — если позволено настройкой — ослабить.

        ПЕРЕНАПРАВЛЕНИЕ, А НЕ ОСЛАБЛЕНИЕ, и это исправление противоречия
        внутри самого проекта. README обещает: «угасает ПУТЬ к
        воспоминанию, а не само воспоминание». Здесь же делалось обратное
        — снижался вес узла и сбрасывалась его стабильность.

        Последствия были тяжёлыми и несимметричными:

          * узел слабел ГЛОБАЛЬНО, то есть выпадал из всех запросов, а не
            только из того, где случилась поправка;
          * становился кандидатом на вытеснение по ёмкости;
          * а при ошибке — и живой разговор дал семь ошибок из семи —
            верный факт повреждался навсегда.

        Само отношение при этом нигде не сохранялось: «это заменило то»
        существовало одно мгновение и исчезало.

        Теперь оно записывается ребром. Узел цел, вес не тронут, а при
        выдаче поиск, встретив устаревшее, знает, чем оно заменено.
        Ошибка стоит почти ничего: связь можно отменить, факт не пострадал.

        `contradiction_weight_penalty = 0` оставляет чистое
        перенаправление; ненулевое значение возвращает прежнее ослабление
        вдобавок к нему.
        """
        row = self.gate.node.get(node_id)
        if row is None:
            return

        if replacement_id is not None and replacement_id != node_id:
            # Ребро направлено ОТ НОВОГО К СТАРОМУ: «я заменяю его».
            self.connect_nodes(
                replacement_id, node_id,
                weight_boost=self.settings.supersede_edge_weight,
                timestamp=timestamp,
                edge_type="supersedes",
            )

        penalty = self.settings.contradiction_weight_penalty
        if penalty <= 0.0:
            return

        new_weight = max(0.0, row["weight"] - penalty)
        stability = (row["stability"] or self.settings.stability_initial)
        new_stability = max(
            self.settings.stability_initial,
            stability * self.settings.contradiction_stability_factor,
        )
        self.gate.node.set_weight(node_id, new_weight)
        self.gate.node.set_stability(node_id, new_stability)

    def save_connection(
        self,
        context: str,
        response: str,
        weight: Optional[float] = None,
        timestamp: Optional[float] = None,
        explicit_correction: bool = False,
        node_type: str = "episodic",
    ) -> int:
        """
        Stores a new link context -> response with an initial weight.

        explicit_correction means the user corrected something outright
        ("no", "that's wrong"). It lowers the bar for superseding stale
        versions.
        """
        initial_weight = weight if weight is not None else self.settings.base_plasticity_threshold

        node_id = self.gate.episodic.insert(
            context=context,
            response=response,
            weight=initial_weight,
            timestamp=timestamp,
            node_type=node_type,
        )

        # ВЕКТОР СЧИТАЕТСЯ ПРИ ЗАПИСИ, а не лениво при первом поиске.
        #
        # Ленивый расчёт растягивал стоимость на первое обращение, и
        # платил за него пользователь: замер показал 781 мс на первом
        # поиске по тысяче узлов против 14 мс на прогретом. То есть
        # человек, открывший бота утром, ждал почти секунду — а потом
        # всё летало, и в отчётах об ошибках это выглядело бы загадкой.
        #
        # При записи та же работа стоит один вызов кодировщика на узел и
        # размазана ровно там, где её ждут.
        if node_id is not None:
            vector = self._encode(f"{context} {response}".strip())
            if vector is not None:
                self.gate.node.set_embedding(node_id, embeddings.to_blob(vector))

        # A newer version of a fact supersedes the older one: otherwise
        # memory piles up mutually exclusive nodes and returns an
        # arbitrary one.
        for stale in self.find_superseded(
            context, exclude_id=node_id, explicit_correction=explicit_correction
        ):
            logger.info(
                "[CONTRADICTION] %r supersedes %r (similarity %.2f, shared words %.2f)",
                context[:40], stale.context[:40], stale.similarity, stale.word_overlap,
            )
            self.supersede_node(stale.id, timestamp=timestamp,
                                replacement_id=node_id)

        logger.info(
            "[SPIKE DETECTED] New link stored id=%s weight=%.3f",
            node_id, initial_weight,
        )
        return node_id

    def touch_node(self, node_id: int, timestamp: Optional[float] = None) -> None:
        ts = timestamp if timestamp is not None else time.time()
        self.gate.node.touch(node_id, timestamp=ts)
        logger.debug("[MEMORY TOUCHED] id=%s last_accessed updated (t=%.2f)", node_id, ts)

    def connect_nodes(
        self,
        node_from: int,
        node_to: int,
        weight_boost: Optional[float] = None,
        timestamp: Optional[float] = None,
        edge_type: Optional[str] = None,
    ) -> float:
        """
        Creates or strengthens an associative edge between two long-term
        nodes.

        Two scenarios use it:
            1. Contextual linking: node A was pulled from memory (a search
               hit) and node B was created or reinforced during the same
               exchange -> the edge A -> B is strengthened.
            2. Co-activation linking: several nodes were used within one
               STM window -> the edges between them grow (see
               reinforce_coactivation).

        The edge is ignored when node_from == node_to; a self-loop means
        nothing. Returns the resulting edge weight.
        """
        if node_from is None or node_to is None or node_from == node_to:
            return 0.0

        # Race protection: one of the nodes may have been deleted — a
        # low-weight syllable node caught by orphan pruning during sleep,
        # say — between the moment its id was recorded and this call. The
        # FOREIGN KEY on edges would otherwise blow up the insert, so the
        # edge is quietly skipped instead.
        if self.gate.node.get(node_from) is None or self.gate.node.get(node_to) is None:
            logger.debug(
                "[ASSOCIATION SKIP] Node %s or %s no longer exists (deleted) -> edge not created",
                node_from, node_to,
            )
            return 0.0

        boost = weight_boost if weight_boost is not None else self.settings.edge_boost_step
        ts = timestamp if timestamp is not None else time.time()

        new_weight = self.gate.edges.upsert(
            node_from=node_from,
            node_to=node_to,
            weight_boost=boost,
            timestamp=ts,
            edge_type=edge_type,
        )

        logger.info(
            "[ASSOCIATION] Node %s -> Node %s (edge_weight=%.2f)",
            node_from, node_to, new_weight,
        )
        return new_weight

    def reinforce_coactivation(
        self,
        node_ids: List[int],
        weight_boost: Optional[float] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Co-activation linking: when several long-term nodes were touched
        or reinforced within ONE STM window, the edges between EVERY pair
        of them grow.

        node_ids is the list of nodes activated in the current window;
        duplicates and None values are filtered out automatically.
        """
        unique_ids = sorted({nid for nid in node_ids if nid is not None})
        if len(unique_ids) < 2:
            return

        boost = weight_boost if weight_boost is not None else self.settings.edge_boost_step
        ts = timestamp if timestamp is not None else time.time()

        for i in range(len(unique_ids)):
            for j in range(i + 1, len(unique_ids)):
                self.connect_nodes(unique_ids[i], unique_ids[j], weight_boost=boost, timestamp=ts)

        logger.info(
            "[COACTIVATION] Edges reinforced between co-activated nodes: %s",
            unique_ids,
        )

    def reinforce_node(self, node_id: int, boost: float = 0.1, timestamp: Optional[float] = None) -> None:
        row = self.gate.node.get(node_id)
        if row is None:
            logger.warning("[MEMORY REINFORCE] Node id=%s not found", node_id)
            return

        new_weight = min(1.0, row["weight"] + boost)
        self.gate.node.set_weight(node_id, new_weight)
        self.touch_node(node_id, timestamp=timestamp)
        logger.info("[MEMORY REINFORCED] id=%s new weight=%.3f", node_id, new_weight)

    def apply_reward(
        self,
        node_id: int,
        valence: float,
        timestamp: Optional[float] = None,
    ) -> Optional[RewardSignal]:
        """
        The dopamine signal: computes the REWARD PREDICTION ERROR for a
        node, updates its expectation and returns the result.

            rpe = actual valence - what this node expected
            expectation += reward_expectation_learning_rate * rpe

        This is the Rescorla-Wagner rule. The point of it: dopamine is
        released not by reward but by UNEXPECTED reward. Without it, the
        pursuit of approval degenerates — the organism would find one word
        that is always praised and repeat it forever. Here, what is
        ALWAYS praised stops producing a signal (rpe -> 0) and the
        organism goes off to try something new.

        Returns None if the node has disappeared: it may have been pruned
        between the action and the rating.
        """
        row = self.gate.node.get(node_id)
        if row is None:
            return None

        expected = row["reward_expectation"] or 0.0
        rpe = valence - expected
        new_expectation = max(-1.0, min(1.0, expected + self.settings.reward_expectation_learning_rate * rpe))

        self.gate.node.set_reward_expectation(node_id, new_expectation)
        # Одобрение поднимает НАКОПЛЕННУЮ СИЛУ, а не только ожидание
        # награды. Вес для этого не годится: он затухает от времени, и
        # через две недели от похвалы не остаётся следа — замерено,
        # похвалённый узел терял 0.95 -> 0.17 за месяц. Сила часам не
        # подчиняется, поэтому одобрение сохраняется столько, сколько
        # его не разбавили новые записи.
        self.gate.node.add_strength(
            node_id,
            valence * self.settings.strength_reward_step,
            self.settings.strength_max,
        )

        logger.info(
            "[DOPAMINE] node=%s valence=%+.2f expected=%+.2f -> rpe=%+.2f "
            "(new expectation %+.2f)",
            node_id, valence, expected, rpe, new_expectation,
        )
        return RewardSignal(
            node_id=node_id,
            valence=valence,
            expected=expected,
            prediction_error=rpe,
            new_expectation=new_expectation,
        )

    def learning_scale(self, prediction_error: float) -> float:
        """
        By how much the reward prediction error accelerates consolidation.

        Dopamine modulates synaptic plasticity: an unexpected outcome
        consolidates strongly, a fully predicted one almost not at all.
        The lower bound (reward_min_learning_scale) keeps learning from
        reaching exactly zero — otherwise a long-mastered node would stop
        receiving even maintenance reinforcement.
        """
        return max(self.settings.reward_min_learning_scale, min(1.0, abs(prediction_error)))

    def penalize_node(
        self,
        node_id: int,
        penalty: float = 0.15,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Penalises a node for negative feedback: lowers its weight and
        deliberately does NOT move last_accessed forward, unlike
        touch_node or reinforce_node. That accelerates the node's relative
        ageing at the next apply_decay, modelling the lower durability of
        a negatively reinforced link.
        """
        row = self.gate.node.get(node_id)
        if row is None:
            logger.warning("[MEMORY PENALIZE] Node id=%s not found", node_id)
            return

        new_weight = max(0.0, row["weight"] - penalty)
        self.gate.node.set_weight(node_id, new_weight)
        logger.info(
            "[MEMORY PENALIZED] id=%s weight %.3f -> %.3f (penalty=%.3f)",
            node_id, row["weight"], new_weight, penalty,
        )
