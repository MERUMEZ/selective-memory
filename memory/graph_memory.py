"""
================================================================================
 GRAPH_MEMORY.PY — Биологический слой памяти "Динамического Мозга" (LTM)
================================================================================
Класс MemoryGraph оперирует поверх Database (memory/database.py) и добавляет
"живую" логику:
    - сохранение новой связи с начальным весом (spike memory)
    - поиск похожего контекста по ключевым словам + нечёткому сходству (search)
    - обновление last_accessed при каждом обращении
    - функция угасания веса (decay) для старых, неиспользуемых связей
    - ИЗБИРАТЕЛЬНАЯ КОНСОЛИДАЦИЯ (Selective Consolidation) — перенос эпизодов
      из кратковременной памяти (STM, WorkingMemory) в долгосрочную (LTM/БД)

Консолидация (consolidate_from_stm) оценивает накопленный эпизод STM по
двум критериям и принимает одно из трёх решений:
    a) ЭМОЦИОНАЛЬНЫЙ УЗЕЛ — высокий emotion_score/total_density -> запись в БД
       с высоким весом (аналог core/amygdala.py, но для целого эпизода).
    b) СТРУКТУРНЫЙ УЗЕЛ — высокая средняя perplexity (много новой информации)
       -> запись в БД с умеренным весом.
    c) РУТИННЫЙ ШУМ — ни то ни другое -> эпизод стирается БЕЗ записи в БД.

Формула затухания (экспоненциальная кривая забывания):
    weight(t) = weight_0 * exp(-DECAY_RATE * dt / AGE_T0)
================================================================================
"""

import math
import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import List, Optional, Set, TYPE_CHECKING

from memory.database import Database
from memory.settings import MemorySettings
from services import embeddings
import logging

if TYPE_CHECKING:
    from memory.working_memory import STMEntry

logger = logging.getLogger(__name__)

STOP_WORDS: Set[str] = {
    "и", "в", "на", "с", "по", "к", "у", "из", "за", "от", "до", "для",
    "что", "как", "это", "то", "я", "ты", "он", "она", "мы", "вы", "они",
    "не", "но", "а", "же", "бы", "ли", "или", "тот", "его", "ее", "их",
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "in", "on",
    "and", "or", "but", "for", "with", "at", "by", "it", "this", "that",
}

WORD_PATTERN = re.compile(r"[^\s\d\W]+", flags=re.UNICODE)

# Гласные (RU + EN) для примитивной слоговой сегментации (heuristic:
# слог = согласные* + гласная(-ые), хвостовые согласные "прилипают"
# к последнему слогу слова).
VOWELS: Set[str] = set("аеёиоуыэюяAEIOUYaeiouy")


@dataclass
class LexicalProcessingResult:
    """Результат побочного процесса лексического освоения языка."""
    words_processed: int
    syllables_processed: int
    new_words: int
    new_syllables: int


@dataclass
class SurpriseResult:
    """
    Ошибка предсказания организма на входящем тексте — СОБСТВЕННОЕ
    удивление, посчитанное по его же графу, а не статистика строки.

    total      — итог [0..1], идёт в спайк-гейт, уверенность, консолидацию
    lexical    — доля новизны от незнакомых СЛОВ
    structural — доля новизны от непривычных СОЧЕТАНИЙ соседних слов
    known_words / total_words — сколько слов входа организм уже знает
    known_pairs / total_pairs — то же для пар соседних слов
    """
    total: float
    lexical: float
    structural: float
    known_words: int
    total_words: int
    known_pairs: int
    total_pairs: int


@dataclass
class MemoryMatch:
    """Результат поиска похожего контекста в графе памяти."""
    id: int
    context: str
    response: str
    weight: float
    similarity: float
    created_at: float
    last_accessed: float


@dataclass
class ConsolidationResult:
    """Результат попытки избирательной консолидации эпизода STM в LTM."""
    decision: str  # "emotional_node" | "structural_node" | "routine_noise"
    node_id: Optional[int] = None
    weight: Optional[float] = None
    reason: str = ""


@dataclass
class KnownSyllable:
    """
    Известный слог с его ID и текущим весом (усвоенностью) — используется
    babbling-подсистемой (InstinctSystem.generate_babble_response) для
    взвешенного выбора слогов и последующего трекинга использованных
    узлов в Reinforcement Loop (Cortex.apply_feedback).
    """
    id: int
    text: str
    weight: float


@dataclass
class RewardSignal:
    """
    Результат дофаминового сигнала на одном узле (см. MemoryGraph.apply_reward).

    prediction_error — то самое "неожиданно похвалили": именно эта
    величина, а не сама валентность, управляет и темпом закрепления, и
    смещением будущего выбора.
    """
    node_id: int
    valence: float
    expected: float
    prediction_error: float
    new_expectation: float


@dataclass
class SupersededNode:
    """
    Воспоминание, вытесненное более новой версией того же факта
    (см. MemoryGraph.find_superseded).

    word_overlap хранится для отладки: по нему видно, почему узел сочли
    другой версией, а не повтором.
    """
    id: int
    context: str
    similarity: float
    word_overlap: float


@dataclass
class KnownWord:
    """
    Освоенное слово, найденное во входящем сообщении (см.
    MemoryGraph.get_mastered_words_in). id нужен, чтобы подкрепление
    (Cortex.apply_feedback) могло усилить именно те СЛОВА, которые бот
    употребил удачно — раньше в контур подкрепления попадали только слоги
    лепета.
    """
    id: int
    text: str
    weight: float
    reward_expectation: float = 0.0
    # По чему организм на самом деле выбирает, что произнести: освоенность
    # плюс склонность к тому, за что хвалили (см. get_mastered_words_in)
    preference: float = 0.0


@dataclass
class AssociatedNode:
    """Узел, подтянутый через Spreading Activation (ассоциативное ребро)."""
    id: int
    context: str
    response: str
    weight: float
    edge_weight: float
    activation_score: float
    source_node_id: int


@dataclass
class ActivationTrace:
    """Одна ассоциативная активация: узел A привёл к подтягиванию узла B."""
    source_id: int
    target_id: int
    edge_weight: float
    activation_score: float




@dataclass
class HubCluster:
    """
    Кластер типа 'звезда вокруг хаба' (Hub-and-Spoke) — доминантный узел
    (hub) и его сильно связанные соседи (spokes), кандидаты на
    семантическую консолидацию во время фазы сна.
    """
    hub_id: int
    hub_context: str
    hub_response: str
    hub_weight: float
    spoke_ids: List[int]
    spoke_contexts: List[str]
    spoke_responses: List[str]
    spoke_weights: List[float]
    edge_weights: List[float]


@dataclass
class PruningReport:
    """Результат синаптического прунинга (Pruning & Edge Cleaning)."""
    edges_pruned: int
    orphan_nodes_pruned: int




class MemoryGraph:
    """
    Высокоуровневый интерфейс к графу долгосрочной памяти (LTM).

    Использование:
        graph = MemoryGraph()
        graph.save_connection("привет", "привет, как дела?", weight=0.9)
        matches = graph.search("как у тебя дела")
        graph.apply_decay()
        result = graph.consolidate_from_stm(stm_entries, timestamp=brain_time)
    """

    def __init__(
        self,
        db: Optional[Database] = None,
        settings: Optional[MemorySettings] = None,
    ):
        # Параметры приходят извне: ядро не должно знать про глобальный
        # config приложения, иначе его нельзя вынести в отдельный пакет.
        # Умолчания в MemorySettings — те же значения, что откалиброваны
        # замерами, поэтому MemoryGraph() без аргументов ведёт себя как
        # раньше.
        self.settings = settings or MemorySettings()
        self.db = db or Database(settings=self.settings)
        self.last_activation_traces: List[ActivationTrace] = []

    # ----------------------------------------------------------------------
    # SELF-MODEL & USER-MODEL — инициализация мета-узлов (Итерация 15)
    # ----------------------------------------------------------------------


    def get_or_create_brain_epoch(self, now: Optional[float] = None) -> float:
        """
        Точка отсчёта субъективного времени, ПЕРЕЖИВАЮЩАЯ перезагрузку.

        Без неё BrainSession при каждом запуске ставил brain_time =
        time.time(), и часы прыгали назад относительно last_decayed_at,
        сохранённых разогнанной шкалой. Забывание после этого молча
        выключалось (в _decay_nodes стоит `if dt <= 0: continue`) — после
        разговора на 100 сообщений ещё на 2.3 часа.

        Хранится тем же механизмом мета-узлов, что и last_sleep_marker:
        отдельная таблица ради одного числа не нужна.
        """
        row = self.db.get_meta_node("brain_epoch")
        if row is not None:
            try:
                return float(row["context"])
            except (TypeError, ValueError):
                logger.warning("[BRAIN EPOCH] Повреждённое значение — ставлю заново")

        # now передаёт приложение — тем же источником времени, который
        # получат часы. Иначе эпоха берётся из настоящих time.time() и
        # отличается от запуска к запуску даже там, где всё остальное
        # зафиксировано: у стенда расходились ответы уже на первом
        # сообщении при побитово одинаковом графе и состоянии генератора.
        epoch = now if now is not None else time.time()
        self.db.upsert_meta_node(
            node_type="brain_epoch", content=str(epoch), weight=1.0, timestamp=epoch,
        )
        logger.info("[BRAIN EPOCH] Точка отсчёта субъективного времени: %.0f", epoch)
        return epoch


    def get_user_model_content(self) -> str:
        """Возвращает текущий текст User-Model (fallback на config-дефолт)."""
        row = self.db.get_meta_node("user_model")
        return row["context"] if row is not None else self.settings.default_user_model



    # ----------------------------------------------------------------------
    # CONCEPT EXTRACTION — семантическая концептуализация (Итерация 16)
    # ----------------------------------------------------------------------

    def create_concept_node(
        self,
        name: str,
        definition: str,
        source_node_id: Optional[int] = None,
        timestamp: Optional[float] = None,
    ) -> int:
        """
        Создаёт (или обновляет, если понятие уже известно) concept-узел и
        выполняет полное связывание в графе знаний (Concept Graph Linking):

            1. Узел сохраняется через upsert_concept_node (node_type='concept').
            2. Автоматически создаётся/укрепляется ребро concept <-> USER_NODE
               (источник знания — Юзер объяснил это понятие).
            3. Если найдены семантически похожие существующие узлы (через
               обычный keyword+fuzzy search()), прокладываются начальные
               ассоциативные рёбра concept <-> similar_node (до
               CONCEPT_MAX_SIMILAR_LINKS штук).
            4. Если передан source_node_id (например, узел эпизода/сообщения,
               из которого извлекли понятие), концепт также связывается с ним.

        Возвращает id concept-узла.
        """
        ts = timestamp if timestamp is not None else time.time()
        normalized_name = name.strip()

        concept_node_id, was_created = self.db.upsert_concept_node(
            name=normalized_name,
            definition=definition.strip(),
            weight=self.settings.concept_node_weight,
            timestamp=ts,
        )

        # --- Связь с USER_MODEL (источник знания) ---
        user_row = self.db.get_meta_node("user_model")
        if user_row is not None:
            self.connect_nodes(
                concept_node_id,
                user_row["id"],
                weight_boost=self.settings.concept_user_edge_weight,
                timestamp=ts,
            )

        # --- Связь с исходным узлом-источником (если передан) ---
        if source_node_id is not None:
            self.connect_nodes(
                concept_node_id,
                source_node_id,
                weight_boost=self.settings.edge_initial_weight,
                timestamp=ts,
            )

        # --- Семантическое связывание с похожими существующими узлами ---
        if was_created:
            self._link_concept_to_similar_nodes(concept_node_id, normalized_name, definition, ts)

        logger.info(
            "[CONCEPT EXTRACTED] Узел '%s' (type=concept) сохранен и связан с User-Model.",
            normalized_name,
        )

        return concept_node_id

    def _link_concept_to_similar_nodes(
        self,
        concept_node_id: int,
        name: str,
        definition: str,
        timestamp: float,
    ) -> None:
        """
        Ищет узлы в графе, семантически перекликающиеся с новым понятием
        (по имени + определению), и прокладывает начальные ассоциативные
        рёбра — до CONCEPT_MAX_SIMILAR_LINKS штук, исключая сам концепт.
        """
        query_text = f"{name} {definition}"
        matches = self.search(
            query_text,
            threshold=self.settings.concept_similarity_link_threshold,
            top_k=self.settings.concept_max_similar_links + 1,  # +1 запас на случай самосовпадения
            timestamp=timestamp,
            with_associations=False,
        )

        linked_count = 0
        for match in matches:
            if match.id == concept_node_id:
                continue
            if linked_count >= self.settings.concept_max_similar_links:
                break

            self.connect_nodes(
                concept_node_id,
                match.id,
                weight_boost=self.settings.concept_similarity_edge_weight,
                timestamp=timestamp,
            )
            linked_count += 1

        if linked_count:
            logger.info(
                "[CONCEPT LINKED] Узел concept_id=%s связан с %d похожими узлами",
                concept_node_id, linked_count,
            )

    # ----------------------------------------------------------------------
    # LEXICAL ACQUISITION — освоение языка "с нуля" (Итерация N+1)
    # ----------------------------------------------------------------------

    def process_language_input(
        self,
        text: str,
        timestamp: Optional[float] = None,
    ) -> LexicalProcessingResult:
        """
        Побочный процесс (не блокирует основной ответ): разбирает входящий
        текст на слова и слоги, накапливая примитивный "словарный запас"
        цифрового ребёнка независимо от того, был MEMORY HIT или MISS.

        Для каждого слова:
            1. upsert word-узла (node_type='word') — частота появления
               повышает вес (усвоенность слова).
            2. Слово разбивается на слоги, каждый слог upsert-ится как
               syllable-узел (node_type='syllable').
            3. Слог связывается ребром с "родительским" словом
               (SYLLABLE_WORD_EDGE_WEIGHT).
        Соседние слова в предложении связываются рёбрами со-встречаемости
        (WORD_COOCCURRENCE_EDGE_WEIGHT) — примитивная "грамматика по
        смежности".

        Если LEXICAL_ACQUISITION_ENABLED=False — сразу возвращает нулевой
        результат без обращений к БД.
        """
        if not self.settings.lexical_acquisition_enabled or not text or not text.strip():
            return LexicalProcessingResult(0, 0, 0, 0)

        ts = timestamp if timestamp is not None else time.time()

        # Та же токенизация, что и в compute_surprise — см. комментарий там:
        # организм обязан удивляться ровно тем единицам, которым учится.
        tokens = self._tokenize_for_lexicon(text)

        if not tokens:
            return LexicalProcessingResult(0, 0, 0, 0)

        words_processed = 0
        syllables_processed = 0
        new_words = 0
        new_syllables = 0
        previous_word_id: Optional[int] = None

        for token in tokens:
            word_id, word_was_created = self.db.upsert_lexical_node(
                node_type="word",
                text=token,
                initial_weight=self.settings.word_node_initial_weight,
                reinforce_step=self.settings.word_node_reinforce_step,
                timestamp=ts,
            )
            words_processed += 1
            new_words += 1 if word_was_created else 0

            for syllable in self._split_into_syllables(token):
                syllable_id, syll_was_created = self.db.upsert_lexical_node(
                    node_type="syllable",
                    text=syllable,
                    initial_weight=self.settings.syllable_node_initial_weight,
                    reinforce_step=self.settings.syllable_node_reinforce_step,
                    timestamp=ts,
                )
                syllables_processed += 1
                new_syllables += 1 if syll_was_created else 0

                self.connect_nodes(
                    syllable_id, word_id,
                    weight_boost=self.settings.syllable_word_edge_weight,
                    timestamp=ts,
                )

            if previous_word_id is not None:
                self.connect_nodes(
                    previous_word_id, word_id,
                    weight_boost=self.settings.word_cooccurrence_edge_weight,
                    timestamp=ts,
                )
            previous_word_id = word_id

        logger.debug(
            "[LEXICAL ACQUISITION] words=%d (new=%d) syllables=%d (new=%d) text=%r",
            words_processed, new_words, syllables_processed, new_syllables, text[:40],
        )

        return LexicalProcessingResult(
            words_processed=words_processed,
            syllables_processed=syllables_processed,
            new_words=new_words,
            new_syllables=new_syllables,
        )

    # ----------------------------------------------------------------------
    # SURPRISE — собственная ошибка предсказания организма
    # ----------------------------------------------------------------------

    def compute_surprise(self, text: str) -> SurpriseResult:
        """
        Считает, насколько входящий текст НЕОЖИДАНЕН для этого конкретного
        организма — по его собственному накопленному графу языка.

        Раньше эту роль играла энтропия Шеннона по символам
        (Cortex.calculate_perplexity): она измеряла свойство строки и была
        полностью слепа к опыту — пустой мозг и мозг после 50 повторений
        фразы выдавали одинаковое число. Спайк-гейт, уверенность,
        структурная консолидация и любопытство — все четыре механизма
        управлялись величиной, которая никогда не менялась от обучения.

        Две составляющие ошибки предсказания:

            лексическая  — знакомы ли САМИ СЛОВА. Знакомость слова растёт
                           с его весом (частота = освоенность) и насыщается
                           на VOCABULARY_MASTERY_MIN_WEIGHT.
            структурная  — привычны ли СОЧЕТАНИЯ соседних слов. Знакомость
                           пары растёт с весом ребра со-встречаемости и
                           насыщается на EDGE_ACTIVATION_THRESHOLD.

        ВАЖНОЕ ОГРАНИЧЕНИЕ: рёбра хранятся ненаправленно — upsert_edge
        нормализует пару по возрастанию id. Поэтому структурная часть
        отвечает на вопрос «встречались ли эти слова рядом», а НЕ «идёт ли
        одно за другим». Это не языковая модель и называть её так нельзя;
        для ошибки предсказания такого разрешения достаточно.

        Токенизация намеренно совпадает с process_language_input — организм
        обязан удивляться ровно тем единицам, которым он учится.

        Краевые случаи:
            пустой текст / нет токенов -> 0.0 (удивляться нечему)
            один токен (пар нет)       -> только лексическая часть
            пустой граф                -> 1.0 (новорождённому всё ново)
        """
        tokens = self._tokenize_for_lexicon(text)
        if not tokens:
            return SurpriseResult(0.0, 0.0, 0.0, 0, 0, 0, 0)

        # --- Лексическая новизна: знакомы ли сами слова ---
        rows = self.db.get_lexical_nodes_by_texts("word", list(set(tokens)))
        known = {row["context"]: (row["id"], row["weight"]) for row in rows}

        mastery = max(1e-9, self.settings.vocabulary_mastery_min_weight)
        familiarities = [
            min(1.0, known[t][1] / mastery) if t in known else 0.0
            for t in tokens
        ]
        lexical_surprise = 1.0 - (sum(familiarities) / len(familiarities))

        # --- Структурная новизна: привычны ли сочетания соседних слов ---
        token_ids = [known[t][0] for t in tokens if t in known]
        edge_weights = {}
        for edge in self.db.get_edges_between(list(set(token_ids))):
            # Пара хранится ненаправленно -> кладём в оба порядка, чтобы
            # искать по фактическому порядку слов во входящем тексте.
            a, b, w = edge["node_from"], edge["node_to"], edge["weight"]
            edge_weights[(a, b)] = w
            edge_weights[(b, a)] = w

        activation = max(1e-9, self.settings.edge_activation_threshold)
        pair_familiarities: List[float] = []
        for left, right in zip(tokens, tokens[1:]):
            if left in known and right in known:
                weight = edge_weights.get((known[left][0], known[right][0]), 0.0)
                pair_familiarities.append(min(1.0, weight / activation))
            else:
                # Хотя бы одно слово пары незнакомо -> сочетание тем более
                pair_familiarities.append(0.0)

        known_words = sum(1 for f in familiarities if f > 0.0)
        known_pairs = sum(1 for f in pair_familiarities if f > 0.0)

        if not pair_familiarities:
            # Один токен: структурной информации нет вообще, поэтому итог
            # определяется только лексикой (перенормировка вместо того,
            # чтобы фиктивно засчитывать структурное удивление как 0 или 1).
            total = lexical_surprise
            structural_surprise = 0.0
        else:
            structural_surprise = 1.0 - (sum(pair_familiarities) / len(pair_familiarities))
            total = (
                self.settings.surprise_lexical_weight * lexical_surprise
                + self.settings.surprise_structural_weight * structural_surprise
            )
            weight_sum = self.settings.surprise_lexical_weight + self.settings.surprise_structural_weight
            if weight_sum > 0:
                total /= weight_sum

        total = max(0.0, min(1.0, total))

        logger.debug(
            "[SURPRISE] text=%r total=%.3f (lex=%.3f structural=%.3f) "
            "known_words=%d/%d known_pairs=%d/%d",
            text[:40], total, lexical_surprise, structural_surprise,
            known_words, len(tokens), known_pairs, len(pair_familiarities),
        )

        return SurpriseResult(
            total=total,
            lexical=lexical_surprise,
            structural=structural_surprise,
            known_words=known_words,
            total_words=len(tokens),
            known_pairs=known_pairs,
            total_pairs=len(pair_familiarities),
        )

    def _tokenize_for_lexicon(self, text: str) -> List[str]:
        """
        Единая токенизация для лексического слоя. Используется И при
        обучении (process_language_input), И при расчёте удивления
        (compute_surprise) — организм должен удивляться ровно тем единицам,
        которые он потом запоминает, иначе измеряется не то, чему учатся.
        """
        if not text or not text.strip():
            return []
        return [
            w.lower() for w in WORD_PATTERN.findall(text)
            if len(w) >= self.settings.lexical_min_token_length
        ][: self.settings.lexical_max_tokens_per_input]

    @staticmethod
    def _split_into_syllables(word: str) -> List[str]:
        """
        Примитивная слоговая сегментация: слог накапливается посимвольно
        и "закрывается" на первой встреченной гласной; хвостовые согласные
        в конце слова прилипают к последнему найденному слогу. Не претендует
        на лингвистическую точность — достаточно для babbling-подсистемы.
        """
        syllables: List[str] = []
        current = ""

        for ch in word:
            current += ch
            if ch in VOWELS:
                syllables.append(current)
                current = ""

        if current:
            if syllables:
                syllables[-1] += current
            else:
                syllables.append(current)

        return syllables if syllables else [word]

    def get_vocabulary_size(self) -> int:
        """
        Возвращает количество ЗАКРЕПЛЁННЫХ (усвоенных) слов — word-узлов
        с weight >= VOCABULARY_MASTERY_MIN_WEIGHT. Слово, услышанное один
        раз, создаёт узел с низким начальным весом и НЕ считается здесь,
        пока не будет повторено пользователем ещё несколько раз (см.
        WORD_NODE_INITIAL_WEIGHT/WORD_NODE_REINFORCE_STEP). Используется
        для гейтинга стадии речевого развития (Cortex._resolve_speech_stage)
        и для пользовательского /status — то есть отражает то, что бот
        реально ОСВОИЛ, а не всё, что когда-либо пролетело через него.
        """
        return self.db.count_mastered_words(self.settings.vocabulary_mastery_min_weight)

    def get_exposed_vocabulary_size(self) -> int:
        """
        Общее количество РАЗЛИЧНЫХ слов, которые бот хотя бы раз услышал,
        независимо от закрепления — "пассивный" словарь. Только для
        статистики/отладки (например, разница между этим числом и
        get_vocabulary_size() показывает, сколько слов ещё "на подходе" к
        усвоению). НЕ используется для гейтинга речевых стадий.
        """
        return self.db.count_nodes_by_type("word")

    def get_mastered_words_in(self, text: str) -> List[KnownWord]:
        """
        Какие слова ВХОДЯЩЕГО сообщения организм действительно освоил —
        в порядке появления в тексте.

        Это первый случай, когда выученный словарь влияет на то, ЧТО бот
        говорит, а не только на счётчик, разрешающий говорить. Раньше
        знание слова existed исключительно как число: бот мог знать
        "привет" лучше всех своих слов (вес 0.747) и всё равно отвечать
        на приветствие случайными слогами, потому что до генерации
        доходил только len(словаря).

        Освоенным считается слово с весом >= VOCABULARY_MASTERY_MIN_WEIGHT,
        то есть та же планка, что и в get_vocabulary_size — иначе бот
        произносил бы слова, которые сам же не считает выученными.
        """
        return self._words_in(text, mastered=True)

    def get_emerging_words_in(self, text: str) -> List[KnownWord]:
        """
        Слова входящей фразы, которые организм УЖЕ СЛЫШАЛ, но ещё НЕ
        ОСВОИЛ — его зона ближайшего развития.

        Это кандидаты на ИССЛЕДОВАНИЕ. До сих пор в архитектуре не было
        вообще никакого механизма попробовать неосвоенное: организм только
        эксплуатировал уже выученное, а всё остальное превращал в лепет.
        Чисто эксплуатирующая система не развивается — она сходится и
        застывает.

        Именно такие слова, а не совсем незнакомые: пробовать то, что
        далеко за пределами текущей компетенции, бессмысленно — попытка
        провалится и ничему не научит. Учение происходит на границе
        освоенного.
        """
        return self._words_in(text, mastered=False)

    def _words_in(self, text: str, mastered: bool) -> List[KnownWord]:
        """
        Общая выборка слов входящей фразы по порогу освоенности.
        mastered=True  -> вес >= VOCABULARY_MASTERY_MIN_WEIGHT (свои слова)
        mastered=False -> вес <  порога (услышанные, но ещё не закреплённые)
        """
        tokens = self._tokenize_for_lexicon(text)
        if not tokens:
            return []

        threshold = self.settings.vocabulary_mastery_min_weight
        rows = self.db.get_lexical_nodes_by_texts("word", list(set(tokens)))
        known = {
            row["context"]: (row["id"], row["weight"], row["reward_expectation"] or 0.0)
            for row in rows
            if (row["weight"] >= threshold) is mastered
        }

        result: List[KnownWord] = []
        seen: Set[str] = set()
        for token in tokens:
            if token in known and token not in seen:
                seen.add(token)
                node_id, weight, expectation = known[token]
                result.append(
                    KnownWord(
                        id=node_id,
                        text=token,
                        weight=weight,
                        reward_expectation=expectation,
                        # Предпочтение = освоенность + склонность к тому, за
                        # что хвалили. Вес остаётся главным критерием, иначе
                        # организм начнёт говорить редкими, но однажды
                        # похваленными словами вместо тех, которыми владеет.
                        preference=weight + expectation * self.settings.reward_preference_weight,
                    )
                )
        return result

    def get_top_words(self, limit: int = 8) -> List["tuple[str, float]"]:
        """
        Самые освоенные слова (текст, вес) — для команды /status.
        Показывает учителю, что реально закрепилось в языке бота.
        """
        rows = self.db.get_top_nodes_by_type("word", limit=limit)
        return [(row["context"], row["weight"]) for row in rows]

    def get_known_syllables(self, limit: Optional[int] = None) -> List[KnownSyllable]:
        """
        Возвращает пул известных слогов (id, text, weight) — кандидатов для
        ВЗВЕШЕННОГО выбора в babbling-подсистеме (InstinctSystem.
        generate_babble_response). Пул случайный на уровне БД (ORDER BY
        RANDOM()), но каждый элемент несёт свой реальный вес — взвешенная
        выборка происходит уже в InstinctSystem, не здесь.

        limit по умолчанию берётся из self.settings.babbling_syllable_pool_size.
        """
        effective_limit = limit if limit is not None else self.settings.babbling_syllable_pool_size
        rows = self.db.get_random_nodes_by_type("syllable", limit=effective_limit)
        return [
            KnownSyllable(id=row["id"], text=row["context"], weight=row["weight"])
            for row in rows
        ]


    # ----------------------------------------------------------------------
    # 1. Сохранение новой связи
    # ----------------------------------------------------------------------

    def find_superseded(
        self,
        text: str,
        exclude_id: Optional[int] = None,
        explicit_correction: bool = False,
    ) -> List["SupersededNode"]:
        """
        Какие существующие воспоминания ВЫТЕСНЯЕТ новое.

        Без этого память копила взаимоисключающие факты и отдавала
        случайный: "мою собаку зовут Рекс", позже "мою собаку зовут Бобик" —
        оба узла равноправны, причём устаревший находился ЛУЧШЕ (0.906
        против 0.875), потому что порядок решает сходство строк, а не время.

        Признак вытеснения — два условия сразу:
          1. высокая СЕМАНТИЧЕСКАЯ близость: речь об одном и том же;
          2. НЕПОЛНОЕ словесное совпадение: значит это другая версия, а не
             повтор. Чистый повтор обязан просто подкреплять узел.

        explicit_correction — пользователь явно поправил ("нет",
        "неправильно"). Это сильное свидетельство, поэтому порог темы
        снижается: без маркера мы осторожничаем, с маркером доверяем.

        Порог намеренно высокий. Ошибиться в сторону "пропустил
        противоречие" дешевле, чем ослабить независимое воспоминание —
        хотя и второе не катастрофа, потому что узлы ослабляются, а не
        удаляются (см. supersede_node).
        """
        query_vector = embeddings.encode(text)
        if query_vector is None:
            # Без семантики отличить "другую версию" от "другой темы"
            # нечем: строковое сходство одинаково высоко и для "зовут
            # Рекс"/"зовут Бобик", и для "зовут Рекс"/"зовут Рекс".
            return []

        threshold = self.settings.contradiction_topic_threshold
        if explicit_correction:
            threshold -= self.settings.contradiction_correction_relief

        new_words = self._extract_keywords(text.lower())
        found: List[SupersededNode] = []

        for row in self.db.fetch_searchable_nodes():
            if row["id"] == exclude_id or row["is_meta"]:
                continue

            # Сравниваем ТОЛЬКО реплики пользователя, без ответов бота.
            # _node_vector считает вектор по паре "вопрос + ответ", и это
            # правильно для поиска, но не здесь: факт живёт в том, что
            # сказал ЧЕЛОВЕК, а реплика бота ("запомнил", "ага") — шум,
            # который сдвигает вектор и решает исход сравнения.
            similarity = embeddings.cosine(
                query_vector, embeddings.encode(row["context"] or "")
            )
            if similarity < threshold:
                continue

            old_words = self._extract_keywords((row["context"] or "").lower())
            overlap = self._keyword_overlap(new_words, old_words)
            if overlap >= self.settings.contradiction_repeat_threshold:
                continue  # это повтор, а не новая версия

            found.append(
                SupersededNode(
                    id=row["id"],
                    context=row["context"],
                    similarity=similarity,
                    word_overlap=overlap,
                )
            )

        return found

    def supersede_node(self, node_id: int, timestamp: Optional[float] = None) -> None:
        """
        Помечает воспоминание как вытесненное: снижает вес и СБРАСЫВАЕТ
        стабильность, то есть возвращает узел в разряд забываемых.

        Именно ослабление, а не удаление. Если факт на самом деле остался
        верным (сработали ложно — "у меня есть кошка" против "у меня есть
        собака"), пользователь упомянет его снова, узел получит touch_node,
        и стабильность отрастёт. Удаление было бы необратимым, а здесь
        ошибка стоит дёшево и исправляется сама.
        """
        row = self.db.get_node(node_id)
        if row is None:
            return

        new_weight = max(0.0, row["weight"] - self.settings.contradiction_weight_penalty)
        stability = (row["stability"] or self.settings.stability_initial)
        new_stability = max(
            self.settings.stability_initial, stability * self.settings.contradiction_stability_factor
        )

        self.db.update_weight(node_id, new_weight)
        self.db.update_stability(node_id, new_stability)

        logger.info(
            "[SUPERSEDED] Узел %s вытеснен новой версией: вес %.3f -> %.3f, "
            "стабильность %.1f -> %.1f",
            node_id, row["weight"], new_weight, stability, new_stability,
        )

    def save_connection(
        self,
        context: str,
        response: str,
        weight: Optional[float] = None,
        timestamp: Optional[float] = None,
        explicit_correction: bool = False,
    ) -> int:
        """
        Сохраняет новую связь context -> response с начальным весом.

        explicit_correction — пользователь явно поправил ("нет",
        "неправильно"). Снижает порог вытеснения устаревших версий.
        """
        initial_weight = weight if weight is not None else self.settings.base_plasticity_threshold

        node_id = self.db.insert_node(
            context=context,
            response=response,
            weight=initial_weight,
            timestamp=timestamp,
        )

        # Новая версия факта вытесняет старую: иначе память копит
        # взаимоисключающие узлы и отдаёт случайный из них.
        for stale in self.find_superseded(
            context, exclude_id=node_id, explicit_correction=explicit_correction
        ):
            logger.info(
                "[CONTRADICTION] %r вытесняет %r (близость %.2f, общих слов %.2f)",
                context[:40], stale.context[:40], stale.similarity, stale.word_overlap,
            )
            self.supersede_node(stale.id, timestamp=timestamp)

        logger.info(
            "[SPIKE DETECTED] Новая связь сохранена id=%s weight=%.3f",
            node_id, initial_weight,
        )
        return node_id

    # ----------------------------------------------------------------------
    # 2. Поиск похожего контекста (ключевые слова + нечёткое сходство)
    # ----------------------------------------------------------------------

    def search(
        self,
        query: str,
        threshold: Optional[float] = None,
        top_k: int = 1,
        timestamp: Optional[float] = None,
        with_associations: bool = True,
    ) -> List[MemoryMatch]:
        """
        Основной метод поиска по графу памяти (keyword overlap + fuzzy).

        Если with_associations=True, после отбора top_k узлов выполняется
        Spreading Activation: для каждого найденного узла подтягиваются
        смежные узлы через рёбра с weight >= EDGE_ACTIVATION_THRESHOLD.
        Ассоциативные узлы подмешиваются в результат как MemoryMatch с
        similarity = activation_score (ослабленной относительно источника).
        """
        effective_threshold = threshold if threshold is not None else self.settings.memory_search_threshold

        query_normalized = query.strip().lower()
        if not query_normalized:
            return []

        query_keywords = self._extract_keywords(query_normalized)
        rows = self.db.fetch_searchable_nodes()

        scored: List[MemoryMatch] = []

        # Вектор запроса считается ОДИН раз на весь поиск. None означает,
        # что семантика недоступна (нет модели/библиотеки) — тогда работают
        # только строковые составляющие, как раньше.
        query_vector = embeddings.encode(query)

        for row in rows:
            context_normalized = row["context"].strip().lower()
            context_keywords = self._extract_keywords(context_normalized)

            keyword_score = self._keyword_overlap(query_keywords, context_keywords)
            fuzzy_score = self._compute_fuzzy_similarity(query_normalized, context_normalized)

            semantic_score = 0.0
            if query_vector is not None:
                semantic_score = max(
                    0.0, embeddings.cosine(query_vector, self._node_vector(row))
                )

            combined_score = (
                keyword_score * self.settings.memory_keyword_weight
                + fuzzy_score * self.settings.memory_fuzzy_weight
                + semantic_score * self.settings.memory_semantic_weight
                + row["weight"] * self.settings.memory_weight_influence
            )
            combined_score = min(1.0, combined_score)

            if combined_score >= effective_threshold:
                scored.append(
                    MemoryMatch(
                        id=row["id"],
                        context=row["context"],
                        response=row["response"],
                        weight=row["weight"],
                        similarity=combined_score,
                        created_at=row["created_at"],
                        last_accessed=row["last_accessed"],
                    )
                )

        scored.sort(key=lambda m: m.similarity, reverse=True)
        top_matches = scored[:top_k]

        for match in top_matches:
            self.touch_node(match.id, timestamp=timestamp)

        if top_matches:
            logger.info(
                "[MEMORY HIT] Найдено %d совпадений для %r (best score=%.3f, id=%s)",
                len(top_matches), query[:50], top_matches[0].similarity, top_matches[0].id,
            )
        else:
            logger.info("[MEMORY MISS] Совпадений не найдено для %r", query[:50])
            return top_matches

        # ------------------------------------------------------------------
        # SPREADING ACTIVATION (Multi-hop RAG)
        # ------------------------------------------------------------------
        self.last_activation_traces = []

        if with_associations:
            existing_ids = {m.id for m in top_matches}
            associative_extras: List[MemoryMatch] = []

            for source_match in top_matches:
                associated = self.get_associated_nodes(
                    source_match.id,
                    min_weight=self.settings.edge_activation_threshold,
                    limit=self.settings.edge_max_hop_nodes,
                    timestamp=timestamp,
                )

                for assoc in associated:
                    if assoc.id in existing_ids:
                        continue

                    activation_score = min(
                        1.0,
                        source_match.similarity * self.settings.edge_activation_decay * assoc.edge_weight,
                    )

                    logger.info(
                        "[ASSOCIATION] Node %s -> Node %s (edge_weight=%.2f, activation_score=%.3f)",
                        source_match.id, assoc.id, assoc.edge_weight, activation_score,
                    )

                    self.last_activation_traces.append(
                        ActivationTrace(
                            source_id=source_match.id,
                            target_id=assoc.id,
                            edge_weight=assoc.edge_weight,
                            activation_score=activation_score,
                        )
                    )

                    associative_extras.append(
                        MemoryMatch(
                            id=assoc.id,
                            context=assoc.context,
                            response=assoc.response,
                            weight=assoc.weight,
                            similarity=activation_score,
                            created_at=0.0,
                            last_accessed=0.0,
                        )
                    )
                    existing_ids.add(assoc.id)

            if associative_extras:
                top_matches = top_matches + associative_extras

        return top_matches

    def _node_vector(self, row):
        """
        Вектор смысла узла, с ЛЕНИВЫМ досчётом.

        Узлы, созданные до появления модели (или до этой правки вообще),
        приходят с embedding=NULL. Вместо разовой тяжёлой миграции всей
        базы вектор считается при первом же обращении к узлу и тут же
        сохраняется — дальше он просто читается.

        Смысл узла берётся из ОБЕИХ его половин: пользователь мог спросить
        одними словами, а суть оказаться в ответе бота.
        """
        vector = embeddings.from_blob(row["embedding"])
        if vector is not None:
            return vector

        text = f"{row['context'] or ''} {row['response'] or ''}".strip()
        vector = embeddings.encode(text)
        if vector is None:
            return None

        self.db.update_embedding(row["id"], embeddings.to_blob(vector))
        return vector

    def _extract_keywords(self, text: str) -> Set[str]:
        words = WORD_PATTERN.findall(text)
        return {
            w for w in words
            if len(w) >= self.settings.memory_min_keyword_length and w not in STOP_WORDS
        }

    @staticmethod
    def _keyword_overlap(query_keywords: Set[str], context_keywords: Set[str]) -> float:
        if not query_keywords or not context_keywords:
            return 0.0
        intersection = query_keywords & context_keywords
        smallest_set_size = min(len(query_keywords), len(context_keywords))
        return len(intersection) / smallest_set_size if smallest_set_size else 0.0

    @staticmethod
    def _compute_fuzzy_similarity(query: str, context: str) -> float:
        return SequenceMatcher(None, query, context).ratio()

    # ----------------------------------------------------------------------
    # 3. Обновление last_accessed при обращении
    # ----------------------------------------------------------------------

    def touch_node(self, node_id: int, timestamp: Optional[float] = None) -> None:
        ts = timestamp if timestamp is not None else time.time()
        self.db.update_last_accessed(node_id, timestamp=ts)
        logger.debug("[MEMORY TOUCHED] id=%s last_accessed обновлён (t=%.2f)", node_id, ts)

    # ----------------------------------------------------------------------
    # 3b. АССОЦИАТИВНЫЕ РЁБРА (Semantic Edges / Spreading Activation)
    # ----------------------------------------------------------------------

    def connect_nodes(
        self,
        node_from: int,
        node_to: int,
        weight_boost: Optional[float] = None,
        timestamp: Optional[float] = None,
    ) -> float:
        """
        Создаёт или укрепляет ассоциативное ребро между двумя узлами LTM.

        Используется в двух сценариях:
            1. Связь по контексту: узел A подтянулся из памяти (MEMORY HIT),
               а в процессе разговора создался/подкрепился узел B ->
               укрепляем ребро A -> B.
            2. Связь по со-активации: несколько узлов были задействованы
               в рамках одного окна STM -> растим вес рёбер между ними
               (см. reinforce_coactivation).

        Ребро игнорируется, если node_from == node_to (нет смысла в петле).
        Возвращает итоговый вес ребра.
        """
        if node_from is None or node_to is None or node_from == node_to:
            return 0.0

        # Защита от гонки: один из узлов мог быть удалён (например, во сне
        # low-weight syllable-узел попал под orphan pruning) в промежутке
        # между тем, как его id был запомнён (last_action_trace.node_ids
        # / связка из STM), и текущим вызовом. FOREIGN KEY на edges иначе
        # уронит вставку исключением — просто тихо пропускаем ребро.
        if self.db.get_node(node_from) is None or self.db.get_node(node_to) is None:
            logger.debug(
                "[ASSOCIATION SKIP] Узел %s или %s больше не существует (удалён) -> ребро не создано",
                node_from, node_to,
            )
            return 0.0

        boost = weight_boost if weight_boost is not None else self.settings.edge_boost_step
        ts = timestamp if timestamp is not None else time.time()

        new_weight = self.db.upsert_edge(
            node_from=node_from,
            node_to=node_to,
            weight_boost=boost,
            timestamp=ts,
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
        Связь по со-активации: если несколько узлов LTM были задействованы
        (touch/reinforce) в рамках ОДНОГО окна STM (обычно 8 сообщений),
        растим вес рёбер между КАЖДОЙ парой этих узлов.

        node_ids — список id узлов, активированных в текущем окне STM
        (дубликаты и None автоматически отфильтровываются).
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
            "[COACTIVATION] Усилены рёбра между узлами со-активации: %s",
            unique_ids,
        )

    def get_associated_nodes(
        self,
        node_id: int,
        min_weight: Optional[float] = None,
        limit: Optional[int] = None,
        timestamp: Optional[float] = None,
    ) -> List[AssociatedNode]:
        """
        Возвращает узлы, смежные с node_id через рёбра с weight >= min_weight,
        отсортированные по весу ребра (сильнейшие ассоциации первыми).

        Используется в search() для Spreading Activation (Multi-hop RAG).
        """
        effective_min_weight = min_weight if min_weight is not None else self.settings.edge_activation_threshold

        edge_rows = self.db.get_edges_for_node(node_id)
        strong_edges = [row for row in edge_rows if row["weight"] >= effective_min_weight]
        strong_edges.sort(key=lambda r: r["weight"], reverse=True)

        if limit is not None:
            strong_edges = strong_edges[:limit]

        results: List[AssociatedNode] = []
        for edge_row in strong_edges:
            neighbor_row = self.db.get_node(edge_row["neighbor_id"])
            if neighbor_row is None:
                continue

            results.append(
                AssociatedNode(
                    id=neighbor_row["id"],
                    context=neighbor_row["context"],
                    response=neighbor_row["response"],
                    weight=neighbor_row["weight"],
                    edge_weight=edge_row["weight"],
                    activation_score=edge_row["weight"],
                    source_node_id=node_id,
                )
            )
            # Обращение к ассоциативному узлу тоже считается "касанием" —
            # он вспомнился, хоть и не был напрямую найден по score.
            self.touch_node(neighbor_row["id"], timestamp=timestamp)

        return results

    # ----------------------------------------------------------------------
    # 4. Decay — угасание веса старых связей
    # ----------------------------------------------------------------------

    def apply_decay(self, now: Optional[float] = None) -> int:
        """
        Применяет экспоненциальное угасание веса ко всем узлам LTM, а также
        к рёбрам ассоциативного графа (Edge Decay). Редкие/неиспользуемые
        рёбра (weight < EDGE_FORGET_THRESHOLD) физически удаляются.
        """
        current_time = now if now is not None else time.time()

        decayed_count = self._decay_nodes(current_time)
        self._decay_edges(current_time)

        return decayed_count

    # Лексические узлы — инфраструктура языка, а не эпизоды разговора,
    # поэтому живут на своей (много более длинной) шкале времени.
    LEXICAL_NODE_TYPES = frozenset({"word", "syllable"})

    def _age_t0_for(self, node_type: Optional[str]) -> float:
        """
        Характерное время жизни узла для формулы decay. Словарь угасает
        по LEXICAL_AGE_T0 (~30 суток), всё остальное — по AGE_T0 (~1 час).

        Без этого разделения освоенное слово теряло статус за ночь, а за
        сутки с небольшим удалялось из БД — словарь не мог накопиться в
        принципе (см. комментарий у self.settings.lexical_age_t0).
        """
        if node_type in MemoryGraph.LEXICAL_NODE_TYPES:
            return self.settings.lexical_age_t0
        return self.settings.age_t0

    def _decay_nodes(self, current_time: float) -> int:
        """Экспоненциальное угасание веса узлов (старая логика apply_decay)."""
        rows = self.db.fetch_all_nodes()

        updates = []
        to_forget = []
        decayed_count = 0
        skipped_meta_count = 0

        for row in rows:
            if row["is_meta"]:
                skipped_meta_count += 1
                continue

            # Защита от NULL: если last_decayed_at не проставлен (пропущенный
            # путь создания/обновления узла) — используем last_accessed как
            # отсчётную точку, вместо падения с TypeError. На этом же проходе
            # last_decayed_at будет проставлен через updates[] ниже.
            last_decayed = row["last_decayed_at"]
            if last_decayed is None:
                last_decayed = row["last_accessed"]

            dt = current_time - last_decayed
            if dt <= 0:
                continue

            old_weight = row["weight"]
            # Эффективное время жизни = базовое для типа узла * стабильность.
            # Стабильность растёт при каждом вспоминании (см. Database.
            # update_last_accessed), поэтому востребованная память
            # сопротивляется времени, а невостребованная уходит быстро.
            stability = row["stability"] if row["stability"] else self.settings.stability_initial
            effective_t0 = self._age_t0_for(row["node_type"]) * max(1e-9, stability)
            decay_factor = math.exp(-self.settings.decay_rate * dt / effective_t0)
            new_weight = old_weight * decay_factor

            if new_weight < self.settings.forget_threshold:
                to_forget.append(row["id"])
            else:
                updates.append({
                    "id": row["id"],
                    "weight": new_weight,
                    "last_decayed_at": current_time,
                })
                decayed_count += 1

        if updates:
            self.db.bulk_update_weights(updates)
            logger.info(
                "[DECAY APPLIED] Обновлено весов: %d узлов (пропущено meta: %d)",
                len(updates), skipped_meta_count,
            )

        for node_id in to_forget:
            self.db.delete_node(node_id)

        if to_forget:
            logger.info("[MEMORY FORGOTTEN] Удалено узлов (вес < FORGET_THRESHOLD): %d", len(to_forget))

        return decayed_count

    def _decay_edges(self, current_time: float) -> int:
        """
        Экспоненциальное угасание веса рёбер (Edge Decay). Рёбра угасают
        быстрее узлов (EDGE_DECAY_RATE обычно > DECAY_RATE), имитируя то,
        что ассоциации между воспоминаниями более хрупкие, чем сами
        воспоминания. Рёбра ниже EDGE_FORGET_THRESHOLD удаляются физически.
        """
        edges = self.db.fetch_all_edges()

        updates = []
        to_forget = []
        decayed_count = 0

        for edge in edges:
            # Защита от NULL — см. комментарий в _decay_nodes.
            last_decayed = edge["last_decayed_at"]
            if last_decayed is None:
                last_decayed = edge["last_activated"]

            dt = current_time - last_decayed
            if dt <= 0:
                continue

            old_weight = edge["weight"]
            decay_factor = math.exp(-self.settings.edge_decay_rate * dt / self.settings.age_t0)
            new_weight = old_weight * decay_factor

            if new_weight < self.settings.edge_forget_threshold:
                to_forget.append(edge["id"])
            else:
                updates.append({
                    "id": edge["id"],
                    "weight": new_weight,
                    "last_decayed_at": current_time,
                })
                decayed_count += 1

        if updates:
            self.db.bulk_update_edge_weights(updates)
            logger.info(
                "[EDGE DECAY APPLIED] Обновлено весов: %d рёбер (t=%.2f)",
                len(updates), current_time,
            )

        for edge_id in to_forget:
            self.db.delete_edge(edge_id)

        if to_forget:
            logger.info(
                "[EDGE FORGOTTEN] Удалено рёбер (вес < EDGE_FORGET_THRESHOLD): %d",
                len(to_forget),
            )

        return decayed_count

    # ----------------------------------------------------------------------
    # 4b. SLEEP CYCLE — синаптический прунинг (Pruning & Edge Cleaning)
    # ----------------------------------------------------------------------

    def prune_weak_edges(self, min_weight: Optional[float] = None) -> int:
        """
        Явный (не decay-based) прунинг рёбер: физически удаляет ВСЕ рёбра
        с weight < min_weight. Используется фазой сна (SleepCycle), в
        отличие от _decay_edges(), который сначала уменьшает вес, а
        удаляет только то, что упало ниже порога В ЭТОМ вызове.

        Возвращает количество удалённых рёбер.
        """
        threshold = min_weight if min_weight is not None else self.settings.edge_forget_threshold
        deleted = self.db.delete_edges_below_weight(threshold)
        return deleted

    def prune_orphan_nodes(
        self,
        min_edge_weight: Optional[float] = None,
        max_node_weight: Optional[float] = None,
    ) -> int:
        """
        Удаляет "осиротевшие" узлы: не имеющие ни одного сильного ребра
        (weight >= min_edge_weight) И при этом сами по себе слабые
        (node.weight < max_node_weight). Сильные изолированные воспоминания
        не трогаем — они самодостаточны, даже если ни с чем не связаны.

        Возвращает количество удалённых узлов.
        """
        effective_edge_weight = (
            min_edge_weight if min_edge_weight is not None else self.settings.edge_activation_threshold
        )
        effective_node_weight = (
            max_node_weight if max_node_weight is not None else self.settings.sleep_orphan_weight_threshold
        )

        orphans = self.db.get_orphan_nodes(
            min_edge_weight=effective_edge_weight,
            max_node_weight=effective_node_weight,
        )

        for orphan in orphans:
            self.db.delete_node(orphan["id"])

        if orphans:
            logger.info(
                "[SLEEP PRUNING] Удалено осиротевших узлов (weight < %.2f, без рёбер >= %.2f): %d",
                effective_node_weight, effective_edge_weight, len(orphans),
            )

        return len(orphans)

    def run_synaptic_pruning(self) -> PruningReport:
        """
        Полный цикл синаптического прунинга: сначала срезаем слабые рёбра,
        ПОТОМ ищем осиротевшие узлы (порядок важен — удаление слабых рёбер
        может "осиротить" узлы, которые держались только на них).
        """
        edges_pruned = self.prune_weak_edges()
        orphan_nodes_pruned = self.prune_orphan_nodes()

        return PruningReport(
            edges_pruned=edges_pruned,
            orphan_nodes_pruned=orphan_nodes_pruned,
        )

    # ----------------------------------------------------------------------
    # 4c. SLEEP CYCLE — кластеризация Hub-and-Spoke
    # ----------------------------------------------------------------------

    def find_hub_clusters(
        self,
        min_edge_weight: Optional[float] = None,
        min_spokes: Optional[float] = None,
        max_spokes: Optional[float] = None,
        limit: int = 1,
        timestamp: Optional[float] = None,
    ) -> List[HubCluster]:
        """
        Ищет кластеры типа 'звезда вокруг хаба': доминантный узел (hub) с
        наибольшей суммой весов сильных рёбер (hub_score), и его top-N
        сильнейших соседей (spokes) как кандидатов на семантическую
        консолидацию (Abstract Node Generation) во время фазы сна.

        Возвращает до `limit` кластеров, отсортированных по hub_score.
        Кластер попадает в результат только если у хаба есть >= min_spokes
        сильных соседей (иначе это не "кластер", а просто пара узлов).
        """
        effective_edge_weight = (
            min_edge_weight if min_edge_weight is not None else self.settings.sleep_hub_min_edge_weight
        )
        effective_min_spokes = (
            min_spokes if min_spokes is not None else self.settings.sleep_min_cluster_spokes
        )
        effective_max_spokes = (
            max_spokes if max_spokes is not None else self.settings.sleep_max_cluster_spokes
        )

        hub_rows = self.db.get_hub_candidates(min_edge_weight=effective_edge_weight)

        clusters: List[HubCluster] = []
        used_node_ids: set = set()

        for hub_row in hub_rows:
            if len(clusters) >= limit:
                break

            hub_id = hub_row["id"]
            if hub_id in used_node_ids:
                continue

            associated = self.get_associated_nodes(
                hub_id,
                min_weight=effective_edge_weight,
                limit=effective_max_spokes,
                timestamp=timestamp,
            )

            # Отсекаем спутников, которые уже "использованы" в другом кластере
            # этого же прогона (чтобы не консолидировать один узел дважды)
            available_spokes = [a for a in associated if a.id not in used_node_ids]

            if len(available_spokes) < effective_min_spokes:
                continue

            cluster = HubCluster(
                hub_id=hub_id,
                hub_context=hub_row["context"],
                hub_response=hub_row["response"],
                hub_weight=hub_row["weight"],
                spoke_ids=[a.id for a in available_spokes],
                spoke_contexts=[a.context for a in available_spokes],
                spoke_responses=[a.response for a in available_spokes],
                spoke_weights=[a.weight for a in available_spokes],
                edge_weights=[a.edge_weight for a in available_spokes],
            )

            clusters.append(cluster)
            used_node_ids.add(hub_id)
            used_node_ids.update(cluster.spoke_ids)

            logger.info(
                "[SLEEP CLUSTER] Найден кластер: hub=%s (weight=%.2f) + spokes=%s",
                hub_id, hub_row["weight"], cluster.spoke_ids,
            )

        return clusters

    def create_abstract_node(
        self,
        summary_context: str,
        summary_response: str,
        source_node_ids: List[int],
        weight: Optional[float] = None,
        timestamp: Optional[float] = None,
    ) -> int:
        """
        Записывает новый 'абстрактный' узел LTM (результат семантической
        консолидации кластера) и АРХИВИРУЕТ исходные узлы кластера:
        снижает их вес умножением на SLEEP_ARCHIVE_WEIGHT_MULTIPLIER (узлы
        не удаляются немедленно — они станут кандидатами на удаление при
        следующем обычном decay/pruning, если так и останутся невостребованными).

        Также связывает новый абстрактный узел ребром со всеми исходными
        узлами кластера (чтобы сохранить след происхождения в графе).

        Возвращает id нового абстрактного узла.
        """
        effective_weight = weight if weight is not None else self.settings.sleep_abstract_node_weight
        ts = timestamp if timestamp is not None else time.time()

        abstract_node_id = self.db.insert_node(
            context=summary_context,
            response=summary_response,
            weight=effective_weight,
            timestamp=ts,
        )

        for source_id in source_node_ids:
            source_row = self.db.get_node(source_id)
            if source_row is None:
                continue

            archived_weight = source_row["weight"] * self.settings.sleep_archive_weight_multiplier
            self.db.update_weight(source_id, archived_weight)

            self.connect_nodes(abstract_node_id, source_id, weight_boost=self.settings.edge_initial_weight, timestamp=ts)

        logger.info(
            "[SLEEP CONSOLIDATION] Абстрактный узел id=%s (weight=%.2f) создан из кластера %s",
            abstract_node_id, effective_weight, source_node_ids,
        )

        return abstract_node_id

        # ----------------------------------------------------------------------
    # 4d. PROACTIVE MEMORY RECALL — выбор узла для инициативного сообщения
    # ----------------------------------------------------------------------



    # ----------------------------------------------------------------------
    # 5. ИЗБИРАТЕЛЬНАЯ КОНСОЛИДАЦИЯ (STM -> LTM)
    # ----------------------------------------------------------------------

    def consolidate_from_stm(
        self,
        entries: List["STMEntry"],
        timestamp: Optional[float] = None,
        already_captured_by_spike: bool = False,
    ) -> ConsolidationResult:
        """
        Оценивает накопленный эпизод кратковременной памяти (STM) и
        принимает решение согласно физике Избирательной Консолидации:

            a) ЭМОЦИОНАЛЬНЫЙ УЗЕЛ — если max(emotion_score) среди записей
               STM >= STM_EMOTIONAL_THRESHOLD -> эпизод упаковывается и
               записывается в LTM с высоким весом (= max emotion_score).

            b) СТРУКТУРНЫЙ УЗЕЛ — если средняя perplexity эпизода >=
               STM_STRUCTURAL_THRESHOLD (много новой информации/фактов) ->
               эпизод упаковывается и записывается в LTM с умеренным
               весом (STM_STRUCTURAL_WEIGHT).

            c) РУТИННЫЙ ШУМ — ни то ни другое -> эпизод просто отбрасывается,
               без записи в БД (экономим память от бытового шума).

        entries — список STMEntry, обычно результат working_memory.consume_all().
        Пустой список -> сразу считается рутинным шумом (нет данных).

        Возвращает ConsolidationResult с решением и деталями для логов/дебага.
        """
        if not entries:
            return ConsolidationResult(decision="routine_noise", reason="STM пуст, нет данных")

        # Защита от дублирования: если этот флеш вызван спайком амигдалы,
        # который уже создал точный узел LTM для текущего обмена (шаг 10 в
        # brain_session.py), а STM на момент флеша содержит только этот же
        # самый обмен (<=2 записи: реплика user + реплика bot), то
        # консолидация создала бы почти идентичный дубль. В этом случае
        # пропускаем запись — контент уже сохранён.
        if already_captured_by_spike and len(entries) <= 2:
            logger.info(
                "[CONSOLIDATION] Пропуск: обмен уже сохранён spike-узлом "
                "(entries=%d)",
                len(entries),
            )
            return ConsolidationResult(
                decision="routine_noise",
                reason="Уже сохранён как spike-узел, дублирование пропущено",
            )

        max_emotion = max(e.emotion_score for e in entries)
        avg_perplexity = sum(e.perplexity for e in entries) / len(entries)

        packed_context, packed_response = self._pack_episode(entries)

        # --- a) Эмоциональный узел (приоритет выше структурного) ---
        if max_emotion >= self.settings.stm_emotional_threshold:
            node_id = self.save_connection(
                context=packed_context,
                response=packed_response,
                weight=max_emotion,
                timestamp=timestamp,
            )
            logger.info(
                "[CONSOLIDATION] Эмоциональный узел id=%s weight=%.3f (max_emotion=%.3f)",
                node_id, max_emotion, max_emotion,
            )
            return ConsolidationResult(
                decision="emotional_node",
                node_id=node_id,
                weight=max_emotion,
                reason=f"max_emotion={max_emotion:.3f} >= {self.settings.stm_emotional_threshold}",
            )

        # --- b) Структурный узел ---
        if avg_perplexity >= self.settings.stm_structural_threshold:
            node_id = self.save_connection(
                context=packed_context,
                response=packed_response,
                weight=self.settings.stm_structural_weight,
                timestamp=timestamp,
            )
            logger.info(
                "[CONSOLIDATION] Структурный узел id=%s weight=%.3f (avg_perplexity=%.3f)",
                node_id, self.settings.stm_structural_weight, avg_perplexity,
            )
            return ConsolidationResult(
                decision="structural_node",
                node_id=node_id,
                weight=self.settings.stm_structural_weight,
                reason=f"avg_perplexity={avg_perplexity:.3f} >= {self.settings.stm_structural_threshold}",
            )

        # --- c) Рутинный шум — отбрасываем без записи в БД ---
        logger.info(
            "[STM FLUSH] Рутинный шум отброшен (max_emotion=%.3f, avg_perplexity=%.3f, %d записей)",
            max_emotion, avg_perplexity, len(entries),
        )
        return ConsolidationResult(
            decision="routine_noise",
            reason=f"max_emotion={max_emotion:.3f}, avg_perplexity={avg_perplexity:.3f} — ниже порогов",
        )

    @staticmethod
    def _pack_episode(entries: List["STMEntry"]) -> "tuple[str, str]":
        """
        Упаковывает список STMEntry в пару (context, response) для хранения
        как единого узла LTM. Реплики user объединяются в context, реплики
        bot — в response (сохраняя порядок появления).
        """
        user_lines = [e.text.strip() for e in entries if e.role == "user"]
        bot_lines = [e.text.strip() for e in entries if e.role != "user"]

        context = " | ".join(user_lines) if user_lines else "(без реплик пользователя)"
        response = " | ".join(bot_lines) if bot_lines else "(без ответов бота)"

        return context, response

    # ----------------------------------------------------------------------
    # Вспомогательные методы
    # ----------------------------------------------------------------------

    def reinforce_node(self, node_id: int, boost: float = 0.1, timestamp: Optional[float] = None) -> None:
        row = self.db.get_node(node_id)
        if row is None:
            logger.warning("[MEMORY REINFORCE] Узел id=%s не найден", node_id)
            return

        new_weight = min(1.0, row["weight"] + boost)
        self.db.update_weight(node_id, new_weight)
        self.touch_node(node_id, timestamp=timestamp)
        logger.info("[MEMORY REINFORCED] id=%s новый вес=%.3f", node_id, new_weight)

    def apply_reward(
        self,
        node_id: int,
        valence: float,
        timestamp: Optional[float] = None,
    ) -> Optional[RewardSignal]:
        """
        Дофаминовый сигнал: считает ОШИБКУ ПРЕДСКАЗАНИЯ награды для узла,
        обновляет его ожидание и возвращает результат.

            rpe = фактическая_валентность - ожидаемая_для_этого_узла
            ожидание += REWARD_EXPECTATION_LEARNING_RATE * rpe

        Это правило Рескорлы-Вагнера. Смысл поправки: дофамин выделяется
        не на награду, а на НЕОЖИДАННУЮ награду. Без неё "стремление к
        одобрению" вырождается — организм нашёл бы одно слово, которое
        всегда хвалят, и повторял бы его вечно. Здесь же то, что хвалят
        ВСЕГДА, перестаёт давать сигнал (rpe -> 0), и организм идёт
        пробовать новое.

        Возвращает None, если узел исчез (мог попасть под прунинг между
        действием и оценкой).
        """
        row = self.db.get_node(node_id)
        if row is None:
            return None

        expected = row["reward_expectation"] or 0.0
        rpe = valence - expected
        new_expectation = max(-1.0, min(1.0, expected + self.settings.reward_expectation_learning_rate * rpe))

        self.db.update_reward_expectation(node_id, new_expectation)

        logger.info(
            "[DOPAMINE] node=%s валентность=%+.2f ожидалось=%+.2f -> rpe=%+.2f "
            "(новое ожидание %+.2f)",
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
        Во сколько раз ошибка предсказания награды ускоряет закрепление.

        Дофамин модулирует синаптическую пластичность: неожиданный исход
        закрепляется сильно, полностью предсказанный — почти никак.
        Нижняя граница (REWARD_MIN_LEARNING_SCALE) не даёт обучению
        обнулиться совсем, иначе давно освоенный узел перестал бы получать
        даже поддерживающее подкрепление.
        """
        return max(self.settings.reward_min_learning_scale, min(1.0, abs(prediction_error)))

    def penalize_node(
        self,
        node_id: int,
        penalty: float = 0.15,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Штрафует узел за негативный фидбэк (Reinforcement Loop): снижает
        вес узла и НЕ обновляет last_accessed на текущий момент (в отличие
        от touch_node/reinforce_node) — это ускоряет относительное старение
        узла при следующем apply_decay(), имитируя пониженную "устойчивость"
        (decay_rate) отрицательно подкреплённой связи.
        """
        row = self.db.get_node(node_id)
        if row is None:
            logger.warning("[MEMORY PENALIZE] Узел id=%s не найден", node_id)
            return

        new_weight = max(0.0, row["weight"] - penalty)
        self.db.update_weight(node_id, new_weight)
        logger.info(
            "[MEMORY PENALIZED] id=%s weight %.3f -> %.3f (penalty=%.3f)",
            node_id, row["weight"], new_weight, penalty,
        )

    def get_top_nodes(self, limit: int = 5) -> List[MemoryMatch]:
        rows = self.db.fetch_all_nodes()
        nodes = [
            MemoryMatch(
                id=row["id"],
                context=row["context"],
                response=row["response"],
                weight=row["weight"],
                similarity=0.0,
                created_at=row["created_at"],
                last_accessed=row["last_accessed"],
            )
            for row in rows
        ]
        nodes.sort(key=lambda n: n.weight, reverse=True)
        return nodes[:limit]

    def count_nodes(self) -> int:
        return len(self.db.fetch_all_nodes())

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "MemoryGraph":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()