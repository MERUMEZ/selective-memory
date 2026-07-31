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

import config
from memory.database import Database
from services.llm import generate_llm_response
from storage.utils.logger import get_logger

if TYPE_CHECKING:
    from memory.working_memory import STMEntry

logger = get_logger(__name__)

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
class ProactiveCandidate:
    """Кандидат на проактивное сообщение с рассчитанным итоговым score."""
    id: int
    context: str
    response: str
    weight: float
    relevance: float
    cooldown_penalty: float
    score: float


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


@dataclass
class SelfModelEvolutionResult:
    """Результат попытки эволюции Self-Model во время фазы сна (Итерация H)."""
    evolved: bool
    old_content: str = ""
    new_content: str = ""
    source_node_ids: List[int] = None
    reason: str = ""

    def __post_init__(self):
        if self.source_node_ids is None:
            self.source_node_ids = []


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

    def __init__(self, db: Optional[Database] = None):
        self.db = db or Database()
        self.last_activation_traces: List[ActivationTrace] = []

    # ----------------------------------------------------------------------
    # SELF-MODEL & USER-MODEL — инициализация мета-узлов (Итерация 15)
    # ----------------------------------------------------------------------

    def ensure_self_and_user_nodes(self) -> "tuple[int, int]":
        """
        Проверяет наличие мета-узлов Self-Model и User-Model в БД. Если
        они отсутствуют — создаёт их со стандартными текстами из config.py
        (DEFAULT_SELF_MODEL / DEFAULT_USER_MODEL) и весом META_NODE_WEIGHT.
        Если уже существуют — просто возвращает их id без изменения контента
        (чтобы не перезатирать личность, если она уже развилась/менялась).

        Вызывается ОДНОКРАТНО при инициализации системы (main.py).
        Возвращает (self_node_id, user_node_id).
        """
        self_row = self.db.get_meta_node("self_model")
        user_row = self.db.get_meta_node("user_model")

        if self_row is None:
            self_node_id = self.db.upsert_meta_node(
                node_type="self_model",
                content=config.DEFAULT_SELF_MODEL,
                weight=config.META_NODE_WEIGHT,
            )
        else:
            self_node_id = self_row["id"]

        if user_row is None:
            user_node_id = self.db.upsert_meta_node(
                node_type="user_model",
                content=config.DEFAULT_USER_MODEL,
                weight=config.META_NODE_WEIGHT,
            )
        else:
            user_node_id = user_row["id"]

        logger.info(
            "[META INIT] Self-Model (id=%s) и User-Model (id=%s) готовы.",
            self_node_id, user_node_id,
        )

        return self_node_id, user_node_id

    def get_self_model_content(self) -> str:
        """Возвращает текущий текст Self-Model (fallback на config-дефолт)."""
        row = self.db.get_meta_node("self_model")
        return row["context"] if row is not None else config.DEFAULT_SELF_MODEL

    def get_user_model_content(self) -> str:
        """Возвращает текущий текст User-Model (fallback на config-дефолт)."""
        row = self.db.get_meta_node("user_model")
        return row["context"] if row is not None else config.DEFAULT_USER_MODEL

    def evolve_self_model(self, timestamp: Optional[float] = None) -> SelfModelEvolutionResult:
        """
        Фаза сна (Итерация H): рефлексия и постепенная эволюция Self-Model.

        Собирает "дайджест" значимых узлов LTM, созданных с момента
        прошлого сна (маркер хранится как отдельный мета-узел
        'last_sleep_marker' — переиспользуем инфраструктуру мета-узлов,
        не меняя схему БД), просит LLM ПОСТЕПЕННО скорректировать текущий
        текст Self-Model с учётом пережитого опыта, и записывает результат.

        Если материала недостаточно (< SELF_MODEL_EVOLUTION_MIN_NODES)
        ИЛИ LLM недоступна — эволюция пропускается, маркер last_sleep НЕ
        обновляется (материал продолжит копиться до следующего сна).
        """
        ts = timestamp if timestamp is not None else time.time()

        marker_row = self.db.get_meta_node("last_sleep_marker")
        min_created_at = float(marker_row["context"]) if marker_row is not None else 0.0

        significant_rows = self.db.get_significant_nodes_since(
            min_created_at, limit=config.SELF_MODEL_EVOLUTION_MAX_NODES
        )

        if len(significant_rows) < config.SELF_MODEL_EVOLUTION_MIN_NODES:
            logger.info(
                "[SELF-MODEL EVOLUTION] Недостаточно опыта для рефлексии "
                "(%d < %d значимых узлов) -> пропуск",
                len(significant_rows), config.SELF_MODEL_EVOLUTION_MIN_NODES,
            )
            return SelfModelEvolutionResult(
                evolved=False,
                reason=f"Недостаточно опыта ({len(significant_rows)} узлов)",
            )

        current_self = self.get_self_model_content()
        digest = self._format_significant_nodes_for_prompt(significant_rows)
        user_message = (
            f"ТЕКУЩИЙ Self-Model:\n{current_self}\n\n"
            f"ДАЙДЖЕСТ значимых событий с прошлого сна:\n{digest}"
        )

        llm_result = generate_llm_response(
            messages=[{"role": "user", "content": user_message}],
            system_prompt=config.SELF_MODEL_EVOLUTION_PROMPT,
        )

        if not llm_result or not llm_result.strip():
            logger.warning("[SELF-MODEL EVOLUTION] LLM недоступна/пустой ответ -> пропуск")
            return SelfModelEvolutionResult(
                evolved=False, reason="LLM недоступна или вернула пустой ответ",
            )

        new_content = llm_result.strip()[: config.SELF_MODEL_MAX_LENGTH]

        self.db.upsert_meta_node(
            node_type="self_model",
            content=new_content,
            weight=config.META_NODE_WEIGHT,
            timestamp=ts,
        )
        # Маркер обновляем ТОЛЬКО при успешной эволюции — иначе накопленный
        # с прошлого раза опыт "потерялся" бы без результата.
        self.db.upsert_meta_node(
            node_type="last_sleep_marker",
            content=str(ts),
            weight=1.0,
            timestamp=ts,
        )

        source_ids = [row["id"] for row in significant_rows]
        logger.info(
            "[SELF-MODEL EVOLUTION] Self-Model обновлён на основе %d узлов: %r -> %r",
            len(significant_rows), current_self[:60], new_content[:60],
        )

        return SelfModelEvolutionResult(
            evolved=True,
            old_content=current_self,
            new_content=new_content,
            source_node_ids=source_ids,
            reason="OK",
        )

    @staticmethod
    def _format_significant_nodes_for_prompt(rows: List["sqlite3.Row"]) -> str:
        """Формирует читаемый текстовый дайджест значимых узлов для LLM."""
        lines = []
        for row in rows:
            ctx = (row["context"] or "").strip().replace("\n", " ")[:100]
            resp = (row["response"] or "").strip().replace("\n", " ")[:100]
            lines.append(f'- (weight={row["weight"]:.2f}) User: "{ctx}" | Bot: "{resp}"')
        return "\n".join(lines)

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
            weight=config.CONCEPT_NODE_WEIGHT,
            timestamp=ts,
        )

        # --- Связь с USER_MODEL (источник знания) ---
        user_row = self.db.get_meta_node("user_model")
        if user_row is not None:
            self.connect_nodes(
                concept_node_id,
                user_row["id"],
                weight_boost=config.CONCEPT_USER_EDGE_WEIGHT,
                timestamp=ts,
            )

        # --- Связь с исходным узлом-источником (если передан) ---
        if source_node_id is not None:
            self.connect_nodes(
                concept_node_id,
                source_node_id,
                weight_boost=config.EDGE_INITIAL_WEIGHT,
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
            threshold=config.CONCEPT_SIMILARITY_LINK_THRESHOLD,
            top_k=config.CONCEPT_MAX_SIMILAR_LINKS + 1,  # +1 запас на случай самосовпадения
            timestamp=timestamp,
            with_associations=False,
        )

        linked_count = 0
        for match in matches:
            if match.id == concept_node_id:
                continue
            if linked_count >= config.CONCEPT_MAX_SIMILAR_LINKS:
                break

            self.connect_nodes(
                concept_node_id,
                match.id,
                weight_boost=config.CONCEPT_SIMILARITY_EDGE_WEIGHT,
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
        if not config.LEXICAL_ACQUISITION_ENABLED or not text or not text.strip():
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
                initial_weight=config.WORD_NODE_INITIAL_WEIGHT,
                reinforce_step=config.WORD_NODE_REINFORCE_STEP,
                timestamp=ts,
            )
            words_processed += 1
            new_words += 1 if word_was_created else 0

            for syllable in self._split_into_syllables(token):
                syllable_id, syll_was_created = self.db.upsert_lexical_node(
                    node_type="syllable",
                    text=syllable,
                    initial_weight=config.SYLLABLE_NODE_INITIAL_WEIGHT,
                    reinforce_step=config.SYLLABLE_NODE_REINFORCE_STEP,
                    timestamp=ts,
                )
                syllables_processed += 1
                new_syllables += 1 if syll_was_created else 0

                self.connect_nodes(
                    syllable_id, word_id,
                    weight_boost=config.SYLLABLE_WORD_EDGE_WEIGHT,
                    timestamp=ts,
                )

            if previous_word_id is not None:
                self.connect_nodes(
                    previous_word_id, word_id,
                    weight_boost=config.WORD_COOCCURRENCE_EDGE_WEIGHT,
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

        mastery = max(1e-9, config.VOCABULARY_MASTERY_MIN_WEIGHT)
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

        activation = max(1e-9, config.EDGE_ACTIVATION_THRESHOLD)
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
                config.SURPRISE_LEXICAL_WEIGHT * lexical_surprise
                + config.SURPRISE_STRUCTURAL_WEIGHT * structural_surprise
            )
            weight_sum = config.SURPRISE_LEXICAL_WEIGHT + config.SURPRISE_STRUCTURAL_WEIGHT
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

    @staticmethod
    def _tokenize_for_lexicon(text: str) -> List[str]:
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
            if len(w) >= config.LEXICAL_MIN_TOKEN_LENGTH
        ][: config.LEXICAL_MAX_TOKENS_PER_INPUT]

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
        return self.db.count_mastered_words(config.VOCABULARY_MASTERY_MIN_WEIGHT)

    def get_exposed_vocabulary_size(self) -> int:
        """
        Общее количество РАЗЛИЧНЫХ слов, которые бот хотя бы раз услышал,
        независимо от закрепления — "пассивный" словарь. Только для
        статистики/отладки (например, разница между этим числом и
        get_vocabulary_size() показывает, сколько слов ещё "на подходе" к
        усвоению). НЕ используется для гейтинга речевых стадий.
        """
        return self.db.count_nodes_by_type("word")

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

        limit по умолчанию берётся из config.BABBLING_SYLLABLE_POOL_SIZE.
        """
        effective_limit = limit if limit is not None else config.BABBLING_SYLLABLE_POOL_SIZE
        rows = self.db.get_random_nodes_by_type("syllable", limit=effective_limit)
        return [
            KnownSyllable(id=row["id"], text=row["context"], weight=row["weight"])
            for row in rows
        ]


    # ----------------------------------------------------------------------
    # 1. Сохранение новой связи
    # ----------------------------------------------------------------------

    def save_connection(
        self,
        context: str,
        response: str,
        weight: Optional[float] = None,
        timestamp: Optional[float] = None,
    ) -> int:
        """Сохраняет новую связь context -> response с начальным весом."""
        initial_weight = weight if weight is not None else config.BASE_PLASTICITY_THRESHOLD

        node_id = self.db.insert_node(
            context=context,
            response=response,
            weight=initial_weight,
            timestamp=timestamp,
        )

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
        effective_threshold = threshold if threshold is not None else config.MEMORY_SEARCH_THRESHOLD

        query_normalized = query.strip().lower()
        if not query_normalized:
            return []

        query_keywords = self._extract_keywords(query_normalized)
        rows = self.db.fetch_searchable_nodes()

        scored: List[MemoryMatch] = []

        for row in rows:
            context_normalized = row["context"].strip().lower()
            context_keywords = self._extract_keywords(context_normalized)

            keyword_score = self._keyword_overlap(query_keywords, context_keywords)
            fuzzy_score = self._compute_fuzzy_similarity(query_normalized, context_normalized)

            combined_score = (
                keyword_score * config.MEMORY_KEYWORD_WEIGHT
                + fuzzy_score * config.MEMORY_FUZZY_WEIGHT
                + row["weight"] * config.MEMORY_WEIGHT_INFLUENCE
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
                    min_weight=config.EDGE_ACTIVATION_THRESHOLD,
                    limit=config.EDGE_MAX_HOP_NODES,
                    timestamp=timestamp,
                )

                for assoc in associated:
                    if assoc.id in existing_ids:
                        continue

                    activation_score = min(
                        1.0,
                        source_match.similarity * config.EDGE_ACTIVATION_DECAY * assoc.edge_weight,
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

    def find_similar_context(
        self,
        query: str,
        threshold: float = 0.35,
        top_k: int = 1,
        timestamp: Optional[float] = None,
    ) -> List[MemoryMatch]:
        """Обратная совместимость со старым API (чистое нечёткое сходство)."""
        return self.search(query, threshold=threshold, top_k=top_k, timestamp=timestamp)

    @staticmethod
    def _extract_keywords(text: str) -> Set[str]:
        words = WORD_PATTERN.findall(text)
        return {
            w for w in words
            if len(w) >= config.MEMORY_MIN_KEYWORD_LENGTH and w not in STOP_WORDS
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

        boost = weight_boost if weight_boost is not None else config.EDGE_BOOST_STEP
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

        boost = weight_boost if weight_boost is not None else config.EDGE_BOOST_STEP
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
        effective_min_weight = min_weight if min_weight is not None else config.EDGE_ACTIVATION_THRESHOLD

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

    @staticmethod
    def _age_t0_for(node_type: Optional[str]) -> float:
        """
        Характерное время жизни узла для формулы decay. Словарь угасает
        по LEXICAL_AGE_T0 (~30 суток), всё остальное — по AGE_T0 (~1 час).

        Без этого разделения освоенное слово теряло статус за ночь, а за
        сутки с небольшим удалялось из БД — словарь не мог накопиться в
        принципе (см. комментарий у config.LEXICAL_AGE_T0).
        """
        if node_type in MemoryGraph.LEXICAL_NODE_TYPES:
            return config.LEXICAL_AGE_T0
        return config.AGE_T0

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
            stability = row["stability"] if row["stability"] else config.STABILITY_INITIAL
            effective_t0 = self._age_t0_for(row["node_type"]) * max(1e-9, stability)
            decay_factor = math.exp(-config.DECAY_RATE * dt / effective_t0)
            new_weight = old_weight * decay_factor

            if new_weight < config.FORGET_THRESHOLD:
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
            decay_factor = math.exp(-config.EDGE_DECAY_RATE * dt / config.AGE_T0)
            new_weight = old_weight * decay_factor

            if new_weight < config.EDGE_FORGET_THRESHOLD:
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
        threshold = min_weight if min_weight is not None else config.EDGE_FORGET_THRESHOLD
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
            min_edge_weight if min_edge_weight is not None else config.EDGE_ACTIVATION_THRESHOLD
        )
        effective_node_weight = (
            max_node_weight if max_node_weight is not None else config.SLEEP_ORPHAN_WEIGHT_THRESHOLD
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
            min_edge_weight if min_edge_weight is not None else config.SLEEP_HUB_MIN_EDGE_WEIGHT
        )
        effective_min_spokes = (
            min_spokes if min_spokes is not None else config.SLEEP_MIN_CLUSTER_SPOKES
        )
        effective_max_spokes = (
            max_spokes if max_spokes is not None else config.SLEEP_MAX_CLUSTER_SPOKES
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
        effective_weight = weight if weight is not None else config.SLEEP_ABSTRACT_NODE_WEIGHT
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

            archived_weight = source_row["weight"] * config.SLEEP_ARCHIVE_WEIGHT_MULTIPLIER
            self.db.update_weight(source_id, archived_weight)

            self.connect_nodes(abstract_node_id, source_id, weight_boost=config.EDGE_INITIAL_WEIGHT, timestamp=ts)

        logger.info(
            "[SLEEP CONSOLIDATION] Абстрактный узел id=%s (weight=%.2f) создан из кластера %s",
            abstract_node_id, effective_weight, source_node_ids,
        )

        return abstract_node_id

        # ----------------------------------------------------------------------
    # 4d. PROACTIVE MEMORY RECALL — выбор узла для инициативного сообщения
    # ----------------------------------------------------------------------

    def select_proactive_node(
        self,
        last_active_node_id: Optional[int],
        brain_time: float,
    ) -> Optional[ProactiveCandidate]:
        """
        Выбирает узел LTM для проактивного сообщения по формуле:

            S(n) = weight_n * relevance_n * cooldown_penalty(t)

        где:
            relevance_n         = PROACTIVE_RELEVANCE_BOOST (1.5), если узел
                                   связан ребром с last_active_node_id,
                                   иначе PROACTIVE_RELEVANCE_BASE (1.0).
            cooldown_penalty(t)  = PROACTIVE_COOLDOWN_PENALTY_VALUE (0.1), если
                                   brain_time - last_accessed < PROACTIVE_COOLDOWN_SECONDS,
                                   иначе 1.0.

        Из top-K (PROACTIVE_TOP_K) кандидатов по score происходит
        вероятностный (softmax) выбор — не берём просто максимум, чтобы
        проактивные сообщения не были всегда предсказуемо про один и тот
        же самый "сильный" узел.

        Возвращает None, если в БД вообще нет узлов (вызывающий код
        должен в этом случае сгенерировать fallback-размышление без узла).
        """
        # ИСПРАВЛЕНИЕ: fetch_all_nodes() возвращал ВСЕ node_type, включая
        # служебные лексические узлы ('word'/'syllable') и мета-узлы
        # (Self-Model/User-Model) — они могли попасть в проактивное
        # сообщение как будто это реальное воспоминание (у лексических
        # узлов context == response == текст слова/слога, что выглядело
        # бы абсурдно в PROACTIVE_PROMPT_TEMPLATE). fetch_searchable_nodes()
        # уже корректно ограничен node_type IN ('episodic', 'concept').
        rows = self.db.fetch_searchable_nodes()
        if not rows:
            logger.info("[PROACTIVE RECALL] LTM пуста — нет кандидатов для проактивного узла")
            return None

        related_ids: set = set()
        if last_active_node_id is not None:
            related_edges = self.db.get_edges_for_node(last_active_node_id)
            related_ids = {
                edge["neighbor_id"] for edge in related_edges
                if edge["weight"] >= config.EDGE_ACTIVATION_THRESHOLD
            }

        candidates: List[ProactiveCandidate] = []

        for row in rows:
            relevance = (
                config.PROACTIVE_RELEVANCE_BOOST
                if row["id"] in related_ids
                else config.PROACTIVE_RELEVANCE_BASE
            )

            seconds_since_touch = brain_time - row["last_accessed"]
            cooldown_penalty = (
                config.PROACTIVE_COOLDOWN_PENALTY_VALUE
                if 0 <= seconds_since_touch < config.PROACTIVE_COOLDOWN_SECONDS
                else 1.0
            )

            score = row["weight"] * relevance * cooldown_penalty

            candidates.append(
                ProactiveCandidate(
                    id=row["id"],
                    context=row["context"],
                    response=row["response"],
                    weight=row["weight"],
                    relevance=relevance,
                    cooldown_penalty=cooldown_penalty,
                    score=score,
                )
            )

        candidates.sort(key=lambda c: c.score, reverse=True)
        top_candidates = candidates[: config.PROACTIVE_TOP_K]

        if not top_candidates:
            return None

        chosen = self._softmax_choice(top_candidates)

        logger.info(
            "[PROACTIVE RECALL] Выбран узел id=%s (score=%.3f, relevance=%.2f, "
            "cooldown_penalty=%.2f) из %d кандидатов",
            chosen.id, chosen.score, chosen.relevance, chosen.cooldown_penalty,
            len(top_candidates),
        )

        # Обращение к узлу засчитывается как "касание" — он всплыл в сознании
        self.touch_node(chosen.id, timestamp=brain_time)

        return chosen

    @staticmethod
    def _softmax_choice(candidates: List[ProactiveCandidate]) -> ProactiveCandidate:
        """
        Вероятностный (softmax) выбор одного кандидата из списка на основе
        их score. Температура берётся из config.PROACTIVE_SOFTMAX_TEMPERATURE.
        """
        import random

        temperature = max(1e-6, config.PROACTIVE_SOFTMAX_TEMPERATURE)
        scores = [c.score for c in candidates]
        max_score = max(scores)

        # Численно стабильный softmax (вычитаем max перед exp)
        exp_scores = [math.exp((s - max_score) / temperature) for s in scores]
        total = sum(exp_scores)

        if total <= 0:
            return candidates[0]

        probabilities = [e / total for e in exp_scores]
        return random.choices(candidates, weights=probabilities, k=1)[0]

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
        if max_emotion >= config.STM_EMOTIONAL_THRESHOLD:
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
                reason=f"max_emotion={max_emotion:.3f} >= {config.STM_EMOTIONAL_THRESHOLD}",
            )

        # --- b) Структурный узел ---
        if avg_perplexity >= config.STM_STRUCTURAL_THRESHOLD:
            node_id = self.save_connection(
                context=packed_context,
                response=packed_response,
                weight=config.STM_STRUCTURAL_WEIGHT,
                timestamp=timestamp,
            )
            logger.info(
                "[CONSOLIDATION] Структурный узел id=%s weight=%.3f (avg_perplexity=%.3f)",
                node_id, config.STM_STRUCTURAL_WEIGHT, avg_perplexity,
            )
            return ConsolidationResult(
                decision="structural_node",
                node_id=node_id,
                weight=config.STM_STRUCTURAL_WEIGHT,
                reason=f"avg_perplexity={avg_perplexity:.3f} >= {config.STM_STRUCTURAL_THRESHOLD}",
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