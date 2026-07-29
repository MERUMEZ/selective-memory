"""
================================================================================

 DATABASE.PY — SQLite-слой хранения памяти "Динамического Мозга"
================================================================================


Отвечает ТОЛЬКО за низкоуровневый доступ к БД: создание схемы, CRUD-операции
над таблицей nodes. Вся "биологическая" логика (decay, spike-пороги, поиск
похожего контекста) находится в memory/graph_memory.py.



Схема таблицы nodes:
    id            INTEGER PRIMARY KEY AUTOINCREMENT
    context       TEXT     -- входящий контекст (сообщение пользователя)
    response      TEXT     -- ответ системы, связанный с этим контекстом
    weight        REAL     -- текущий "вес" связи (сила памяти), 0..1+
    created_at    REAL     -- unix timestamp создания узла
    last_accessed REAL     -- unix timestamp последнего обращения к узлу
================================================================================
"""

import sqlite3
import time
import config
from pathlib import Path
from typing import Optional, List, Dict, Any
from storage.utils.logger import get_logger

logger = get_logger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    context       TEXT NOT NULL,
    response      TEXT NOT NULL,
    weight        REAL NOT NULL DEFAULT 1.0,
    created_at    REAL NOT NULL,
    last_accessed REAL NOT NULL
);
"""

INDEX_CONTEXT = """
CREATE INDEX IF NOT EXISTS idx_nodes_context ON nodes(context);
"""

# --------------------------------------------------------------------------
# Схема таблицы edges — ассоциативные связи между узлами LTM
# (Semantic Edges / Spreading Activation)
# --------------------------------------------------------------------------
EDGES_SCHEMA = """
CREATE TABLE IF NOT EXISTS edges (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    node_from      INTEGER NOT NULL,
    node_to        INTEGER NOT NULL,
    weight         REAL NOT NULL DEFAULT 0.2,
    last_activated REAL NOT NULL,
    FOREIGN KEY (node_from) REFERENCES nodes(id) ON DELETE CASCADE,
    FOREIGN KEY (node_to)   REFERENCES nodes(id) ON DELETE CASCADE
);
"""

INDEX_EDGE_FROM = """
CREATE INDEX IF NOT EXISTS idx_edges_from ON edges(node_from);
"""

INDEX_EDGE_TO = """
CREATE INDEX IF NOT EXISTS idx_edges_to ON edges(node_to);
"""

# Уникальность направленной пары (node_from, node_to). Ненаправленность
# (A->B эквивалентно B->A) обеспечивается на уровне логики graph_memory.py
# через нормализацию порядка id перед вставкой (см. Database.upsert_edge).
UNIQUE_EDGE_PAIR = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_edges_unique_pair ON edges(node_from, node_to);
"""

# --------------------------------------------------------------------------
# Миграция: мета-узлы Self-Model / User-Model (Итерация 15)
# --------------------------------------------------------------------------
ALTER_ADD_IS_META = """
ALTER TABLE nodes ADD COLUMN is_meta INTEGER DEFAULT 0;
"""

ALTER_ADD_NODE_TYPE = """
ALTER TABLE nodes ADD COLUMN node_type TEXT DEFAULT NULL;
"""
# --------------------------------------------------------------------------
# Миграция: отдельные "часы" decay (last_decayed_at), НЕ совпадающие с
# last_accessed/last_activated (фикс компаундинга decay).
#
# last_accessed / last_activated = "когда узел/ребро РЕАЛЬНО использовали"
#   (нужно для proactive cooldown, скоринга релевантности и т.п.)
# last_decayed_at                = "с какого момента отсчитывать угасание
#   веса" — чисто техническая метка для формулы decay.
#
# Раньше decay считал dt от last_accessed, но apply_decay гоняется на
# КАЖДОЕ сообщение для ВСЕХ узлов, а last_accessed двигается только при
# реальном касании — из-за этого угасание накапливалось некорректно
# (квадратично по числу сообщений вместо линейного по времени).
# --------------------------------------------------------------------------
ALTER_ADD_LAST_DECAYED_NODES = """
ALTER TABLE nodes ADD COLUMN last_decayed_at REAL DEFAULT NULL;
"""
ALTER_ADD_LAST_DECAYED_EDGES = """
ALTER TABLE edges ADD COLUMN last_decayed_at REAL DEFAULT NULL;
"""
class Database:
    """
    Тонкая обёртка над SQLite для таблицы `nodes`.

    Использование:
        db = Database()
        db.insert_node(context="привет", response="привет!", weight=0.8)
        rows = db.fetch_all_nodes()
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or config.BRAIN_DB_PATH
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # Включаем поддержку внешних ключей — без этого SQLite ИГНОРИРУЕТ
        # "ON DELETE CASCADE" в схеме edges, и при удалении узла (delete_node)
        # его рёбра остались бы "осиротевшими" записями в таблице edges.
        self._conn.execute("PRAGMA foreign_keys = ON;")
        self._init_schema()

    def _init_schema(self) -> None:
        """Создаёт таблицы nodes/edges и нужные индексы, если их ещё нет."""
        cursor = self._conn.cursor()
        cursor.execute(SCHEMA)
        cursor.execute(INDEX_CONTEXT)
        cursor.execute(EDGES_SCHEMA)
        cursor.execute(INDEX_EDGE_FROM)
        cursor.execute(INDEX_EDGE_TO)
        cursor.execute(UNIQUE_EDGE_PAIR)
        self._conn.commit()
        self._migrate_meta_columns()
        self._migrate_decay_columns()
        logger.info("[DB INIT] Схема nodes + edges готова (%s)", self.db_path)

    def _migrate_meta_columns(self) -> None:
        """
        Миграция таблицы nodes: добавляет колонки is_meta/node_type, если
        они ещё не существуют (для БД, созданных до Итерации 15). ALTER
        TABLE в SQLite не поддерживает IF NOT EXISTS для колонок, поэтому
        перехватываем sqlite3.OperationalError ("duplicate column name").
        """
        cursor = self._conn.cursor()
        migrated_any = False

        try:
            cursor.execute(ALTER_ADD_IS_META)
            migrated_any = True
        except sqlite3.OperationalError as exc:
            if "duplicate column name" not in str(exc):
                raise

        try:
            cursor.execute(ALTER_ADD_NODE_TYPE)
            migrated_any = True
        except sqlite3.OperationalError as exc:
            if "duplicate column name" not in str(exc):
                raise

        self._conn.commit()

        if migrated_any:
            logger.info("[MIGRATION] Таблица nodes обновлена (is_meta, node_type)")

    def _migrate_decay_columns(self) -> None:
        """
        Миграция: добавляет колонку last_decayed_at в nodes и edges —
        отдельные "часы" для decay, НЕ совпадающие с last_accessed/
        last_activated (см. комментарий у ALTER_ADD_LAST_DECAYED_*).

        Существующие строки backfill'ятся текущим значением
        last_accessed/last_activated — decay-часы для них просто
        начинают тикать с этого момента, без искусственного "долга".
        """
        cursor = self._conn.cursor()
        migrated_any = False
        try:
            cursor.execute(ALTER_ADD_LAST_DECAYED_NODES)
            migrated_any = True
        except sqlite3.OperationalError as exc:
            if "duplicate column name" not in str(exc):
                raise
        try:
            cursor.execute(ALTER_ADD_LAST_DECAYED_EDGES)
            migrated_any = True
        except sqlite3.OperationalError as exc:
            if "duplicate column name" not in str(exc):
                raise
        cursor.execute(
            "UPDATE nodes SET last_decayed_at = last_accessed WHERE last_decayed_at IS NULL"
        )
        cursor.execute(
            "UPDATE edges SET last_decayed_at = last_activated WHERE last_decayed_at IS NULL"
        )
        self._conn.commit()
        if migrated_any:
            logger.info("[MIGRATION] Таблица nodes/edges обновлена (last_decayed_at)")

    # ----------------------------------------------------------------------
    # CRUD операции
    # ----------------------------------------------------------------------

    def insert_node(
        self,
        context: str,
        response: str,
        weight: float = 1.0,
        timestamp: Optional[float] = None,
        node_type: str = "episodic",
    ) -> int:
        """
        Добавляет новый узел памяти. Возвращает id созданной записи.

        node_type классифицирует природу узла:
            "episodic" (по умолчанию) — обычное воспоминание/эпизод
            "concept"                 — явно преподанное понятие/термин/правило
            "meta"                    — служебные супер-узлы (обрабатываются
                                         отдельно через is_meta/upsert_meta_node,
                                         сюда практически не попадает напрямую)
        """
        ts = timestamp if timestamp is not None else time.time()
        cursor = self._conn.cursor()
        cursor.execute(
            """
            INSERT INTO nodes (context, response, weight, created_at, last_accessed, last_decayed_at, node_type)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (context, response, weight, ts, ts, ts, node_type),
        )
        self._conn.commit()
        node_id = cursor.lastrowid
        logger.info(
            "[MEMORY SAVED] id=%s weight=%.3f node_type=%s context=%r",
            node_id, weight, node_type, context[:50],
        )
        return node_id

    def get_node(self, node_id: int) -> Optional[sqlite3.Row]:
        """Возвращает один узел по id, либо None."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM nodes WHERE id = ?", (node_id,))
        return cursor.fetchone()

    def fetch_all_nodes(self) -> List[sqlite3.Row]:
        """Возвращает все узлы памяти (ВСЕ node_type — используется decay/sleep_cycle,
        где угасание/забывание должно применяться одинаково к любому типу узла)."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM nodes")
        return cursor.fetchall()

    def fetch_searchable_nodes(self) -> List[sqlite3.Row]:
        """
        Возвращает только узлы, пригодные для семантического поиска
        диалогового контекста (MemoryGraph.search) — то есть настоящие
        воспоминания/эпизоды и явно преподанные понятия.

        ИСКЛЮЧАЕТ 'word'/'syllable' (служебные узлы лексического
        усвоения — у них пустой response, и участие в поиске приводило
        к ложным MEMORY HIT на любом повторно использованном слове,
        см. баг: "почему" находило само себя с score=1.0) и
        'self_model'/'user_model' (мета-узлы, подмешиваются в промпт
        отдельным механизмом, не через search).
        """
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT * FROM nodes WHERE node_type IN ('episodic', 'concept')"
        )
        return cursor.fetchall()

    def update_weight(self, node_id: int, new_weight: float) -> None:
        """Обновляет вес узла (например, после decay или подкрепления)."""
        cursor = self._conn.cursor()
        cursor.execute(
            "UPDATE nodes SET weight = ? WHERE id = ?",
            (new_weight, node_id),
        )
        self._conn.commit()

    def update_last_accessed(self, node_id: int, timestamp: Optional[float] = None) -> None:
        """
        Обновляет метку последнего обращения к узлу. Также сбрасывает
        last_decayed_at — реальное использование узла законно "обновляет"
        и точку отсчёта для decay (см. _migrate_decay_columns).
        """
        ts = timestamp if timestamp is not None else time.time()
        cursor = self._conn.cursor()
        cursor.execute(
            "UPDATE nodes SET last_accessed = ?, last_decayed_at = ? WHERE id = ?",
            (ts, ts, node_id),
        )
        self._conn.commit()

    def delete_node(self, node_id: int) -> None:
        """Физически удаляет узел (используется во время sleep_cycle при забывании)."""
        cursor = self._conn.cursor()
        cursor.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
        self._conn.commit()
        logger.info("[MEMORY FORGOTTEN] id=%s удалён из БД", node_id)

    def bulk_update_weights(self, updates: List[Dict[str, Any]]) -> None:
        """
        Массовое обновление весов. updates = [{"id": 1, "weight": 0.4,
        "last_decayed_at": 12345.0}, ...]
        Используется decay-циклом для эффективного батч-обновления.
        Каждая запись ОБЯЗАНА содержать last_decayed_at (новую точку
        отсчёта decay) — иначе следующий decay-проход пересчитает dt от
        старой метки и угасание накопится некорректно.
        """
        cursor = self._conn.cursor()
        cursor.executemany(
            "UPDATE nodes SET weight = :weight, last_decayed_at = :last_decayed_at WHERE id = :id",
            updates,
        )
        self._conn.commit()

    def close(self) -> None:
        """Закрывает соединение с БД."""
        self._conn.close()

    # ----------------------------------------------------------------------
    # EDGES — ассоциативные связи между узлами (Semantic Edges)
    # ----------------------------------------------------------------------

    def upsert_edge(
        self,
        node_from: int,
        node_to: int,
        weight_boost: float,
        timestamp: Optional[float] = None,
        max_weight: float = 1.0,
    ) -> float:
        """
        Создаёт новое ребро node_from -> node_to с начальным весом weight_boost,
        либо, если ребро уже существует, УСИЛИВАЕТ его вес на weight_boost
        (ограничивая сверху max_weight). Обновляет last_activated.

        Пара нормализуется (node_from, node_to) так, чтобы связь была
        симметричной по факту хранения — всегда сохраняем меньший id первым,
        избегая дублей A->B и B->A как разных строк.

        Возвращает итоговый вес ребра после операции.
        """
        ts = timestamp if timestamp is not None else time.time()
        a, b = (node_from, node_to) if node_from <= node_to else (node_to, node_from)

        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT id, weight FROM edges WHERE node_from = ? AND node_to = ?",
            (a, b),
        )
        row = cursor.fetchone()

        if row is None:
            initial_weight = min(max_weight, max(0.0, weight_boost))
            try:
                cursor.execute(
                    """
                    INSERT INTO edges (node_from, node_to, weight, last_activated, last_decayed_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (a, b, initial_weight, ts, ts),
                )
                self._conn.commit()
            except sqlite3.IntegrityError:
                # Последний рубеж защиты: один из узлов был удалён между
                # проверкой существования (MemoryGraph.connect_nodes) и
                # этой вставкой (крайне узкое окно гонки) — FOREIGN KEY
                # constraint. Не даём этому уронить весь обработчик
                # сообщения, просто пропускаем создание ребра.
                self._conn.rollback()
                logger.warning(
                    "[EDGE SKIP] FOREIGN KEY constraint при создании ребра %s <-> %s "
                    "(узел удалён?) -> пропущено",
                    a, b,
                )
                return 0.0
            logger.info(
                "[EDGE CREATED] %s <-> %s weight=%.3f", a, b, initial_weight
            )
            return initial_weight

        new_weight = min(max_weight, row["weight"] + weight_boost)
        cursor.execute(
            "UPDATE edges SET weight = ?, last_activated = ?, last_decayed_at = ? WHERE id = ?",
            (new_weight, ts, ts, row["id"]),
        )
        self._conn.commit()
        logger.info(
            "[EDGE REINFORCED] %s <-> %s weight %.3f -> %.3f",
            a, b, row["weight"], new_weight,
        )
        return new_weight

    def get_edges_for_node(self, node_id: int) -> List[sqlite3.Row]:
        """
        Возвращает все рёбра, инцидентные node_id (в любом направлении
        хранения node_from/node_to), вместе с id соседнего узла как
        отдельным вычисляемым полем `neighbor_id`.
        """
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT
                id,
                weight,
                last_activated,
                node_from,
                node_to,
                CASE WHEN node_from = ? THEN node_to ELSE node_from END AS neighbor_id
            FROM edges
            WHERE node_from = ? OR node_to = ?
            """,
            (node_id, node_id, node_id),
        )
        return cursor.fetchall()

    def fetch_all_edges(self) -> List[sqlite3.Row]:
        """Возвращает все рёбра графа (используется decay-циклом)."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM edges")
        return cursor.fetchall()

    def bulk_update_edge_weights(self, updates: List[Dict[str, Any]]) -> None:
        """
        Массовое обновление весов рёбер. updates = [{"id": 1, "weight": 0.1}, ...]
        Используется decay-циклом рёбер.
        """
        if not updates:
            return
        cursor = self._conn.cursor()
        cursor.executemany(
            "UPDATE edges SET weight = :weight, last_decayed_at = :last_decayed_at WHERE id = :id",
            updates,
        )
        self._conn.commit()

    def delete_edge(self, edge_id: int) -> None:
        """Физически удаляет ребро (используется при decay/забывании связи)."""
        cursor = self._conn.cursor()
        cursor.execute("DELETE FROM edges WHERE id = ?", (edge_id,))
        self._conn.commit()
        logger.info("[EDGE FORGOTTEN] id=%s удалено из БД", edge_id)

    # ----------------------------------------------------------------------
    # SLEEP CYCLE — батч-прунинг и поиск орфанов
    # ----------------------------------------------------------------------

    def delete_edges_below_weight(self, min_weight: float) -> int:
        """
        Батч-удаление ВСЕХ рёбер с weight < min_weight. Используется при
        синаптическом прунинге во время фазы сна (SleepCycle.run_sleep_cycle).

        Возвращает количество удалённых рёбер.
        """
        cursor = self._conn.cursor()
        cursor.execute("SELECT COUNT(*) as cnt FROM edges WHERE weight < ?", (min_weight,))
        count_row = cursor.fetchone()
        deleted_count = count_row["cnt"] if count_row else 0

        if deleted_count:
            cursor.execute("DELETE FROM edges WHERE weight < ?", (min_weight,))
            self._conn.commit()
            logger.info(
                "[SLEEP PRUNING] Удалено рёбер с weight < %.3f: %d",
                min_weight, deleted_count,
            )

        return deleted_count

    def get_orphan_nodes(self, min_edge_weight: float, max_node_weight: float) -> List[sqlite3.Row]:
        """
        Возвращает узлы, у которых НЕТ ни одного ребра с weight >= min_edge_weight
        (то есть узел фактически изолирован от ассоциативной сети), И
        текущий вес самого узла ниже max_node_weight (слабые, малозначимые
        воспоминания — сильные изолированные узлы НЕ считаются орфанами
        и не удаляются, даже без связей). Мета-узлы (is_meta=1) полностью
        иммунны и никогда не попадают в этот запрос.

        node_type='word' ТАКЖЕ иммунен: словарный запас — это медленно
        растущая инфраструктура (vocabulary_size), а не транзиентное
        эпизодическое воспоминание. Свежесозданный word-узел почти всегда
        технически "осиротевший" на границе EDGE_ACTIVATION_THRESHOLD сразу
        после первого decay-тика (SYLLABLE_WORD_EDGE_WEIGHT граничит с этим
        порогом) — без иммунитета слово удалялось бы до того, как успеет
        повториться и закрепиться, и vocabulary_size никогда бы не рос.
        Естественное забывание для word-узлов всё равно происходит — через
        обычный decay до FORGET_THRESHOLD (медленнее и мягче, чем orphan-
        прунинг сна).

        Используется при синаптическом прунинге фазы сна для удаления
        "мусорных" узлов, которые не входят ни в один ассоциативный кластер
        и сами по себе не представляют большой ценности.
        """
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT n.*
            FROM nodes n
            WHERE n.is_meta = 0
              AND n.node_type != 'word'
              AND n.weight < ?
              AND NOT EXISTS (
                  SELECT 1 FROM edges e
                  WHERE (e.node_from = n.id OR e.node_to = n.id)
                    AND e.weight >= ?
              )
            """,
            (max_node_weight, min_edge_weight),
        )
        return cursor.fetchall()

    def get_hub_candidates(self, min_edge_weight: float) -> List[sqlite3.Row]:
        """
        Возвращает узлы, отсортированные по СУММЕ весов их рёбер, где
        каждое учитываемое ребро имеет weight >= min_edge_weight.

        Используется для поиска "хабов" — доминантных очагов активности,
        вокруг которых строится звёздная кластеризация (Hub-and-Spoke)
        в фазе сна.

        ВАЖНО: ограничено node_type IN ('episodic', 'concept') и is_meta=0.
        Без этого фильтра хабами могли становиться служебные лексические
        узлы ('word'/'syllable') — их рёбра (SYLLABLE_WORD_EDGE_WEIGHT,
        WORD_COOCCURRENCE_EDGE_WEIGHT) укрепляются очень быстро при
        повторном употреблении слова и легко перегоняли по hub_score
        настоящие эпизодические воспоминания, из-за чего семантическая
        консолидация во сне "обобщала" бессмысленные пары вида
        User: "привет" | Bot: "привет" (context == response у лексических
        узлов) вместо реальных диалогов.

        Возвращает строки вида: {id, context, response, weight,
        created_at, last_accessed, hub_score}, отсортированные по
        hub_score DESC.
        """
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT
                n.*,
                COALESCE(SUM(
                    CASE WHEN e.weight >= ? THEN e.weight ELSE 0 END
                ), 0) AS hub_score,
                COALESCE(SUM(
                    CASE WHEN e.weight >= ? THEN 1 ELSE 0 END
                ), 0) AS strong_edge_count
            FROM nodes n
            LEFT JOIN edges e ON (e.node_from = n.id OR e.node_to = n.id)
            WHERE n.is_meta = 0
              AND n.node_type IN ('episodic', 'concept')
            GROUP BY n.id
            HAVING strong_edge_count > 0
            ORDER BY hub_score DESC
            """,
            (min_edge_weight, min_edge_weight),
        )
        return cursor.fetchall()

    def get_significant_nodes_since(
        self, min_created_at: float, limit: int
    ) -> List[sqlite3.Row]:
        """
        Возвращает до `limit` "значимых" узлов LTM (node_type='episodic',
        is_meta=0 — то есть спайк-узлы, эмоциональные/структурные узлы
        консолидации и абстрактные узлы сна, но НЕ лексика/концепты/мета-
        узлы), созданных ПОСЛЕ min_created_at, отсортированных по весу
        (сильнейшие/самые эмоционально значимые — первыми).

        Используется эволюцией Self-Model во время сна (Итерация H) —
        строит "дайджест" опыта, накопленного с прошлого сна.
        """
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT * FROM nodes
            WHERE is_meta = 0
              AND node_type = 'episodic'
              AND created_at > ?
            ORDER BY weight DESC
            LIMIT ?
            """,
            (min_created_at, limit),
        )
        return cursor.fetchall()

    # ----------------------------------------------------------------------
    # SELF-MODEL & USER-MODEL — мета-узлы самосознания (Итерация 15)
    # ----------------------------------------------------------------------

    def get_meta_node(self, node_type: str) -> Optional[sqlite3.Row]:
        """
        Возвращает мета-узел заданного типа ('self_model' или 'user_model'),
        либо None, если он ещё не создан.
        """
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT * FROM nodes WHERE is_meta = 1 AND node_type = ? LIMIT 1",
            (node_type,),
        )
        return cursor.fetchone()

    def upsert_meta_node(
        self,
        node_type: str,
        content: str,
        weight: float = 0.95,
        timestamp: Optional[float] = None,
    ) -> int:
        """
        Создаёт мета-узел заданного типа, если он не существует, либо
        обновляет его содержимое (context/response), если уже существует.
        context и response дублируют одно и то же содержимое мета-узла —
        для мета-узлов различие context/response не имеет смысла, это
        просто единый "слот" самосознания/образа пользователя.

        Возвращает id мета-узла.
        """
        ts = timestamp if timestamp is not None else time.time()
        existing = self.get_meta_node(node_type)

        cursor = self._conn.cursor()

        if existing is None:
            cursor.execute(
                """
                INSERT INTO nodes (context, response, weight, created_at, last_accessed, is_meta, node_type)
                VALUES (?, ?, ?, ?, ?, 1, ?)
                """,
                (content, content, weight, ts, ts, node_type),
            )
            self._conn.commit()
            node_id = cursor.lastrowid
            logger.info(
                "[META NODE CREATED] type=%s id=%s weight=%.2f",
                node_type, node_id, weight,
            )
            return node_id

        cursor.execute(
            """
            UPDATE nodes
            SET context = ?, response = ?, weight = ?, last_accessed = ?
            WHERE id = ?
            """,
            (content, content, weight, ts, existing["id"]),
        )
        self._conn.commit()
        logger.info(
            "[META NODE UPDATED] type=%s id=%s weight=%.2f",
            node_type, existing["id"], weight,
        )
        return existing["id"]

    # ----------------------------------------------------------------------
    # CONCEPT EXTRACTION — семантические понятия (Итерация 16)
    # ----------------------------------------------------------------------

    def get_concept_node_by_name(self, name: str) -> Optional[sqlite3.Row]:
        """
        Возвращает concept-узел по точному совпадению нормализованного
        имени (хранится в поле context), либо None, если такого понятия
        ещё нет в графе знаний.
        """
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT * FROM nodes WHERE node_type = 'concept' AND context = ? LIMIT 1",
            (name,),
        )
        return cursor.fetchone()

    def upsert_concept_node(
        self,
        name: str,
        definition: str,
        weight: float = 0.7,
        timestamp: Optional[float] = None,
    ) -> "tuple[int, bool]":
        """
        Создаёт новый concept-узел (context=name, response=definition,
        node_type='concept'), либо, если понятие с таким именем уже
        существует, ОБНОВЛЯЕТ его определение (response) и слегка
        усиливает вес (повторное объяснение того же термина укрепляет
        память о нём).

        Возвращает (node_id, was_created), где was_created=True, если
        узел был создан впервые, False — если это обновление существующего.
        """
        ts = timestamp if timestamp is not None else time.time()
        existing = self.get_concept_node_by_name(name)

        cursor = self._conn.cursor()

        if existing is None:
            cursor.execute(
                """
                INSERT INTO nodes (context, response, weight, created_at, last_accessed, last_decayed_at, node_type)
                VALUES (?, ?, ?, ?, ?, ?, 'concept')
                """,
                (name, definition, weight, ts, ts, ts),
            )
            self._conn.commit()
            node_id = cursor.lastrowid
            logger.info(
                "[CONCEPT CREATED] id=%s name=%r weight=%.2f",
                node_id, name, weight,
            )
            return node_id, True

        new_weight = min(1.0, existing["weight"] + 0.05)
        cursor.execute(
            """
            UPDATE nodes
            SET response = ?, weight = ?, last_accessed = ?, last_decayed_at = ?
            WHERE id = ?
            """,
            (definition, new_weight, ts, ts, existing["id"]),
        )
        self._conn.commit()
        logger.info(
            "[CONCEPT UPDATED] id=%s name=%r weight %.2f -> %.2f",
            existing["id"], name, existing["weight"], new_weight,
        )
        return existing["id"], False

    # ----------------------------------------------------------------------
    # LEXICAL ACQUISITION — word/syllable узлы (освоение языка "с нуля")
    # ----------------------------------------------------------------------

    def get_lexical_node(self, node_type: str, text: str) -> Optional[sqlite3.Row]:
        """
        Возвращает лексический узел (node_type='word' или 'syllable') по
        точному совпадению нормализованного текста (хранится в context),
        либо None, если такой узел ещё не создан.
        """
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT * FROM nodes WHERE node_type = ? AND context = ? LIMIT 1",
            (node_type, text),
        )
        return cursor.fetchone()

    def upsert_lexical_node(
        self,
        node_type: str,
        text: str,
        initial_weight: float,
        reinforce_step: float,
        timestamp: Optional[float] = None,
        max_weight: float = 1.0,
    ) -> "tuple[int, bool]":
        """
        Создаёт новый лексический узел (context=response=text, node_type=
        'word'/'syllable') с весом initial_weight, либо, если такой токен
        уже встречался ранее, УСИЛИВАЕТ его вес на reinforce_step (не выше
        max_weight) — имитация постепенного "усвоения" слова/слога через
        повторение (частотность = освоенность).

        Возвращает (node_id, was_created).
        """
        ts = timestamp if timestamp is not None else time.time()
        existing = self.get_lexical_node(node_type, text)

        cursor = self._conn.cursor()

        if existing is None:
            cursor.execute(
                """
                INSERT INTO nodes (context, response, weight, created_at, last_accessed, last_decayed_at, node_type)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (text, text, initial_weight, ts, ts, ts, node_type),
            )
            self._conn.commit()
            node_id = cursor.lastrowid
            logger.debug(
                "[LEXICAL CREATED] type=%s id=%s text=%r weight=%.3f",
                node_type, node_id, text, initial_weight,
            )
            return node_id, True

        new_weight = min(max_weight, existing["weight"] + reinforce_step)
        cursor.execute(
            "UPDATE nodes SET weight = ?, last_accessed = ?, last_decayed_at = ? WHERE id = ?",
            (new_weight, ts, ts, existing["id"]),
        )
        self._conn.commit()
        logger.debug(
            "[LEXICAL REINFORCED] type=%s id=%s text=%r weight %.3f -> %.3f",
            node_type, existing["id"], text, existing["weight"], new_weight,
        )
        return existing["id"], False

    def count_nodes_by_type(self, node_type: str) -> int:
        """Возвращает количество узлов заданного node_type (например, 'word')."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT COUNT(*) as cnt FROM nodes WHERE node_type = ?", (node_type,))
        row = cursor.fetchone()
        return row["cnt"] if row else 0

    def count_mastered_words(self, min_weight: float) -> int:
        """
        Возвращает количество word-узлов с weight >= min_weight — то есть
        слов, которые были ЗАКРЕПЛЕНЫ повторным употреблением, а не просто
        услышаны один раз. Используется вместо count_nodes_by_type('word')
        там, где важно честное "усвоение" языка (гейтинг речевых стадий),
        а не сырой факт разового контакта со словом.
        """
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) as cnt FROM nodes WHERE node_type = 'word' AND weight >= ?",
            (min_weight,),
        )
        row = cursor.fetchone()
        return row["cnt"] if row else 0

    def get_random_nodes_by_type(self, node_type: str, limit: int) -> List[sqlite3.Row]:
        """
        Возвращает до `limit` случайных узлов заданного node_type —
        используется инстинктом лепета (babbling) для выбора известных
        слогов при генерации "лепетного" ответа.
        """
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT * FROM nodes WHERE node_type = ? ORDER BY RANDOM() LIMIT ?",
            (node_type, limit),
        )
        return cursor.fetchall()

    # ----------------------------------------------------------------------

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()