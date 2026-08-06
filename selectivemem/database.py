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
 DATABASE.PY — The SQLite storage layer
================================================================================
Responsible ONLY for low-level database access: creating the schema and
CRUD over the `nodes` table. All the "biological" logic — decay, spike
thresholds, finding similar context — lives in graph_memory.py.

Schema of the `nodes` table:
    id            INTEGER PRIMARY KEY AUTOINCREMENT
    context       TEXT     -- incoming context (the user's message)
    response      TEXT     -- the system's reply tied to that context
    weight        REAL     -- current "weight" of the link (memory strength), 0..1+
    created_at    REAL     -- unix timestamp of creation
    last_accessed REAL     -- unix timestamp of the last touch
================================================================================
"""

import random
import sqlite3
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


# ============================================================================
#  ДВА ХРАНИЛИЩА: ГИППОКАМП И КОРА
# ============================================================================
# Разделение не косметическое, и вот довод, который решает.
#
# В таблице nodes колонка `weight` означала РАЗНОЕ для разных строк:
#   * у эпизода — важность события, и она затухает по часам;
#   * у слова   — освоенность, и она РАСТЁТ от повторений.
#
# Одна колонка, два несовместимых смысла. Разные сроки жизни
# (age_t0 семь часов против lexical_age_t0 тридцати суток) были заплаткой
# ровно поверх этого.
#
# В мозге это два вещества с разными свойствами, и не по прихоти: сеть с
# общими весами не может одновременно писать с одного раза и не разрушать
# старое. Гиппокамп пишет мгновенно и раздельно, кора накапливает медленно
# и с перекрытием. Отсюда два органа, а не один.
#
# НОМЕРА ОБЩИЕ. Рёбра пересекают границу — слово к слову, эпизод к
# понятию, схема к источнику, — поэтому идентификатор обязан оставаться
# уникальным по обоим хранилищам. Их раздаёт node_seq.
# ============================================================================


# ВИД `nodes` НАД ДВУМЯ ХРАНИЛИЩАМИ.
#
# Сорок три запроса читают из `nodes`, и переписывать их все — риск на
# ровном месте. Вид оставляет чтение как есть и приводит два разных
# набора колонок к одному: у коры нет силы спайка и ожидания награды, у
# эпизода нет числа встреч.
#
# is_meta выводится из рода: модель себя, модель собеседника и эпоха
# мозга — это состояние, а не воспоминание, и вытеснение по ёмкости их
# не трогает.
NODES_VIEW = """
CREATE VIEW IF NOT EXISTS nodes AS
    SELECT id, context, response, weight, created_at, last_accessed,
           0 AS is_meta, 'episodic' AS node_type, last_decayed_at,
           stability, reward_expectation, embedding, spike_strength, strength
    FROM episodes
    UNION ALL
    SELECT id, text AS context, meaning AS response, weight, created_at,
           last_accessed,
           is_meta,
           kind AS node_type, last_decayed_at,
           1.0 AS stability, reward_expectation, embedding,
           NULL AS spike_strength, strength
    FROM cortex;
"""

EPISODES_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
    context, response, content='episodes', content_rowid='id',
    tokenize='unicode61'
);
"""

NODE_SEQ_SCHEMA = """
CREATE TABLE IF NOT EXISTS node_seq (
    id INTEGER PRIMARY KEY AUTOINCREMENT
);
"""

EPISODES_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id                 INTEGER PRIMARY KEY,
    context            TEXT NOT NULL,
    response           TEXT NOT NULL,
    weight             REAL DEFAULT 1.0,
    strength           REAL,
    created_at         REAL,
    last_accessed      REAL,
    last_decayed_at    REAL,
    stability          REAL DEFAULT 1.0,
    spike_strength     REAL,
    reward_expectation REAL DEFAULT 0.0,
    embedding          BLOB
);
"""

CORTEX_SCHEMA = """
CREATE TABLE IF NOT EXISTS cortex (
    id                 INTEGER PRIMARY KEY,
    kind               TEXT NOT NULL,
    text               TEXT NOT NULL,
    meaning            TEXT NOT NULL DEFAULT '',
    weight             REAL DEFAULT 1.0,
    strength           REAL,
    occurrences        INTEGER DEFAULT 1,
    reward_expectation REAL DEFAULT 0.0,
    is_meta            INTEGER DEFAULT 0,
    created_at         REAL,
    last_accessed      REAL,
    last_decayed_at    REAL,
    embedding          BLOB
);
"""

# `occurrences` — то, чего у эпизода нет и быть не может: событие
# случилось один раз. Корковая запись именно НАКАПЛИВАЕТ, и до сих пор
# этот счёт был свален в `weight` вместе с важностью.

INDEX_CORTEX_KIND = """
CREATE INDEX IF NOT EXISTS idx_cortex_kind ON cortex(kind, text);
"""

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
# Schema of the `edges` table — associative links between long-term nodes
# (semantic edges / spreading activation)
# --------------------------------------------------------------------------
EDGES_SCHEMA = """
CREATE TABLE IF NOT EXISTS edges (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    node_from      INTEGER NOT NULL,
    node_to        INTEGER NOT NULL,
    weight         REAL NOT NULL DEFAULT 0.2,
    last_activated REAL NOT NULL
);
-- ВНЕШНЕГО КЛЮЧА ЗДЕСЬ НЕТ, и это следствие разъезда на два хранилища.
-- Связь может вести из эпизода в понятие, из слова в слово, из схемы в
-- источник — то есть в ЛЮБОЕ из двух хранилищ. Ссылка на одну таблицу
-- этого не выражает, а ссылаться на вид SQLite не умеет.
--
-- Целостность держится тем, что номера раздаёт node_seq и они уникальны
-- по обоим хранилищам, а удаление узла чистит связи явно.
"""

INDEX_EDGE_FROM = """
CREATE INDEX IF NOT EXISTS idx_edges_from ON edges(node_from);
"""

INDEX_EDGE_TO = """
CREATE INDEX IF NOT EXISTS idx_edges_to ON edges(node_to);
"""

# Uniqueness of the directed pair (node_from, node_to). Undirectedness
# (A->B equals B->A) is enforced in graph_memory.py by normalising the id
# order before insertion (see Database.upsert_edge).
UNIQUE_EDGE_PAIR = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_edges_unique_pair ON edges(node_from, node_to);
"""

# --------------------------------------------------------------------------
# Полнотекстовый индекс — чтобы ЗАПИСЬ перестала перебирать всю базу.
#
# Проверка на устаревание вызывается при КАЖДОЙ записи и сканировала все
# узлы, считая косинус для каждого. Замер: три тысячи узлов не удавалось
# записать за две минуты, тогда как прогретый ПОИСК по тем же трём тысячам
# занимает миллисекунды.
#
# То есть при росте памяти первой становится невыносимой запись, а не
# чтение, — и индексировать надо ради вставки.
#
# FTS5 входит в сам SQLite (проверено: версия 3.45.1, модуль доступен),
# поэтому обещание "ноль зависимостей" остаётся в силе. Токенизатор
# unicode61 работает с кириллицей.
#
# content='nodes' означает внешнее хранилище: индекс не дублирует тексты,
# а ссылается на строки таблицы nodes по rowid.
# --------------------------------------------------------------------------
FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
    context, response, content='nodes', content_rowid='id', tokenize='unicode61'
);
"""

# --------------------------------------------------------------------------
# Migration: meta-nodes for the self model and the user model
# --------------------------------------------------------------------------
ALTER_ADD_IS_META = """
ALTER TABLE nodes ADD COLUMN is_meta INTEGER DEFAULT 0;
"""

ALTER_ADD_NODE_TYPE = """
ALTER TABLE nodes ADD COLUMN node_type TEXT DEFAULT NULL;
"""
# --------------------------------------------------------------------------
# Migration: a separate decay "clock" (last_decayed_at) that does NOT
# coincide with last_accessed/last_activated — the fix for compounding decay.
#
# last_accessed / last_activated = "when the node/edge was REALLY used"
#   (needed for proactive cooldown, relevance scoring and so on)
# last_decayed_at                = "from which moment to count the weight
#   fading" — a purely technical mark for the decay formula.
#
# Decay used to measure dt from last_accessed, but apply_decay runs on
# EVERY message for EVERY node, while last_accessed only moves on a real
# touch. Fading therefore accumulated incorrectly — quadratically in the
# number of messages instead of linearly in time.
# --------------------------------------------------------------------------
ALTER_ADD_LAST_DECAYED_NODES = """
ALTER TABLE nodes ADD COLUMN last_decayed_at REAL DEFAULT NULL;
"""
ALTER_ADD_LAST_DECAYED_EDGES = """
ALTER TABLE edges ADD COLUMN last_decayed_at REAL DEFAULT NULL;
"""
# --------------------------------------------------------------------------
# Migration: stability — a node's resistance to being forgotten.
#
# weight    = "how vividly it is remembered right now"
# stability = "how slowly it fades" (a multiplier on AGE_T0)
#
# There used to be only weight, and every episode aged on one scale
# regardless of how useful it had been. Stability now grows on every
# recall (see update_last_accessed), so what proved useful lives for
# months while an accidental exchange is gone within a day.
# --------------------------------------------------------------------------
ALTER_ADD_STABILITY = """
ALTER TABLE nodes ADD COLUMN stability REAL DEFAULT 1.0;
"""
# --------------------------------------------------------------------------
# Migration: reward_expectation — the approval expected for using a node.
#
# Needed for the dopamine signal: dopamine is released not by reward but
# by UNEXPECTED reward, so the organism needs something to compare the
# user's actual rating against. Updated by the Rescorla-Wagner rule (see
# MemoryGraph.apply_reward); the range matches valence: [-1.0, 1.0].
# --------------------------------------------------------------------------
ALTER_ADD_REWARD_EXPECTATION = """
ALTER TABLE nodes ADD COLUMN reward_expectation REAL DEFAULT 0.0;
"""
# --------------------------------------------------------------------------
# Migration: spike_strength — the density of the event that CREATED the node.
#
# The initial weight already equals that density, but weight decays, so by
# the time forgetting has to decide anything the birth signal is gone.
#
# Why it is needed. Decay does not merely lower a weight, it DELETES a node
# once it falls below forget_threshold, and measurement showed what that
# costs: for LongMemEval knowledge-update questions the evidence is about
# 16 days older than the question, and every evidence node was erased —
# 12 of 12 across five instances — while only a tenth of memory went. A
# flat floor for everything fixed retrieval (18% -> 85%) and destroyed
# selectivity in the same move (the praised-over-routine gap fell from
# +40 pp to zero): with it, nothing is ever forgotten.
#
# So the floor has to remember how hard the event hit. A strong spike earns
# protection from age; routine that barely cleared the gate earns none.
# --------------------------------------------------------------------------
ALTER_ADD_SPIKE_STRENGTH = """
ALTER TABLE nodes ADD COLUMN spike_strength REAL DEFAULT NULL;
"""
# --------------------------------------------------------------------------
# Migration: edges.edge_type — what kind of link this is.
#
# Everything used to fade at one rate, and the rate was calibrated for the
# WORD graph, where a link is reinforced by every repetition of a phrase
# and must go stale quickly if the word stops being said.
#
# Associations between episodes have no such repetition: a conversation
# does not return to the same PAIR of remarks ten times. Measured: links
# created during a conversation vanish entirely between the fourth and the
# eleventh day, taking multi-hop retrieval with them — the benefit falls
# from +6.7 points to zero after a week of silence.
#
# NULL means the old kind (lexical/concept) and keeps the old rate.
# --------------------------------------------------------------------------
ALTER_ADD_EDGE_TYPE = """
ALTER TABLE edges ADD COLUMN edge_type TEXT DEFAULT NULL;
"""
# --------------------------------------------------------------------------
# Migration: nodes.strength — importance that a CLOCK CANNOT TOUCH.
#
# `weight` has been doing three jobs at once: recency (it decays),
# importance (reinforcement raises it) and retrieval strength (it is a term
# in the search score). Nearly every defect measured over these two days
# grew out of that conflation:
#
#   - re-ranking BY IMPORTANCE turned out to be re-ranking BY AGE, because
#     with no feedback weight is nothing but elapsed time. Widening the
#     band dropped R@1 from 32% to 18%;
#   - age-based deletion erased the evidence for every knowledge-update
#     question — 12 nodes of 12 — while removing only a tenth of memory;
#   - sleep marks archived sources by LOWERING weight, and capacity
#     eviction cannot hear it, because a score built on age would bring
#     the same disease back.
#
# strength accumulates from reinforcement and from being useful, and never
# falls because time passed. What a node is WORTH at retrieval time is its
# SHARE of the total, computed lazily over the candidates — so forgetting
# becomes competition (interference) rather than a clock (decay).
#
# That is also the better-supported theory of human forgetting: we lose
# memories mostly because new learning competes with them, not because a
# timer runs down.
# --------------------------------------------------------------------------
ALTER_ADD_STRENGTH = """
ALTER TABLE nodes ADD COLUMN strength REAL DEFAULT NULL;
"""
# --------------------------------------------------------------------------
# Migration: embedding — a node's meaning vector for semantic search.
#
# Stored as a BLOB (float32) and filled lazily: nodes created before the
# model appeared get their vector on the first search that touches them.
# NULL is a legitimate value meaning "no semantics for this node yet", and
# search then falls back to string similarity.
# --------------------------------------------------------------------------
ALTER_ADD_EMBEDDING = """
ALTER TABLE nodes ADD COLUMN embedding BLOB DEFAULT NULL;
"""
class Database:
    """
    A thin wrapper over SQLite for the `nodes` table.

    Usage:
        db = Database()
        db.insert_node(context="hello", response="hi there!", weight=0.8)
        rows = db.fetch_all_nodes()
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        settings: Optional["MemorySettings"] = None,
    ):
        from selectivemem.settings import MemorySettings

        self.settings = settings or MemorySettings()
        self.db_path = db_path or self.settings.db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # Enable foreign keys — without this SQLite IGNORES the
        # "ON DELETE CASCADE" in the edges schema, and deleting a node
        # would leave orphaned rows behind in `edges`.
        self._conn.execute("PRAGMA foreign_keys = ON;")
        self._init_schema()

    def _init_schema(self) -> None:
        """Creates the nodes/edges tables and their indexes if absent."""
        cursor = self._conn.cursor()
        cursor.execute(SCHEMA)
        # ИНДЕКС СТАВИТСЯ, ТОЛЬКО ПОКА `nodes` — ТАБЛИЦА.
        #
        # На свежей базе SCHEMA создаёт её таблицей, и индекс законен. Но
        # после разъезда на два хранилища `nodes` становится ВИДОМ, а вид
        # индексировать нельзя — SQLite отвечает "views may not be
        # indexed". Значит при КАЖДОМ повторном открытии базы с данными
        # библиотека падала прямо в конструкторе.
        #
        # Не поймал ни один тест: все они открывают ":memory:", то есть
        # всегда свежую базу. Поймал живой запуск примера, где ассистент
        # закрыл базу и открыл её снова — то есть на самом обычном
        # сценарии из всех возможных.
        if self._nodes_is_table():
            cursor.execute(INDEX_CONTEXT)
        cursor.execute(EDGES_SCHEMA)
        cursor.execute(INDEX_EDGE_FROM)
        cursor.execute(INDEX_EDGE_TO)
        cursor.execute(UNIQUE_EDGE_PAIR)
        cursor.execute(FTS_SCHEMA)
        # Разовое наполнение для баз, созданных до появления индекса.
        cursor.execute("SELECT count(*) FROM nodes_fts")
        if cursor.fetchone()[0] == 0:
            cursor.execute(
                "INSERT INTO nodes_fts(rowid, context, response) "
                "SELECT id, context, response FROM nodes"
            )
        self._conn.commit()
        self._migrate_meta_columns()
        self._migrate_decay_columns()
        self._migrate_stability_column()
        self._migrate_two_stores()
        logger.info("[DB INIT] Schema nodes + edges ready (%s)", self.db_path)



    # ------------------------------------------------------------------
    # ЗАПИСЬ В ДВА ХРАНИЛИЩА
    # ------------------------------------------------------------------
    def _nodes_is_table(self) -> bool:
        """
        Старые миграции правят таблицу `nodes`. После разъезда она стала
        видом, и править его нельзя — да и незачем: колонки давно на месте.
        """
        row = self._conn.execute(
            "SELECT type FROM sqlite_master WHERE name = 'nodes'"
        ).fetchone()
        return row is not None and row[0] == "table"

    def _both(self, sql: str, params) -> None:
        """
        Один и тот же запрос по обеим таблицам.

        Работает потому, что НОМЕРА УНИКАЛЬНЫ ПО ОБОИМ ХРАНИЛИЩАМ: их
        раздаёт node_seq. Значит `WHERE id = ?` попадёт ровно в одну
        строку, а вторая таблица честно ответит «ноль изменений».

        Дешевле, чем сначала спрашивать «где лежит этот узел»: оба запроса
        идут по первичному ключу.
        """
        cursor = self._conn.cursor()
        for table in ("episodes", "cortex"):
            cursor.execute(sql.replace("{table}", table), params)
        self._conn.commit()

    def _next_id(self) -> int:
        """Общий номер для обоих хранилищ — на него ссылаются рёбра."""
        cursor = self._conn.cursor()
        cursor.execute("INSERT INTO node_seq DEFAULT VALUES")
        return int(cursor.lastrowid)

    def _migrate_two_stores(self) -> None:
        """
        Разъезд на два хранилища: эпизоды в `episodes`, накопленное в
        `cortex`.

        ПЕРЕНОС ОДНОРАЗОВЫЙ И ПРОВЕРЯЕМЫЙ. Строки копируются по типу, счёт
        сверяется, и если сходится — старая таблица остаётся на месте до
        следующей версии. Терять чужие базы ради стройности нельзя.

        Номера сохраняются как есть: на них ссылаются рёбра, и
        переназначение сломало бы связи молча. node_seq подхватывает
        счётчик с максимума, чтобы новые узлы не столкнулись со старыми.
        """
        cursor = self._conn.cursor()
        cursor.execute(NODE_SEQ_SCHEMA)
        cursor.execute(EPISODES_SCHEMA)
        cursor.execute(CORTEX_SCHEMA)
        cursor.execute(INDEX_CORTEX_KIND)
        self._conn.commit()

        # `nodes` уже вид? Значит разъезд состоялся раньше.
        kind = cursor.execute(
            "SELECT type FROM sqlite_master WHERE name = 'nodes'"
        ).fetchone()
        if kind is None or kind[0] == "view":
            return

        total = cursor.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        if total:
            cursor.execute("""
                INSERT INTO episodes (id, context, response, weight, strength,
                                      created_at, last_accessed, last_decayed_at,
                                      stability, spike_strength, reward_expectation,
                                      embedding)
                SELECT id, context, response, weight, strength, created_at,
                       last_accessed, last_decayed_at, stability, spike_strength,
                       reward_expectation, embedding
                FROM nodes WHERE node_type = 'episodic'
            """)
            cursor.execute("""
                INSERT INTO cortex (id, kind, text, meaning, weight, strength,
                                    occurrences, created_at, last_accessed,
                                    last_decayed_at, embedding)
                SELECT id, COALESCE(node_type, 'concept'), context, response,
                       weight, strength, 1, created_at, last_accessed,
                       last_decayed_at, embedding
                FROM nodes WHERE node_type IS NULL OR node_type <> 'episodic'
            """)
            # Счётчик номеров продолжает с максимума: иначе новый узел
            # получил бы уже занятый номер и ребро указало бы не туда.
            top = cursor.execute("SELECT COALESCE(MAX(id), 0) FROM nodes").fetchone()[0]
            if top:
                cursor.execute("INSERT INTO node_seq (id) VALUES (?)", (top,))
            self._conn.commit()

            ep = cursor.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
            cx = cursor.execute("SELECT COUNT(*) FROM cortex").fetchone()[0]
            if ep + cx != total:
                logger.error(
                    "[TWO STORES] Перенос неполон: было %d, стало %d + %d. "
                    "Старая таблица не тронута, разъезд отменён.", total, ep, cx,
                )
                cursor.execute("DELETE FROM episodes")
                cursor.execute("DELETE FROM cortex")
                self._conn.commit()
                return
            logger.info(
                "[TWO STORES] Разъезд: %d эпизодов, %d корковых узлов из %d",
                ep, cx, total,
            )

        self._replace_nodes_with_view()

    def _replace_nodes_with_view(self) -> None:
        """
        Старая таблица уступает место ВИДУ над двумя хранилищами.

        Делается только после того, как перенос сошёлся по счёту. Чтение
        от этого не меняется: сорок три запроса продолжают спрашивать
        `nodes` и получают объединение.

        Полнотекстовый индекс переезжает на `episodes`: искать по тексту
        нужно события, а не словарь — его узлы служебные и однажды уже
        давали ложные совпадения на любом повторённом слове.
        """
        cursor = self._conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS nodes_fts")
        # Рёбра пересобираются без внешнего ключа: он указывал на таблицу
        # nodes, которая становится видом, а связь теперь может вести в
        # любое из двух хранилищ.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS edges_new (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                node_from      INTEGER NOT NULL,
                node_to        INTEGER NOT NULL,
                weight         REAL NOT NULL DEFAULT 0.2,
                last_activated REAL NOT NULL,
                last_decayed_at REAL DEFAULT NULL,
                edge_type      TEXT DEFAULT NULL
            )
        """)
        cols = [r[1] for r in cursor.execute("PRAGMA table_info(edges)")]
        shared = [c for c in cols if c in
                  ("id", "node_from", "node_to", "weight", "last_activated",
                   "last_decayed_at", "edge_type")]
        cursor.execute(
            f"INSERT INTO edges_new ({','.join(shared)}) "
            f"SELECT {','.join(shared)} FROM edges"
        )
        cursor.execute("DROP TABLE edges")
        cursor.execute("ALTER TABLE edges_new RENAME TO edges")
        cursor.execute(INDEX_EDGE_FROM)
        cursor.execute(INDEX_EDGE_TO)
        cursor.execute(UNIQUE_EDGE_PAIR)
        cursor.execute("DROP TABLE IF EXISTS nodes")
        cursor.execute(NODES_VIEW)
        cursor.execute(EPISODES_FTS_SCHEMA)
        cursor.execute(
            "INSERT INTO episodes_fts(rowid, context, response) "
            "SELECT id, context, response FROM episodes"
        )
        self._conn.commit()
        logger.info("[TWO STORES] nodes стал видом; индекс переехал на episodes")

    def _migrate_meta_columns(self) -> None:
        """
        Adds the is_meta/node_type columns if they do not exist yet (for
        databases created before they were introduced). SQLite's ALTER
        TABLE has no IF NOT EXISTS for columns, so sqlite3.OperationalError
        ("duplicate column name") is caught instead.
        """
        if not self._nodes_is_table():
            return
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
            logger.info("[MIGRATION] Table nodes updated (is_meta, node_type)")

    def _migrate_decay_columns(self) -> None:
        """
        Adds last_decayed_at to nodes and edges — a separate decay clock
        that does NOT coincide with last_accessed/last_activated (see the
        comment near ALTER_ADD_LAST_DECAYED_*).

        Existing rows are backfilled with their current
        last_accessed/last_activated: their decay clock simply starts
        ticking from that moment, with no artificial "debt".
        """
        if not self._nodes_is_table():
            return
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
            logger.info("[MIGRATION] Tables nodes/edges updated (last_decayed_at)")

    def _migrate_stability_column(self) -> None:
        """
        Adds nodes.stability — a multiplier on a node's characteristic
        lifetime (see the comment near ALTER_ADD_STABILITY).

        Existing nodes get STABILITY_INITIAL: they are not punished for
        predating the mechanism, but they get no undeserved immortality
        either — from here on, usefulness decides.
        """
        if not self._nodes_is_table():
            return
        cursor = self._conn.cursor()
        migrated = False
        for statement in (ALTER_ADD_STABILITY, ALTER_ADD_REWARD_EXPECTATION,
                          ALTER_ADD_EMBEDDING, ALTER_ADD_SPIKE_STRENGTH,
                          ALTER_ADD_EDGE_TYPE, ALTER_ADD_STRENGTH):
            try:
                cursor.execute(statement)
                migrated = True
            except sqlite3.OperationalError as exc:
                if "duplicate column name" not in str(exc):
                    raise

        # Explicit backfill: also guards against NULLs if the column was
        # once added without a DEFAULT.
        cursor.execute(
            "UPDATE nodes SET stability = ? WHERE stability IS NULL",
            (self.settings.stability_initial,),
        )
        cursor.execute(
            "UPDATE nodes SET reward_expectation = 0.0 WHERE reward_expectation IS NULL"
        )
        # Узлы, созданные до перехода на интерференцию, получают свою
        # СЕГОДНЯШНЮЮ силу из веса: это лучшее, что о них известно, и
        # переносить их в новую модель с нуля было бы несправедливо.
        cursor.execute("UPDATE nodes SET strength = weight WHERE strength IS NULL")
        self._conn.commit()

        if migrated:
            logger.info(
                "[MIGRATION] Table nodes updated (stability, reward_expectation, embedding)"
            )

    # ----------------------------------------------------------------------
    # CRUD operations
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
        Inserts a new memory node. Returns the id of the created row.

        node_type classifies the nature of the node:
            "episodic" (default) — an ordinary memory or episode
            "concept"            — an explicitly taught notion, term or rule
            "meta"               — service super-nodes (handled separately
                                    through is_meta/upsert_meta_node; they
                                    practically never arrive here directly)
        """
        ts = timestamp if timestamp is not None else time.time()
        cursor = self._conn.cursor()
        # РАЗВИЛКА ПО ХРАНИЛИЩУ, и она здесь единственная в коде.
        #
        # Эпизод — то, что случилось однажды: у него есть сила спайка при
        # рождении и ожидание награды. Корковый узел накапливает: у него
        # есть число встреч, а силы спайка быть не может, потому что не
        # было единичного события.
        node_id = self._next_id()
        if node_type == "episodic":
            cursor.execute(
                """
                INSERT INTO episodes (id, context, response, weight, created_at,
                                      last_accessed, last_decayed_at,
                                      spike_strength, strength)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                # spike_strength — вес при рождении, и он не меняется.
                # Weight продолжит затухать, а забывание должно уметь
                # спросить позже, насколько сильно ударило тогда.
                #
                # strength стартует с того же числа, но живёт по другим
                # правилам: растёт от подкрепления и пользы и НИКОГДА не
                # падает оттого, что прошло время.
                (node_id, context, response, weight, ts, ts, ts, weight, weight),
            )
            cursor.execute(
                "INSERT INTO episodes_fts(rowid, context, response) VALUES (?, ?, ?)",
                (node_id, context, response),
            )
        else:
            cursor.execute(
                """
                INSERT INTO cortex (id, kind, text, meaning, weight, strength,
                                    occurrences, created_at, last_accessed,
                                    last_decayed_at)
                VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
                """,
                (node_id, node_type or "concept", context, response,
                 weight, weight, ts, ts, ts),
            )
        self._conn.commit()
        logger.info(
            "[MEMORY SAVED] id=%s weight=%.3f node_type=%s context=%r",
            node_id, weight, node_type, context[:50],
        )
        return node_id

    def get_node(self, node_id: int) -> Optional[sqlite3.Row]:
        """Returns a single node by id, or None."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM nodes WHERE id = ?", (node_id,))
        return cursor.fetchone()

    def fetch_all_nodes(self) -> List[sqlite3.Row]:
        """All memory nodes (EVERY node_type — used by decay and the sleep
        cycle, where fading must apply identically to any type of node)."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM nodes")
        return cursor.fetchall()

    def document_frequency(self, words: List[str]) -> Dict[str, int]:
        """
        Сколько записей содержит каждое из слов — ПО ИНДЕКСУ, не перебором.

        Нужно для взвешивания совпавших слов по редкости. Считать это
        перебором всех узлов можно, но тогда исчезает смысл предотбора:
        замер на трёх тысячах узлов дал 753 мс против 2206 мс, то есть
        поиск втрое медленнее — ровно тот расход, ради устранения которого
        предотбор и писался.

        Полнотекстовый индекс отвечает на тот же вопрос одним COUNT на
        слово.
        """
        result: Dict[str, int] = {}
        cursor = self._conn.cursor()
        for word in words:
            term = word.replace('"', "")
            if len(term) < 2:
                result[word] = 0
                continue
            try:
                # ТОЛЬКО КОЛОНКА context, и это не мелочь. Индекс покрывает
                # и context, и response, а поиск сравнивает запрос ТОЛЬКО с
                # context. Считать частоту по обоим полям значит взвешивать
                # слова по тому, чего поиск не видит.
                #
                # На стендах разница мала — ответы там «понятно» и «ага». У
                # настоящего ассистента ответы содержательные, и редкость
                # слова оказалась бы систематически заниженной.
                row = cursor.execute(
                    "SELECT COUNT(*) AS n FROM episodes_fts WHERE episodes_fts MATCH ?",
                    ('context: "' + term + '"',),
                ).fetchone()
                result[word] = int(row["n"]) if row else 0
            except sqlite3.OperationalError:
                # Индекса нет (старая база до миграции) — частота
                # неизвестна, и это честнее, чем выдумать её нулём.
                result[word] = 0
        return result

    def fetch_candidates_by_text(self, words: List[str], limit: int) -> List[sqlite3.Row]:
        """
        Узлы, делящие с запросом хоть одно слово, — через полнотекстовый
        индекс, а не перебором.

        Ради этого индекс и заведён: проверка на устаревание идёт при
        КАЖДОЙ записи и раньше сканировала всю базу с косинусом на каждый
        узел. Замер: три тысячи узлов не записывались за две минуты.

        Пустой список означает "ни одного общего слова" — тогда звать
        нечего, и это ЧЕСТНЫЙ ответ: вытеснение всё равно требует
        пересечения не ниже contradiction_min_overlap.
        """
        terms = [w for w in words if len(w) >= 2]
        if not terms:
            return []
        # Кавычки вокруг каждого слова: иначе FTS примет его за оператор.
        query = " OR ".join('"' + t.replace('"', '') + '"' for t in terms)
        cursor = self._conn.cursor()
        try:
            cursor.execute(
                "SELECT n.* FROM episodes_fts f JOIN nodes n ON n.id = f.rowid "
                "WHERE episodes_fts MATCH ? AND n.node_type IN ('episodic', 'concept') "
                "ORDER BY rank LIMIT ?",
                (query, limit),
            )
            return cursor.fetchall()
        except sqlite3.OperationalError:
            # Непарсимый запрос не должен ронять запись.
            return []

    def fetch_summary_nodes(self) -> List[sqlite3.Row]:
        """Только свёрнутые эпизоды — схемы разговора."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM nodes WHERE node_type = 'episode_summary'")
        return cursor.fetchall()

    def fetch_superseded_by(self, node_ids):
        """
        Какие из этих узлов кем-то ЗАМЕНЕНЫ: {старый: новый}.

        Нужно поиску: устаревшее не ослабляется, а перенаправляется, и
        решение принимается в момент выдачи, а не порчей самого узла.
        Один запрос на поиск, а не по одному на кандидата.

        НАПРАВЛЕНИЕ ВЫВОДИТСЯ ИЗ НОМЕРОВ, А НЕ ИЗ РЁБЕР. `upsert_edge`
        нормализует пару по идентификатору — меньший номер всегда в
        `node_from`, — то есть «кто кого заменил» в ребре не хранится.
        Первая версия этого не учла и вычёркивала из выдачи как раз
        АКТУАЛЬНЫЙ факт: два теста поймали это сразу.
        
        Номера выдаёт общий счётчик с автоинкрементом, поэтому у более
        позднего узла номер всегда больше. Этого и достаточно: в паре
        «заменяет» старший номер — замена.
        """
        if not node_ids:
            return {}
        marks = ",".join("?" * len(node_ids))
        cursor = self._conn.cursor()
        rows = cursor.execute(
            f"SELECT node_from, node_to FROM edges "
            f"WHERE edge_type = 'supersedes' "
            f"AND (node_from IN ({marks}) OR node_to IN ({marks}))",
            list(node_ids) + list(node_ids),
        ).fetchall()
        pairs = {}
        for row in rows:
            older, newer = row["node_from"], row["node_to"]
            if older > newer:
                older, newer = newer, older
            pairs[older] = newer
        return pairs

    def fetch_searchable_nodes(self) -> List[sqlite3.Row]:
        """
        Only the nodes fit for semantic search over conversational
        context (MemoryGraph.search) — that is, real memories and
        explicitly taught concepts.

        EXCLUDES 'word'/'syllable' (service nodes of lexical acquisition:
        their response is empty, and letting them into search produced
        false MEMORY HITs on any reused word — the bug where "why" found
        itself with score=1.0) and 'self_model'/'user_model' (meta-nodes
        mixed into the prompt by a separate mechanism, not through
        search).
        """
        cursor = self._conn.cursor()
        cursor.execute(
            # 'fact' — выведенное корой из повторяющегося. Оно участвует в
            # поиске наравне с эпизодами, и это принципиально: кора должна
            # уметь ОТВЕЧАТЬ, когда отдельного события уже не найти.
            #
            # Опасность известна и учтена: свёртки эпизодов в общем поиске
            # роняли R@1 с 76% до 42%, потому что содержали столько слов,
            # что совпадали почти с любым запросом. Факт устроен иначе — он
            # индексируется ОДНОЙ темой, а текст случая несёт в ответе.
            "SELECT * FROM nodes WHERE node_type IN ('episodic', 'concept', 'fact')"
        )
        return cursor.fetchall()

    def update_weight(self, node_id: int, new_weight: float) -> None:
        """Updates a node's weight (after decay or reinforcement, say)."""
        cursor = self._conn.cursor()
        self._both("UPDATE {table} SET weight = ? WHERE id = ?", (new_weight, node_id))

    def update_last_accessed(self, node_id: int, timestamp: Optional[float] = None) -> None:
        """
        Registers a RECALL of the node. Does three things in one query:

            1. last_accessed    — when the node was actually used
                                  (needed for proactive cooldown and scoring);
            2. last_decayed_at  — the decay origin moves forward
                                  (see _migrate_decay_columns);
            3. stability        — GROWS multiplicatively, up to STABILITY_MAX.

        Point 3 is the spacing effect: every successful retrieval makes a
        memory not only fresher but STURDIER, increasing its effective
        lifetime (AGE_T0 * stability). Thanks to that an accidental
        exchange is gone within a day, while what the organism genuinely
        returned to lives for months.

        A single UPDATE instead of read-then-write is deliberate:
        touch_node is called for every search hit and for every node
        pulled in by spreading activation.
        """
        ts = timestamp if timestamp is not None else time.time()
        cursor = self._conn.cursor()
        cursor.execute(
            """
            UPDATE episodes
            SET last_accessed = ?,
                last_decayed_at = ?,
                stability = MIN(?, COALESCE(stability, ?) * ?),
                strength = MIN(?, COALESCE(strength, weight) + ?)
            WHERE id = ?
            """,
            (ts, ts, self.settings.stability_max, self.settings.stability_initial,
             self.settings.stability_growth_factor, self.settings.strength_max,
             self.settings.strength_use_step, node_id),
        )
        # У КОРЫ СТАБИЛЬНОСТИ НЕТ, и это не упущение: стабильность —
        # свойство следа о событии, продлевающее его срок. Корковое знание
        # живёт не сроком, а числом встреч.
        cursor.execute(
            """
            UPDATE cortex
            SET last_accessed = ?,
                last_decayed_at = ?,
                strength = MIN(?, COALESCE(strength, weight) + ?)
            WHERE id = ?
            """,
            (ts, ts, self.settings.strength_max,
             self.settings.strength_use_step, node_id),
        )
        self._conn.commit()

    def add_strength(self, node_id: int, delta: float, cap: float) -> None:
        """
        Меняет накопленную силу узла. Отдельно от веса намеренно: вес
        затухает от времени, сила — нет. Отрицательная дельта допустима,
        ниже нуля не опускаемся.
        """
        cursor = self._conn.cursor()
        self._both(
            "UPDATE {table} SET strength = "
            "MAX(0.0, MIN(?, COALESCE(strength, weight) + ?)) WHERE id = ?",
            (cap, delta, node_id),
        )

    def update_embedding(self, node_id: int, blob: Optional[bytes]) -> None:
        """
        Stores a node's meaning vector. A separate method rather than
        part of insertion: nodes created before the model existed are
        filled in lazily on the first search.
        """
        cursor = self._conn.cursor()
        self._both("UPDATE {table} SET embedding = ? WHERE id = ?", (blob, node_id))

    def update_stability(self, node_id: int, stability: float) -> None:
        """
        Sets a node's resistance to forgetting directly. Needed by
        supersession: a stale version of a fact returns to the forgettable
        pile rather than being deleted.
        """
        cursor = self._conn.cursor()
        cursor.execute("UPDATE episodes SET stability = ? WHERE id = ?", (stability, node_id))
        self._conn.commit()

    def update_reward_expectation(self, node_id: int, expectation: float) -> None:
        """
        Stores the new expected approval for using a node.

        Deliberately does NOT touch last_accessed/last_decayed_at:
        receiving a rating is not the same as recalling. Otherwise any
        reaction from the user would extend a node's life, including
        criticism.
        """
        cursor = self._conn.cursor()
        self._both(
            "UPDATE {table} SET reward_expectation = ? WHERE id = ?",
            (expectation, node_id),
        )

    def delete_node(self, node_id: int) -> None:
        """Physically deletes a node (used by the sleep cycle when forgetting)."""
        cursor = self._conn.cursor()
        cursor.execute("DELETE FROM episodes_fts WHERE rowid = ?", (node_id,))
        cursor.execute("DELETE FROM episodes WHERE id = ?", (node_id,))
        cursor.execute("DELETE FROM cortex WHERE id = ?", (node_id,))
        self._conn.commit()
        logger.info("[MEMORY FORGOTTEN] id=%s deleted from the database", node_id)

    def bulk_update_weights(self, updates: List[Dict[str, Any]]) -> None:
        """
        Bulk weight update. updates = [{"id": 1, "weight": 0.4,
        "last_decayed_at": 12345.0}, ...]
        Used by the decay cycle for an efficient batch update. Every row
        MUST carry last_decayed_at (the new decay origin) — otherwise the
        next decay pass would measure dt from the old mark and fading
        would accumulate incorrectly.
        """
        cursor = self._conn.cursor()
        cursor.executemany(
            "UPDATE episodes SET weight = :weight, last_decayed_at = :last_decayed_at WHERE id = :id",
            updates,
        )
        # СЛОВАРЬ ТОЖЕ ЗАТУХАЕТ, только медленнее: тридцать суток против
        # семи часов. Направить затухание только в эпизоды значило бы
        # сделать словарь вечным.
        cursor.executemany(
            "UPDATE cortex SET weight = :weight, last_decayed_at = :last_decayed_at WHERE id = :id",
            updates,
        )
        self._conn.commit()

    def close(self) -> None:
        """Closes the database connection."""
        self._conn.close()

    # ----------------------------------------------------------------------
    # EDGES — associative links between nodes (semantic edges)
    # ----------------------------------------------------------------------

    def upsert_edge(
        self,
        node_from: int,
        node_to: int,
        weight_boost: float,
        timestamp: Optional[float] = None,
        max_weight: float = 1.0,
        edge_type: Optional[str] = None,
    ) -> float:
        """
        Creates a new edge node_from -> node_to with initial weight
        weight_boost, or, if the edge exists, INCREASES its weight by
        weight_boost (capped at max_weight). Updates last_activated.

        The pair is normalised so storage is symmetric — the smaller id is
        always saved first, which avoids A->B and B->A becoming two rows.

        Returns the resulting edge weight.
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
                    INSERT INTO edges (node_from, node_to, weight, last_activated,
                                       last_decayed_at, edge_type)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (a, b, initial_weight, ts, ts, edge_type),
                )
                self._conn.commit()
            except sqlite3.IntegrityError:
                # Last line of defence: one of the nodes was deleted
                # between the existence check (MemoryGraph.connect_nodes)
                # and this insert — an extremely narrow race window —
                # tripping the FOREIGN KEY constraint. Rather than let
                # that take down the whole message handler, the edge is
                # simply skipped.
                self._conn.rollback()
                logger.warning(
                    "[EDGE SKIP] FOREIGN KEY constraint while creating edge %s <-> %s "
                    "(node deleted?) -> skipped",
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
        All edges incident to node_id (in either stored direction),
        together with the neighbouring node's id as a computed field
        `neighbor_id`.
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
        """All edges in the graph (used by the decay cycle)."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM edges")
        return cursor.fetchall()

    def bulk_update_edge_weights(self, updates: List[Dict[str, Any]]) -> None:
        """
        Bulk edge weight update. updates = [{"id": 1, "weight": 0.1}, ...]
        Used by the edge decay cycle.
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
        """Physically deletes an edge (used when a link decays away)."""
        cursor = self._conn.cursor()
        cursor.execute("DELETE FROM edges WHERE id = ?", (edge_id,))
        self._conn.commit()
        logger.info("[EDGE FORGOTTEN] id=%s deleted from the database", edge_id)

    # ----------------------------------------------------------------------
    # SLEEP CYCLE — batch pruning and orphan search
    # ----------------------------------------------------------------------

    def delete_edges_below_weight(self, min_weight: float) -> int:
        """
        Bulk deletion of ALL edges with weight < min_weight. Used for
        synaptic pruning during the sleep phase.

        Returns the number of edges deleted.
        """
        cursor = self._conn.cursor()
        cursor.execute("SELECT COUNT(*) as cnt FROM edges WHERE weight < ?", (min_weight,))
        count_row = cursor.fetchone()
        deleted_count = count_row["cnt"] if count_row else 0

        if deleted_count:
            cursor.execute("DELETE FROM edges WHERE weight < ?", (min_weight,))
            self._conn.commit()
            logger.info(
                "[SLEEP PRUNING] Edges removed with weight < %.3f: %d",
                min_weight, deleted_count,
            )

        return deleted_count

    def get_orphan_nodes(self, min_edge_weight: float, max_node_weight: float) -> List[sqlite3.Row]:
        """
        Nodes that have NO edge with weight >= min_edge_weight (that is,
        the node is effectively cut off from the associative network) AND
        whose own weight is below max_node_weight (weak, unimportant
        memories — strong isolated nodes are NOT orphans and are not
        deleted even without links). Meta-nodes (is_meta=1) are fully
        immune and never appear in this query.

        LEXICAL nodes ('word' AND 'syllable') are immune: vocabulary is
        slowly growing infrastructure, not a transient episodic memory. A
        freshly created word node is almost always technically "orphaned"
        right at EDGE_ACTIVATION_THRESHOLD after the first decay tick
        (SYLLABLE_WORD_EDGE_WEIGHT sits on that boundary) — without
        immunity a word would be deleted before it had a chance to recur
        and take hold, and the vocabulary would never grow.

        'syllable' was NOT immune originally, though by the same logic it
        should have been: a syllable is held by an edge of
        SYLLABLE_WORD_EDGE_WEIGHT=0.45, but edges decay faster than nodes
        (EDGE_DECAY_RATE=0.08 against DECAY_RATE=0.05), so the link sooner
        or later drops below EDGE_ACTIVATION_THRESHOLD=0.3 — and the
        syllable was deleted during sleep, eroding the very substrate that
        pre-verbal speech is built from.

        Natural forgetting still applies to vocabulary — through ordinary
        decay down to FORGET_THRESHOLD, but on its own timescale
        (lexical_age_t0, ~30 days) rather than through the sleep phase's
        hard orphan pruning.

        Used during synaptic pruning to remove "junk" nodes that belong to
        no associative cluster and carry little value on their own.
        """
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT n.*
            FROM nodes n
            WHERE n.is_meta = 0
              AND n.node_type NOT IN ('word', 'syllable')
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
        Nodes sorted by the SUM of their edge weights, counting only
        edges with weight >= min_edge_weight.

        Used to find "hubs" — dominant centres of activity around which
        hub-and-spoke clustering is built during sleep.

        IMPORTANT: restricted to node_type IN ('episodic', 'concept') and
        is_meta=0. Without that filter, service lexical nodes could become
        hubs: their edges strengthen very quickly on repeated use of a word
        and easily outscored real episodic memories, so semantic
        consolidation during sleep "generalised" meaningless pairs like
        User: "hello" | Bot: "hello" (context == response for lexical
        nodes) instead of actual conversations.

        Returns rows of {id, context, response, weight, created_at,
        last_accessed, hub_score}, sorted by
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
        Up to `limit` "significant" long-term nodes (node_type='episodic',
        is_meta=0 — spike nodes, emotional and structural consolidation
        nodes, abstract sleep nodes, but NOT vocabulary, concepts or
        meta-nodes) created AFTER min_created_at, sorted by weight
        (strongest and most emotionally charged first).

        Used by the self-model evolution during sleep to build a digest of
        the experience accumulated since the previous sleep.
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
    # SELF-MODEL & USER-MODEL — meta-nodes of self-awareness
    # ----------------------------------------------------------------------

    def get_meta_node(self, node_type: str) -> Optional[sqlite3.Row]:
        """
        The meta-node of a given type ('self_model' or 'user_model'), or
        None if it has not been created yet.
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
        Creates the meta-node of a given type if it does not exist, or
        updates its content otherwise. context and response hold the same
        text — for meta-nodes the context/response distinction is
        meaningless; it is simply one slot for the self image or the image
        of the user.

        Returns the meta-node's id.
        """
        ts = timestamp if timestamp is not None else time.time()
        existing = self.get_meta_node(node_type)

        cursor = self._conn.cursor()

        if existing is None:
            cursor.execute(
                """
                INSERT INTO cortex (id, text, meaning, weight, strength,
                                    created_at, last_accessed, last_decayed_at,
                                    kind, is_meta)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                """,
                # is_meta ХРАНИТСЯ, а не выводится из списка родов.
                # Первая версия выводила его из перечисления, и в
                # перечисление не попал last_decision — витрина перестала
                # видеть решение. Выводить то, что раньше хранилось,
                # значит терять всё, чего не предусмотрел.
                (nid := self._next_id(), content, content, weight, weight,
                 ts, ts, ts, node_type),
            )
            self._conn.commit()
            node_id = nid
            logger.info(
                "[META NODE CREATED] type=%s id=%s weight=%.2f",
                node_type, node_id, weight,
            )
            return node_id

        cursor.execute(
            """
            UPDATE cortex
            SET text = ?, meaning = ?, weight = ?, last_accessed = ?,
                occurrences = occurrences + 1
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
    # CONCEPT EXTRACTION — semantic concepts
    # ----------------------------------------------------------------------

    def get_concept_node_by_name(self, name: str) -> Optional[sqlite3.Row]:
        """
        A concept node matched by its exact normalised name (stored in
        the context field), or None if the graph holds no such concept.
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
        Creates a concept node (context=name, response=definition,
        node_type='concept'), or, if a concept with that name exists,
        UPDATES its definition and slightly raises its weight — explaining
        the same term again strengthens the memory of it.

        Returns (node_id, was_created), where was_created is True on first
        creation and False when an existing node was updated.
        """
        ts = timestamp if timestamp is not None else time.time()
        existing = self.get_concept_node_by_name(name)

        cursor = self._conn.cursor()

        if existing is None:
            cursor.execute(
                """
                INSERT INTO cortex (id, text, meaning, weight, strength, created_at, last_accessed, last_decayed_at, kind)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (nid := self._next_id(), name, definition, weight, weight,
                 ts, ts, ts, "concept"),
            )
            self._conn.commit()
            node_id = nid
            logger.info(
                "[CONCEPT CREATED] id=%s name=%r weight=%.2f",
                node_id, name, weight,
            )
            return node_id, True

        new_weight = min(1.0, existing["weight"] + 0.05)
        cursor.execute(
            """
            UPDATE cortex
            SET meaning = ?, weight = ?, last_accessed = ?, last_decayed_at = ?,
                occurrences = occurrences + 1
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
    # LEXICAL ACQUISITION — word/syllable nodes (learning a language from zero)
    # ----------------------------------------------------------------------

    def get_lexical_node(self, node_type: str, text: str) -> Optional[sqlite3.Row]:
        """
        A lexical node (node_type='word' or 'syllable') matched by its
        exact normalised text (stored in context), or None if no such node
        exists yet.
        """
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT * FROM nodes WHERE node_type = ? AND context = ? LIMIT 1",
            (node_type, text),
        )
        return cursor.fetchone()

    def upsert_cortex_fact(
        self,
        theme: str,
        text: str,
        meaning: str,
        strength_step: float,
        cap: float,
        timestamp: Optional[float] = None,
    ) -> Optional[int]:
        """
        Корковый факт, выведенный из повторяющейся темы.

        Ключ — ТЕМА (набор общих слов), а не текст очередного случая:
        «люблю кофе по утрам» и «кофе мой любимый» должны дать один факт,
        а не два узла.

        Сила растёт ЛОГАРИФМИЧЕСКИ от числа встреч. Это не украшение:
        линейный рост сделал бы часто повторяемое непобедимым в выдаче, а
        кора учится с насыщением — десятая встреча добавляет заметно
        меньше второй.
        """
        import math

        ts = timestamp if timestamp is not None else time.time()
        cursor = self._conn.cursor()
        existing = cursor.execute(
            "SELECT id, occurrences FROM cortex WHERE kind = 'fact' AND text = ?",
            (theme,),
        ).fetchone()

        if existing is None:
            node_id = self._next_id()
            cursor.execute(
                """
                INSERT INTO cortex (id, kind, text, meaning, weight, strength,
                                    occurrences, created_at, last_accessed,
                                    last_decayed_at)
                VALUES (?, 'fact', ?, ?, ?, ?, 1, ?, ?, ?)
                """,
                (node_id, theme, meaning, strength_step, strength_step, ts, ts, ts),
            )
            self._conn.commit()
            logger.info("[CORTEX FACT] Тема %r выведена впервые", theme[:40])
            return node_id

        node_id = int(existing["id"])
        seen = int(existing["occurrences"] or 1) + 1
        strength = min(cap, strength_step * math.log(1.0 + seen))
        cursor.execute(
            """
            UPDATE cortex
            SET occurrences = ?, strength = ?, weight = ?,
                meaning = ?, last_accessed = ?, last_decayed_at = ?
            WHERE id = ?
            """,
            (seen, strength, strength, meaning, ts, ts, node_id),
        )
        self._conn.commit()
        logger.info(
            "[CORTEX FACT] Тема %r встречена %d раз, сила %.3f",
            theme[:40], seen, strength,
        )
        return node_id

    def fetch_cortex_facts(self, limit: int = 10) -> List[sqlite3.Row]:
        """Выведенное корой, по убыванию числа встреч."""
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT * FROM cortex WHERE kind = 'fact' "
            "ORDER BY occurrences DESC LIMIT ?",
            (limit,),
        )
        return cursor.fetchall()

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
        Creates a lexical node (context=response=text,
        node_type='word'/'syllable') with weight initial_weight, or, if the
        token has been seen before, RAISES its weight by reinforce_step
        (capped at max_weight) — modelling the gradual acquisition of a
        word or syllable through repetition, where frequency equals
        mastery.

        Returns (node_id, was_created).
        """
        ts = timestamp if timestamp is not None else time.time()
        existing = self.get_lexical_node(node_type, text)

        cursor = self._conn.cursor()

        if existing is None:
            cursor.execute(
                """
                INSERT INTO cortex (id, text, meaning, weight, strength, created_at, last_accessed, last_decayed_at, kind)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (nid := self._next_id(), text, text, initial_weight,
                 initial_weight, ts, ts, ts, node_type),
            )
            self._conn.commit()
            node_id = nid
            logger.debug(
                "[LEXICAL CREATED] type=%s id=%s text=%r weight=%.3f",
                node_type, node_id, text, initial_weight,
            )
            return node_id, True

        new_weight = min(max_weight, existing["weight"] + reinforce_step)
        cursor.execute(
            "UPDATE cortex SET weight = ?, last_accessed = ?, last_decayed_at = ?, "
            "occurrences = occurrences + 1 WHERE id = ?",
            (new_weight, ts, ts, existing["id"]),
        )
        self._conn.commit()
        logger.debug(
            "[LEXICAL REINFORCED] type=%s id=%s text=%r weight %.3f -> %.3f",
            node_type, existing["id"], text, existing["weight"], new_weight,
        )
        return existing["id"], False

    def count_memory_nodes(self) -> int:
        """
        The number of nodes that are MEMORIES — episodes and concepts.
        Excludes lexical infrastructure ('word'/'syllable') and meta-nodes.

        Needed for the automatic sleep trigger. It used to call
        count_nodes() over EVERY node, while vocabulary accumulates
        hundreds of nodes within the first dozen messages — so the
        threshold of 150 was crossed by the ninth message and sleep fired
        on EVERY message after that. The consequences were severe: STM was
        wiped every message (the bot lost the thread of the conversation),
        stress was reset every message, pruning ran every message, and in
        production every message cost TWO LLM calls (cluster consolidation
        plus self-model evolution).
        """
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) as cnt FROM nodes "
            "WHERE is_meta = 0 AND node_type IN ('episodic', 'concept')"
        )
        row = cursor.fetchone()
        return row["cnt"] if row else 0

    def count_nodes_by_type(self, node_type: str) -> int:
        """The number of nodes of a given node_type (for example 'word')."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT COUNT(*) as cnt FROM nodes WHERE node_type = ?", (node_type,))
        row = cursor.fetchone()
        return row["cnt"] if row else 0

    def count_mastered_words(self, min_weight: float) -> int:
        """
        The number of word nodes with weight >= min_weight — words that
        were CONSOLIDATED by repeated use rather than merely heard once.
        Used instead of count_nodes_by_type('word') wherever genuine
        acquisition matters (gating speech stages) rather than the raw
        fact of a single encounter with a word.
        """
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) as cnt FROM nodes WHERE node_type = 'word' AND weight >= ?",
            (min_weight,),
        )
        row = cursor.fetchone()
        return row["cnt"] if row else 0

    def get_lexical_nodes_by_texts(self, node_type: str, texts: List[str]) -> List[sqlite3.Row]:
        """
        Lexical nodes for a LIST of texts in a single query.

        Needed by the surprise computation (MemoryGraph.compute_surprise):
        it checks the familiarity of every word of every incoming message,
        and calling get_lexical_node one by one would mean up to
        lexical_max_tokens_per_input round trips to SQLite per message,
        twice over.

        Duplicates in `texts` are fine: SQL returns one row per node
        anyway, and the caller arranges the result into a dictionary.
        """
        if not texts:
            return []

        placeholders = ",".join("?" for _ in texts)
        cursor = self._conn.cursor()
        cursor.execute(
            f"SELECT * FROM nodes WHERE node_type = ? AND context IN ({placeholders})",
            (node_type, *texts),
        )
        return cursor.fetchall()

    def get_edges_between(self, node_ids: List[int]) -> List[sqlite3.Row]:
        """
        All edges with BOTH ends inside node_ids, in a single query. Used
        by the surprise computation to check how familiar the pairings of
        neighbouring words in an incoming message are.

        Direction is not stored (see upsert_edge — the pair is normalised
        by ascending id), so the caller must look a pair up in both
        orders.
        """
        if len(node_ids) < 2:
            return []

        placeholders = ",".join("?" for _ in node_ids)
        cursor = self._conn.cursor()
        cursor.execute(
            f"""
            SELECT node_from, node_to, weight FROM edges
            WHERE node_from IN ({placeholders}) AND node_to IN ({placeholders})
            """,
            (*node_ids, *node_ids),
        )
        return cursor.fetchall()

    def get_degrees(self, node_ids: List[int]) -> Dict[int, int]:
        """
        How many edges each of these nodes has, in ONE query.

        Connectivity is a candidate importance signal: a memory woven into
        many others is, in memory research, better retained than an
        isolated one ("depth of processing"). The graph has held this all
        along and nothing has ever asked it.

        A single query rather than a loop for the same reason as
        get_edges_between: this runs inside search, on every recall.
        """
        if not node_ids:
            return {}

        placeholders = ",".join("?" for _ in node_ids)
        cursor = self._conn.cursor()
        cursor.execute(
            f"""
            SELECT node_id, COUNT(*) AS degree FROM (
                SELECT node_from AS node_id FROM edges
                 WHERE node_from IN ({placeholders})
                UNION ALL
                SELECT node_to AS node_id FROM edges
                 WHERE node_to IN ({placeholders})
            ) GROUP BY node_id
            """,
            (*node_ids, *node_ids),
        )
        return {row["node_id"]: row["degree"] for row in cursor.fetchall()}

    def get_nodes_by_ids(self, node_ids: List[int]) -> List[sqlite3.Row]:
        """The given nodes in one query — for re-ranking, which needs
        columns MemoryMatch does not carry (stability, spike_strength)."""
        if not node_ids:
            return []
        placeholders = ",".join("?" for _ in node_ids)
        cursor = self._conn.cursor()
        cursor.execute(f"SELECT * FROM nodes WHERE id IN ({placeholders})", tuple(node_ids))
        return cursor.fetchall()

    def get_top_nodes_by_type(self, node_type: str, limit: int) -> List[sqlite3.Row]:
        """
        Up to `limit` nodes of a given node_type sorted by descending
        weight — the best MASTERED words or syllables. Unlike
        get_random_nodes_by_type (a random pool for babbling), what is
        needed here is the top: a status command shows the teacher what
        the bot has learned best.
        """
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT * FROM nodes WHERE node_type = ? ORDER BY weight DESC LIMIT ?",
            (node_type, limit),
        )
        return cursor.fetchall()

    def get_random_nodes_by_type(self, node_type: str, limit: int) -> List[sqlite3.Row]:
        """
        Up to `limit` random nodes of a given node_type — used by the
        babbling instinct to pick known syllables when generating a
        pre-verbal reply.

        THE SAMPLING IS DONE IN PYTHON, not via ORDER BY RANDOM(). It used
        to be the latter, and SQLite's generator is seeded by nothing: it
        takes entropy from the system. That made the benchmarks
        irreproducible — two runs of the same code over the same data,
        with an identical graph, identical time and a bit-identical
        `random` state, diverged on the very first message because
        babbling received a different pool of syllables. The final
        retention figures wandered along with it.

        Python's `random` module is seeded by the benchmark, so sampling
        here is controllable. The distribution is the same — uniform
        without replacement.
        """
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT * FROM nodes WHERE node_type = ? ORDER BY id",
            (node_type,),
        )
        rows = cursor.fetchall()
        if limit >= len(rows):
            # Sampling would return everything anyway — but the order
            # must stay random: the caller draws from the pool by weight,
            # and a stable order would bias the choice towards low ids.
            rows = list(rows)
            random.shuffle(rows)
            return rows
        return random.sample(rows, limit)

    # ----------------------------------------------------------------------

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()