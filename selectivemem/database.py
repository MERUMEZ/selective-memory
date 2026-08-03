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

# Uniqueness of the directed pair (node_from, node_to). Undirectedness
# (A->B equals B->A) is enforced in graph_memory.py by normalising the id
# order before insertion (see Database.upsert_edge).
UNIQUE_EDGE_PAIR = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_edges_unique_pair ON edges(node_from, node_to);
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
        cursor.execute(INDEX_CONTEXT)
        cursor.execute(EDGES_SCHEMA)
        cursor.execute(INDEX_EDGE_FROM)
        cursor.execute(INDEX_EDGE_TO)
        cursor.execute(UNIQUE_EDGE_PAIR)
        self._conn.commit()
        self._migrate_meta_columns()
        self._migrate_decay_columns()
        self._migrate_stability_column()
        logger.info("[DB INIT] Schema nodes + edges ready (%s)", self.db_path)

    def _migrate_meta_columns(self) -> None:
        """
        Adds the is_meta/node_type columns if they do not exist yet (for
        databases created before they were introduced). SQLite's ALTER
        TABLE has no IF NOT EXISTS for columns, so sqlite3.OperationalError
        ("duplicate column name") is caught instead.
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
        cursor = self._conn.cursor()
        migrated = False
        for statement in (ALTER_ADD_STABILITY, ALTER_ADD_REWARD_EXPECTATION,
                          ALTER_ADD_EMBEDDING, ALTER_ADD_SPIKE_STRENGTH,
                          ALTER_ADD_EDGE_TYPE):
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
        cursor.execute(
            """
            INSERT INTO nodes (context, response, weight, created_at, last_accessed,
                               last_decayed_at, node_type, spike_strength)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            # spike_strength is the birth weight, kept as it was. Weight goes
            # on decaying; this stays, because forgetting has to be able to
            # ask later how hard the event hit at the time.
            (context, response, weight, ts, ts, ts, node_type, weight),
        )
        self._conn.commit()
        node_id = cursor.lastrowid
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
            "SELECT * FROM nodes WHERE node_type IN ('episodic', 'concept')"
        )
        return cursor.fetchall()

    def update_weight(self, node_id: int, new_weight: float) -> None:
        """Updates a node's weight (after decay or reinforcement, say)."""
        cursor = self._conn.cursor()
        cursor.execute(
            "UPDATE nodes SET weight = ? WHERE id = ?",
            (new_weight, node_id),
        )
        self._conn.commit()

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
            UPDATE nodes
            SET last_accessed = ?,
                last_decayed_at = ?,
                stability = MIN(?, COALESCE(stability, ?) * ?)
            WHERE id = ?
            """,
            (
                ts, ts,
                self.settings.stability_max,
                self.settings.stability_initial,
                self.settings.stability_growth_factor,
                node_id,
            ),
        )
        self._conn.commit()

    def update_embedding(self, node_id: int, blob: Optional[bytes]) -> None:
        """
        Stores a node's meaning vector. A separate method rather than
        part of insertion: nodes created before the model existed are
        filled in lazily on the first search.
        """
        cursor = self._conn.cursor()
        cursor.execute("UPDATE nodes SET embedding = ? WHERE id = ?", (blob, node_id))
        self._conn.commit()

    def update_stability(self, node_id: int, stability: float) -> None:
        """
        Sets a node's resistance to forgetting directly. Needed by
        supersession: a stale version of a fact returns to the forgettable
        pile rather than being deleted.
        """
        cursor = self._conn.cursor()
        cursor.execute("UPDATE nodes SET stability = ? WHERE id = ?", (stability, node_id))
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
        cursor.execute(
            "UPDATE nodes SET reward_expectation = ? WHERE id = ?",
            (expectation, node_id),
        )
        self._conn.commit()

    def delete_node(self, node_id: int) -> None:
        """Physically deletes a node (used by the sleep cycle when forgetting)."""
        cursor = self._conn.cursor()
        cursor.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
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
            "UPDATE nodes SET weight = :weight, last_decayed_at = :last_decayed_at WHERE id = :id",
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