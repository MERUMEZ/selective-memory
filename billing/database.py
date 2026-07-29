"""
================================================================================
 BILLING/DATABASE.PY — Низкоуровневый слой SQLite для подписок и квот
================================================================================
Отдельная БД (config.BILLING_DB_PATH), НЕ смешивается с per-user brain.db —
платёжные данные и когнитивные данные "мозга" имеют разные жизненные циклы
и разные требования к консистентности (см. audit.txt, раздел B).

Три таблицы:
    subscriptions  — текущий статус Premium по user_id (1 строка на юзера).
    payments_log   — журнал ВСЕХ успешных платежей (audit trail, никогда
                     не удаляется и не перезаписывается).
    usage_daily    — счётчик сообщений бесплатного тарифа по дню (UTC).

Все временные метки — unix-время (float), даты в usage_daily — строки
'YYYY-MM-DD' в UTC (не зависят от локального часового пояса сервера).
================================================================================
"""

import sqlite3
import time
from typing import Optional

import config
from storage.utils.logger import get_logger

logger = get_logger(__name__)


class BillingDatabase:
    """Инкапсулирует всё SQL для подписок/квот. Синхронный (как memory.database.Database)."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or config.BILLING_DB_PATH
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        logger.info("[BILLING DB] Инициализирована (db_path=%s)", self.db_path)

    def _init_schema(self) -> None:
        cursor = self._conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS subscriptions (
                user_id           INTEGER PRIMARY KEY,
                is_premium        INTEGER NOT NULL DEFAULT 0,
                premium_until     REAL,
                stars_spent_total INTEGER NOT NULL DEFAULT 0,
                updated_at        REAL NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS payments_log (
                id                          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id                     INTEGER NOT NULL,
                stars_amount                INTEGER NOT NULL,
                payload                     TEXT NOT NULL,
                telegram_payment_charge_id  TEXT,
                created_at                  REAL NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS usage_daily (
                user_id       INTEGER NOT NULL,
                usage_date    TEXT NOT NULL,
                message_count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (user_id, usage_date)
            )
            """
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ----------------------------------------------------------------------
    # SUBSCRIPTIONS
    # ----------------------------------------------------------------------

    def get_subscription(self, user_id: int) -> Optional[sqlite3.Row]:
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM subscriptions WHERE user_id = ?", (user_id,))
        return cursor.fetchone()

    def is_premium(self, user_id: int, now: Optional[float] = None) -> bool:
        """Premium активен, если premium_until задан и строго больше текущего времени."""
        ts = now if now is not None else time.time()
        row = self.get_subscription(user_id)
        if row is None or row["premium_until"] is None:
            return False
        return row["premium_until"] > ts

    def get_premium_expiry(self, user_id: int) -> Optional[float]:
        row = self.get_subscription(user_id)
        if row is None:
            return None
        return row["premium_until"]

    def grant_premium(
        self,
        user_id: int,
        days: int,
        stars_amount: int,
        payload: str,
        charge_id: Optional[str],
        timestamp: Optional[float] = None,
    ) -> float:
        """
        Атомарно: продлевает Premium (от текущего premium_until, если
        подписка ещё активна на момент оплаты — стэкинг дней, а не потеря
        оплаченного времени; иначе активирует от now) И логирует платёж в
        payments_log в той же транзакции. Возвращает новый premium_until
        (unix-время).
        """
        ts = timestamp if timestamp is not None else time.time()
        cursor = self._conn.cursor()

        existing = self.get_subscription(user_id)
        base_ts = ts
        if existing is not None and existing["premium_until"] is not None and existing["premium_until"] > ts:
            base_ts = existing["premium_until"]

        new_until = base_ts + days * 86400.0
        new_stars_total = (existing["stars_spent_total"] if existing is not None else 0) + stars_amount

        cursor.execute(
            """
            INSERT INTO subscriptions (user_id, is_premium, premium_until, stars_spent_total, updated_at)
            VALUES (?, 1, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                is_premium = 1,
                premium_until = excluded.premium_until,
                stars_spent_total = excluded.stars_spent_total,
                updated_at = excluded.updated_at
            """,
            (user_id, new_until, new_stars_total, ts),
        )
        cursor.execute(
            """
            INSERT INTO payments_log (user_id, stars_amount, payload, telegram_payment_charge_id, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, stars_amount, payload, charge_id, ts),
        )
        self._conn.commit()

        logger.info(
            "[BILLING] Premium выдан user_id=%s: +%dd (stars=%d) -> premium_until=%.0f (stars_total=%d)",
            user_id, days, stars_amount, new_until, new_stars_total,
        )
        return new_until

    # ----------------------------------------------------------------------
    # USAGE_DAILY — счётчик сообщений бесплатного тарифа по дню (UTC)
    # ----------------------------------------------------------------------

    def get_usage_today(self, user_id: int, usage_date: str) -> int:
        """usage_date в формате 'YYYY-MM-DD' (UTC, см. subscription_manager)."""
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT message_count FROM usage_daily WHERE user_id = ? AND usage_date = ?",
            (user_id, usage_date),
        )
        row = cursor.fetchone()
        return row["message_count"] if row is not None else 0

    def increment_usage(self, user_id: int, usage_date: str) -> int:
        """
        Атомарно увеличивает счётчик сообщений за день на 1 (UPSERT) и
        возвращает НОВОЕ значение счётчика.
        """
        cursor = self._conn.cursor()
        cursor.execute(
            """
            INSERT INTO usage_daily (user_id, usage_date, message_count)
            VALUES (?, ?, 1)
            ON CONFLICT(user_id, usage_date) DO UPDATE SET
                message_count = message_count + 1
            """,
            (user_id, usage_date),
        )
        self._conn.commit()
        return self.get_usage_today(user_id, usage_date)