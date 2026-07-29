"""
================================================================================
 BILLING/SUBSCRIPTION_MANAGER.PY — Высокоуровневый API биллинга
================================================================================
Обёртка над billing.database.BillingDatabase с бизнес-логикой:
    - проверка/выдача Premium-статуса;
    - дневные квоты для бесплатного тарифа (сброс по UTC-дате, а не по
      скользящим 24 часам — проще для пользователя понять "лимит на сегодня").

Синхронный класс (как SessionManager) — вызывается из bot.py через
asyncio.to_thread, никакого asyncio здесь.

BrainSession/MemoryGraph НИЧЕГО не знают об этом модуле — проверка лимитов
происходит ДО вызова session.process_message, на уровне bot.py/middleware
(разделение ответственности, см. audit.txt раздел B).
================================================================================
"""

import threading
import time
from datetime import datetime, timezone
from typing import Optional

import config
from billing.database import BillingDatabase
from storage.utils.logger import get_logger

logger = get_logger(__name__)


class SubscriptionManager:
    def __init__(self, db_path: Optional[str] = None):
        self.db = BillingDatabase(db_path=db_path or config.BILLING_DB_PATH)
        self._lock = threading.Lock()
        logger.info("[SUBSCRIPTION MANAGER] Инициализирован")

    # ----------------------------------------------------------------------
    # Premium
    # ----------------------------------------------------------------------

    def is_premium(self, user_id: int) -> bool:
        return self.db.is_premium(user_id, now=time.time())

    def get_premium_expiry(self, user_id: int) -> Optional[float]:
        return self.db.get_premium_expiry(user_id)

    def grant_premium(
        self,
        user_id: int,
        days: int,
        stars_amount: int,
        payload: str,
        charge_id: Optional[str],
    ) -> float:
        """Возвращает новый premium_until (unix-время)."""
        return self.db.grant_premium(
            user_id=user_id,
            days=days,
            stars_amount=stars_amount,
            payload=payload,
            charge_id=charge_id,
            timestamp=time.time(),
        )

    # ----------------------------------------------------------------------
    # Дневная квота (бесплатный тариф)
    # ----------------------------------------------------------------------

    def check_and_increment_quota(self, user_id: int) -> bool:
        """
        Проверяет, не исчерпан ли дневной лимит бесплатных сообщений, и
        если нет — атомарно увеличивает счётчик и возвращает True.
        Если лимит уже исчерпан — возвращает False, счётчик НЕ трогается.

        Premium-пользователей вызывающий код (middleware) не должен сюда
        пускать вообще — эта проверка только для бесплатного тарифа.
        """
        with self._lock:
            today = self._today_str()
            current = self.db.get_usage_today(user_id, today)
            if current >= config.FREE_TIER_DAILY_MESSAGE_LIMIT:
                return False
            self.db.increment_usage(user_id, today)
            return True

    def get_remaining_quota_today(self, user_id: int) -> int:
        today = self._today_str()
        used = self.db.get_usage_today(user_id, today)
        return max(0, config.FREE_TIER_DAILY_MESSAGE_LIMIT - used)

    # ----------------------------------------------------------------------
    # Внутренние хелперы
    # ----------------------------------------------------------------------

    @staticmethod
    def _today_str(now: Optional[float] = None) -> str:
        """Дата в UTC, формат 'YYYY-MM-DD' — не зависит от таймзоны сервера."""
        ts = now if now is not None else time.time()
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")

    def close(self) -> None:
        self.db.close()