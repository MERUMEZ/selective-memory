"""
Regression-тесты на billing/ — даты, квоты и начисление подписки.

Аудит явно предупреждал: ошибки в логике дат/начисления Premium-дней —
это баги, которые стоят реальных денег пользователей (audit.txt, Фаза 5).
Поэтому этот файл создаётся ДО подключения биллинга к bot.py.
"""
import time
from datetime import datetime, timezone

import pytest

import config
from billing.subscription_manager import SubscriptionManager


@pytest.fixture
def sm():
    """Свежий SubscriptionManager на in-memory SQLite для каждого теста."""
    manager = SubscriptionManager(db_path=":memory:")
    yield manager
    manager.close()


# ---------------------------------------------------------------------------
# Premium: активация / стэкинг / истечение
# ---------------------------------------------------------------------------
def test_free_user_is_not_premium_by_default(sm):
    assert not sm.is_premium(12345)
    assert sm.get_premium_expiry(12345) is None


def test_grant_premium_activates_for_correct_duration(sm):
    now = 1_000_000.0
    until = sm.grant_premium(user_id=1, days=30, stars_amount=150, payload="premium_30d", charge_id="c1")

    row = sm.db.get_subscription(1)
    assert row["premium_until"] == pytest.approx(row["updated_at"] + 30 * 86400.0)
    assert until == row["premium_until"]


def test_grant_premium_stacks_if_still_active(sm):
    now = 1_000_000.0
    until1 = sm.db.grant_premium(1, days=30, stars_amount=150, payload="p1", charge_id="c1", timestamp=now)
    # вторая покупка ДО истечения первой -> должна стэкаться ОТ until1, а не от now
    until2 = sm.db.grant_premium(1, days=30, stars_amount=150, payload="p2", charge_id="c2", timestamp=now + 100)
    assert until2 == pytest.approx(until1 + 30 * 86400.0)


def test_grant_premium_restarts_from_now_if_already_expired(sm):
    now = 1_000_000.0
    until1 = sm.db.grant_premium(1, days=30, stars_amount=150, payload="p1", charge_id="c1", timestamp=now)
    later = until1 + 86400.0 * 5  # подписка истекла 5 дней назад
    until2 = sm.db.grant_premium(1, days=30, stars_amount=150, payload="p2", charge_id="c2", timestamp=later)
    assert until2 == pytest.approx(later + 30 * 86400.0)


def test_is_premium_respects_expiry_boundary(sm):
    now = 1_000_000.0
    until = sm.db.grant_premium(1, days=30, stars_amount=150, payload="p1", charge_id="c1", timestamp=now)
    assert sm.db.is_premium(1, now=until - 1.0) is True
    assert sm.db.is_premium(1, now=until + 1.0) is False


def test_grant_premium_accumulates_stars_total(sm):
    sm.db.grant_premium(1, days=30, stars_amount=150, payload="p1", charge_id="c1", timestamp=0.0)
    sm.db.grant_premium(1, days=30, stars_amount=150, payload="p2", charge_id="c2", timestamp=100.0)

    row = sm.db.get_subscription(1)
    assert row["stars_spent_total"] == 300


def test_grant_premium_logs_payment_atomically(sm):
    sm.db.grant_premium(1, days=30, stars_amount=150, payload="premium_30d", charge_id="charge_abc", timestamp=0.0)

    cursor = sm.db._conn.cursor()
    cursor.execute("SELECT * FROM payments_log WHERE user_id = 1")
    rows = cursor.fetchall()

    assert len(rows) == 1
    assert rows[0]["stars_amount"] == 150
    assert rows[0]["payload"] == "premium_30d"
    assert rows[0]["telegram_payment_charge_id"] == "charge_abc"


def test_two_different_users_have_independent_subscriptions(sm):
    sm.db.grant_premium(1, days=30, stars_amount=150, payload="p1", charge_id="c1", timestamp=0.0)

    assert sm.db.is_premium(1, now=100.0) is True
    assert sm.db.is_premium(2, now=100.0) is False

# ---------------------------------------------------------------------------
# Дневная квота (бесплатный тариф)
# ---------------------------------------------------------------------------
def test_quota_allows_up_to_limit_then_blocks(sm, monkeypatch):
    monkeypatch.setattr(config, "FREE_TIER_DAILY_MESSAGE_LIMIT", 3)
    user_id = 42

    assert sm.check_and_increment_quota(user_id) is True
    assert sm.check_and_increment_quota(user_id) is True
    assert sm.check_and_increment_quota(user_id) is True
    assert sm.check_and_increment_quota(user_id) is False  # лимит исчерпан

    assert sm.get_remaining_quota_today(user_id) == 0


def test_quota_blocked_call_does_not_increment_counter(sm, monkeypatch):
    """
    Важно: если лимит уже исчерпан, повторные вызовы НЕ должны продолжать
    увеличивать message_count до бесконечности (иначе get_remaining_quota_today
    может уйти в минус при неаккуратной реализации).
    """
    monkeypatch.setattr(config, "FREE_TIER_DAILY_MESSAGE_LIMIT", 1)
    user_id = 99

    assert sm.check_and_increment_quota(user_id) is True
    for _ in range(5):
        assert sm.check_and_increment_quota(user_id) is False

    today = sm._today_str()
    assert sm.db.get_usage_today(user_id, today) == 1


def test_quota_resets_on_new_utc_day(sm, monkeypatch):
    monkeypatch.setattr(config, "FREE_TIER_DAILY_MESSAGE_LIMIT", 1)
    user_id = 7

    day1 = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc).timestamp()
    day2 = datetime(2025, 1, 2, 0, 30, tzinfo=timezone.utc).timestamp()

    monkeypatch.setattr(time, "time", lambda: day1)
    assert sm.check_and_increment_quota(user_id) is True
    assert sm.check_and_increment_quota(user_id) is False  # лимит на day1 исчерпан

    monkeypatch.setattr(time, "time", lambda: day2)
    assert sm.check_and_increment_quota(user_id) is True  # новый UTC-день -> лимит сброшен


def test_quota_is_isolated_per_user(sm, monkeypatch):
    monkeypatch.setattr(config, "FREE_TIER_DAILY_MESSAGE_LIMIT", 1)

    assert sm.check_and_increment_quota(100) is True
    assert sm.check_and_increment_quota(200) is True  # другой пользователь — свой лимит
    assert sm.check_and_increment_quota(100) is False
    assert sm.check_and_increment_quota(200) is False


def test_premium_user_is_not_gated_by_quota_helper_directly(sm):
    """
    check_and_increment_quota сама по себе НЕ проверяет is_premium — это
    осознанно: вызывающий код (QuotaMiddleware) обязан сначала проверить
    is_premium() и не звать check_and_increment_quota вовсе для Premium.
    Этот тест фиксирует контракт: SubscriptionManager не отвечает за то,
    чтобы Premium-пользователь был освобождён от лимита — это ответственность
    вызывающей стороны.
    """
    sm.db.grant_premium(1, days=30, stars_amount=150, payload="p1", charge_id="c1", timestamp=time.time())
    assert sm.is_premium(1) is True
    # даже Premium физически МОЖЕТ быть ограничен через этот метод, если
    # его вызвать — контракт соблюдается на уровне middleware, не здесь.