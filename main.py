"""
================================================================================
 MAIN.PY — Консольный интерфейс "Динамического Мозга" (CLI)
================================================================================
Локальная точка входа для тестирования всей архитектуры целиком:

    Perception -> Amygdala(spike + feedback valence) -> InstinctSystem
        -> WorkingMemory (STM) -> Cortex(response, LTM+STM) -> GraphMemory (LTM)

ПОДСОЗНАТЕЛЬНОЕ ПОДКРЕПЛЕНИЕ (Reinforcement / Feedback Loop):
    Триада Input -> Action -> Feedback реализована так:
        1. На шаге N бот отвечает на user_input_N -> Cortex фиксирует
           last_action_trace (Input_N -> Action_N).
        2. На шаге N+1, ДО генерации нового ответа, Amygdala сканирует
           user_input_(N+1) на маркеры одобрения/порицания (valence).
        3. Если valence != 0 -> Cortex.apply_feedback(valence) применяет
           обратную связь к last_action_trace (Action_N), подкрепляя или
           штрафуя задействованный узел LTM и корректируя склонность к
           эхолалии для похожего контекста.
        4. Обычный цикл продолжается: генерируется новый ответ на
           user_input_(N+1), формируется новый last_action_trace.

Двухслойная модель памяти (STM + LTM) и дискретные "внутренние часы"
(brain_time) — без изменений относительно предыдущей версии.

Выход из цикла: команды "exit", "quit" или Ctrl+C.
================================================================================
"""

import sys
import threading
import time
import config

from typing import Optional
from core.amygdala import Amygdala
from core.cortex import Cortex
from core.debug_formatting import format_debug_block
from core.drives import BoredomDrive
from core.instincts import InstinctSystem
from core.perception import Perception
from core.async_console import AsyncConsole
from memory.graph_memory import MemoryGraph
from memory.sleep_cycle import SleepCycle
from memory.working_memory import WorkingMemory
from storage.utils.logger import get_logger

logger = get_logger(__name__)

EXIT_COMMANDS = {"exit", "quit", "выход"}
SLEEP_COMMANDS = {"/sleep", "спать", "сон"}

TICK_SECONDS = config._get_float("TICK_SECONDS", 120.0) if hasattr(config, "_get_float") else 120.0

# Сколько последовательных сообщений без spike/consolidation считается
# "продолжительным молчанием по существу" -> авто-триггер фазы сна.
AUTO_SLEEP_IDLE_TICKS = config._get_int("AUTO_SLEEP_IDLE_TICKS", 15) if hasattr(config, "_get_int") else 15


def print_debug_block(
    brain_time: float,
    session_elapsed: float,
    emotion_score: float,
    perplexity: float,
    total_density: float,
    confidence: float,
    stress_state,
    spike_triggered: bool,
    memory_written: bool,
    response_source: str,
    decayed_nodes: int,
    total_nodes: int,
    top_nodes,
    stm_status: str,
    consolidation_event: str = None,
    prompt_context: str = None,
    reward_trace: str = None,
    reward_eval: str = None,
    activation_traces: list = None,
    mood_state=None,
) -> None:
    """
    Единый системный дебаг-блок. Печатается ТОЛЬКО здесь, синхронно,
    сразу после ответа на сообщение пользователя — никаких фоновых print().

    Само форматирование строки теперь живёт в core/debug_formatting.py
    (общий модуль, используемый также Telegram-ботом в core/brain_session.py) —
    здесь остаётся только print() итоговой строки.
    """
    print(
        format_debug_block(
            brain_time=brain_time,
            session_elapsed=session_elapsed,
            emotion_score=emotion_score,
            perplexity=perplexity,
            total_density=total_density,
            confidence=confidence,
            stress_state=stress_state,
            spike_triggered=spike_triggered,
            memory_written=memory_written,
            response_source=response_source,
            decayed_nodes=decayed_nodes,
            total_nodes=total_nodes,
            top_nodes=top_nodes,
            stm_status=stm_status,
            consolidation_event=consolidation_event,
            prompt_context=prompt_context,
            reward_trace=reward_trace,
            reward_eval=reward_eval,
            activation_traces=activation_traces,
            mood_state=mood_state,
        )
    )


class SharedActivationState:
    """
    Потокобезопасный контейнер для last_active_node_id и окна со-активации
    STM (stm_window_node_ids), разделяемый между главным потоком и фоновым
    Idle Sleep потоком (который читает last_active_node_id при выборе
    проактивного узла в select_proactive_node).
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._last_active_node_id: Optional[int] = None
        self._stm_window_node_ids: list = []

    def set_last_active(self, node_id: Optional[int]) -> None:
        with self._lock:
            self._last_active_node_id = node_id

    def get_last_active(self) -> Optional[int]:
        with self._lock:
            return self._last_active_node_id

    def append_window_node(self, node_id: int) -> None:
        with self._lock:
            self._stm_window_node_ids.append(node_id)

    def get_window_snapshot(self) -> list:
        with self._lock:
            return list(self._stm_window_node_ids)

    def clear_window(self) -> None:
        with self._lock:
            self._stm_window_node_ids.clear()


class SharedBrainClock:
    """
    Потокобезопасный контейнер для brain_time и SystemState, разделяемый
    между главным потоком (обработка input()) и фоновым Idle Sleep потоком.

    ВСЕ чтения и записи brain_time/state проходят через threading.Lock —
    ни один из двух потоков не должен читать/писать эти поля напрямую.
    """

    def __init__(self, initial_brain_time: float):
        self._lock = threading.Lock()
        self._brain_time = initial_brain_time
        self.last_user_msg_time = initial_brain_time
        self.last_activity_time = time.time()

    def get_brain_time(self) -> float:
        with self._lock:
            return self._brain_time

    def advance_by(self, delta_seconds: float) -> float:
        with self._lock:
            self._brain_time += delta_seconds
            return self._brain_time

    def register_user_message(self, delta_seconds: float) -> float:
        with self._lock:
            self._brain_time += delta_seconds
            self.last_user_msg_time = self._brain_time
            self.last_activity_time = time.time()
            return self._brain_time

    def register_activity(self) -> None:
        """Сбрасывает Idle-таймер на любое нажатие клавиши (без учёта как
        отправки сообщения и без продвижения brain_time)."""
        with self._lock:
            self.last_activity_time = time.time()

    def seconds_since_last_user_message(self) -> float:
        with self._lock:
            return self._brain_time - self.last_user_msg_time

    def seconds_since_last_activity(self) -> float:
        """Реальные секунды с последнего keystroke/сообщения — используется
        для решения 'засыпать или нет' (защита от сна во время печати)."""
        with self._lock:
            return time.time() - self.last_activity_time


def run() -> None:
    """Запускает основной консольный цикл взаимодействия с 'мозгом'."""
    logger.info("[BRAIN STATE] Инициализация системы...")

    memory = MemoryGraph()
    self_node_id, user_node_id = memory.ensure_self_and_user_nodes()
    logger.info(
        "[BRAIN STATE] Мета-узлы загружены: Self-Model (id=%s), User-Model (id=%s)",
        self_node_id, user_node_id,
    )

    stm = WorkingMemory()
    instincts = InstinctSystem()
    amygdala = Amygdala()
    cortex = Cortex(instincts=instincts, memory=memory, stm=stm, amygdala=amygdala)
    sleep_cycle = SleepCycle(memory=memory, stm=stm, instincts=instincts)
    boredom_drive = BoredomDrive()

    session_start_real = time.time()
    clock = SharedBrainClock(initial_brain_time=session_start_real)
    async_console = AsyncConsole(brain_clock=clock)

    # Потокобезопасный контейнер last_active_node_id + окно со-активации STM
    # (заменяет обычные list/int — читаются и пишутся из двух потоков).
    activation_state = SharedActivationState()

    # Счётчик тиков без spike/consolidation — используется для автоматического
    # триггера фазы сна при "продолжительном молчании по существу" (см. ниже).
    idle_ticks_without_event: int = 0

    # Событие для остановки фонового потока при выходе из программы.
    stop_event = threading.Event()

    idle_thread = threading.Thread(
        target=_idle_sleep_background_loop,
        args=(
            clock, sleep_cycle, boredom_drive, memory, cortex,
            stop_event, activation_state, async_console,
        ),
        daemon=True,
        name="IdleSleepThread",
    )
    idle_thread.start()

    print("=" * 60)
    print(" DYNAMIC AI BRAIN — локальный CLI-прототип (STM + LTM + Reinforcement + Idle Sleep)")
    print(f" Дискретный тик времени: +{TICK_SECONDS:.0f}s внутренних часов на каждое сообщение")
    print(f" Idle Sleep после {config.IDLE_SLEEP_THRESHOLD_SECONDS:.0f}s бездействия")
    print(f" Boredom threshold: {config.BOREDOM_THRESHOLD:.2f}")
    print(f" STM capacity: {stm.capacity} сообщений")
    print(" Введите сообщение ('exit' для выхода)")
    print("=" * 60)

    try:
        while True:
            try:
                user_input = async_console.get_input()
            except EOFError:
                break

            if not user_input:
                continue

            if user_input.lower() in EXIT_COMMANDS:
                print("Bot > Завершение сессии. До связи.")
                break

            # Любое сообщение пользователя -> пробуждение (сбрасывает boredom,
            # переводит состояние в AWAKE). Делается под тем же логическим
            # порядком, что и продвижение времени, чтобы избежать гонки с
            # фоновым потоком, который может засыпать/будить систему.
            brain_time = clock.register_user_message(TICK_SECONDS)
            boredom_drive.on_user_message(brain_time)

            if user_input.lower() in SLEEP_COMMANDS:
                print("\nBot > ...засыпаю, провожу консолидацию памяти...")
                sleep_summary = sleep_cycle.run_sleep_cycle(timestamp=brain_time)
                print(f"\n{sleep_summary.to_report_string()}")
                activation_state.clear_window()
                idle_ticks_without_event = 0
                continue

            logger.info("[BRAIN CLOCK] Tick: brain_time=%.1f (+%.1fs)", brain_time, TICK_SECONDS)

            # 2. ПОДСОЗНАТЕЛЬНОЕ ПОДКРЕПЛЕНИЕ — Триада Input -> Action -> Feedback.
            reward_trace_log = None
            reward_eval_log = None

            feedback_signal = amygdala.detect_feedback_signal(user_input)
            feedback_valence = feedback_signal.valence
            if feedback_valence != 0.0 and cortex.last_action_trace is not None:
                trace = cortex.last_action_trace
                reward_trace_log = (
                    f'[REWARD TRACE] User Trigger: "{trace.user_input}" | '
                    f'Bot Action: "{trace.bot_output}"'
                )
                print(f"\n{reward_trace_log}")

                feedback_result = cortex.apply_feedback(
                    feedback_valence,
                    timestamp=brain_time,
                    matched_markers=feedback_signal.matched_markers,
                )

                sign = "+" if feedback_valence > 0 else ""
                reward_eval_log = (
                    f"[REWARD EVAL] Feedback Valence: {sign}{feedback_valence:.2f} -> "
                    f"{'Rewarding' if feedback_valence > 0 else 'Penalizing'} Node ID: {feedback_result.node_id}"
                )
                print(reward_eval_log)

                # RETROSPECTIVE CORRECTION: если этот фидбэк опроверг ранее
                # применённую оценку (сарказм/самокоррекция пользователя) —
                # выводим отдельную строку и подмешиваем её в дебаг-блок.
                retro = feedback_result.retrospective_correction
                if retro is not None and retro.triggered:
                    retro_log = (
                        f"[RETROSPECTIVE CORRECTION] Опровергнута прошлая оценка "
                        f"(node_id={retro.reversed_entry.node_id}, "
                        f"old_valence={retro.reversed_entry.valence:+.2f}, "
                        f"reversal_delta={retro.reversal_delta:+.3f}, "
                        f"penalized_markers={retro.penalized_markers})"
                    )
                    print(retro_log)
                    reward_eval_log = f"{reward_eval_log}\n{retro_log}"

            # 3. Perception
            perception_result = Perception().analyze(user_input)
            emotion_score = perception_result.emotion_score

            # 4. Perplexity
            perplexity = cortex.calculate_perplexity(user_input)

            # 5. Состояние инстинктов
            stress_state = instincts.get_state()

            # 6. Amygdala — spike detection
            amygdala_result = amygdala.evaluate(
                emotion_score=emotion_score,
                perplexity=perplexity,
                stress_level=stress_state.current_stress,
            )

            instincts.accumulate_stress(emotion_score)

            # 7. Записываем реплику пользователя в STM
            stm.add_message(
                role="user",
                text=user_input,
                emotion_score=emotion_score,
                perplexity=perplexity,
                timestamp=brain_time,
            )

            # 8. Cortex — генерация ответа.
            cortex_response = cortex.generate_response(user_input, timestamp=brain_time)

            # 8b. Если Cortex нашёл узел через MEMORY HIT (memory_retrieval),
            #     запоминаем его id в окне со-активации и как last_active_node_id.
            if (
                cortex.last_action_trace is not None
                and cortex.last_action_trace.action_type == "memory_retrieval"
                and cortex.last_action_trace.node_id is not None
            ):
                activation_state.append_window_node(cortex.last_action_trace.node_id)
                activation_state.set_last_active(cortex.last_action_trace.node_id)

            # 9. Записываем ответ бота в STM
            stm.add_message(
                role="bot",
                text=cortex_response.text,
                emotion_score=0.0,
                perplexity=cortex_response.perplexity,
                timestamp=brain_time,
            )

            # 10. Если произошёл spike — классический немедленный spike-узел в LTM
            memory_written = False
            if amygdala_result.is_spike_triggered:
                new_node_id = memory.save_connection(
                    context=user_input,
                    response=cortex_response.text,
                    weight=amygdala_result.total_density,
                    timestamp=brain_time,
                )
                memory_written = True
                activation_state.set_last_active(new_node_id)
                logger.info("[BRAIN STATE] Spike Triggered! New memory node formed.")
                print("\n[BRAIN STATE] ⚡ Spike Triggered! New memory node formed.")

                # СВЯЗЬ ПО КОНТЕКСТУ: если в этом же обмене был найден узел
                # через MEMORY HIT, связываем его с новым spike-узлом.
                window_snapshot = activation_state.get_window_snapshot()
                if window_snapshot:
                    source_node_id = window_snapshot[-1]
                    memory.connect_nodes(source_node_id, new_node_id, timestamp=brain_time)

                activation_state.append_window_node(new_node_id)

            # 11. ИЗБИРАТЕЛЬНАЯ КОНСОЛИДАЦИЯ (STM -> LTM)
            consolidation_event = None
            if stm.is_full() or amygdala_result.is_spike_triggered:
                episode = stm.consume_all()
                result = memory.consolidate_from_stm(episode, timestamp=brain_time)

                if result.decision == "emotional_node":
                    consolidation_event = f"[CONSOLIDATION] Эмоциональный узел записан в БД (id={result.node_id}, weight={result.weight:.2f})"
                    print(f"\n{consolidation_event}")
                elif result.decision == "structural_node":
                    consolidation_event = f"[CONSOLIDATION] Структурный узел записан в БД (id={result.node_id}, weight={result.weight:.2f})"
                    print(f"\n{consolidation_event}")
                else:
                    consolidation_event = "[STM FLUSH] Рутинный шум отброшен"
                    print(f"\n{consolidation_event}")

                # СВЯЗЬ ПО КОНТЕКСТУ: если консолидация создала новый узел,
                # связываем его с последним MEMORY HIT-узлом этого окна.
                window_snapshot = activation_state.get_window_snapshot()
                if result.node_id is not None and window_snapshot:
                    source_node_id = window_snapshot[-1]
                    memory.connect_nodes(source_node_id, result.node_id, timestamp=brain_time)
                    activation_state.append_window_node(result.node_id)
                    activation_state.set_last_active(result.node_id)

                # СВЯЗЬ ПО СО-АКТИВАЦИИ: все узлы, задействованные в рамках
                # этого окна STM, получают усиленные рёбра друг с другом.
                memory.reinforce_coactivation(activation_state.get_window_snapshot(), timestamp=brain_time)
                activation_state.clear_window()

                idle_ticks_without_event = 0
            else:
                idle_ticks_without_event += 1

            # 12. Decay применяется синхронно, используя виртуальное brain_time
            decayed_nodes = memory.apply_decay(now=brain_time)
            total_nodes = memory.count_nodes()

            # 12b. АВТОМАТИЧЕСКИЙ ТРИГГЕР ФАЗЫ СНА: переполнение памяти ИЛИ
            # продолжительное "молчание по существу" -> запускаем /sleep.
            # (Idle Sleep по времени бездействия обрабатывается отдельно
            # фоновым потоком _idle_sleep_background_loop, не здесь.)
            auto_sleep_reason = None
            if total_nodes >= config.SLEEP_AUTO_TRIGGER_NODE_COUNT:
                auto_sleep_reason = f"переполнение памяти ({total_nodes} >= {config.SLEEP_AUTO_TRIGGER_NODE_COUNT} узлов)"
            elif idle_ticks_without_event >= AUTO_SLEEP_IDLE_TICKS:
                auto_sleep_reason = f"продолжительное молчание ({idle_ticks_without_event} тиков без событий)"

            if auto_sleep_reason:
                print(f"\n[BRAIN STATE] 💤 Автоматический триггер фазы сна: {auto_sleep_reason}")
                auto_sleep_summary = sleep_cycle.run_sleep_cycle(timestamp=brain_time)
                print(f"\n{auto_sleep_summary.to_report_string()}")
                idle_ticks_without_event = 0
                total_nodes = memory.count_nodes()

            # 12c. Настроение (Mood) затухает к базовому уровню на каждом тике,
            # компенсируя стимулы, накопленные в generate_response/apply_feedback.
            mood_snapshot = cortex.mood.decay(log=False)

            # 13. Вывод ответа
            print(f"\nBot > {cortex_response.text}")

            # 14. Обновлённое состояние стресса
            updated_stress_state = instincts.get_state()

            # 15. Единый системный дебаг-блок (включая REWARD TRACE/EVAL)
            print_debug_block(
                brain_time=brain_time,
                session_elapsed=brain_time - session_start_real,
                emotion_score=emotion_score,
                perplexity=amygdala_result.perplexity,
                total_density=amygdala_result.total_density,
                confidence=cortex_response.confidence,
                stress_state=updated_stress_state,
                spike_triggered=amygdala_result.is_spike_triggered,
                memory_written=memory_written,
                response_source=cortex_response.source,
                decayed_nodes=decayed_nodes,
                total_nodes=total_nodes,
                top_nodes=memory.get_top_nodes(limit=5),
                stm_status=stm.get_status_string(),
                consolidation_event=consolidation_event,
                prompt_context=cortex_response.prompt_context,
                reward_trace=reward_trace_log,
                reward_eval=reward_eval_log,
                activation_traces=cortex_response.activation_traces,
                mood_state=mood_snapshot,
            )

    except KeyboardInterrupt:
        print("\nBot > Прервано пользователем (Ctrl+C). Завершение сессии.")

    finally:
        stop_event.set()
        idle_thread.join(timeout=2.0)
        cortex.close()
        logger.info("[BRAIN STATE] Система остановлена, ресурсы освобождены.")


def _idle_sleep_background_loop(
    clock: SharedBrainClock,
    sleep_cycle: SleepCycle,
    boredom_drive: BoredomDrive,
    memory: MemoryGraph,
    cortex: Cortex,
    stop_event: threading.Event,
    activation_state: "SharedActivationState",
    async_console: AsyncConsole,
) -> None:
    """
    Фоновый поток Idle Sleep Thread. Раз в IDLE_THREAD_REAL_INTERVAL_SECONDS
    реальных секунд (по умолчанию 1с = 1 виртуальная секунда brain_time):

        1. Продвигает brain_time на реальный интервал сна потока (Continuous
           Brain Time — виртуальное время течёт САМО ПО СЕБЕ, даже когда
           пользователь молчит и главный поток блокирован на input()).
        2. Если Δt с последнего сообщения пользователя >= IDLE_SLEEP_THRESHOLD_SECONDS
           и система ещё не спит (AWAKE) -> запускает run_sleep_cycle() один раз,
           переводит BoredomDrive в SLEEPING.
        3. Если система в SLEEPING -> пересчитывает boredom; при достижении
           BOREDOM_THRESHOLD -> выбирает узел (select_proactive_node),
           генерирует проактивное сообщение через cortex, печатает его в
           консоль и переводит BoredomDrive в WAITING_FOR_USER (Anti-Spam Guard).

    ВСЕ обращения к brain_time/state идут через clock (SharedBrainClock),
    защищённый внутренним threading.Lock — гонок с главным потоком не возникает.
    """
    while not stop_event.is_set():
        time.sleep(config.IDLE_THREAD_REAL_INTERVAL_SECONDS)

        brain_time = clock.advance_by(config.IDLE_THREAD_REAL_INTERVAL_SECONDS)
        idle_seconds = clock.seconds_since_last_activity()

        if boredom_drive.is_awake() and idle_seconds >= config.IDLE_SLEEP_THRESHOLD_SECONDS:
            logger.info(
                "[IDLE SLEEP] Бездействие %.1fs >= %.1fs -> запуск фазы сна",
                idle_seconds, config.IDLE_SLEEP_THRESHOLD_SECONDS,
            )
            async_console.safe_print(f"\n[BRAIN STATE] 💤 Idle Sleep: {idle_seconds:.0f}s бездействия -> засыпаю...")

            sleep_summary = sleep_cycle.run_sleep_cycle(timestamp=brain_time)
            async_console.safe_print(f"\n{sleep_summary.to_report_string()}")

            activation_state.clear_window()
            boredom_drive.enter_sleeping(brain_time)
            continue

        if boredom_drive.is_sleeping():
            snapshot = boredom_drive.update(brain_time)

            logger.debug(
                "[BOREDOM] state=%s boredom=%.3f (t_sleep=%.1fs)",
                snapshot.state.value, snapshot.boredom, snapshot.seconds_since_sleep_start,
            )

            if snapshot.boredom >= config.BOREDOM_THRESHOLD:
                last_node_id = activation_state.get_last_active()
                node = memory.select_proactive_node(
                    last_active_node_id=last_node_id,
                    brain_time=brain_time,
                )

                proactive_message = cortex.generate_proactive_message(node, timestamp=brain_time)

                if proactive_message is not None:
                    block = (
                        "\n" + "─" * 60 +
                        f"\n[BRAIN STATE] 💭 Boredom Spike ({snapshot.boredom:.2f}) — бот инициирует разговор" +
                        f"\nBot > {proactive_message.text}" +
                        f"\n[PROACTIVE DEBUG] source_node_id={proactive_message.source_node_id}" +
                        "\n" + "─" * 60
                    )
                    async_console.safe_print(block)
                    boredom_drive.trigger_proactive()
                else:
                    logger.warning(
                        "[PROACTIVE FALLBACK] Не удалось сгенерировать проактивное "
                        "сообщение (LLM недоступна) — остаёмся в SLEEPING"
                    )


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:  # noqa: BLE001
        logger.exception("[BRAIN STATE] Критическая ошибка в главном цикле: %s", exc)
        sys.exit(1)