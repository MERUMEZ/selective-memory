"""
Точечный ручной тест SessionManager (Этап 2). Не входит в CI, не трогает main.py/bot.py.
Запуск: python test_session_manager_manual.py
"""
from core.session_manager import SessionManager

EXIT_COMMANDS = {"exit", "quit", "выход"}


def main():
    manager = SessionManager(db_dir="storage/test_brains")
    print("SessionManager запущен. Формат ввода: <user_id> <текст>. 'exit' — выход.")
    print("Пример: 111 привет   /   222 привет   -> разные 'мозги', разные БД.")

    try:
        while True:
            raw = input("\n> ")
            if raw.lower() in EXIT_COMMANDS:
                break

            try:
                user_id_str, text = raw.split(" ", 1)
                user_id = int(user_id_str)
            except ValueError:
                print("Формат: <user_id> <текст>")
                continue

            session = manager.get_or_create(user_id)
            response = session.process_message(text)
            print(f"[user_id={user_id}] Bot > {response.text}")
            print(f"[active_sessions={manager.active_count()}]")
    finally:
        manager.close_all()


if __name__ == "__main__":
    main()