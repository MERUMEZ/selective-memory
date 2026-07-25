"""
Точечный ручной тест BrainSession (Этап 1). Не входит в CI, не трогает main.py.
Запуск: python test_brain_session_manual.py
"""
from core.brain_session import BrainSession

EXIT_COMMANDS = {"exit", "quit", "выход"}


def main():
    session = BrainSession(db_path="storage/test_brain_session.db")
    print("BrainSession запущен. 'exit' — выход.")
    try:
        while True:
            text = input("\nUser > ")
            if text.lower() in EXIT_COMMANDS:
                break
            response = session.process_message(text)
            print(f"Bot > {response.text}")
            print(f"[debug] {response.debug}")
    finally:
        session.close()


if __name__ == "__main__":
    main()