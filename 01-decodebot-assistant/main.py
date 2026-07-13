"""Command-line entry point for DecodeBot."""

from chatbot import get_reply, is_exit_command

WELCOME_MESSAGE = (
    "Hello! I'm DecodeBot, your DecodeLabs internship assistant. "
    "Type 'help' to see the available commands."
)
INTERRUPTED_MESSAGE = "Goodbye! Keep building and learning."


def run_chatbot() -> None:
    """Run the chatbot until the user enters an exit command."""
    print(f"DecodeBot: {WELCOME_MESSAGE}")
    previous_intent: str | None = None

    while True:
        try:
            user_input = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print(f"\nDecodeBot: {INTERRUPTED_MESSAGE}")
            break

        reply = get_reply(user_input, previous_intent)
        print(f"DecodeBot: {reply.content}")

        if reply.matched and reply.intent not in {"empty", "exit", "fallback"}:
            previous_intent = reply.intent

        if is_exit_command(user_input):
            break


if __name__ == "__main__":
    run_chatbot()
