"""Core response logic for DecodeBot."""

GREETING_COMMANDS = {"hello", "hi", "hey"}
HELP_COMMANDS = {"help", "commands", "what can you do"}
PROJECT_COMMANDS = {"project", "project 1", "what is project 1"}
REQUIREMENT_COMMANDS = {
    "requirements",
    "project requirements",
    "what are the requirements",
}
SKILL_COMMANDS = {"skills", "what skills will i learn", "what will i learn"}
EXIT_COMMANDS = {"exit", "quit", "bye", "goodbye"}


def normalize_input(user_input: str) -> str:
    """Normalize user input so predefined commands can be matched reliably."""
    return user_input.strip().lower()


def is_exit_command(user_input: str) -> bool:
    """Return whether the provided input should end the conversation."""
    return normalize_input(user_input) in EXIT_COMMANDS


def get_response(user_input: str) -> str:
    """Return a deterministic response for a predefined user intent."""
    clean_input = normalize_input(user_input)

    if not clean_input:
        return "Please enter a message or type 'help' to see the available commands."
    elif clean_input in GREETING_COMMANDS:
        return "Hello! I'm DecodeBot, your DecodeLabs internship assistant."
    elif clean_input in HELP_COMMANDS:
        return (
            "Available commands: ask about the project, requirements, or skills; "
            "type 'exit' to end the conversation."
        )
    elif clean_input in PROJECT_COMMANDS:
        return (
            "Project 1 is a rule-based chatbot built with explicit decision logic."
        )
    elif clean_input in REQUIREMENT_COMMANDS:
        return (
            "The chatbot must handle greetings and exit commands, use if-else "
            "logic, and run in a continuous loop."
        )
    elif clean_input in SKILL_COMMANDS:
        return (
            "This project practices control flow, decision-making logic, input "
            "normalization, and basic AI concepts."
        )
    elif clean_input in EXIT_COMMANDS:
        return "Goodbye! Keep building and learning."
    else:
        return "I don't understand that yet. Type 'help' to see what you can ask."