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

RESPONSES = {
    "empty": (
        "Please enter a message or type 'help' to see the available commands."
    ),
    "greeting": (
        "Hello! I'm DecodeBot, your DecodeLabs internship assistant."
    ),
    "help": (
        "Available commands: ask about the project, requirements, or skills; "
        "type 'exit' to end the conversation."
    ),
    "project": (
        "Project 1 is a rule-based chatbot built with explicit decision logic."
    ),
    "requirements": (
        "The chatbot must handle greetings and exit commands, use if-else "
        "logic, and run in a continuous loop."
    ),
    "skills": (
        "This project practices control flow, decision-making logic, input "
        "normalization, and basic AI concepts."
    ),
    "exit": "Goodbye! Keep building and learning.",
    "fallback": (
        "I don't understand that yet. Type 'help' to see what you can ask."
    ),
}


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
        intent = "empty"
    elif clean_input in GREETING_COMMANDS:
        intent = "greeting"
    elif clean_input in HELP_COMMANDS:
        intent = "help"
    elif clean_input in PROJECT_COMMANDS:
        intent = "project"
    elif clean_input in REQUIREMENT_COMMANDS:
        intent = "requirements"
    elif clean_input in SKILL_COMMANDS:
        intent = "skills"
    elif clean_input in EXIT_COMMANDS:
        intent = "exit"
    else:
        intent = "fallback"

    return RESPONSES.get(intent, RESPONSES["fallback"])