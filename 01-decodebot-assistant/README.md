# DecodeBot Assistant

DecodeBot Assistant is a command-line, rule-based chatbot created for Project 1
of the DecodeLabs Artificial Intelligence internship program.

The project demonstrates how explicit rules, input normalization, and control
flow can produce predictable conversational behavior without machine learning
or external AI services.

## Features

- Runs continuously until the user enters an exit command.
- Normalizes uppercase letters and surrounding spaces.
- Recognizes greetings and six predefined intent categories.
- Explains the project, requirements, and skills practiced.
- Shows available commands through a help response.
- Returns a fallback response for unsupported input.
- Handles `Ctrl+C` and end-of-file interruptions cleanly.
- Includes automated tests for the core response logic.

## Project structure

```text
01-decodebot-assistant/
|-- 01-decodebot-assistant.pdf
|-- README.md
|-- chatbot.py
|-- main.py
`-- test_chatbot.py
```

- `chatbot.py` contains input normalization, intent recognition, and responses.
- `main.py` contains the command-line interface and continuous conversation loop.
- `test_chatbot.py` verifies normalization, intents, exit commands, empty input,
  and fallback behavior.
- `01-decodebot-assistant.pdf` contains the original project specification.

## Requirements

- Python 3.8 or newer
- No third-party packages

## Run the chatbot

From the project directory, run:

```bash
python main.py
```

Example conversation:

```text
DecodeBot: Hello! I'm DecodeBot, your DecodeLabs internship assistant. Type 'help' to see the available commands.
You: hello
DecodeBot: Hello! I'm DecodeBot, your DecodeLabs internship assistant.
You: project 1
DecodeBot: Project 1 is a rule-based chatbot built with explicit decision logic.
You: exit
DecodeBot: Goodbye! Keep building and learning.
```

## Supported inputs

| Intent | Example inputs |
| --- | --- |
| Greeting | `hello`, `hi`, `hey` |
| Help | `help`, `commands`, `what can you do` |
| Project | `project`, `project 1`, `what is project 1` |
| Requirements | `requirements`, `project requirements` |
| Skills | `skills`, `what will i learn` |
| Exit | `exit`, `quit`, `bye`, `goodbye` |

Any unsupported input receives a fallback response instead of stopping the
program.

## Run the tests

The tests use Python's built-in `unittest` framework:

```bash
python -m unittest -v
```

The suite covers:

- Case and whitespace normalization
- Every supported intent category
- Exit-command detection
- Empty input
- Unknown input and fallback behavior

## DecodeLabs requirement coverage

| Project requirement | Implementation |
| --- | --- |
| Handle greetings | Greeting branch in `get_response()` |
| Handle exit commands | `is_exit_command()` and the exit response branch |
| Use if-else logic | Explicit `if-elif-else` decision chain in `get_response()` |
| Run continuously | `while True` loop in `run_chatbot()` |
| Sanitize input | `.strip().lower()` in `normalize_input()` |
| Provide at least five intents | Greeting, help, project, requirements, skills, and exit |
| Provide a fallback | Final `else` branch in `get_response()` |
| Exit cleanly | `break` in `run_chatbot()` |

## Design decisions

The response logic is separated from the input/output loop. This makes the
chatbot easier to test and allows a future interface to reuse the same core
logic. Responses remain deterministic so every supported input produces a
predictable and verifiable result.

## Limitations

DecodeBot matches predefined inputs. It does not infer meaning, learn from
conversations, retain conversation history, or call a large language model.
These limitations are intentional because Project 1 focuses on rule-based
decision logic.

## Author

Jean Franck Loa Rojas