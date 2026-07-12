# DecodeBot Assistant

DecodeBot is a polished, rule-based AI assistant created for Project 1 of the
DecodeLabs Artificial Intelligence internship. It presents the required
`if-elif-else` decision logic through a modern Streamlit experience inspired by
contemporary conversational products, without pretending to be a large language
model.

## Product experience

- Professional responsive chat interface built with Streamlit.
- Multiple conversations with automatic titles.
- Persistent local history backed by SQLite.
- Conversation export in JSON format.
- Clear empty state, assistant identity, and local-memory indicator.
- Deterministic answers for greetings, help, project, requirements, skills,
  and exit intents.
- Command-line mode retained as a lightweight alternative.

## Project structure

```text
01-decodebot-assistant/
|-- .streamlit/config.toml
|-- 01-project.pdf
|-- app.py
|-- chatbot.py
|-- conversation_store.py
|-- main.py
|-- requirements.txt
|-- test_chatbot.py
`-- test_conversation_store.py
```

The conversational rules stay in `chatbot.py`; Streamlit only handles the web
experience, while `conversation_store.py` owns local persistence. This keeps the
core logic reusable and independently testable.

## Run the web application

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open `http://localhost:8501` if it does not launch automatically.

## Run the CLI version

```bash
python main.py
```

## Run the tests

```bash
python -m unittest -v
```

The test suite covers normalization, every supported intent, exit detection,
fallback behavior, conversation creation, message persistence, renaming, and
deletion.

## DecodeLabs requirement coverage

| Project requirement | Implementation |
| --- | --- |
| Handle greetings | Greeting branch in `get_response()` |
| Handle exit commands | Exit branch and `is_exit_command()` |
| Use if-else logic | Explicit decision chain in `get_response()` |
| Run continuously | `while True` loop in the CLI and persistent web session |
| Sanitize input | `.strip().lower()` in `normalize_input()` |
| Provide at least five intents | Six predefined intent categories |
| Provide a fallback | Final `else` branch in `get_response()` |

## Scope and limitations

DecodeBot remains intentionally rule-based, matching the Project 1 brief. The
application remembers and exports conversations, but it does not infer unseen
intent, learn from messages, or call an LLM. The sophistication is concentrated
in product design, usability, architecture, and persistence so the submission
looks professional without misrepresenting its AI capabilities.

## Author

Jean Franck Loa Rojas
