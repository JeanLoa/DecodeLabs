# DecodeBot Assistant

DecodeBot is a polished, rule-based AI assistant created for Project 1 of the
DecodeLabs Artificial Intelligence internship. It presents the required
deterministic decision logic through a modern Streamlit experience inspired by
contemporary conversational products, without pretending to be a large language
model. Its implementation follows the PDF's professional recommendation: a
normalized phrase index and constant-time dictionary lookup instead of a growing
`if-elif` ladder.

## Product experience

- Professional responsive chat interface built with Streamlit.
- Multiple conversations with automatic titles.
- Persistent local history backed by SQLite.
- Conversation export in JSON format.
- Clear empty state, assistant identity, and local-memory indicator.
- Eight deterministic intents with Spanish and English phrase catalogs.
- Robust normalization for case, accents, punctuation, and repeated whitespace.
- Direct O(1) phrase-to-intent lookup with controlled fallback behavior.
- Visible quick prompts, supported phrases, and decision trace.
- Lightweight contextual follow-ups such as “Cuéntame más”.
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

The conversational rules and phrase index stay in `chatbot.py`; Streamlit only
handles the web experience, while `conversation_store.py` owns local persistence.
This keeps the core logic reusable and independently testable.

## Deterministic conversation model

DecodeBot follows the Input-Process-Output blueprint from the project brief:

```text
User input
  -> normalize case, accents, punctuation, and whitespace
  -> direct phrase-to-intent lookup
  -> localized controlled response or fallback
  -> persist conversation and display the latest decision trace
```

The sidebar exposes the supported phrases by intent. They are complete
utterances rather than a loose word allowlist, preventing accidental matches
while keeping every response explainable. A follow-up like `Cuéntame más` uses
the latest recognized intent in that conversation; this is bounded state, not
semantic inference or an LLM call.

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
| Handle greetings | Spanish and English greeting phrases in the knowledge base |
| Handle exit commands | Exit intent and `is_exit_command()` |
| Use deterministic decision logic | Direct dictionary lookup in `resolve_intent()` |
| Run continuously | `while True` loop in the CLI and persistent web session |
| Sanitize input | Case, accent, punctuation, and whitespace normalization |
| Provide at least five intents | Eight predefined intent categories |
| Provide a fallback | Localized controlled fallback in `get_reply()` |

## Scope and limitations

DecodeBot remains intentionally rule-based, matching the Project 1 brief. The
application remembers and exports conversations and can resolve bounded
follow-ups from the latest recognized intent. It does not infer unseen intent,
learn from messages, or call an LLM. The sophistication is concentrated in
product design, usability, architecture, traceability, and persistence so the
submission looks professional without misrepresenting its AI capabilities.

## Author

Jean Franck Loa Rojas
