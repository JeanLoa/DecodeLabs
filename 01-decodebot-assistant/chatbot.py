"""Deterministic intent engine for DecodeBot."""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


@dataclass(frozen=True)
class IntentRule:
    """A traceable rule with supported phrases and localized responses."""

    label: str
    phrases_es: tuple[str, ...]
    phrases_en: tuple[str, ...]
    response_es: str
    response_en: str
    detail_es: str
    detail_en: str


@dataclass(frozen=True)
class IntentMatch:
    intent: str
    language: str


@dataclass(frozen=True)
class BotReply:
    content: str
    intent: str
    language: str
    matched: bool
    contextual: bool = False


INTENT_RULES: dict[str, IntentRule] = {
    "greeting": IntentRule(
        label="Saludo",
        phrases_es=("hola", "buenos dias", "buenas tardes", "hey", "que tal"),
        phrases_en=("hello", "hi", "good morning", "good afternoon", "hey there"),
        response_es="¡Hola! Soy DecodeBot, tu asistente determinista de DecodeLabs. ¿Qué quieres explorar?",
        response_en="Hello! I'm DecodeBot, your deterministic DecodeLabs assistant. What would you like to explore?",
        detail_es="Puedo explicarte el proyecto, sus requisitos, habilidades, arquitectura y memoria local.",
        detail_en="I can explain the project, requirements, skills, architecture, and local memory.",
    ),
    "help": IntentRule(
        label="Ayuda",
        phrases_es=("ayuda", "comandos", "que puedes hacer", "como funciona", "opciones"),
        phrases_en=("help", "commands", "what can you do", "how does this work", "options"),
        response_es="Puedes preguntarme por el proyecto, requisitos, habilidades, arquitectura o memoria. También puedes escribir “salir”.",
        response_en="Ask me about the project, requirements, skills, architecture, or memory. You can also type 'exit'.",
        detail_es="Abre “Frases disponibles” en la barra lateral para ver el catálogo exacto que reconoce el motor.",
        detail_en="Open 'Supported phrases' in the sidebar to inspect the exact catalog recognized by the engine.",
    ),
    "project": IntentRule(
        label="Proyecto",
        phrases_es=("proyecto", "proyecto 1", "que es el proyecto 1", "explica el proyecto", "objetivo del proyecto"),
        phrases_en=("project", "project 1", "what is project 1", "explain the project", "project goal"),
        response_es="El Proyecto 1 es un chatbot basado en reglas que transforma entradas predefinidas en respuestas trazables.",
        response_en="Project 1 is a rule-based chatbot that maps predefined inputs to traceable responses.",
        detail_es="Su flujo sigue el modelo IPO: normaliza la entrada, resuelve una intención mediante un diccionario y genera una respuesta controlada.",
        detail_en="Its flow follows the IPO model: normalize input, resolve an intent through a dictionary, and generate a controlled response.",
    ),
    "requirements": IntentRule(
        label="Requisitos",
        phrases_es=("requisitos", "requisitos del proyecto", "cuales son los requisitos", "que debo implementar", "criterios del proyecto"),
        phrases_en=("requirements", "project requirements", "what are the requirements", "what should i implement", "project criteria"),
        response_es="Debe operar continuamente, normalizar entradas, manejar 5 o más intenciones, responder con fallback y terminar con un comando de salida limpio.",
        response_en="It must run continuously, normalize input, handle at least five intents, provide a fallback, and stop with a clean exit command.",
        detail_es="Esta versión también conserva conversaciones, expone frases admitidas y mantiene la lógica separada de la interfaz.",
        detail_en="This version also preserves conversations, exposes supported phrases, and keeps logic separate from the interface.",
    ),
    "skills": IntentRule(
        label="Habilidades",
        phrases_es=("habilidades", "que habilidades aprendere", "que voy a aprender", "aprendizajes", "competencias"),
        phrases_en=("skills", "what skills will i learn", "what will i learn", "learning outcomes", "competencies"),
        response_es="Practicas control de flujo, normalización, diccionarios, diseño de reglas, persistencia local, pruebas y conceptos básicos de IA.",
        response_en="You practice control flow, normalization, dictionaries, rule design, local persistence, testing, and foundational AI concepts.",
        detail_es="La idea central es dominar sistemas deterministas y trazables antes de avanzar hacia modelos probabilísticos.",
        detail_en="The core idea is to master deterministic, traceable systems before moving to probabilistic models.",
    ),
    "architecture": IntentRule(
        label="Arquitectura",
        phrases_es=("arquitectura", "como esta construido", "motor de reglas", "diccionario de intenciones", "barreras de seguridad"),
        phrases_en=("architecture", "how is it built", "rule engine", "intent dictionary", "guardrails"),
        response_es="DecodeBot separa la interfaz Streamlit, el motor determinista y la persistencia SQLite para mantener responsabilidades claras.",
        response_en="DecodeBot separates the Streamlit interface, deterministic engine, and SQLite persistence to keep responsibilities clear.",
        detail_es="Las frases normalizadas apuntan a intenciones mediante un hash map de acceso O(1); cada intención conserva respuestas y trazabilidad.",
        detail_en="Normalized phrases map to intents through an O(1) hash map; every intent keeps controlled responses and traceability.",
    ),
    "memory": IntentRule(
        label="Memoria",
        phrases_es=("memoria", "recuerdas conversaciones", "como guardas los chats", "historial", "persistencia"),
        phrases_en=("memory", "do you remember conversations", "how do you save chats", "history", "persistence"),
        response_es="Sí. El historial se guarda localmente en SQLite y puedes reabrir o exportar cada conversación.",
        response_en="Yes. History is stored locally in SQLite, and each conversation can be reopened or exported.",
        detail_es="La memoria contextual es deliberadamente limitada: recuerda la última intención para responder seguimientos, pero no aprende ni llama a un LLM.",
        detail_en="Contextual memory is intentionally limited: it remembers the last intent for follow-ups, but it does not learn or call an LLM.",
    ),
    "exit": IntentRule(
        label="Salida",
        phrases_es=("salir", "cerrar", "adios", "hasta luego", "terminar"),
        phrases_en=("exit", "quit", "bye", "goodbye", "close"),
        response_es="¡Hasta luego! Tu conversación quedó guardada localmente.",
        response_en="Goodbye! Your conversation has been saved locally.",
        detail_es="Usa una nueva conversación cuando quieras continuar.",
        detail_en="Start a new conversation whenever you want to continue.",
    ),
}

EMPTY_RESPONSES = {
    "es": "Escribe un mensaje o consulta las frases disponibles en la barra lateral.",
    "en": "Enter a message or inspect the supported phrases in the sidebar.",
}
FALLBACK_RESPONSES = {
    "es": "Aún no tengo una regla exacta para eso. Prueba con “ayuda”, “proyecto”, “requisitos”, “habilidades”, “arquitectura” o “memoria”.",
    "en": "I do not have an exact rule for that yet. Try 'help', 'project', 'requirements', 'skills', 'architecture', or 'memory'.",
}
FOLLOW_UP_PHRASES = {
    "es": ("cuentame mas", "dime mas", "amplia la respuesta", "por que", "mas detalles"),
    "en": ("tell me more", "say more", "expand the answer", "why", "more details"),
}


def normalize_input(user_input: str) -> str:
    """Normalize casing, accents, punctuation, and whitespace."""
    decomposed = unicodedata.normalize("NFKD", user_input.casefold())
    without_accents = "".join(character for character in decomposed if not unicodedata.combining(character))
    without_punctuation = re.sub(r"[^\w\s]", " ", without_accents, flags=re.UNICODE)
    return " ".join(without_punctuation.split())


def _build_phrase_index() -> dict[str, IntentMatch]:
    phrase_index: dict[str, IntentMatch] = {}
    for intent, rule in INTENT_RULES.items():
        for language, phrases in (("es", rule.phrases_es), ("en", rule.phrases_en)):
            for phrase in phrases:
                normalized = normalize_input(phrase)
                if normalized in phrase_index:
                    raise ValueError(f"Duplicated supported phrase: {phrase}")
                phrase_index[normalized] = IntentMatch(intent, language)
    return phrase_index


PHRASE_TO_INTENT = _build_phrase_index()
FOLLOW_UP_TO_LANGUAGE = {
    normalize_input(phrase): language
    for language, phrases in FOLLOW_UP_PHRASES.items()
    for phrase in phrases
}


def resolve_intent(user_input: str) -> IntentMatch | None:
    """Resolve an exact supported phrase in constant expected time."""
    return PHRASE_TO_INTENT.get(normalize_input(user_input))


def _fallback_language(clean_input: str) -> str:
    spanish_hints = {
        "que",
        "como",
        "cual",
        "hola",
        "proyecto",
        "puedes",
        "ayuda",
        "requisitos",
        "cuentame",
        "dime",
        "quiero",
    }
    return "es" if spanish_hints.intersection(clean_input.split()) else "en"


def get_reply(user_input: str, previous_intent: str | None = None) -> BotReply:
    """Return response content plus transparent rule-engine metadata."""
    clean_input = normalize_input(user_input)
    if not clean_input:
        return BotReply(EMPTY_RESPONSES["es"], "empty", "es", matched=False)

    match = PHRASE_TO_INTENT.get(clean_input)
    if match:
        rule = INTENT_RULES[match.intent]
        content = rule.response_es if match.language == "es" else rule.response_en
        return BotReply(content, match.intent, match.language, matched=True)

    follow_up_language = FOLLOW_UP_TO_LANGUAGE.get(clean_input)
    if follow_up_language and previous_intent in INTENT_RULES and previous_intent != "exit":
        rule = INTENT_RULES[previous_intent]
        content = rule.detail_es if follow_up_language == "es" else rule.detail_en
        return BotReply(content, previous_intent, follow_up_language, matched=True, contextual=True)

    language = _fallback_language(clean_input)
    return BotReply(FALLBACK_RESPONSES[language], "fallback", language, matched=False)


def get_response(user_input: str, previous_intent: str | None = None) -> str:
    """Backward-compatible response API for the CLI and tests."""
    return get_reply(user_input, previous_intent).content


def is_exit_command(user_input: str) -> bool:
    """Return whether the provided input maps to the exit intent."""
    match = resolve_intent(user_input)
    return bool(match and match.intent == "exit")


def supported_phrases(language: str = "es") -> dict[str, tuple[str, ...]]:
    """Expose the knowledge base without leaking implementation details to UI."""
    attribute = "phrases_es" if language == "es" else "phrases_en"
    return {rule.label: getattr(rule, attribute) for rule in INTENT_RULES.values()}


def previous_supported_intent(messages: list[dict[str, str]]) -> str | None:
    """Recover the latest recognized user intent from persisted history."""
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        match = resolve_intent(message.get("content", ""))
        if match and match.intent != "exit":
            return match.intent
    return None
