"""Interfaz web profesional de DecodeBot construida con Streamlit."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from chatbot import get_response
from conversation_store import ConversationStore


APP_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="DecodeBot", page_icon="✦", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: radial-gradient(circle at 50% -15%, #24143d 0, #09090b 35%); }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] { border-right: 1px solid #27272a; }
    .block-container { max-width: 900px; padding-top: 2.2rem; padding-bottom: 8rem; }
    .brand { display:flex; gap:.7rem; align-items:center; font-weight:750; font-size:1.1rem; }
    .brand-mark { display:grid; place-items:center; width:34px; height:34px; border-radius:11px;
      background:linear-gradient(135deg,#a78bfa,#6d28d9); box-shadow:0 8px 30px #7c3aed55; }
    .eyebrow { color:#a78bfa; letter-spacing:.14em; font-size:.7rem; font-weight:700; text-transform:uppercase; }
    .hero { text-align:center; padding:10vh 1rem 3rem; }
    .hero-icon { margin:auto; display:grid; place-items:center; width:62px; height:62px; border-radius:20px;
      font-size:1.7rem; background:linear-gradient(135deg,#a78bfa,#6d28d9); box-shadow:0 16px 60px #7c3aed66; }
    .hero h1 { font-size:2.6rem; margin:.9rem 0 .45rem; letter-spacing:-.045em; }
    .hero p { max-width:590px; margin:auto; color:#a1a1aa; line-height:1.65; }
    .capabilities { display:flex; justify-content:center; flex-wrap:wrap; gap:.55rem; margin-top:1.4rem; }
    .capability { border:1px solid #3f3f46; border-radius:999px; padding:.4rem .75rem; color:#d4d4d8; font-size:.8rem; }
    [data-testid="stChatMessage"] { border:1px solid #27272a; border-radius:18px; padding:.45rem .75rem; margin:.65rem 0; background:#101012aa; }
    [data-testid="stChatInput"] { border-color:#52525b; box-shadow:0 10px 40px #0008; }
    .status { border:1px solid #27272a; border-radius:12px; padding:.7rem .8rem; color:#a1a1aa; font-size:.76rem; }
    .status-dot { display:inline-block; width:7px; height:7px; border-radius:50%; background:#34d399; margin-right:.45rem; }
    #MainMenu, footer { visibility:hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_store() -> ConversationStore:
    return ConversationStore(APP_DIR / "data" / "decodebot.db")


store = get_store()

if "conversation_id" not in st.session_state:
    conversations = store.list_conversations()
    st.session_state.conversation_id = conversations[0].id if conversations else store.create_conversation()


def new_conversation() -> None:
    st.session_state.conversation_id = store.create_conversation()


with st.sidebar:
    st.markdown('<div class="brand"><span class="brand-mark">✦</span>DecodeBot</div>', unsafe_allow_html=True)
    st.caption("Rule-based AI assistant")
    st.button("＋  Nueva conversación", use_container_width=True, on_click=new_conversation)
    st.markdown("#### Conversaciones")

    for conversation in store.list_conversations():
        selected = conversation.id == st.session_state.conversation_id
        label = f"{'●' if selected else '○'}  {conversation.title}"
        if st.button(label, key=f"open_{conversation.id}", use_container_width=True):
            st.session_state.conversation_id = conversation.id
            st.rerun()

    st.markdown("---")
    st.markdown('<div class="status"><span class="status-dot"></span>Memoria local activa<br>Los chats se guardan en este dispositivo.</div>', unsafe_allow_html=True)


conversation_id = st.session_state.conversation_id
messages = store.get_messages(conversation_id)

left, right = st.columns([5, 1])
with left:
    st.markdown('<div class="eyebrow">Asistente inteligente local</div>', unsafe_allow_html=True)
with right:
    export_payload = json.dumps(messages, ensure_ascii=False, indent=2)
    st.download_button("Exportar", export_payload, "decodebot-chat.json", "application/json", use_container_width=True)

if not messages:
    st.markdown(
        """
        <section class="hero">
          <div class="hero-icon">✦</div>
          <h1>¿En qué puedo ayudarte?</h1>
          <p>Explora DecodeLabs, consulta los requisitos del proyecto o descubre las habilidades que demuestra este asistente.</p>
          <div class="capabilities">
            <span class="capability">Respuestas inmediatas</span>
            <span class="capability">Historial persistente</span>
            <span class="capability">Privacidad local</span>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
else:
    for message in messages:
        avatar = "👤" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

prompt = st.chat_input("Escribe un mensaje para DecodeBot…")
if prompt:
    is_first_message = not messages
    store.add_message(conversation_id, "user", prompt)
    if is_first_message:
        title = " ".join(prompt.split())[:34]
        store.rename_conversation(conversation_id, title + ("…" if len(prompt) > 34 else ""))

    response = get_response(prompt)
    store.add_message(conversation_id, "assistant", response)
    st.rerun()
