"""DecodePath: a transparent, content-based technology career recommender."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

import streamlit as st

from src.catalog import load_catalog
from src.models import UserProfile
from src.recommender import (
    MINIMUM_SKILLS,
    InsufficientProfileError,
    TechStackRecommender,
)
from src.storage import RecommendationStore

ROOT = Path(__file__).resolve().parent
CATALOG_PATH = ROOT / "data" / "raw" / "raw_skills.csv"
DATABASE_PATH = ROOT / "data" / "decodepath.db"

GOALS = [
    "Choose my first technology career",
    "Move into an AI role",
    "Transition to cloud and infrastructure",
    "Build complete digital products",
    "Work more closely with data",
    "Strengthen my current profile",
]
EXPERIENCE_LEVELS = ["Exploring", "Beginner", "Intermediate", "Advanced"]


st.set_page_config(
    page_title="DecodePath · Tech Stack Recommender",
    page_icon="✦",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root{
      --canvas:#f8fafd;--sidebar:#f0f4f9;--surface:#ffffff;--surface-2:#f3f6fc;
      --ink:#1f1f1f;--muted:#5f6368;--line:#dfe3eb;--blue:#0b57d0;
      --blue-soft:#d3e3fd;--violet:#7b4eff;--red:#d93025;--green:#0f9d58;
    }
    .stApp{background:var(--canvas);color:var(--ink)}
    [data-testid="stHeader"]{background:transparent}
    [data-testid="stSidebar"]{
      background:var(--sidebar);border-right:0;min-width:276px;max-width:276px;
    }
    [data-testid="stSidebar"]>div{padding:1.15rem .85rem}
    .block-container{max-width:1040px;padding:1.15rem 2rem 8rem}
    .brand{display:flex;align-items:center;gap:.72rem;font-size:1.08rem;font-weight:730}
    .brand-mark{
      display:grid;place-items:center;width:36px;height:36px;border-radius:12px;color:white;
      background:linear-gradient(135deg,#4285f4 8%,#9b72cb 48%,#d96570 76%,#fbbc04);
      box-shadow:0 8px 22px #4285f433;
    }
    .brand-sub{color:var(--muted);font-size:.74rem;margin:.45rem 0 1.1rem 3rem}
    .topbar{
      min-height:46px;display:flex;align-items:center;justify-content:space-between;
      margin-bottom:1rem;color:var(--muted);font-size:.8rem;
    }
    .topbar strong{color:var(--ink);font-size:.92rem}
    .local-badge{
      display:inline-flex;align-items:center;gap:.38rem;padding:.34rem .7rem;
      border:1px solid var(--line);border-radius:999px;background:#fff;
    }
    .local-dot{width:7px;height:7px;border-radius:50%;background:var(--green)}
    .welcome{padding:8vh 0 2.2rem;max-width:790px;margin:auto}
    .welcome h1{
      margin:0;font-size:3.45rem;font-weight:540;letter-spacing:-.055em;line-height:1.08;
      background:linear-gradient(90deg,#4285f4 0%,#9b72cb 37%,#d96570 70%,#d27a00 100%);
      -webkit-background-clip:text;background-clip:text;color:transparent;
    }
    .welcome h2{
      margin:.25rem 0 0;font-size:3.45rem;font-weight:520;letter-spacing:-.055em;
      color:#c4c7c5;line-height:1.08;
    }
    .welcome p{color:var(--muted);font-size:1rem;line-height:1.65;max-width:680px;margin:1.2rem 0 0}
    .section-label{
      margin:.25rem 0 .75rem;color:var(--muted);font-size:.74rem;font-weight:700;
      letter-spacing:.09em;text-transform:uppercase;
    }
    .profile-card{
      border:1px solid var(--line);background:var(--surface);border-radius:24px;
      padding:1.25rem 1.35rem .35rem;box-shadow:0 2px 10px #3c40430a;
    }
    .profile-head{display:flex;align-items:flex-start;gap:.85rem;margin-bottom:.55rem}
    .profile-icon{
      display:grid;place-items:center;width:36px;height:36px;border-radius:12px;
      color:var(--blue);background:var(--blue-soft);font-weight:800;
    }
    .profile-head b{display:block;font-size:.95rem}
    .profile-head span{color:var(--muted);font-size:.8rem;line-height:1.45}
    .prompt-card{
      min-height:116px;border:1px solid var(--line);background:var(--surface);
      border-radius:18px;padding:1rem;transition:transform .16s ease,box-shadow .16s ease;
    }
    .prompt-card:hover{transform:translateY(-2px);box-shadow:0 8px 22px #3c404312}
    .prompt-card b{font-size:.83rem;line-height:1.4}
    .prompt-card p{color:var(--muted);font-size:.75rem;line-height:1.45;margin:.42rem 0 0}
    .assistant-note{
      display:flex;gap:.8rem;margin:1.1rem 0;padding:.2rem 0;max-width:820px;
    }
    .assistant-avatar{
      flex:0 0 auto;display:grid;place-items:center;width:34px;height:34px;border-radius:11px;
      color:#fff;background:linear-gradient(135deg,#4285f4,#9b72cb,#d96570);
    }
    .assistant-copy{line-height:1.62;font-size:.93rem;padding-top:.25rem}
    .assistant-copy small{display:block;color:var(--muted);margin-top:.2rem}
    .result-card{
      position:relative;overflow:hidden;border:1px solid var(--line);background:var(--surface);
      border-radius:22px;padding:1.25rem 1.3rem;margin:.85rem 0;
      box-shadow:0 2px 8px #3c40430a;
    }
    .result-card:before{
      content:"";position:absolute;left:0;top:0;bottom:0;width:4px;
      background:linear-gradient(#4285f4,#9b72cb);
    }
    .result-top{display:flex;justify-content:space-between;gap:1rem;align-items:flex-start}
    .rank{color:var(--blue);font-size:.69rem;font-weight:800;letter-spacing:.1em}
    .result-card h3{font-size:1.18rem;margin:.18rem 0 .25rem;letter-spacing:-.02em}
    .category{color:var(--muted);font-size:.75rem}
    .score{
      flex:0 0 auto;display:grid;place-items:center;width:64px;height:64px;border-radius:50%;
      background:conic-gradient(var(--blue) var(--score),#e8eaed 0);
    }
    .score-inner{
      display:grid;place-items:center;width:52px;height:52px;border-radius:50%;
      background:#fff;font-size:.77rem;font-weight:780;color:var(--blue);
    }
    .result-summary{color:#3c4043;line-height:1.58;font-size:.86rem;margin:.85rem 0}
    .chip-row{display:flex;flex-wrap:wrap;gap:.42rem;margin-top:.55rem}
    .chip{
      display:inline-block;padding:.29rem .58rem;border-radius:999px;font-size:.69rem;
      background:var(--blue-soft);color:#174ea6;
    }
    .chip.gap{background:#f1f3f4;color:#5f6368}
    .why{
      margin-top:.9rem;padding:.8rem .9rem;border-radius:13px;background:var(--surface-2);
      color:var(--muted);font-size:.76rem;line-height:1.5;
    }
    .method-hero{
      padding:1.8rem;border:1px solid var(--line);border-radius:24px;background:#fff;
      margin-bottom:1rem;
    }
    .method-hero h1{font-size:2rem;letter-spacing:-.04em;margin:.2rem 0 .55rem}
    .method-hero p{color:var(--muted);line-height:1.65;max-width:760px}
    .method-step{
      height:100%;border:1px solid var(--line);border-radius:18px;background:#fff;padding:1rem;
    }
    .method-no{color:var(--blue);font-size:.7rem;font-weight:800;letter-spacing:.1em}
    .method-step b{display:block;margin:.45rem 0 .25rem}
    .method-step p{color:var(--muted);font-size:.79rem;line-height:1.5;margin:0}
    .metric-box{
      border:1px solid var(--line);border-radius:18px;background:#fff;padding:1rem;
      text-align:center;
    }
    .metric-box b{display:block;font-size:1.5rem;color:var(--blue)}
    .metric-box span{color:var(--muted);font-size:.75rem}
    [data-testid="stChatInput"]{
      border:1px solid #c7c9cf;border-radius:28px;background:#fff;
      box-shadow:0 8px 28px #3c404318;
    }
    [data-testid="stChatInput"] textarea{font-size:.9rem}
    [data-testid="stMultiSelect"] [data-baseweb="select"]>div,
    [data-testid="stTextInput"] input{
      background:#fff;border-color:#c7c9cf;border-radius:13px;
    }
    div[data-testid="stButton"] button,div[data-testid="stFormSubmitButton"] button{
      border-radius:999px;min-height:2.45rem;font-weight:650;border-color:var(--line);
    }
    div[data-testid="stFormSubmitButton"] button{
      background:var(--blue);color:#fff;border-color:var(--blue);
    }
    [data-testid="stExpander"]{border:1px solid var(--line);border-radius:16px;background:#fff}
    [data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:16px;overflow:hidden}
    #MainMenu,footer{visibility:hidden}
    @media(max-width:800px){
      .block-container{padding:1rem 1rem 7rem}
      .welcome{padding:3rem 0 1.5rem}
      .welcome h1,.welcome h2{font-size:2.35rem}
      .result-top{gap:.5rem}.score{width:58px;height:58px}.score-inner{width:47px;height:47px}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_services() -> tuple[TechStackRecommender, RecommendationStore]:
    roles = load_catalog(CATALOG_PATH)
    return (
        TechStackRecommender(roles),
        RecommendationStore(DATABASE_PATH),
    )


try:
    recommender, store = get_services()
except Exception as error:
    st.error(f"DecodePath could not initialize: {error}")
    st.stop()


def set_preset(skills: list[str], goal: str) -> None:
    st.session_state.profile_skills = skills
    st.session_state.profile_goal = goal


def clear_recommendation() -> None:
    st.session_state.pop("result_payload", None)
    st.session_state.pop("profile_payload", None)
    st.session_state.pop("user_copy", None)
    st.session_state.pop("notice", None)


def run_recommendation(
    skills: list[str] | tuple[str, ...],
    goal: str,
    interests: str,
    experience: str,
    user_copy: str,
) -> None:
    canonical = recommender.canonicalize_skills(skills)
    profile = UserProfile(
        skills=canonical,
        goal=goal.strip(),
        interests=interests.strip(),
        experience=experience,
    )
    try:
        recommendations = recommender.recommend(profile)
    except InsufficientProfileError:
        missing = max(0, MINIMUM_SKILLS - len(canonical))
        st.session_state.notice = (
            f"I found {len(canonical)} recognizable skill"
            f"{'' if len(canonical) == 1 else 's'}. Add {missing} more so the "
            "personalized comparison has enough signal."
        )
        return

    store.save(profile, recommendations)
    st.session_state.profile_payload = profile.to_dict()
    st.session_state.result_payload = [item.to_dict() for item in recommendations]
    st.session_state.user_copy = user_copy
    st.session_state.pop("notice", None)


def render_chips(values: list[str] | tuple[str, ...], kind: str = "") -> str:
    if not values:
        return '<span class="chip gap">No direct overlap</span>'
    return "".join(
        f'<span class="chip {kind}">{escape(value)}</span>' for value in values
    )


def render_result(item: dict[str, Any]) -> None:
    role = item["role"]
    score = float(item["score"])
    score_percent = round(score * 100)
    matched = item.get("matched_skills", [])
    gaps = item.get("skill_gaps", [])
    st.markdown(
        f"""
        <article class="result-card">
          <div class="result-top">
            <div>
              <div class="rank">MATCH {int(item["rank"]):02d}</div>
              <h3>{escape(role["name"])}</h3>
              <div class="category">{escape(role["category"])}</div>
            </div>
            <div class="score" style="--score:{score_percent}%">
              <div class="score-inner">{score_percent}%</div>
            </div>
          </div>
          <p class="result-summary">{escape(role["summary"])}</p>
          <div class="section-label">Skills you already bring</div>
          <div class="chip-row">{render_chips(matched)}</div>
          <div class="section-label" style="margin-top:.85rem">Skills to grow next</div>
          <div class="chip-row">{render_chips(gaps, "gap")}</div>
          <div class="why"><b>Why this appeared</b><br>{escape(item["rationale"])}</div>
        </article>
        """,
        unsafe_allow_html=True,
    )
    with st.expander(f"Explore the {role['name']} path"):
        detail_columns = st.columns(2, gap="large")
        with detail_columns[0]:
            st.markdown("**Useful tools**")
            st.write(" · ".join(role["tools"]))
        with detail_columns[1]:
            st.markdown("**Suggested learning path**")
            for index, step in enumerate(role["learning_path"], start=1):
                st.markdown(f"{index}. {step}")


if "profile_skills" not in st.session_state:
    st.session_state.profile_skills = []
if "profile_goal" not in st.session_state:
    st.session_state.profile_goal = GOALS[0]

with st.sidebar:
    st.markdown(
        '<div class="brand"><span class="brand-mark">✦</span>'
        "<span>DecodePath</span></div>"
        '<div class="brand-sub">Tech Stack Recommender</div>',
        unsafe_allow_html=True,
    )
    st.button(
        "＋  New recommendation",
        use_container_width=True,
        on_click=clear_recommendation,
    )
    page = st.radio(
        "Navigate",
        ["Discover", "How it works", "Career catalog"],
        captions=[
            "Find your strongest paths",
            "Inspect the recommendation logic",
            "Explore the local dataset",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("RECENT EXPLORATIONS")
    recent = store.list_recent(limit=6)
    if not recent:
        st.caption("Your recommendations will appear here.")
    for saved in recent:
        if st.button(
            f"↗  {saved.title}",
            key=f"history_{saved.id}",
            use_container_width=True,
        ):
            st.session_state.profile_payload = saved.profile
            st.session_state.result_payload = saved.recommendations
            skills_copy = ", ".join(saved.profile.get("skills", []))
            st.session_state.user_copy = f"My skills: {skills_copy}"
            st.session_state.pop("notice", None)
            st.rerun()
    st.markdown("---")
    st.caption("PROJECT 03 · DECODELABS")
    st.markdown(
        '<div class="local-badge"><span class="local-dot"></span>'
        "Local and deterministic</div>",
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="topbar"><strong>AI Recommendation Logic</strong>'
    '<span class="local-badge">TF-IDF · cosine similarity · Top 3</span></div>',
    unsafe_allow_html=True,
)

if page == "Discover":
    has_results = bool(st.session_state.get("result_payload"))

    if not has_results:
        st.markdown(
            """
            <section class="welcome">
              <h1>Hello, explorer.</h1>
              <h2>Where could your skills take you?</h2>
              <p>Tell DecodePath what you know and what you want to build.
              It will compare your profile against a local career catalog and
              explain the three closest paths—without a black box.</p>
            </section>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="section-label">Try a starting point</div>', unsafe_allow_html=True)
        prompt_columns = st.columns(3, gap="medium")
        presets = [
            (
                "I enjoy data and prediction",
                "Python · SQL · statistics · machine learning",
                ["Python", "SQL", "Statistics", "Machine Learning"],
                "Move into an AI role",
            ),
            (
                "I want to build cloud systems",
                "AWS · Linux · Terraform · Kubernetes",
                ["AWS", "Linux", "Terraform", "Kubernetes"],
                "Transition to cloud and infrastructure",
            ),
            (
                "I like creating web products",
                "JavaScript · TypeScript · React · APIs",
                ["JavaScript", "TypeScript", "React", "REST APIs"],
                "Build complete digital products",
            ),
        ]
        for column, preset in zip(prompt_columns, presets, strict=True):
            title, copy, skills, goal = preset
            with column:
                st.markdown(
                    f'<div class="prompt-card"><b>{escape(title)}</b>'
                    f"<p>{escape(copy)}</p></div>",
                    unsafe_allow_html=True,
                )
                st.button(
                    "Use this profile",
                    key=f"preset_{goal}",
                    use_container_width=True,
                    on_click=set_preset,
                    args=(skills, goal),
                )

    st.markdown('<div class="section-label" style="margin-top:1.3rem">Your profile</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="profile-card"><div class="profile-head">'
        '<div class="profile-icon">01</div><div><b>Give the engine real signal</b>'
        f"<span>Choose at least {MINIMUM_SKILLS} skills. Your goal and interests "
        "refine the ranking but never replace the explicit profile.</span></div>"
        "</div></div>",
        unsafe_allow_html=True,
    )
    with st.form("recommendation_profile"):
        skills = st.multiselect(
            "Skills and tools",
            recommender.skill_options,
            key="profile_skills",
            placeholder="Search Python, React, AWS, SQL…",
            help="At least three distinct selections are required by the project brief.",
        )
        goal = st.selectbox("Primary goal", GOALS, key="profile_goal")
        form_columns = st.columns([1, 1.4], gap="large")
        with form_columns[0]:
            experience = st.select_slider(
                "Current experience",
                EXPERIENCE_LEVELS,
                value="Exploring",
            )
        with form_columns[1]:
            interests = st.text_input(
                "What kind of problems do you enjoy?",
                placeholder="Automation, visual products, scalable systems…",
            )
        submitted = st.form_submit_button(
            "Find my career paths",
            use_container_width=True,
        )
    if submitted:
        user_copy = (
            f"I know {', '.join(skills)}. My goal is to "
            f"{goal.lower()}. {interests}".strip()
        )
        run_recommendation(skills, goal, interests, experience, user_copy)
        st.rerun()

    if st.session_state.get("notice"):
        st.markdown(
            '<div class="assistant-note"><div class="assistant-avatar">✦</div>'
            f'<div class="assistant-copy">{escape(st.session_state["notice"])}'
            "<small>You can use the selector above or describe your stack below.</small>"
            "</div></div>",
            unsafe_allow_html=True,
        )

    if has_results:
        st.chat_message("user").write(st.session_state.get("user_copy", "Analyze my profile."))
        st.markdown(
            '<div class="assistant-note"><div class="assistant-avatar">✦</div>'
            '<div class="assistant-copy">I compared your profile with every role in the '
            "catalog. These are the three closest content matches."
            "<small>Scores are cosine similarity values—not probabilities or guarantees.</small>"
            "</div></div>",
            unsafe_allow_html=True,
        )
        results = st.session_state["result_payload"]
        for result in results:
            render_result(result)

        export_data = {
            "project": "DecodePath",
            "method": "TF-IDF + cosine similarity",
            "profile": st.session_state.get("profile_payload", {}),
            "recommendations": results,
        }
        st.download_button(
            "Download this recommendation as JSON",
            json.dumps(export_data, ensure_ascii=False, indent=2),
            file_name="decodepath-recommendation.json",
            mime="application/json",
            use_container_width=True,
        )

    natural_prompt = st.chat_input(
        "Describe your stack naturally: “I use Python, AWS and Kubernetes…”"
    )
    if natural_prompt:
        extracted = recommender.extract_skills(natural_prompt)
        run_recommendation(
            extracted,
            "Strengthen my current profile",
            natural_prompt,
            "Exploring",
            natural_prompt,
        )
        st.rerun()

elif page == "How it works":
    st.markdown(
        """
        <section class="method-hero">
          <div class="section-label">Transparent by design</div>
          <h1>A recommendation you can inspect.</h1>
          <p>DecodePath is a content-based system. It compares explicit user
          preferences with career metadata; it does not use other users,
          hidden ratings, an LLM, or remote APIs.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    steps = [
        ("01 · INGEST", "Build the profile", "Collect at least three skills plus an optional goal, interest and experience signal."),
        ("02 · WEIGHT", "Apply TF-IDF", "Reward distinctive terms and reduce the influence of generic terms repeated across many roles."),
        ("03 · SCORE", "Measure cosine similarity", "Compare vector direction instead of raw magnitude and keep every score between zero and one."),
        ("04 · RANK", "Return the Top 3", "Sort deterministically from strongest to weakest match and expose the evidence behind each result."),
    ]
    step_columns = st.columns(4, gap="medium")
    for column, (number, title, copy) in zip(step_columns, steps, strict=True):
        with column:
            st.markdown(
                f'<div class="method-step"><div class="method-no">{number}</div>'
                f"<b>{escape(title)}</b><p>{escape(copy)}</p></div>",
                unsafe_allow_html=True,
            )

    st.subheader("Cold-start strategy")
    st.write(
        "Before a user provides three skills, the system avoids inventing a "
        "personalized score. It uses a clearly labeled popularity fallback, "
        "while the onboarding prompts and catalog metadata help the user create "
        "a meaningful first profile."
    )
    trending_columns = st.columns(3, gap="medium")
    for column, role in zip(trending_columns, recommender.trending(), strict=True):
        with column:
            st.markdown(
                f'<div class="metric-box"><b>{role.popularity}</b>'
                f"<span>{escape(role.name)} · popularity fallback</span></div>",
                unsafe_allow_html=True,
            )

    st.subheader("What the score means")
    st.info(
        "A 0.52 score means the weighted profile vector and role vector have a "
        "cosine similarity of 0.52. It is not a 52% chance of getting a job."
    )
    with st.expander("Inspect implementation boundaries"):
        st.markdown(
            """
            - **Input:** explicit skills, a goal, interests and experience.
            - **Process:** local CSV → validation → TF-IDF → cosine similarity → sorting.
            - **Output:** three roles, raw scores, overlaps, gaps and learning steps.
            - **Privacy:** only recommendation history is stored in local SQLite.
            - **Limitation:** this educational catalog is curated and is not a labor-market forecast.
            """
        )

else:
    st.markdown(
        """
        <section class="method-hero">
          <div class="section-label">Versioned local evidence</div>
          <h1>Career catalog</h1>
          <p>Every recommendation comes from this inspectable dataset. Search
          the roles, compare their skills and verify that no item enters the
          model without usable metadata.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    search = st.text_input("Search roles or skills", placeholder="Cloud, Python, security…")
    query = search.casefold().strip()
    filtered_roles = [
        role
        for role in recommender.roles
        if not query
        or query
        in " ".join(
            [role.name, role.category, *role.skills, *role.tools]
        ).casefold()
    ]
    catalog_rows = [
        {
            "Career path": role.name,
            "Category": role.category,
            "Core skills": " · ".join(role.skills),
            "Tools": " · ".join(role.tools),
            "Popularity": role.popularity,
        }
        for role in filtered_roles
    ]
    st.dataframe(
        catalog_rows,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Popularity": st.column_config.ProgressColumn(
                "Cold-start popularity",
                min_value=0,
                max_value=100,
                format="%d",
            )
        },
    )
    st.caption(
        f"Showing {len(filtered_roles)} of {len(recommender.roles)} versioned roles "
        "from data/raw/raw_skills.csv."
    )
