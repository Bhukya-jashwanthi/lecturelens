"""LectureLens web app.

    streamlit run app.py

Streamlit re-runs this whole script on every interaction, so:
  * state that must survive between clicks lives in st.session_state
  * expensive objects (LLM client, vector store) are created once with st.cache_resource
"""

import tempfile
from pathlib import Path

import streamlit as st
from pydantic import ValidationError

from lecturelens.config import get_settings
from lecturelens.models import format_timestamp
from lecturelens.rag_chain import Answer, LectureQA
from lecturelens.retry import describe_api_error
from lecturelens.transcript import TranscriptError, import_caption_file, load_transcript
from lecturelens.vectorstore import delete_lecture, index_transcript, list_lectures

st.set_page_config(page_title="LectureLens", page_icon="🎓", layout="wide")


# --- setup -------------------------------------------------------------------------------


def check_configuration() -> None:
    """Stop with setup instructions instead of a stack trace when the API key is missing."""
    try:
        get_settings()
    except ValidationError:
        st.error("**GOOGLE_API_KEY is not set.** Copy `.env.example` to `.env`, add your Gemini key, and restart.")
        st.stop()


@st.cache_resource
def get_qa() -> LectureQA:
    return LectureQA()


def init_state() -> None:
    st.session_state.setdefault("messages", [])  # list of {"role", "content", "answer"}
    st.session_state.setdefault("player", None)  # (video_id, start_seconds) or None


# --- sidebar: add and manage lectures --------------------------------------------------


def add_lecture_sidebar() -> None:
    st.sidebar.header("➕ Add a lecture")
    from_link, from_file = st.sidebar.tabs(["YouTube link", "Caption file"])

    with from_link:
        url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
        if st.button("Add lecture", disabled=not url, width="stretch"):
            index_with_feedback(lambda: load_transcript(url))

    with from_file:
        upload = st.file_uploader("SRT, VTT or copied YouTube transcript", type=["srt", "vtt", "txt"])
        link = st.text_input("YouTube URL (optional, makes timestamps playable)", key="file_link")
        if st.button("Import file", disabled=upload is None, width="stretch"):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / upload.name
                path.write_bytes(upload.getvalue())
                index_with_feedback(lambda: import_caption_file(path, youtube_url=link or None))


def index_with_feedback(load) -> None:
    """Load a transcript, index it, and report the outcome in the sidebar."""
    with st.sidebar:
        with st.spinner("Fetching transcript and creating embeddings..."):
            try:
                transcript = load()
                count = index_transcript(transcript)
            except TranscriptError as exc:
                st.error(str(exc))
                return
            except Exception as exc:  # embedding API failure: quota, outage, bad key...
                st.error(f"Could not index the lecture. {describe_api_error(exc)}")
                return
        st.success(f"Added **{transcript.title}** ({count} chunks).")


def lectures_sidebar() -> list[str]:
    """List indexed lectures with checkboxes; return the IDs of the selected ones."""
    st.sidebar.header("📚 Your lectures")
    lectures = list_lectures()
    if not lectures:
        st.sidebar.info("No lectures yet. Add one above.")
        return []

    selected = []
    for lec in lectures:
        check, remove = st.sidebar.columns([5, 1])
        if check.checkbox(lec["title"], value=True, key=f"use_{lec['video_id']}", help=f"{lec['chunks']} chunks"):
            selected.append(lec["video_id"])
        if remove.button("🗑", key=f"del_{lec['video_id']}", help="Remove this lecture"):
            delete_lecture(lec["video_id"])
            st.rerun()
    return selected


# --- main area: player and chat ------------------------------------------------------


def video_player() -> None:
    if st.session_state.player:
        video_id, start = st.session_state.player
        st.video(f"https://www.youtube.com/watch?v={video_id}", start_time=int(start))


def show_sources(answer: Answer, message_index: int) -> None:
    """Play buttons for the cited moments, and the retrieval details behind the answer."""
    cited = [answer.sources[n - 1] for n in answer.cited]
    playable = [r for r in cited if r.chunk.youtube]
    if playable:
        cols = st.columns(min(len(playable), 4))
        for i, r in enumerate(playable[:4]):
            label = f"▶ Play {r.chunk.timestamp}"
            if cols[i].button(label, key=f"play_{message_index}_{r.chunk.chunk_id}", help=r.chunk.title):
                st.session_state.player = (r.chunk.video_id, r.chunk.start)
                st.rerun()

    with st.expander("How this answer was found"):
        if answer.search_query != answer.question:
            st.caption(f"Searched for: *{answer.search_query}*")
        for n, r in enumerate(answer.sources, start=1):
            marker = "✅ cited" if n in answer.cited else "not cited"
            st.markdown(
                f"**[{n}] {r.chunk.title} @ {format_timestamp(r.chunk.start)}** · "
                f"semantic #{r.semantic_rank or '–'}, keyword #{r.keyword_rank or '–'} · {marker}"
            )
            st.caption(r.chunk.text[:300] + ("..." if len(r.chunk.text) > 300 else ""))


def chat(selected_ids: list[str]) -> None:
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("answer"):
                show_sources(message["answer"], i)

    question = st.chat_input("Ask about your lectures...", disabled=not selected_ids)
    if not question:
        return

    history = [
        (st.session_state.messages[i]["content"], st.session_state.messages[i + 1]["answer"].raw_text)
        for i in range(0, len(st.session_state.messages) - 1, 2)
    ]
    st.session_state.messages.append({"role": "user", "content": question})
    # Show the question right away, and the spinner where the answer will appear.
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"), st.spinner("Searching the lectures..."):
        try:
            answer = get_qa().ask(question, history=history, video_ids=selected_ids)
        except Exception as exc:  # API outage after retries, quota exhausted, network down...
            st.session_state.messages.pop()  # let the user simply ask again
            st.error(f"Sorry, I couldn't answer that. {describe_api_error(exc)}")
            return
    st.session_state.messages.append({"role": "assistant", "content": answer.text, "answer": answer})

    # Jump the player to the first cited moment of the newest answer.
    first = next((answer.sources[n - 1].chunk for n in answer.cited if answer.sources[n - 1].chunk.youtube), None)
    if first:
        st.session_state.player = (first.video_id, first.start)
    st.rerun()


def main() -> None:
    check_configuration()
    init_state()

    add_lecture_sidebar()
    selected_ids = lectures_sidebar()
    if st.sidebar.button("Clear chat", width="stretch"):
        st.session_state.messages = []
        st.session_state.player = None
        st.rerun()

    st.title("🎓 LectureLens")
    st.caption("Ask questions across your lectures. Every answer links to the exact moment it comes from.")
    video_player()
    chat(selected_ids)


main()
