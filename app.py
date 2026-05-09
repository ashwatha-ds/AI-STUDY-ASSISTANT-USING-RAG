import streamlit as st
from utils.pdf_processor import process_pdfs
from utils.vector_store import build_vector_store, search_documents
from utils.llm import ask_groq, summarize_document, generate_quiz
from utils.helpers import format_sources, export_chat
import os


st.set_page_config(
    page_title="StudyMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


with open("styles/main.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

if not GROQ_API_KEY:
    st.error("⚠️ GROQ_API_KEY not found. Add it to `.streamlit/secrets.toml` or environment variables.")
    st.stop()


if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "doc_texts" not in st.session_state:
    st.session_state.doc_texts = {}
if "uploaded_names" not in st.session_state:
    st.session_state.uploaded_names = []
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "chat"


with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <span class="logo">🧠</span>
        <div>
            <div class="app-title">StudyMind AI</div>
            <div class="app-subtitle">RAG-Powered Study Assistant</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📂 Upload Study Materials")

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        new_names = [f.name for f in uploaded_files]
        if new_names != st.session_state.uploaded_names:
            with st.spinner("🔍 Processing documents..."):
                chunks, doc_texts = process_pdfs(uploaded_files)
                vs = build_vector_store(chunks)
                st.session_state.vector_store = vs
                st.session_state.doc_texts = doc_texts
                st.session_state.uploaded_names = new_names
                st.session_state.messages = []
            st.success(f"✅ {len(uploaded_files)} document(s) ready!")

        st.markdown("**Loaded Documents:**")
        for name in st.session_state.uploaded_names:
            st.markdown(f"<div class='doc-badge'>📄 {name}</div>", unsafe_allow_html=True)

    st.markdown("---")

    if st.session_state.messages:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        chat_export = export_chat(st.session_state.messages)
        st.download_button(
            "💾 Export Chat",
            data=chat_export,
            file_name="studymind_chat.txt",
            mime="text/plain",
            use_container_width=True,
        )

    st.markdown("---")
    st.markdown("""
    <div class="sidebar-info">
        <b>How it works:</b><br>
        📄 Answers from your docs<br>
        🧠 Falls back to AI for follow-ups<br>
        🔗 Sources always shown
    </div>
    """, unsafe_allow_html=True)


st.markdown("""
<div class="main-header">
    <h1>📚 StudyMind <span class="accent">AI</span></h1>
    <p>Upload your notes & textbooks — ask anything, get cited answers instantly.</p>
</div>
""", unsafe_allow_html=True)


tab1, tab2, tab3 = st.tabs(["💬 Chat", "📝 Summarize", "🧪 Quiz Mode"])




with tab1:
    if not st.session_state.vector_store:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">📂</div>
            <div class="empty-title">No documents loaded</div>
            <div class="empty-sub">Upload PDFs from the sidebar to get started.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.messages:
                role = msg["role"]
                with st.chat_message(role):
                    st.markdown(msg["content"])
                    if role == "assistant" and "sources" in msg and msg["sources"]:
                        with st.expander("🔗 Sources"):
                            st.markdown(msg["sources"])
                    if role == "assistant" and "mode" in msg:
                        badge = "📄 From your documents" if msg["mode"] == "doc" else "🧠 From AI knowledge"
                        color = "#2563eb" if msg["mode"] == "doc" else "#7c3aed"
                        st.markdown(
                            f'<span style="font-size:0.75rem;color:{color};font-weight:600;">{badge}</span>',
                            unsafe_allow_html=True,
                        )

        
        if prompt := st.chat_input("Ask anything about your documents..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    
                    history = st.session_state.messages[-10:]

                    # Search vector store
                    results, score = search_documents(st.session_state.vector_store, prompt)

                    THRESHOLD = 0.45  

                    if results and score >= THRESHOLD:
                        
                        context = "\n\n".join([r.page_content for r in results])
                        answer = ask_groq(
                            prompt, context=context, history=history,
                            api_key=GROQ_API_KEY, mode="rag"
                        )
                        sources = format_sources(results)
                        mode = "doc"
                    else:
                        
                        answer = ask_groq(
                            prompt, context=None, history=history,
                            api_key=GROQ_API_KEY, mode="general"
                        )
                        sources = None
                        mode = "ai"

                    st.markdown(answer)
                    if sources:
                        with st.expander("🔗 Sources"):
                            st.markdown(sources)
                    badge = "📄 From your documents" if mode == "doc" else "🧠 From AI knowledge"
                    color = "#2563eb" if mode == "doc" else "#7c3aed"
                    st.markdown(
                        f'<span style="font-size:0.75rem;color:{color};font-weight:600;">{badge}</span>',
                        unsafe_allow_html=True,
                    )

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "mode": mode,
            })


with tab2:
    if not st.session_state.doc_texts:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">📝</div>
            <div class="empty-title">No documents loaded</div>
            <div class="empty-sub">Upload PDFs from the sidebar first.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("### 📝 Document Summarizer")
        st.markdown("Get a concise, structured summary of any uploaded document.")

        selected_doc = st.selectbox(
            "Choose a document to summarize:",
            options=list(st.session_state.doc_texts.keys()),
        )

        summary_style = st.radio(
            "Summary style:",
            ["Concise (bullet points)", "Detailed (paragraph)", "Key Concepts Only"],
            horizontal=True,
        )

        if st.button("✨ Generate Summary", use_container_width=True):
            with st.spinner(f"Summarizing {selected_doc}..."):
                text = st.session_state.doc_texts[selected_doc]
                summary = summarize_document(text, style=summary_style, api_key=GROQ_API_KEY)
            st.markdown("---")
            st.markdown(f"#### Summary of: *{selected_doc}*")
            st.markdown(summary)
            st.download_button(
                "💾 Download Summary",
                data=summary,
                file_name=f"summary_{selected_doc}.txt",
                mime="text/plain",
            )


with tab3:
    if not st.session_state.doc_texts:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🧪</div>
            <div class="empty-title">No documents loaded</div>
            <div class="empty-sub">Upload PDFs from the sidebar first.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("### 🧪 Quiz Mode")
        st.markdown("Auto-generate MCQ questions from your study material.")

        col1, col2 = st.columns(2)
        with col1:
            quiz_doc = st.selectbox(
                "Select document:",
                options=list(st.session_state.doc_texts.keys()),
                key="quiz_doc",
            )
        with col2:
            num_questions = st.slider("Number of questions:", 3, 10, 5)

        difficulty = st.select_slider(
            "Difficulty:",
            options=["Easy", "Medium", "Hard"],
            value="Medium",
        )

        if st.button("🎯 Generate Quiz", use_container_width=True):
            with st.spinner("Crafting questions..."):
                text = st.session_state.doc_texts[quiz_doc]
                quiz_data = generate_quiz(
                    text,
                    num_questions=num_questions,
                    difficulty=difficulty,
                    api_key=GROQ_API_KEY,
                )

            st.markdown("---")
            if "error" in quiz_data:
                st.error(quiz_data["error"])
            else:
                st.markdown(f"#### 📋 Quiz on: *{quiz_doc}*")
                score_tracker = []

                for i, q in enumerate(quiz_data.get("questions", []), 1):
                    st.markdown(f"**Q{i}. {q['question']}**")
                    user_ans = st.radio(
                        f"q{i}",
                        options=q["options"],
                        key=f"quiz_q_{i}",
                        label_visibility="collapsed",
                    )
                    score_tracker.append({
                        "answer": user_ans,
                        "correct": q["answer"],
                        "explanation": q.get("explanation", ""),
                    })
                    st.markdown("")

                if st.button("✅ Submit Quiz", use_container_width=True):
                    correct = sum(
                        1 for s in score_tracker if s["answer"] == s["correct"]
                    )
                    total = len(score_tracker)
                    pct = int(correct / total * 100)
                    emoji = "🎉" if pct >= 80 else "📚" if pct >= 50 else "💪"
                    st.markdown(
                        f"<div class='score-card'>{emoji} You scored <b>{correct}/{total}</b> ({pct}%)</div>",
                        unsafe_allow_html=True,
                    )
                    for i, s in enumerate(score_tracker, 1):
                        if s["answer"] == s["correct"]:
                            st.success(f"Q{i} ✅ Correct!")
                        else:
                            st.error(f"Q{i} ❌ Correct answer: **{s['correct']}**")
                        if s["explanation"]:
                            st.caption(f"💡 {s['explanation']}")
