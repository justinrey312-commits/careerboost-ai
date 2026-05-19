import streamlit as st
import tempfile
import os
import json

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI



# Load environment variables
load_dotenv()

try:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass  # Running locally, use .env instead

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CareerBoost AI",
    page_icon="📄",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}
    .stApp { background: #0f1117; color: #e8e8e8; }
    .career-title {
        font-family: 'Georgia', serif;
        font-size: 2.4rem;
        font-weight: 700;
        color: #f0c040;
        letter-spacing: -0.5px;
        margin-bottom: 0;
    }
    .career-subtitle {
        font-size: 1rem;
        color: #888;
        margin-top: 2px;
        margin-bottom: 1.5rem;
    }
    .chip-row { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 1.2rem; }
    .chip {
        background: #1e2130; border: 1px solid #2e3250;
        border-radius: 20px; padding: 5px 14px;
        font-size: 0.82rem; color: #aab4d4;
    }
    .score-card {
        background: #1a1d2e;
        border: 1px solid #2e3250;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .score-number {
        font-size: 3.5rem;
        font-weight: 800;
        line-height: 1;
    }
    .score-label { font-size: 0.9rem; color: #888; margin-top: 4px; }
    .match-bar-bg {
        background: #1e2130;
        border-radius: 8px;
        height: 22px;
        width: 100%;
        overflow: hidden;
        margin: 8px 0;
    }
    .match-bar-fill {
        height: 100%;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="career-title">📄 CareerBoost AI</p>', unsafe_allow_html=True)
st.markdown('<p class="career-subtitle">AI-powered resume analysis using RAG · Powered by OpenAI + LangChain</p>', unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    **CareerBoost AI** uses Retrieval-Augmented Generation (RAG)
    to analyze your resume and give you actionable feedback.

    **How it works:**
    1. Upload your resume as a PDF
    2. The app chunks it and creates embeddings
    3. Your question retrieves the most relevant chunks
    4. GPT-4o-mini answers using only your resume content
    """)
    st.divider()
    st.caption("Built with LangChain · FAISS · HuggingFace · OpenAI")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "last_uploaded_filename" not in st.session_state:
    st.session_state.last_uploaded_filename = None
if "resume_text" not in st.session_state:
    st.session_state.resume_text = None

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

def get_resume_context(question, k=5):
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(question)
    return "\n\n".join([doc.page_content for doc in docs])

def parse_json_response(raw):
    return json.loads(raw.strip().replace("```json", "").replace("```", ""))

# ── File uploader ─────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload Resume PDF", type="pdf")

if uploaded_file:
    if uploaded_file.name != st.session_state.last_uploaded_filename:
        with st.spinner("🔍 Reading and indexing your resume..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                temp_path = tmp.name

            loader = PyPDFLoader(temp_path)
            docs = loader.load()
            st.session_state.resume_text = "\n\n".join([d.page_content for d in docs])

            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(docs)

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)
            st.session_state.last_uploaded_filename = uploaded_file.name
            st.session_state.messages = []
            os.unlink(temp_path)

        st.success(f"✅ **{uploaded_file.name}** indexed successfully!")
    else:
        st.success(f"✅ **{uploaded_file.name}** is loaded and ready.")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "💬 Chat",
    "📊 Resume Score",
    "💡 Improvements",
    "🎯 Job Matcher"
])

# ════════════════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("""
    <div class="chip-row">
      <span class="chip">💼 Summarize this resume</span>
      <span class="chip">🛠️ What skills does this person have?</span>
      <span class="chip">🎯 What jobs fit this applicant?</span>
      <span class="chip">⚠️ What are the weaknesses?</span>
      <span class="chip">🔑 What ATS keywords are missing?</span>
    </div>
    """, unsafe_allow_html=True)

    for role, content in st.session_state.messages:
        with st.chat_message(role):
            st.write(content)

    user_question = st.chat_input("Ask something about the resume...")

    if user_question:
        if st.session_state.vectorstore is None:
            st.warning("⚠️ Please upload a resume PDF first.")
            st.stop()

        st.session_state.messages.append(("user", user_question))
        with st.chat_message("user"):
            st.write(user_question)

        context = get_resume_context(user_question)

        full_prompt = f"""You are CareerBoost AI, an expert resume coach and career advisor.
You ONLY answer questions related to the resume provided.
If the user asks anything unrelated to the resume, politely decline and remind them to ask about the resume.
Be specific, constructive, and actionable. Always base answers strictly on the resume content provided.
If the resume doesn't contain enough information to answer, say so clearly.

--- RESUME CONTENT ---
{context}
--- END OF RESUME ---

User Question: {user_question}

Answer helpfully and concisely:"""

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_llm().invoke(full_prompt)
                answer = response.content
                st.write(answer)

        st.session_state.messages.append(("assistant", answer))

# ════════════════════════════════════════════════════════════════════════
# TAB 2 — RESUME SCORE
# ════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 Resume Score")
    st.write("Get an overall quality score for your resume across key dimensions.")

    if st.session_state.resume_text is None:
        st.warning("⚠️ Please upload a resume PDF first.")
    else:
        if st.button("⚡ Score My Resume", use_container_width=True):
            with st.spinner("Analyzing your resume..."):
                prompt = f"""You are an expert resume evaluator. Analyze the resume below and score it.

Resume:
{st.session_state.resume_text}

Return ONLY a valid JSON object, no extra text:
{{
  "overall": <int 1-10>,
  "dimensions": {{
    "Clarity & Formatting": <int 1-10>,
    "Work Experience": <int 1-10>,
    "Skills": <int 1-10>,
    "Impact & Achievements": <int 1-10>
  }},
  "verdict": "<one sentence overall verdict>",
  "summary": "<2-3 sentence explanation of the score>"
}}"""

                response = get_llm().invoke(prompt)
                try:
                    data = parse_json_response(response.content)
                    overall = data["overall"]
                    dims = data["dimensions"]

                    if overall >= 8:
                        color, grade = "#4ade80", "Excellent 🌟"
                    elif overall >= 6:
                        color, grade = "#f0c040", "Good 👍"
                    elif overall >= 4:
                        color, grade = "#fb923c", "Needs Work ⚠️"
                    else:
                        color, grade = "#f87171", "Poor 🔴"

                    st.markdown(f"""
                    <div class="score-card">
                        <div class="score-number" style="color:{color}">{overall}<span style="font-size:1.5rem;color:#888">/10</span></div>
                        <div class="score-label">{grade}</div>
                        <p style="color:#ccc;margin-top:10px;font-size:0.95rem"><em>{data.get('verdict','')}</em></p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("#### Breakdown")
                    for dim, score in dims.items():
                        pct = score * 10
                        bc = "#4ade80" if score >= 8 else "#f0c040" if score >= 6 else "#fb923c" if score >= 4 else "#f87171"
                        st.markdown(f"**{dim}** — {score}/10")
                        st.markdown(f"""
                        <div class="match-bar-bg">
                          <div class="match-bar-fill" style="width:{pct}%;background:{bc};"></div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown(f"\n> {data.get('summary', '')}")

                except Exception:
                    st.error("Could not parse score. Raw response:")
                    st.code(response.content)

# ════════════════════════════════════════════════════════════════════════
# TAB 3 — IMPROVEMENT SUGGESTIONS
# ════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 💡 Improvement Suggestions")
    st.write("Get a specific, actionable list of things to fix in your resume.")

    if st.session_state.resume_text is None:
        st.warning("⚠️ Please upload a resume PDF first.")
    else:
        if st.button("🔍 Analyze & Suggest Improvements", use_container_width=True):
            with st.spinner("Reviewing your resume..."):
                prompt = f"""You are an expert resume coach. Review the resume and provide improvement suggestions.

Resume:
{st.session_state.resume_text}

Return ONLY a valid JSON object, no extra text:
{{
  "critical": ["<issue 1>", "<issue 2>"],
  "improvements": ["<suggestion 1>", "<suggestion 2>", "<suggestion 3>"],
  "strengths": ["<strength 1>", "<strength 2>"],
  "quick_wins": ["<quick fix 1>", "<quick fix 2>"]
}}

- critical: serious problems that must be fixed (max 3)
- improvements: specific things to add or rewrite (max 5)
- strengths: what is already good (max 3)
- quick_wins: small easy fixes with immediate impact (max 3)"""

                response = get_llm().invoke(prompt)
                try:
                    data = parse_json_response(response.content)

                    if data.get("critical"):
                        st.markdown("#### 🔴 Critical Issues")
                        for item in data["critical"]:
                            st.error(f"• {item}")

                    if data.get("improvements"):
                        st.markdown("#### 🟡 Suggested Improvements")
                        for item in data["improvements"]:
                            st.warning(f"• {item}")

                    if data.get("quick_wins"):
                        st.markdown("#### ⚡ Quick Wins")
                        for item in data["quick_wins"]:
                            st.info(f"• {item}")

                    if data.get("strengths"):
                        st.markdown("#### ✅ Strengths")
                        for item in data["strengths"]:
                            st.success(f"• {item}")

                except Exception:
                    st.error("Could not parse suggestions. Raw response:")
                    st.code(response.content)

# ════════════════════════════════════════════════════════════════════════
# TAB 4 — JOB MATCHER
# ════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 🎯 Job Description Matcher")
    st.write("Paste a job posting to see how well your resume matches it.")

    if st.session_state.resume_text is None:
        st.warning("⚠️ Please upload a resume PDF first.")
    else:
        job_description = st.text_area(
            "Paste Job Description here",
            height=200,
            placeholder="Paste the full job posting here..."
        )

        if st.button("🎯 Check Match", use_container_width=True):
            if not job_description.strip():
                st.warning("Please paste a job description first.")
            else:
                with st.spinner("Comparing your resume to the job posting..."):
                    prompt = f"""You are an expert recruiter. Compare the resume to the job description.

Resume:
{st.session_state.resume_text}

Job Description:
{job_description}

Return ONLY a valid JSON object, no extra text:
{{
  "match_score": <int 0-100>,
  "matched_keywords": ["<keyword 1>", "<keyword 2>"],
  "missing_keywords": ["<keyword 1>", "<keyword 2>"],
  "verdict": "<one sentence overall match verdict>",
  "advice": "<2-3 sentences on how to tailor the resume for this job>"
}}"""

                    response = get_llm().invoke(prompt)
                    try:
                        data = parse_json_response(response.content)
                        score = data["match_score"]

                        if score >= 75:
                            bar_color, label = "#4ade80", "Strong Match 🌟"
                        elif score >= 50:
                            bar_color, label = "#f0c040", "Moderate Match 👍"
                        else:
                            bar_color, label = "#f87171", "Weak Match ⚠️"

                        st.markdown(f"""
                        <div class="score-card">
                            <div class="score-number" style="color:{bar_color}">{score}<span style="font-size:1.5rem;color:#888">%</span></div>
                            <div class="score-label">{label}</div>
                            <p style="color:#ccc;margin-top:10px;font-size:0.95rem"><em>{data.get('verdict','')}</em></p>
                        </div>
                        <div class="match-bar-bg">
                          <div class="match-bar-fill" style="width:{score}%;background:{bar_color};"></div>
                        </div>
                        """, unsafe_allow_html=True)

                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("#### ✅ Matched Keywords")
                            for kw in data.get("matched_keywords", []):
                                st.success(f"• {kw}")

                        with col2:
                            st.markdown("#### ❌ Missing Keywords")
                            for kw in data.get("missing_keywords", []):
                                st.error(f"• {kw}")

                        st.markdown("#### 📝 How to Tailor Your Resume")
                        st.info(data.get("advice", ""))

                    except Exception:
                        st.error("Could not parse match results. Raw response:")
                        st.code(response.content)