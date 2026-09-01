import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq
import textwrap

st.set_page_config(
    page_title="Groww Mutual Fund FAQ Assistant",
    page_icon="📈",
    layout="centered"
)

st.markdown(
    textwrap.dedent(
    """
    <style>
    /* Main background */
    [data-testid="stAppViewContainer"] {
        background: #07111f;
        color: #f4f7fb;
    }

    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }

    .block-container {
        max-width: 900px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }

    /* Header */
    .hero-card {
        background: linear-gradient(135deg, #101d33, #13263f);
        border: 1px solid #263b58;
        border-radius: 22px;
        padding: 34px 32px;
        margin-bottom: 18px;
        box-shadow: 0 12px 35px rgba(0,0,0,0.22);
    }

    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 8px;
        line-height: 1.15;
    }

    .hero-subtitle {
        font-size: 1.15rem;
        color: #aebbd0;
        margin: 0;
    }

    /* Warning */
    .warning-box {
        background: #211b12;
        border: 1px solid #795b20;
        border-left: 5px solid #f2c14e;
        color: #f7cf68;
        border-radius: 14px;
        padding: 18px 20px;
        margin: 18px 0 22px 0;
        font-size: 1rem;
        line-height: 1.55;
    }

    /* Scope */
    .scope-box {
        background: #101a2b;
        border: 1px solid #2a3950;
        border-radius: 13px;
        padding: 15px 18px;
        color: #cbd5e5;
        margin-bottom: 22px;
        font-weight: 600;
    }

    /* Example questions */
    .questions-card {
        background: #101a2b;
        border: 1px solid #273851;
        border-radius: 20px;
        padding: 24px;
        margin-bottom: 25px;
    }

    .questions-title {
        font-size: 1.25rem;
        font-weight: 750;
        color: white;
        margin-bottom: 16px;
    }

    .question-item {
        background: #0c1626;
        border: 1px solid #263750;
        color: #b8c3d4;
        padding: 16px 18px;
        border-radius: 13px;
        margin: 10px 0;
        line-height: 1.45;
    }

    /* Input */
    div[data-testid="stTextInput"] input {
        background: #101a2b !important;
        color: white !important;
        border: 1px solid #31435f !important;
        border-radius: 13px !important;
        min-height: 54px;
    }

    div[data-testid="stTextInput"] input::placeholder {
        color: #7f8ba0 !important;
    }

    /* Button */
    div[data-testid="stFormSubmitButton"] button {
        background: #0f8f72;
        color: white;
        border: none;
        border-radius: 13px;
        min-height: 52px;
        font-weight: 700;
        width: 100%;
    }

    div[data-testid="stFormSubmitButton"] button:hover {
        background: #12a383;
        color: white;
    }

    /* Answer area */
    h2, h3 {
        color: #ffffff !important;
    }

    a {
        color: #55c8ac !important;
    }

    /* Footer */
    .footer-box {
        margin-top: 35px;
        background: #101a2b;
        border-top: 1px solid #293a52;
        padding: 20px;
        border-radius: 15px;
        color: #98a6bb;
        font-size: 0.92rem;
        line-height: 1.55;
    }

    @media (max-width: 600px) {
        .hero-title {
            font-size: 2rem;
        }

        .hero-card {
            padding: 26px 20px;
        }

        .questions-card {
            padding: 18px;
        }
    }
    </style>

    <div class="hero-card">
        <div class="hero-title">Groww Mutual Fund FAQ Assistant</div>
        <p class="hero-subtitle">
            Facts-only mutual fund information. No investment advice.
        </p>
    </div>

    <div class="warning-box">
        ⚠️ <b>Do not enter PAN, Aadhaar, OTPs, account numbers,
        phone numbers, email addresses, or other personal/account information.</b>
    </div>

    <div class="scope-box">
        ▾ &nbsp; Covers 4 Groww Mutual Fund schemes
    </div>

    <div class="questions-card">
<div class="questions-title">Try asking</div>
<p class="question-item">What is the minimum SIP amount for Groww Large Cap Fund?</p>
<p class="question-item">What is the lock-in period of Groww ELSS Tax Saver Fund?</p>
<p class="question-item">What is the Riskometer level of Groww Value Fund?</p>
<p class="question-item">What does Groww Aggressive Hybrid Fund mainly invest in?</p>
</div>
    """),
    unsafe_allow_html=True
)


@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_model()


@st.cache_data
def load_knowledge_base():
    with open("groww_faq.txt", "r", encoding="utf-8") as file:
        text = file.read()

    raw_chunks = [
        chunk.strip()
        for chunk in text.split("\n\n")
        if chunk.strip()
    ]

    chunks = []

    for chunk in raw_chunks:
        source_name = ""
        source_url = ""
        content_lines = []

        for line in chunk.split("\n"):
            if line.startswith("Source:"):
                source_name = line.replace("Source:", "").strip()
            elif line.startswith("Source URL:"):
                source_url = line.replace("Source URL:", "").strip()
            else:
                content_lines.append(line)

        content = "\n".join(content_lines).strip()

        if not content and chunks:
            if source_name:
                chunks[-1]["source_name"] = source_name
            if source_url:
                chunks[-1]["source_url"] = source_url

        elif content:
            chunks.append({
                "text": content,
                "source_name": source_name,
                "source_url": source_url
            })

    return chunks

chunks = load_knowledge_base()


@st.cache_resource
def create_vector_index(chunks):
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(texts)
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index


index = create_vector_index(chunks)


def retrieve_context(question, top_k=3):
    question_embedding = model.encode([question])
    question_embedding = np.array(question_embedding).astype("float32")

    distances, indices = index.search(
        question_embedding,
        top_k
    )

    retrieved_chunks = [chunks[i] for i in indices[0]]

    context = "\n\n".join(
        chunk["text"] for chunk in retrieved_chunks
    )

    best_chunk = retrieved_chunks[0]

    return (
        context,
        best_chunk["source_name"],
        best_chunk["source_url"]
    )


def get_source(question):
    q = question.lower()

    if "expense" in q:
        return (
            "Groww Mutual Fund – Expense Ratio",
            "https://www.growwmf.in/downloads/expense-ratio"
        )

    if "risk" in q or "riskometer" in q:
        return (
            "Groww Mutual Fund – Riskometer",
            "https://www.growwmf.in/downloads/riskometer"
        )

    if "sid" in q or "scheme document" in q:
        return (
            "Groww Mutual Fund – Scheme Information Documents",
            "https://www.growwmf.in/downloads/sid"
        )

    if "performance" in q or "return" in q:
        return (
            "Groww Mutual Fund – Scheme Performance",
            "https://www.growwmf.in/downloads/scheme-performance"
        )

    if "portfolio" in q:
        return (
            "Groww Mutual Fund – Portfolio Disclosure",
            "https://www.growwmf.in/statutory-disclosure/portfolio"
        )

    return (
        "Groww Mutual Fund – Official Website",
        "https://www.growwmf.in/"
    )


api_key = st.secrets.get("GROQ_API_KEY", "")

with st.form("question_form"):
    user_question = st.text_input(
        "Ask a factual question",
        placeholder="Ask a factual question about the four Groww schemes..."
    )

    ask_button = st.form_submit_button("Ask")

if not ask_button:
    user_question = ""


if user_question:

    question_lower = user_question.lower()

    advice_words = [
        "should i buy",
        "should i sell",
        "should i invest",
        "best fund",
        "recommend",
        "which fund should"
    ]

    pii_words = [
    "pan number",
    "pan",
    "aadhaar number",
    "aadhaar",
    "otp",
    "account number",
    "phone number",
    "email",
    "email address"
]

    comparison_words = [
        "compare returns",
        "highest return",
        "best return",
        "future return",
        "predict return"
    ]

    source_name, source_url = get_source(
        user_question
    )

    if any(word in question_lower for word in pii_words):

        st.subheader("Answer")
        st.write(
           
            "Please do not share personal or sensitive information. "
"This chatbot does not require PAN, Aadhaar, OTP, "
"account numbers, phone numbers or email addresses."

        )
        st.markdown(
            f"*Source:* [{source_name}]({source_url})"
        )

    elif any(word in question_lower for word in advice_words):

        st.subheader("Answer")
        st.write(
            "I can provide factual mutual-fund information, "
            "but I cannot recommend whether you should buy, "
            "sell or invest in a particular fund."
        )

        st.markdown(
    "*Educational source:* [SEBI - Mutual Funds Investor Education]"
    "(https://investor.sebi.gov.in/pdf/reference-material/MFunds.pdf)"
)

    elif any(word in question_lower for word in comparison_words):

        st.subheader("Answer")
        st.write(
            "I do not predict, rank or compare mutual-fund returns. "
            "Please refer to the official scheme information."
        )

        st.markdown(
    "*Source:* [Groww Mutual Fund - Official Fact Sheets]"
    "(https://www.growwmf.in/downloads/fact-sheet)"
)
        

    elif not api_key:

        st.error(
            "Groq API key is missing. "
            "Add GROQ_API_KEY in Streamlit secrets."
        )

    else:

        context, source_name, source_url = retrieve_context(
            user_question
        )

        client = Groq(
            api_key=api_key
        )

        prompt = f"""
You are a facts-only assistant for Groww Mutual Fund FAQs.

Scope:
- Groww Large Cap Fund
- Groww Value Fund
- Groww ELSS Tax Saver Fund
- Groww Aggressive Hybrid Fund
- General mutual fund educational questions included in the knowledge base

Answer only from the provided context.

Rules:
- Give factual educational information only.
- Do not give investment advice.
- Do not recommend buying or selling funds.
- Do not predict, rank or compare returns.
- Do not ask for, repeat or expose PAN, Aadhaar, OTP, account number, phone number or email address.
- Keep the answer to a maximum of 3 sentences.
- If the question is about another scheme outside the selected scope, say that the scheme is outside this chatbot's current knowledge base.
- If the answer is unavailable, say:
"I could not find this information in the Groww FAQ knowledge base."

Context:
{context}

Question:
{user_question}
"""

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        answer = response.choices[0].message.content

        st.subheader("Answer")
        st.write(answer)

        st.markdown(
            f"*Source:* [{source_name}]({source_url})"
        )

        st.caption(
            "Last updated from sources: 31 August 2026"
        )

        with st.expander(
            "Retrieved RAG Context"
        ):
            st.write(context)


st.markdown("---")
st.caption(
    "Facts-only educational assistant. "
    "No investment advice. Mutual fund investments "
    "are subject to market risks."
)
