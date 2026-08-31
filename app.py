import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq

st.set_page_config(
    page_title="Groww Mutual Fund RAG",
    page_icon="📈"
)

st.title("📈 Groww Mutual Fund RAG Chatbot")
st.write("Ask factual questions about mutual funds and Groww.")

st.info("Facts-only. No investment advice.")

st.write("*Example questions:*")
st.write("• What is SIP?")
st.write("• What is an expense ratio?")
st.write("• What is ELSS?")

st.caption(
    "Please do not enter PAN, Aadhaar, OTP, account number, "
    "email address or phone number."
)


@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_model()


@st.cache_data
def load_knowledge_base():
    with open("groww_faq.txt", "r", encoding="utf-8") as file:
        text = file.read()

    raw_chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]

    chunks = []

    for chunk in raw_chunks:
        source_name = ""
        source_url = ""

        lines = chunk.split("\n")
        content_lines = []

        for line in lines:
            if line.startswith("Source:"):
                source_name = line.replace("Source:", "").strip()
            elif line.startswith("Source URL:"):
                source_url = line.replace("Source URL:", "").strip()
            else:
                content_lines.append(line)

        chunks.append({
            "text": "\n".join(content_lines).strip(),
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

user_question = st.text_input(
    "Ask your question"
)


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
