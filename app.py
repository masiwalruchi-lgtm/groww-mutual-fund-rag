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

    chunks = [
        chunk.strip()
        for chunk in text.split("\n\n")
        if chunk.strip()
    ]

    return chunks


chunks = load_knowledge_base()


@st.cache_resource
def create_vector_index(chunks):
    embeddings = model.encode(chunks)
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index


index = create_vector_index(chunks)


def retrieve_context(question, top_k=3):
    question_embedding = model.encode([question])
    question_embedding = np.array(
        question_embedding
    ).astype("float32")

    distances, indices = index.search(
        question_embedding,
        top_k
    )

    retrieved_chunks = [
        chunks[i] for i in indices[0]
    ]

    return "\n\n".join(retrieved_chunks)


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
        "aadhaar number",
        "otp",
        "account number",
        "phone number"
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
            "account numbers or phone numbers."
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
            f"*Educational source:* [{source_name}]({source_url})"
        )

    elif any(word in question_lower for word in comparison_words):

        st.subheader("Answer")
        st.write(
            "I do not predict, rank or compare mutual-fund returns. "
            "Please refer to the official scheme information."
        )

        st.markdown(
            f"*Source:* [{source_name}]({source_url})"
        )

    elif not api_key:

        st.error(
            "Groq API key is missing. "
            "Add GROQ_API_KEY in Streamlit secrets."
        )

    else:

        context = retrieve_context(
            user_question
        )

        client = Groq(
            api_key=api_key
        )

        prompt = f"""
You are a facts-only assistant for Groww mutual fund FAQs.

Answer only from the provided context.

Rules:
- Give factual educational information only.
- Do not give investment advice.
- Do not recommend buying or selling funds.
- Do not predict or guarantee returns.
- Keep the answer to a maximum of 3 sentences.
- If the answer is unavailable, say:
"I could not find this information in the Groww FAQ knowledge base."

Context:
{context}

User Question:
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
            "Last updated from sources: 30 August 2026"
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
