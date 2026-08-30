import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq

st.set_page_config(page_title="Groww Mutual Fund RAG", page_icon="📈")

st.title("📈 Groww Mutual Fund RAG Chatbot")
st.write("Ask questions about mutual funds and Groww.")

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

@st.cache_data
def load_knowledge_base():
    with open("groww_faq.txt", "r", encoding="utf-8") as file:
        text = file.read()

    chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
    return chunks

chunks = load_knowledge_base()

@st.cache_resource
def create_vector_index(chunks):
    embeddings = model.encode(chunks)
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, embeddings

index, embeddings = create_vector_index(chunks)

def retrieve_context(question, top_k=3):
    question_embedding = model.encode([question])
    question_embedding = np.array(question_embedding).astype("float32")

    distances, indices = index.search(question_embedding, top_k)

    retrieved_chunks = [chunks[i] for i in indices[0]]
    return "\n\n".join(retrieved_chunks)

api_key = st.secrets.get("GROQ_API_KEY", "")

user_question = st.text_input("Ask your question")

if user_question:
    context = retrieve_context(user_question)

    if not api_key:
        st.error("Groq API key is missing. Add GROQ_API_KEY in Streamlit secrets.")
    else:
        client = Groq(api_key=api_key)

        prompt = f"""
You are a helpful assistant for Groww mutual fund FAQs.

Answer the user's question only using the context below.
If the answer is not available in the context, say:
"I could not find this information in the Groww FAQ knowledge base."

Context:
{context}

User Question:
{user_question}

Answer clearly and simply.
"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        answer = response.choices[0].message.content

        st.subheader("Answer")
        st.write(answer)

        with st.expander("Retrieved RAG Context"):
            st.write(context)
