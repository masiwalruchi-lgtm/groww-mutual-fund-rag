# Groww Mutual Fund RAG FAQ Chatbot

## Project Overview

This project is a facts-only Mutual Fund FAQ assistant built using Retrieval-Augmented Generation (RAG) and an LLM.

The chatbot retrieves relevant information from a mutual fund knowledge base and uses an LLM to generate concise factual answers.

## Product Chosen

Groww Mutual Fund

## Scope

The project focuses on Groww Mutual Fund as the AMC and covers the following four schemes:

1. Groww Large Cap Fund
2. Groww Value Fund
3. Groww ELSS Tax Saver Fund
4. Groww Aggressive Hybrid Fund



The chatbot can answer factual questions about topics such as SIP, NAV, expense ratio, exit load, ELSS lock-in, Riskometer, scheme characteristics and official scheme documents.

## How RAG Works

1. Mutual fund information is stored in groww_faq.txt.
2. Sentence Transformers convert the knowledge-base chunks into embeddings.
3. FAISS creates a vector index of the embeddings.
4. A user's question is converted into an embedding.
5. FAISS retrieves the most relevant knowledge-base chunks.
6. The retrieved context is provided to the LLM.
7. The LLM generates a concise factual answer based on the retrieved context.

## Technology Used

- Python
- Streamlit
- Sentence Transformers
- FAISS
- Groq LLM API
- Retrieval-Augmented Generation (RAG)

## Setup Steps

1. Clone or download this GitHub repository.
2. Install the dependencies from requirements.txt.
3. Add a Groq API key as GROQ_API_KEY in Streamlit secrets.
4. Run the application using streamlit run app.py.
5. Enter a factual mutual-fund question in the chatbot.

## Public Sources

The project uses official/public mutual-fund information.

The complete list of public sources is available in:

sources.md

Sources are limited to official Groww Mutual Fund, AMFI and SEBI resources.

## Safety and Scope

This chatbot is a facts-only educational assistant.

It does not provide personalized investment advice, buy/sell recommendations, fund rankings or predictions of future returns.

Users should not enter sensitive information such as PAN, Aadhaar, OTP, account number, email address or phone number.

## Known Limitations

- The chatbot answers using information available in its knowledge base.
- It does not provide personalized portfolio recommendations.
- It does not predict or guarantee mutual fund returns.
- Mutual fund information may change and should be verified using the linked official sources.
- The knowledge base needs periodic updates when official scheme information changes.

## Project Files

- app.py - Main Streamlit RAG application
- groww_faq.txt - RAG knowledge base
- requirements.txt - Python dependencies
- sources.md - Official public source list
- sample_qa.md - Sample questions and answers
- README.md - Project documentation

## Disclaimer

Facts-only educational assistant. No investment advice. Mutual fund investments are subject to market risks. Verify important information using the linked official sources.
