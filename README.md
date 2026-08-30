# Groww Mutual Fund RAG FAQ Chatbot

## Project Overview
This project is a facts-only Mutual Fund FAQ assistant built using RAG (Retrieval-Augmented Generation) and an LLM.

The assistant retrieves relevant information from its mutual-fund knowledge base and uses an LLM to generate a simple factual response.

## Product Chosen
Groww

## Purpose
The goal is to help users quickly find factual information related to mutual funds, such as:

- SIP
- Expense ratio
- Exit load
- ELSS and lock-in
- Risk level / Riskometer
- NAV
- KYC
- Mutual fund redemption
- Mutual fund terminology

The assistant is designed for factual and educational queries only.

## How RAG Works

1. Mutual fund information is stored in a knowledge base.
2. The Sentence Transformer model converts the information into embeddings.
3. FAISS stores and searches these embeddings.
4. When a user asks a question, the most relevant chunks are retrieved.
5. The retrieved context is sent to the LLM.
6. The LLM generates a concise answer based on the retrieved context.

## Technology Used

- Python
- Streamlit
- Sentence Transformers
- FAISS
- Groq LLM API
- RAG (Retrieval-Augmented Generation)

## Public Sources

The project uses public mutual-fund information sources.

The source list is available in:

sources.md

Sources are limited to official/public resources such as AMC, AMFI and SEBI information.

## Safety and Scope

This assistant is intended for facts-only educational information.

It does not provide personalized investment advice or buy/sell recommendations.

Users should not enter personal or sensitive information such as PAN, Aadhaar, OTP, account number, email address or phone number.

## Known Limitations

- The assistant can answer only from information available in its knowledge base.
- It does not provide personalized portfolio recommendations.
- It does not predict or guarantee mutual fund returns.
- Information may need to be updated when official source information changes.

## Project Files

- app.py - Main Streamlit RAG application
- groww_faq.txt - Knowledge base
- requirements.txt - Python dependencies
- sources.md - Public source list
- README.md - Project documentation

## Disclaimer

Facts-only educational assistant. No investment advice. Mutual fund investments are subject to market risks. Verify important information using the linked official sources.
