<<<<<<< HEAD
# 📚 LangChain-Powered RAG Conversational AI Assistant

An intelligent Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDF documents, generate AI-powered summaries, and ask context-aware questions using semantic search and local LLM inference.

Built with **LangChain**, **FAISS**, **Sentence Transformers**, **Groq API**, **Llama 3.3 70B**, and **Streamlit**.

---
## 🌐 Live Demo

🚀 **Live App:** https://langchain-rag-pdf-assistant.streamlit.app

---

## 🚀 Features

✅ Upload and process PDF documents

✅ Semantic document search using vector embeddings

✅ Context-aware question answering

✅ Conversational memory for follow-up questions

✅ AI-generated document summaries

✅ Source-backed responses

✅  Ultra-fast AI responses using Groq Llama 3.3

✅ Interactive Streamlit UI

---

## 🏗️ System Architecture

```text
                    ┌──────────────┐
                    │   PDF File   │
                    └──────┬───────┘
                           │
                           ▼
                ┌────────────────────┐
                │ Text Extraction    │
                │ (PyPDF)            │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ Text Chunking      │
                │ LangChain Splitter │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ MiniLM Embeddings  │
                │ HuggingFace        │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ FAISS Vector Store │
                │ (LangChain)        │
                └─────────┬──────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ LangChain       │
                 │ Retriever       │
                 └───────┬─────────┘
                         │
         User Query      ▼
      ─────────────► Relevant Chunks
                         │
                         ▼
                ┌────────────────────┐
                │ Ollama (Llama 3.2) │
                └─────────┬──────────┘
                          │
                          ▼
                   Final Response
```

---

## 🧠 How It Works

1. User uploads a PDF document.
2. PDF text is extracted and split into semantic chunks.
3. Chunks are converted into vector embeddings using MiniLM.
4. Embeddings are stored in a FAISS vector database.
5. User asks a question.
6. LangChain Retriever finds the most relevant chunks.
7. Retrieved context is sent to Ollama (Llama 3.2).
8. The model generates a source-grounded answer.

---

## 📊 Performance Observations

| Document Size | Chunks Generated |
| ------------- | ----------------- |
| 9 Pages       | 6 Chunks          |
| 14 Pages      | 23 Chunks         |
| 32 Pages      | 56 Chunks         |

Chunk count scales roughly linearly with document length, confirming the chunking strategy holds up consistently across short and long documents.

### Performance

• Semantic search over FAISS vector store
• Fast response generation via Groq's hosted Llama 3.3 model
• Supports conversational follow-up questions
• Efficient retrieval using MiniLM embeddings
---

## 🛠️ Tech Stack

* Python
* Streamlit
* LangChain
* FAISS
* HuggingFace Embeddings
* Ollama
* Llama 3.2
* PyPDF

---

## 📂 Project Structure

```text
rag-chatbot/
│
├── src/
│   ├── app.py
│   ├── ingest.py
│   ├── retriever.py
│   ├── config.py
│
├── vectorstore/
│
├── requirements.txt
│
└── README.md
```

---

## 📸 Screenshots

### PDF Upload
![PDF Upload](assets/upload.png)

### Question Answering
![Question Answering](assets/chat.png)

### Document Summary
![Document Summary](assets/summary.png)
---

## 🔮 Future Enhancements

* Multi-PDF Support
* Citation with Page Numbers
* ChromaDB / Pinecone Integration
* Advanced Memory Management
* Document Comparison
* Multi-Agent RAG Workflow
* GPU-accelerated / quantized inference for faster response times

---

## 👩‍💻 Author

**Hibza Kudari**

AI & Machine Learning Student

Passionate about Generative AI, RAG Systems, LLM Applications, and Intelligent Document Processing.
=======
# StudyMate-AI
>>>>>>> 7ecc19025141cfa1af5863a5faec06111df40276
