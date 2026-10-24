"""
RAG Pipeline using Gemini + ChromaDB for deeplearning.pdf
=========================================================
This script implements a full Retrieval-Augmented Generation (RAG) pipeline:
  1. Loads deeplearning.pdf from the project root
  2. Splits it into overlapping chunks suitable for large documents
  3. Embeds each chunk using Google's Gemini embedding model
  4. Persists the vector store to disk (one-time cost)
  5. Builds a retrieval chain and answers questions about the document
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ── Environment ──────────────────────────────────────────────────────────────
load_dotenv()

# Resolve paths relative to this file so the script works from any cwd
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # …/langchain_models/
PDF_PATH = PROJECT_ROOT / "deeplearning.pdf"
CHROMA_DIR = PROJECT_ROOT / "gemini_chroma_db"

# ── 1. Models ─────────────────────────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
)

# Stay consistent with the embedding model used elsewhere in this project
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

# ── 2. Load PDF ───────────────────────────────────────────────────────────────
print(f"📄 Loading: {PDF_PATH}")
loader = PyPDFLoader(str(PDF_PATH))
documents = loader.load()
print(f"   Loaded {len(documents)} page(s).")

# ── 3. Chunk the document ─────────────────────────────────────────────────────
# For a large PDF, 800-char chunks with 10 % overlap gives good retrieval
# precision without fragmenting sentences too aggressively.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,       # characters per chunk  (~250 tokens)
    chunk_overlap=150,     # ~15 % overlap to preserve cross-boundary context
    length_function=len,
    separators=["\n\n", "\n", " ", ""],
)
chunks = text_splitter.split_documents(documents)
print(f"   Split into {len(chunks)} chunk(s).")

# Optional: sanity-check the first chunk
print("\n── Sample chunk (first) ──────────────────────────────────────────────")
print(chunks[0].page_content[:400], "…")
print("──────────────────────────────────────────────────────────────────────\n")

# ── 4. Vector store (embed once, reuse on subsequent runs) ────────────────────
if CHROMA_DIR.exists():
    # Load the already-persisted store — no API calls needed for embeddings
    print(f"♻️  Loading existing ChromaDB from: {CHROMA_DIR}")
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )
else:
    # First run: embed all chunks (will make many Gemini Embedding API calls)
    print(f"🔄 Embedding {len(chunks)} chunks and saving to: {CHROMA_DIR}")
    print("   This may take a while for a large document — runs only once.")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    print("✅ Vector store created and persisted.")

# ── 5. Retriever ──────────────────────────────────────────────────────────────
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5},  # fetch 5 most relevant chunks per query
)

# ── 6. RAG Chain ──────────────────────────────────────────────────────────────
def format_docs(docs):
    """Join retrieved chunks into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


prompt = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Answer the question using ONLY the context
provided below. If the answer is not contained in the context, say
"I don't have enough information in the document to answer that."

Context:
{context}

Question: {question}

Answer:"""
)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ── 7. Query ──────────────────────────────────────────────────────────────────
QUESTIONS = [
    "What is the main topic of this document?",
    "What are the key concepts covered in deeplearning?",
    "Summarize the most important ideas from the document.",
]

for question in QUESTIONS:
    print(f"\n❓ Question: {question}")
    answer = rag_chain.invoke(question)
    print(f"💬 Answer:\n{answer}")
    print("─" * 70)
