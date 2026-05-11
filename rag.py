# =============================================
# MODULO A: Ingestione PDF, RAG Pipeline e Q&A
# =============================================
# Responsabilità:
#   - Parsing del PDF (in memoria, mai su disco)
#   - Chunking strategico con RecursiveCharacterTextSplitter
#   - Embeddings locali con HuggingFace (zero costo, zero API)
#   - Salvataggio temporaneo in ChromaDB (ephemeral)
#   - Q&A interattivo sul documento via RetrievalQA
# =============================================

import io
import os

import pdfplumber
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv

from llm_utils import _invoke_with_backoff

load_dotenv()

# --- Configurazione da .env ---
CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP", 200))
GROQ_API_KEY    = os.getenv("GROQ_API_KEY")
LLM_MODEL       = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

# Embeddings locali — scaricati una volta, poi cached localmente

import streamlit as st

@st.cache_resource
def _get_embeddings() -> HuggingFaceEmbeddings:
    """
    Carica il modello HuggingFace una sola volta per l'intera
    durata del processo Streamlit — persiste tra sessioni e
    ricaricamenti della pagina. Zero overhead dopo il primo avvio.
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# -------------------------------------------------------
# FUNZIONE INTERNA: Estrazione testo dal PDF (in memoria)
# -------------------------------------------------------

def _extract_text_from_bytes(pdf_bytes: bytes) -> str:
    """
    Estrae il testo dal PDF ricevuto come bytes.
    Strategia a due livelli:
      1. pdfplumber (più preciso per layout complessi e tabelle)
      2. PyPDF2 come fallback automatico
    Il file non viene mai scritto su disco.
    """
    text = ""

    # Tentativo 1 — pdfplumber
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            pages_text = [
                page.extract_text()
                for page in pdf.pages
                if page.extract_text()
            ]
            text = "\n\n".join(pages_text)
    except Exception:
        text = ""

    # Tentativo 2 — PyPDF2 (fallback)
    if not text.strip():
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            pages_text = [
                page.extract_text()
                for page in reader.pages
                if page.extract_text()
            ]
            text = "\n\n".join(pages_text)
        except Exception as e:
            raise ValueError(
                "Impossibile estrarre testo dal PDF. "
                "Il file potrebbe essere una scansione senza OCR oppure danneggiato."
            ) from e

    if not text.strip():
        raise ValueError(
            "Il PDF non contiene testo estraibile. "
            "Potrebbe essere una scansione senza OCR. "
            "Carica un PDF con testo selezionabile."
        )

    return text


# -------------------------------------------------------
# FUNZIONE PUBBLICA: Ingestione completa del PDF
# -------------------------------------------------------

def ingest_pdf(pdf_bytes: bytes) -> tuple:
    """
    Pipeline completa di ingestione:
      1. Estrae il testo dal PDF (in memoria).
      2. Applica chunking strategico.
      3. Genera embeddings HuggingFace (locali) e crea ChromaDB ephemeral.

    Args:
        pdf_bytes: Il contenuto del PDF come bytes (da st.file_uploader).

    Returns:
        (vectorstore, collection_id): Il vector store e un ID univoco di sessione.

    Raises:
        ValueError: Se il PDF non è leggibile o non contiene testo.
        Exception:  Se il modello HuggingFace non è scaricabile.
    """
    import uuid

    raw_text = _extract_text_from_bytes(pdf_bytes)

    splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ".", " "],
)
    chunks = splitter.split_text(raw_text)

    if not chunks:
        raise ValueError("Nessun chunk generato dal documento. Controlla il contenuto del PDF.")

    embeddings    = _get_embeddings()
    collection_id = str(uuid.uuid4())

    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        collection_name=collection_id,
        # Nessun persist_directory = ephemeral, zero file su disco
    )

    return vectorstore, collection_id


# -------------------------------------------------------
# FUNZIONE PUBBLICA: Query al Vector Store
# -------------------------------------------------------

def query_vector_db(vectorstore: Chroma, query: str, k: int = 6) -> list:
    """
    Interroga il Vector Store e restituisce i chunk più rilevanti.

    Args:
        vectorstore: Il vector store ChromaDB della sessione corrente.
        query:       La domanda o il tema da ricercare.
        k:           Numero di chunk da restituire (default: 6).

    Returns:
        Lista di stringhe con i chunk più rilevanti.
    """
    results = vectorstore.similarity_search(query=query, k=k)
    return [doc.page_content for doc in results]


# -------------------------------------------------------
# FUNZIONE PUBBLICA: Q&A Interattivo sul Documento
# -------------------------------------------------------

QA_PROMPT_IT = ChatPromptTemplate.from_messages([
    ("system", """Sei un assistente legale esperto. Rispondi alle domande
dell'utente basandoti ESCLUSIVAMENTE sul testo del contratto fornito come contesto.

Regole:
- Se la risposta non è nel contratto, dillo esplicitamente: "Questa informazione non è presente nel contratto."
- Cita sempre la parte del contratto da cui hai tratto la risposta.
- Sii preciso, conciso e professionale.
- Rispondi in italiano.

Contesto dal contratto:
{context}"""),
    ("human", "{question}"),
])

QA_PROMPT_EN = ChatPromptTemplate.from_messages([
    ("system", """You are an expert legal assistant. Answer the user's questions
based EXCLUSIVELY on the contract text provided as context.

Rules:
- If the answer is not in the contract, state it explicitly: "This information is not present in the contract."
- Always cite the part of the contract from which you drew the answer.
- Be precise, concise and professional.
- Answer in English.

Context from the contract:
{context}"""),
    ("human", "{question}"),
])


def ask_document(query: str, vectorstore, lingua: str = "Italiano",
                 chat_history: list[dict] | None = None) -> str:
    """
    Risponde a una domanda sul documento usando RAG con memoria conversazionale.
    La chat_history viene iniettata nel prompt per dare all'LLM il contesto
    dei turni precedenti — elimina l'amnesia tra domande consecutive.

    Args:
        query:        La domanda dell'utente in linguaggio naturale.
        vectorstore:  Il ChromaDB ephemeral della sessione corrente.
        lingua:       "Italiano" o "English" (default: "Italiano").
        chat_history: Lista di dict {"role": "user"|"assistant", "content": "..."}

    Returns:
        La risposta del LLM basata sul contratto e sulla conversazione.
    """
    # Recupera i chunk più rilevanti
    relevant_chunks = query_vector_db(vectorstore, query, k=6)
    context = "\n\n---\n\n".join(relevant_chunks)

    # Formatta la chat history come stringa leggibile dall'LLM
    history_text = ""
    if chat_history:
        turns = []
        for msg in chat_history:
            role  = "Utente" if msg["role"] == "user" else "Assistente"
            turns.append(f"{role}: {msg['content']}")
        history_text = "\n".join(turns)

    # Prompt con memoria — sezione history opzionale
    if lingua == "English":
        system = """You are an expert legal assistant with memory of the conversation.
Answer questions based EXCLUSIVELY on the contract context provided.

Rules:
- Use conversation history to understand follow-up questions and references.
- If the answer is not in the contract, say: "This information is not present in the contract."
- Always cite the relevant part of the contract.
- Be precise, concise and professional.

Contract context:
{context}

Conversation history:
{history}"""
    else:
        system = """Sei un assistente legale esperto con memoria della conversazione.
Rispondi alle domande basandoti ESCLUSIVAMENTE sul testo del contratto fornito.

Regole:
- Usa la cronologia della conversazione per capire domande di follow-up e riferimenti.
- Se la risposta non e' nel contratto, dillo: "Questa informazione non e' presente nel contratto."
- Cita sempre la parte del contratto da cui hai tratto la risposta.
- Sii preciso, conciso e professionale.

Contesto dal contratto:
{context}

Cronologia conversazione:
{history}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "{question}"),
    ])

    llm = ChatGroq(
        model_name=LLM_MODEL,
        temperature=0,
        api_key=GROQ_API_KEY,
    )

    chain    = prompt | llm
    response = _invoke_with_backoff(chain, {
    "context":  context,
    "history":  history_text if history_text else "Nessuna conversazione precedente.",
    "question": query,})
    return response.content