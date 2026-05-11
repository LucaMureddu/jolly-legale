# =============================================
# IL JOLLY LEGALE - Entry Point (UI Only)
# =============================================

import io
import streamlit as st
from PyPDF2 import PdfReader, PdfWriter
from rag import ingest_pdf, ask_document
from agents import run_analysis_pipeline
from report_generator import markdown_to_pdf_bytes
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Il Jolly Legale",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

if not os.getenv("GROQ_API_KEY"):
    st.error(
        "**Chiave API Groq mancante.** "
        "Crea un file `.env` nella cartella del progetto con `GROQ_API_KEY=gsk_...`. "
        "Ottieni la tua chiave gratuita su [console.groq.com](https://console.groq.com)."
    )
    st.stop()

# -------------------------------------------------------
# CSS — Tema Corporate Dark Navy / Oro
# -------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;600;700&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --navy:      #0d1b2a;
    --navy-mid:  #112236;
    --navy-soft: #1a2f45;
    --gold:      #c9a84c;
    --gold-light:#e8c96a;
    --cream:     #f5f0e8;
    --white:     #ffffff;
    --gray-400:  #9aa5b4;
    --gray-600:  #6b7280;
    --success:   #4caf7d;
}

.stApp {
    background-color: var(--navy);
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stSidebar"] {
    background-color: var(--navy-mid) !important;
    border-right: 1px solid rgba(201,168,76,0.2);
}
[data-testid="stSidebar"] * { color: var(--cream) !important; }

/* Header trasparente — il bottone sidebar rimane visibile */
#MainMenu, footer { visibility: hidden; }
[data-testid="stHeader"] { background: transparent !important; }

.jl-hero {
    padding: 2.5rem 0 1.5rem 0;
    border-bottom: 1px solid rgba(201,168,76,0.25);
    margin-bottom: 2rem;
}
.jl-logo {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.8rem;
    font-weight: 700;
    color: var(--white);
    letter-spacing: -0.02em;
    line-height: 1;
}
.jl-logo span { color: var(--gold); }
.jl-tagline {
    font-size: 0.85rem;
    color: var(--gray-400);
    font-weight: 300;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 0.5rem;
}
.jl-step {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--gold);
    border: 1px solid rgba(201,168,76,0.35);
    padding: 0.25rem 0.75rem;
    border-radius: 2px;
    margin-bottom: 0.75rem;
}
.jl-card {
    background: var(--navy-mid);
    border: 1px solid rgba(201,168,76,0.15);
    border-radius: 4px;
    padding: 1.5rem;
    margin-bottom: 1.25rem;
}
.jl-guardrail {
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    padding: 0.75rem 1rem;
    background: rgba(201,168,76,0.06);
    border-left: 3px solid var(--gold);
    border-radius: 0 3px 3px 0;
    margin: 0.5rem 0;
    font-size: 0.82rem;
    color: var(--cream);
}
.jl-agent {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.9rem 1rem;
    background: var(--navy-soft);
    border-radius: 3px;
    margin-bottom: 0.5rem;
    border: 1px solid rgba(255,255,255,0.05);
}
.jl-agent-num {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--gold);
    width: 1.8rem;
    flex-shrink: 0;
}
.jl-agent-name { font-size: 0.88rem; font-weight: 500; color: var(--white); }
.jl-agent-desc { font-size: 0.75rem; color: var(--gray-400); margin-top: 0.1rem; }

.jl-metrics {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.75rem;
    margin: 1rem 0;
}
.jl-metric {
    background: var(--navy-soft);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 3px;
    padding: 0.75rem 1rem;
    text-align: center;
}
.jl-metric-val {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--gold);
    line-height: 1;
}
.jl-metric-label {
    font-size: 0.7rem;
    color: var(--gray-400);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.3rem;
}

[data-testid="stFileUploader"] {
    background: var(--navy-soft) !important;
    border: 1.5px dashed rgba(201,168,76,0.3) !important;
    border-radius: 4px !important;
}

.stButton > button {
    background: linear-gradient(135deg, var(--gold) 0%, #b8922a 100%) !important;
    color: var(--navy) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 2px !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

[data-testid="stDownloadButton"] > button {
    background: transparent !important;
    color: var(--gold) !important;
    border: 1.5px solid var(--gold) !important;
    border-radius: 2px !important;
}

.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--gold) 0%, var(--gold-light) 100%) !important;
}

/* Chat Q&A */
.jl-chat-hint {
    font-size: 0.78rem;
    color: var(--gray-400);
    margin-bottom: 1rem;
    padding: 0.6rem 0.9rem;
    background: rgba(201,168,76,0.06);
    border-left: 3px solid var(--gold);
    border-radius: 0 3px 3px 0;
}

[data-testid="stChatMessage"] {
    background: var(--navy-soft) !important;
    border-radius: 4px !important;
    margin-bottom: 0.5rem !important;
}
[data-testid="stChatInput"] {
    background: var(--navy-soft) !important;
    border: 1px solid rgba(201,168,76,0.3) !important;
    border-radius: 3px !important;
}

p, li, label { color: var(--cream) !important; }
h1, h2, h3   { color: var(--white) !important; }
hr { border-color: rgba(201,168,76,0.2) !important; }

.jl-report-preview {
    background: var(--navy-soft);
    border: 1px solid rgba(201,168,76,0.15);
    border-radius: 4px;
    padding: 2rem;
    color: var(--cream);
    line-height: 1.75;
    max-height: 600px;
    overflow-y: auto;
}
.jl-report-preview h1 {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.6rem;
    color: var(--white) !important;
    border-bottom: 2px solid var(--gold);
    padding-bottom: 0.5rem;
    margin-bottom: 1.25rem;
}
.jl-report-preview h2 {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.2rem;
    color: var(--gold) !important;
    margin-top: 1.5rem;
    margin-bottom: 0.5rem;
}
.jl-report-preview table { width: 100%; border-collapse: collapse; font-size: 0.85rem; margin: 1rem 0; }
.jl-report-preview th {
    background: var(--navy);
    color: var(--gold) !important;
    padding: 0.5rem 0.75rem;
    text-align: left;
    font-size: 0.78rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.jl-report-preview td {
    padding: 0.5rem 0.75rem;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    color: var(--cream) !important;
}
.jl-report-preview tr:nth-child(even) td { background: rgba(255,255,255,0.03); }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------
# COSTANTI
# -------------------------------------------------------
PAGE_LIMIT = int(os.getenv("PAGE_LIMIT", 30))


# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------
def render_sidebar(lingua: str) -> str:
    with st.sidebar:
        st.markdown("""
        <div style="padding:0.5rem 0 1.5rem 0; border-bottom:1px solid rgba(201,168,76,0.2); margin-bottom:1.5rem;">
            <div style="font-family:'Cormorant Garamond',serif; font-size:1.3rem; font-weight:700; color:#ffffff;">
                ⚖️ Il Jolly Legale
            </div>
            <div style="font-size:0.7rem; color:#9aa5b4; letter-spacing:0.1em; text-transform:uppercase; margin-top:0.25rem;">
                Legal AI Assistant
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**🌐 Lingua del Report**")
        lingua = st.selectbox(
            label="lingua",
            options=["Italiano", "English"],
            index=0 if lingua == "Italiano" else 1,
            label_visibility="collapsed",
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**🤖 Architettura Multi-Agente**")
        st.markdown('<div style="font-size:0.78rem;color:#9aa5b4;margin-bottom:0.75rem;">Pipeline LangGraph — 3 agenti + Q&A</div>', unsafe_allow_html=True)

        agenti = [
            ("01",  "Reader",   "ChromaDB — 12 query tematiche"),
            ("02",  "Reviewer", "Groq llama3-70b — analisi bilanciata"),
            ("03",  "Reporter", "Formattazione report Markdown"),
            ("Q&A", "Chat",     "RAG interattivo sul documento"),
        ]
        for num, nome, desc in agenti:
            st.markdown(f"""
            <div class="jl-agent">
                <div class="jl-agent-num">{num}</div>
                <div>
                    <div class="jl-agent-name">{nome}</div>
                    <div class="jl-agent-desc">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**🛡️ Guardrail Zero Trust**")
        guardrail = [
            ("🔒", "PDF in memoria — mai su disco"),
            ("🔑", "API keys solo in .env"),
            ("🚫", "ChromaDB ephemeral"),
            ("🧹", "Clean Architecture"),
            ("⚡", "LLM: Groq (gratuito)"),
            ("🤗", "Embeddings: HuggingFace (locale)"),
        ]
        for icon, testo in guardrail:
            st.markdown(f'<div class="jl-guardrail"><span>{icon}</span><span>{testo}</span></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**⚙️ Tech Stack**")
        stack = {
            "LLM":          "Groq llama3-70b",
            "Embeddings":   "multilingual-MiniLM",
            "Orchestrazione": "LangGraph",
            "Vector DB":    "ChromaDB",
            "UI":           "Streamlit",
            "PDF":          "fpdf2",
        }
        for k, v in stack.items():
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;padding:0.3rem 0;
                        border-bottom:1px solid rgba(255,255,255,0.05);font-size:0.78rem;">
                <span style="color:#9aa5b4;">{k}</span>
                <span style="color:#c9a84c;font-weight:500;">{v}</span>
            </div>
            """, unsafe_allow_html=True)

    return lingua


# -------------------------------------------------------
# HELPER: Card agente con stato
# -------------------------------------------------------
def agent_card(col, num: str, nome: str, desc: str, stato: str) -> None:
    color = {"wait": "#6b7280", "run": "#c9a84c", "done": "#4caf7d"}[stato]
    icon  = {"wait": "○", "run": "◉", "done": "✓"}[stato]
    with col:
        st.markdown(f"""
        <div style="background:#1a2f45; border:1px solid {color}40;
                    border-top:2px solid {color}; border-radius:3px;
                    padding:1rem; text-align:center;">
            <div style="font-size:1.2rem; color:{color};">{icon}</div>
            <div style="font-family:'Cormorant Garamond',serif; font-size:0.95rem;
                        color:#ffffff; font-weight:600; margin:0.3rem 0;">
                {num} · {nome}
            </div>
            <div style="font-size:0.72rem; color:#9aa5b4;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)


# -------------------------------------------------------
# HELPER: Controllo e troncamento pagine PDF
# -------------------------------------------------------
def _check_and_truncate_pdf(uploaded_file) -> tuple[bytes, int, bool]:
    """
    Conta le pagine del PDF e lo tronca se supera PAGE_LIMIT.

    Returns:
        (pdf_bytes, num_pages, was_truncated)
    """
    raw_bytes = uploaded_file.getvalue()
    reader    = PdfReader(io.BytesIO(raw_bytes))
    num_pages = len(reader.pages)

    if num_pages <= PAGE_LIMIT:
        return raw_bytes, num_pages, False

    # Tronca alle prime PAGE_LIMIT pagine
    writer = PdfWriter()
    for i in range(PAGE_LIMIT):
        writer.add_page(reader.pages[i])
    buffer = io.BytesIO()
    writer.write(buffer)
    buffer.seek(0)
    return buffer.read(), num_pages, True


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
def main() -> None:
    # Inizializzazione session state
    if "lingua"      not in st.session_state: st.session_state.lingua      = "Italiano"
    if "messages"    not in st.session_state: st.session_state.messages    = []
    if "vectorstore" not in st.session_state: st.session_state.vectorstore = None
    if "report_md"   not in st.session_state: st.session_state.report_md   = None
    if "report_pdf"  not in st.session_state: st.session_state.report_pdf  = None
    if "filename"    not in st.session_state: st.session_state.filename    = None
    if "pdf_bytes"   not in st.session_state: st.session_state.pdf_bytes   = None

    lingua = render_sidebar(st.session_state.lingua)
    if lingua != st.session_state.lingua:
        st.session_state.lingua = lingua

    # Hero
    st.markdown("""
    <div class="jl-hero">
        <div class="jl-logo">Il Jolly <span>Legale</span></div>
        <div class="jl-tagline">AI-Powered Contract Analysis · Enterprise Grade · Zero Trust · Powered by Groq</div>
    </div>
    """, unsafe_allow_html=True)

    # ── STEP 01: Upload ──────────────────────────────────
    st.markdown('<div class="jl-step">◆ Step 01 — Documento</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        label="Carica il contratto PDF",
        type=["pdf"],
        help="Elaborato interamente in memoria. Nessun dato salvato su disco.",
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        st.markdown(f"""
        <div class="jl-card" style="text-align:center; padding:2.5rem; color:#9aa5b4;">
            <div style="font-size:2rem; margin-bottom:0.75rem;">📄</div>
            <div style="font-size:0.9rem;">Carica un contratto PDF per iniziare l'analisi</div>
            <div style="font-size:0.75rem; margin-top:0.5rem; color:#6b7280;">
                Supporta contratti fino a {PAGE_LIMIT} pagine · Analisi in ~30 secondi con Groq
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # --- Controllo e troncamento pagine ---
    pdf_bytes, num_pages, was_truncated = _check_and_truncate_pdf(uploaded_file)
    st.session_state.pdf_bytes = pdf_bytes

    if was_truncated:
        st.warning(
            f"⚠️ **Documento troppo lungo ({num_pages} pagine).** "
            f"Questa demo gratuita supporta fino a **{PAGE_LIMIT} pagine**. "
            f"Il documento è stato troncato automaticamente per garantire stabilità "
            f"e rispettare i limiti del free tier Groq. "
            f"Per contratti più lunghi contattaci per l'infrastruttura **Premium** ☎️"
        )
    else:
        st.success(f"✓ **{uploaded_file.name}** · {num_pages} pagine · caricato in memoria")

    size_kb = len(pdf_bytes) / 1024
    st.markdown(f"""
    <div class="jl-metrics">
        <div class="jl-metric">
            <div class="jl-metric-val">{num_pages if not was_truncated else PAGE_LIMIT}</div>
            <div class="jl-metric-label">Pagine</div>
        </div>
        <div class="jl-metric">
            <div class="jl-metric-val">{lingua[:2].upper()}</div>
            <div class="jl-metric-label">Lingua</div>
        </div>
        <div class="jl-metric">
            <div class="jl-metric-val">Groq</div>
            <div class="jl-metric-label">Engine</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── STEP 02: Analisi ─────────────────────────────────
    st.markdown('<div class="jl-step" style="margin-top:1.5rem;">◆ Step 02 — Analisi</div>', unsafe_allow_html=True)

    label_btn = "Avvia Analisi" if lingua == "Italiano" else "Start Analysis"
    if st.button(f"⚡ {label_btn}", type="primary", use_container_width=True):

        # Reset stato per nuova analisi
        st.session_state.messages    = []
        st.session_state.vectorstore = None
        st.session_state.report_md   = None
        st.session_state.report_pdf  = None
        st.session_state.filename    = uploaded_file.name

        progress = st.progress(0)
        col1, col2, col3 = st.columns(3)

        try:
            agent_card(col1, "01", "Reader",   "Indicizzazione vettoriale", "run")
            agent_card(col2, "02", "Reviewer", "In attesa",                 "wait")
            agent_card(col3, "03", "Reporter", "In attesa",                 "wait")
            progress.progress(10)

            vectorstore, _ = ingest_pdf(st.session_state.pdf_bytes)
            st.session_state.vectorstore = vectorstore
            progress.progress(35)

            agent_card(col1, "01", "Reader",   "Completato",       "done")
            agent_card(col2, "02", "Reviewer", "Analisi clausole", "run")
            agent_card(col3, "03", "Reporter", "In attesa",        "wait")
            progress.progress(50)

            final_report_md = run_analysis_pipeline(vectorstore, lingua=lingua)
            st.session_state.report_md = final_report_md
            progress.progress(80)

            agent_card(col1, "01", "Reader",   "Completato",    "done")
            agent_card(col2, "02", "Reviewer", "Completato",    "done")
            agent_card(col3, "03", "Reporter", "Genera PDF...", "run")

            pdf_bytes_out = markdown_to_pdf_bytes(final_report_md)
            st.session_state.report_pdf = pdf_bytes_out
            progress.progress(100)

            agent_card(col1, "01", "Reader",   "Completato", "done")
            agent_card(col2, "02", "Reviewer", "Completato", "done")
            agent_card(col3, "03", "Reporter", "Completato", "done")

        except ValueError as e:
            st.error(f"**Errore nel documento:** {str(e)}")
            return
        except Exception as e:
            st.error(f"**Errore durante l'analisi:** {str(e)}")
            return

    # ── STEP 03: Report ──────────────────────────────────
    if st.session_state.report_md:
        st.markdown('<div class="jl-step" style="margin-top:2rem;">◆ Step 03 — Report</div>', unsafe_allow_html=True)

        with st.expander("📋 Anteprima Report", expanded=True):
            import markdown as md_lib
            html_preview = md_lib.markdown(
                st.session_state.report_md,
                extensions=["tables", "fenced_code", "nl2br", "sane_lists"],
            )
            st.markdown(f'<div class="jl-report-preview">{html_preview}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        nome_file = (st.session_state.filename or "contratto").replace(".pdf", "")
        st.download_button(
            label="⬇  Scarica Report PDF",
            data=st.session_state.report_pdf,
            file_name=f"analisi_{nome_file}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    # ── STEP 04: Q&A Interattivo ─────────────────────────
    if st.session_state.vectorstore is not None:
        st.markdown('<div class="jl-step" style="margin-top:2rem;">◆ Step 04 — Q&A Interattivo</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="jl-chat-hint">
            Fai domande dirette al contratto. L'assistente risponde basandosi
            esclusivamente sul testo del documento caricato.
        </div>
        """, unsafe_allow_html=True)

        # Mostra cronologia chat
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Input utente
        placeholder = (
            "Es: Qual è la penale per ritardo nel pagamento?"
            if lingua == "Italiano"
            else "E.g.: What is the penalty for late payment?"
        )

        if user_input := st.chat_input(placeholder):
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Consultando il contratto..."):
                    try:
                        answer = ask_document(
                            query=user_input,
                            vectorstore=st.session_state.vectorstore,
                            lingua=lingua,
                            chat_history=st.session_state.messages[:-1],
                        )
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        err = f"Errore nella risposta: {str(e)}"
                        st.error(err)
                        st.session_state.messages.append({"role": "assistant", "content": err})

        if st.session_state.messages:
            if st.button("🗑 Cancella cronologia chat", use_container_width=False):
                st.session_state.messages = []
                st.rerun()

    st.markdown("""
    <div style="text-align:center; margin-top:2rem; font-size:0.75rem; color:#6b7280;">
        🔒 Sessione Zero Trust · Documento in memoria · Nessun dato persistente
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()