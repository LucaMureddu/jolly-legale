# 🏛️ PROJECT BLUEPRINT: Assistente B2B Analisi Contratti ("Il Jolly Legale")

## 1. VISIONE E OBIETTIVO DI BUSINESS
Sviluppare un sistema RAG (Retrieval-Augmented Generation) Multi-Agente di livello Enterprise. L'obiettivo è automatizzare e velocizzare l'analisi di contratti legali lunghi (50+ pagine) per studi legali e società di consulenza (Big 4), riducendo i tempi di lettura e identificando chirurgicamente clausole nascoste e rischi legali. L'output finale è un report PDF direzionale.

## 2. TECH STACK AUTORIZZATO (Blueprint Originale)
Nessuna deviazione da questo stack senza esplicita autorizzazione del CTO:
- **Linguaggio:** Python 3.10+
- **Frontend / UI:** Streamlit
- **Orchestrazione AI:** LangChain (e LangGraph per workflow multi-agente)
- **Vector Database:** ChromaDB (locale/ephemeral per MVP, pronto per scaling)
- **LLM Provider:** OpenAI API (GPT-4o per ragionamento, text-embedding-3 per i vettori)
- **Parsing PDF:** `pdfplumber` o `PyPDF2` (con fallback se necessario)
- **Generazione Report:** libreria Python per Markdown-to-PDF (es. `md2pdf` o `WeasyPrint`)

## 3. ARCHITETTURA DI SISTEMA (3 Moduli Principali)

### Modulo A: Ingestione e RAG Pipeline (Sicura)
- L'utente carica il PDF tramite Streamlit.
- **Chunking Strategico:** Il testo estratto non deve essere tagliato brutalmente. Obbligatorio usare `RecursiveCharacterTextSplitter` con overlap intelligente (es. chunk 1000, overlap 200) per preservare il contesto legale.
- Generazione embeddings e salvataggio temporaneo in ChromaDB.

### Modulo B: Orchestrazione Multi-Agente (Il Motore Logico)
Il sistema deve implementare 3 Agenti specializzati (tramite LangChain/LangGraph):
1. **Agente Ricercatore (Reader):** Interroga il Vector DB estraendo il contesto più rilevante.
2. **Agente Revisore Legale (Reviewer):** Analizza il contesto con un prompt altamente specializzato per individuare: Clausole penali, Rischi di rescissione, Scadenze critiche, Asimmetrie legali.
3. **Agente Formattatore (Reporter):** Prende gli insight grezzi e li formatta in un report Markdown professionale, pronto per essere convertito in PDF.

### Modulo C: Interfaccia Utente (Streamlit "Premium")
- **Asincronia e Feedback:** L'elaborazione LLM richiede tempo. L'UI DEVE mostrare spinner, barre di progresso (`st.progress`) e messaggi di stato (es. "Agente Revisore all'opera..."). Nessun blocco silenzioso dell'interfaccia.
- Download finale del report in PDF.

## 4. REQUISITI NON NEGOZIABILI (GUARDRAILS)
1. **Sicurezza "Zero Trust":** I contratti sono strettamente confidenziali. I file PDF caricati NON DEVONO mai essere salvati permanentemente sul disco. Devono essere elaborati in memoria o in cartelle temporanee eliminate immediatamente dopo la sessione.
2. **Gestione Segreti:** Tutte le chiavi API devono risiedere in un file `.env`. Il file `.env` deve essere inserito nel `.gitignore` dal primo commit.
3. **Robustezza Server-Side:** Gestione elegante delle eccezioni (try/except). Se le API di OpenAI vanno in timeout o il PDF è illeggibile (es. scansione senza OCR), mostrare all'utente un messaggio formattato in Streamlit, MAI un traceback Python.
4. **Separazione delle Preoccupazioni (Clean Architecture):** Il file dell'interfaccia (es. `app.py`) non deve contenere la logica di orchestrazione LLM. La logica AI va in moduli separati (es. `rag.py`, `agents.py`).

---

## 5. ARCHITECTURAL DECISION LOG (ADL)

> Questa sezione documenta le deviazioni consapevoli dal blueprint originale,
> le motivazioni tecniche e commerciali, e le conseguenze architetturali.
> È la prova che ogni scelta è stata ragionata, non casuale.

---

### ADL-001 — Sostituzione OpenAI con Groq + HuggingFace

**Data:** Fase di sviluppo MVP
**Stato:** Implementato e in produzione

#### Contesto
Il blueprint originale prevedeva OpenAI API per entrambi i layer AI:
- GPT-4o per il ragionamento degli agenti
- text-embedding-3-small per la vettorializzazione

Durante lo sviluppo è emerso un problema strutturale: ogni analisi di un contratto
da 30 pagine generava un costo stimato di ~$0.08-0.15, accettabile in produzione
enterprise ma incompatibile con un MVP dimostrabile a costo zero.

#### Decisione
Pivot verso uno stack ibrido a costo operativo zero:

| Layer | Blueprint | Implementato | Motivazione |
|---|---|---|---|
| LLM | GPT-4o (OpenAI) | llama-3.3-70b-versatile (Groq) | Free tier generoso, latenza sub-secondo grazie al LPU |
| Embeddings | text-embedding-3-small (API) | paraphrase-multilingual-MiniLM-L12-v2 (locale) | Zero API call, privacy totale, supporto 50+ lingue |

#### Conseguenze Positive
- **Costo operativo:** $0.00 per analisi (da ~$0.10)
- **Privacy by architecture:** gli embeddings girano localmente — il testo del contratto non lascia mai la macchina dell'utente per la fase di vettorializzazione
- **Latenza:** Groq LPU riduce il tempo di analisi da ~90s a ~25s
- **Multilingua:** il modello HuggingFace scelto gestisce sinonimi e parafrasi in italiano molto meglio del modello inglese originale (es. "subaffitto" → "sublocazione")

#### Conseguenze Negative / Trade-off
- Prima esecuzione richiede download del modello HuggingFace (~470MB, una-tantum)
- Il free tier Groq ha rate limit (14.400 req/giorno) — mitigato con exponential backoff in `llm_utils.py`
- Qualità del ragionamento leggermente inferiore a GPT-4o su contratti molto complessi — accettabile per MVP

#### Percorso di Upgrade
Lo stack è progettato per scalare senza riscrivere il codice:
- Sostituire `ChatGroq` con `ChatOpenAI` in `agents.py` è una modifica di 2 righe
- ChromaDB supporta persistenza su disco per sessioni multi-documento (rimuovere il commento `persist_directory`)

---

### ADL-002 — Sostituzione WeasyPrint con fpdf2

**Data:** Fase di sviluppo MVP
**Stato:** Implementato e in produzione

#### Contesto
Il blueprint indicava WeasyPrint come libreria per la generazione PDF.
Durante l'installazione su macOS (Apple Silicon) è emersa una dipendenza
da librerie di sistema native (libgobject, Cairo, Pango) non disponibili
senza Homebrew, rendendo il setup complesso e fragile.

#### Decisione
Sostituzione con `fpdf2` — libreria puro Python senza dipendenze di sistema.

#### Conseguenze
- Setup su qualsiasi OS (Mac, Linux, Windows) senza prerequisiti di sistema
- Font Unicode (DejaVu) inclusi manualmente nella cartella `fonts/` per supporto caratteri italiani e simboli speciali
- Implementazione manuale del word-wrap nelle tabelle tramite `multi_cell()` con calcolo dinamico dell'altezza delle righe

---

### ADL-003 — Aggiunta Sezione B al Prompt del Reviewer (Bias Correction)

**Data:** Fase di ottimizzazione
**Stato:** Implementato e in produzione

#### Contesto
Il prompt originale del Reviewer istruiva l'agente a "trovare i rischi".
Durante i test su contratti reali è emerso un bias cognitivo sistematico:
l'agente ignorava le clausole favorevoli al cliente (es. scomputi dal canone,
garanzie ricevute) perché il suo framework mentale era orientato solo al negativo.

#### Decisione
Riscrittura del prompt in due macro-sezioni obbligatorie:
- **Sezione A:** Rischi e clausole critiche (invariata)
- **Sezione B:** Clausole favorevoli e mitigazioni finanziarie (nuova, marcata OBBLIGATORIA)

#### Conseguenze
- Report bilanciato che riflette il reale quadro contrattuale
- Il Formattatore produce due tabelle indice separate (rischi / opportunità)
- Sommario Esecutivo istruito a sintetizzare entrambe le dimensioni per il C-level

---

### ADL-004 — Introduzione di llm_utils.py per Risolvere Circular Import

**Data:** Fase di hardening
**Stato:** Implementato e in produzione

#### Contesto
L'implementazione del retry con exponential backoff (`tenacity`) richiedeva
una funzione `_invoke_with_backoff` condivisa tra `agents.py` e `rag.py`.
Il posizionamento iniziale in `agents.py` con import da `rag.py` ha generato
un circular import (rag → agents → rag).

#### Decisione
Estrazione in un modulo neutro `llm_utils.py` senza dipendenze dai moduli
applicativi, importato da entrambi senza creare cicli.

#### Grafo delle dipendenze risultante
```
app.py → rag.py      → llm_utils.py
app.py → agents.py   → llm_utils.py
                     → rag.py (solo query_vector_db)
```