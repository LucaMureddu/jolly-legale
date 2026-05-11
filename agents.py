# =============================================
# MODULO B: Orchestrazione Multi-Agente (LangGraph)
# =============================================
# Agenti:
#   1. Agente Ricercatore (Reader)    — interroga il Vector DB
#   2. Agente Revisore Legale         — analizza rischi E opportunità
#   3. Agente Formattatore (Reporter) — genera report Markdown bilanciato
# =============================================

import os
from typing import TypedDict

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from dotenv import load_dotenv
from rag import query_vector_db
from llm_utils import _invoke_with_backoff

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL    = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

# -------------------------------------------------------
# STATO CONDIVISO TRA GLI AGENTI
# -------------------------------------------------------

class ContractAnalysisState(TypedDict):
    vectorstore:    object  # ChromaDB ephemeral della sessione
    lingua:         str     # "Italiano" o "English"
    raw_context:    str     # Contesto estratto dal Ricercatore
    legal_analysis: str     # Analisi grezza del Revisore Legale
    final_report:   str     # Report Markdown finale del Formattatore


# -------------------------------------------------------
# AGENTE 1: RICERCATORE (Reader)
# -------------------------------------------------------

RESEARCH_QUERIES = [
    # Rischi
    "clausole penali e sanzioni",
    "condizioni di rescissione e recesso",
    "scadenze termini e rinnovi automatici",
    "limitazioni di responsabilità e indennizzi",
    "obblighi delle parti e asimmetrie contrattuali",
    "foro competente e legge applicabile",
    "riservatezza e non concorrenza",
    # Clausole favorevoli e mitigazioni
    "sconti sul canone rimborsi e compensazioni",
    "lavori a carico del locatore o scomputati dal canone",
    "diritti e facoltà del conduttore o dell'acquirente",
    "garanzie a favore del cliente o dell'acquirente",
    "incentivi agevolazioni e clausole migliorative",
]

def agent_reader(state: ContractAnalysisState) -> ContractAnalysisState:
    """
    Agente Ricercatore: interroga ChromaDB con query tematiche
    bilanciate (rischi + opportunità) e aggrega il contesto.
    """
    vectorstore = state["vectorstore"]
    all_chunks: list[str] = []

    for query in RESEARCH_QUERIES:
        chunks = query_vector_db(vectorstore, query, k=3)
        all_chunks.extend(chunks)

    seen: set[str] = set()
    unique_chunks: list[str] = []
    for chunk in all_chunks:
        if chunk not in seen:
            seen.add(chunk)
            unique_chunks.append(chunk)

    raw_context = "\n\n---\n\n".join(unique_chunks)
    return {**state, "raw_context": raw_context}


# -------------------------------------------------------
# AGENTE 2: REVISORE LEGALE
# -------------------------------------------------------

LEGAL_REVIEW_PROMPT_IT = ChatPromptTemplate.from_messages([
    ("system", """Sei un avvocato d'affari senior con 20 anni di esperienza in diritto
commerciale e contrattualistica presso studi legali di primo piano e Big 4.

Il tuo mandato NON è solo trovare i rischi: è fornire al cliente
una lettura completa e bilanciata del contratto, come farebbe
un vero giurista d'impresa che deve consigliare se firmare o rinegoziare.

Conduci la tua analisi in due macro-sezioni:

SEZIONE A - RISCHI E CLAUSOLE CRITICHE

Analizza con occhio critico le seguenti categorie:

1. CLAUSOLE PENALI E SANZIONI
   Importi, condizioni di attivazione, proporzionalità rispetto al mercato.

2. RISCHI DI RESCISSIONE E RECESSO
   Condizioni che permettono alla controparte di recedere unilateralmente.
   Asimmetrie tra i diritti di recesso delle parti.

3. SCADENZE CRITICHE
   Termini perentori, rinnovi automatici, finestre di disdetta.
   Evidenzia i preavvisi particolarmente onerosi.

4. ASIMMETRIE LEGALI E CLAUSOLE VESSATORIE
   Obblighi sbilanciati, deroghe al codice civile sfavorevoli al cliente.

5. LIMITAZIONI DI RESPONSABILITÀ
   Cap sui danni, esclusioni, indennizzi.

6. RISERVATEZZA E NON CONCORRENZA
   Durata, perimetro geografico, sanzioni previste.

7. RISCHI NASCOSTI
   Qualsiasi altra clausola che potrebbe danneggiare il cliente
   in modo non immediatamente evidente.

Per ogni elemento: cita brevemente la clausola, descrivi il rischio
concreto e assegna: ALTO / MEDIO / BASSO

SEZIONE B - CLAUSOLE FAVOREVOLI E MITIGAZIONI FINANZIARIE

Questa sezione è OBBLIGATORIA. Identifica esplicitamente:

8. SCONTI, RIMBORSI E COMPENSAZIONI
   Qualsiasi meccanismo che riduce l'esposizione economica del cliente:
   scomputi dal canone, rimborsi per lavori, incentivi, periodi di
   franchigia, contributi a carico della controparte.

9. DIRITTI E FACOLTÀ DEL CLIENTE
   Opzioni di acquisto, diritti di prelazione, facoltà di sublocazione,
   possibilità di cessione del contratto, clausole di uscita agevolata.

10. GARANZIE E TUTELE CONTRATTUALI
    Garanzie ricevute dalla controparte, manleve, polizze assicurative
    obbligatorie a carico dell'altra parte, depositi cauzionali restituibili.

11. CLAUSOLE MIGLIORATIVE RISPETTO AGLI USI DI MERCATO
    Qualsiasi condizione più favorevole rispetto allo standard
    del settore o alle disposizioni di legge supplettive.

Per ogni elemento: descrivi il beneficio concreto e quantificalo
economicamente se possibile.

ISTRUZIONI GENERALI:
- Se una categoria non è presente nel contratto, scrivilo esplicitamente.
- Non inventare nulla: ogni affermazione deve essere supportata dal testo.
- Scrivi in italiano, con tono professionale e diretto."""),
    ("human", "Analizza il seguente testo contrattuale:\n\n{context}")
])

LEGAL_REVIEW_PROMPT_EN = ChatPromptTemplate.from_messages([
    ("system", """You are a senior business lawyer with 20 years of experience in commercial
and contract law at top international law firms and Big 4 consulting firms.

Your mandate is NOT only to find risks: it is to provide the client with
a complete and balanced reading of the contract, as a true business lawyer
would do when advising whether to sign or renegotiate.

Conduct your analysis in two macro-sections:

SECTION A - RISKS AND CRITICAL CLAUSES

Critically analyze the following categories:

1. PENALTY CLAUSES AND SANCTIONS
   Amounts, activation conditions, proportionality vs. market standards.
   Example: automatic termination for payment delays of only 10 days.

2. TERMINATION AND WITHDRAWAL RISKS
   Conditions allowing the counterparty to unilaterally withdraw.
   Asymmetries between the parties' withdrawal rights.
   Example: tenant can only withdraw for "serious reasons" while landlord has broader rights.

3. CRITICAL DEADLINES
   Mandatory terms, automatic renewals, cancellation windows.
   Highlight particularly burdensome notice periods.
   Example: 12-month notice required to avoid automatic renewal.

4. LEGAL ASYMMETRIES AND UNFAIR TERMS
   Unbalanced obligations, derogations from statutory law unfavorable to the client.
   Example: waiver of statutory reimbursement rights for improvements to the property.

5. LIABILITY LIMITATIONS
   Damage caps, exclusions, indemnities.
   Example: liability capped at 3 months of fees regardless of actual damage.

6. CONFIDENTIALITY AND NON-COMPETE
   Duration, geographic scope, sanctions.
   Example: 2-year non-compete covering the entire national territory.

7. HIDDEN RISKS
   Any clause that could harm the client in a non-immediately obvious way.
   Example: automatic indexation clauses, change-of-control provisions.

For each item: briefly quote the clause, describe the concrete risk,
assign: HIGH / MEDIUM / LOW

SECTION B - FAVORABLE CLAUSES AND FINANCIAL MITIGATIONS (MANDATORY)

This section is MANDATORY and often overlooked: a contract is never purely
unfavorable. Explicitly identify:

8. DISCOUNTS, REIMBURSEMENTS AND COMPENSATIONS
   Any mechanism reducing the client's economic exposure:
   rent deductions, reimbursements for fit-out works, incentives,
   rent-free periods, contributions charged to the counterparty.
   Example: initial fit-out costs deducted from monthly rent instalments.

9. CLIENT RIGHTS AND OPTIONS
   Purchase options, rights of first refusal, subletting rights,
   contract assignment possibilities, facilitated exit clauses.
   Example: right of first refusal if the property is put up for sale.

10. CONTRACTUAL GUARANTEES AND PROTECTIONS
    Guarantees received from the counterparty, indemnities,
    insurance policies mandatory for the other party,
    refundable security deposits, warranties on the asset condition.
    Example: security deposit fully refundable within 30 days of contract end.

11. CLAUSES BETTER THAN MARKET STANDARD
    Any condition more favorable than sector standard
    or than the applicable suppletive statutory provisions.
    Example: fixed rent for first 3 years with no indexation.

For each item: describe the concrete benefit and quantify it
economically if possible (e.g. "deduction of X months of rent").

GENERAL INSTRUCTIONS:
- If a category is not present in the contract, state it explicitly.
- Do not invent anything: every statement must be supported by the text.
- Write in English, with a professional and direct tone."""),
    ("human", "Analyze the following contractual text:\n\n{context}")
])


def agent_reviewer(state: ContractAnalysisState) -> ContractAnalysisState:
    """
    Agente Revisore Legale: analisi bilanciata rischi + clausole favorevoli.
    Usa ChatGroq con exponential backoff su rate limit.
    """
    llm    = ChatGroq(model_name=LLM_MODEL, temperature=0, api_key=GROQ_API_KEY)
    prompt = LEGAL_REVIEW_PROMPT_EN if state["lingua"] == "English" else LEGAL_REVIEW_PROMPT_IT
    chain  = prompt | llm

    response = _invoke_with_backoff(chain, {"context": state["raw_context"]})
    return {**state, "legal_analysis": response.content}


# -------------------------------------------------------
# AGENTE 3: FORMATTATORE (Reporter)
# -------------------------------------------------------

REPORT_PROMPT_IT = ChatPromptTemplate.from_messages([
    ("system", """Sei un consulente senior che produce report direzionali per studi legali.

Trasforma l'analisi legale grezza in un report Markdown professionale e bilanciato.
Rifletti fedelmente ENTRAMBE le sezioni: rischi E clausole favorevoli.

Struttura OBBLIGATORIA:

# Report di Analisi Contrattuale

## Sommario Esecutivo
(4-5 righe: sintetizza rischi principali E punti favorevoli per un C-level)

## Indice dei Rischi
Tabella Markdown: Categoria | Livello | Sintesi

## Indice delle Clausole Favorevoli
Tabella Markdown: Categoria | Beneficio Economico | Sintesi

## Analisi Dettagliata - Rischi
(Sezioni per ogni categoria di rischio)

## Analisi Dettagliata - Clausole Favorevoli
(Sezioni per ogni beneficio, con quantificazione economica ove possibile)

## Raccomandazioni
(3-5 azioni concrete: cosa rinegoziare, accettare, monitorare)

## Note Metodologiche
(2-3 righe sul metodo)

---
*Report generato da Il Jolly Legale*

Scrivi in italiano. Markdown pulito. Non usare caratteri speciali come frecce o linee decorative."""),
    ("human", "Trasforma questa analisi in un report professionale:\n\n{analysis}")
])

REPORT_PROMPT_EN = ChatPromptTemplate.from_messages([
    ("system", """You are a senior consultant producing executive reports for law firms.

Transform the raw legal analysis into a professional, balanced Markdown report.
Reflect BOTH sections faithfully: risks AND favorable clauses.

MANDATORY structure:

# Contract Analysis Report
## Executive Summary
## Risk Index (table: Category | Level | Summary)
## Favorable Clauses Index (table: Category | Economic Benefit | Summary)
## Detailed Analysis - Risks
## Detailed Analysis - Favorable Clauses
## Recommendations (3-5 prioritized actions)
## Methodological Notes

---
*Report generated by Il Jolly Legale*

Write in English. Clean Markdown. No special decorative characters."""),
    ("human", "Transform this analysis into a professional report:\n\n{analysis}")
])


def agent_reporter(state: ContractAnalysisState) -> ContractAnalysisState:
    """
    Agente Formattatore: converte l'analisi in report Markdown professionale.
    Usa ChatGroq con temperature=0.2 per output più fluido.
    """
    llm    = ChatGroq(model_name=LLM_MODEL, temperature=0.2, api_key=GROQ_API_KEY)
    prompt = REPORT_PROMPT_EN if state["lingua"] == "English" else REPORT_PROMPT_IT
    chain  = prompt | llm

    response = _invoke_with_backoff(chain, {"analysis": state["legal_analysis"]})
    return {**state, "final_report": response.content}


# -------------------------------------------------------
# GRAFO LANGGRAPH
# -------------------------------------------------------

def build_analysis_graph() -> StateGraph:
    """Costruisce il grafo LangGraph con i 3 agenti in sequenza."""
    graph = StateGraph(ContractAnalysisState)

    graph.add_node("reader",   agent_reader)
    graph.add_node("reviewer", agent_reviewer)
    graph.add_node("reporter", agent_reporter)

    graph.set_entry_point("reader")
    graph.add_edge("reader",   "reviewer")
    graph.add_edge("reviewer", "reporter")
    graph.add_edge("reporter", END)

    return graph.compile()


# -------------------------------------------------------
# FUNZIONE PUBBLICA
# -------------------------------------------------------

def run_analysis_pipeline(vectorstore: object, lingua: str = "Italiano") -> str:
    """
    Esegue la pipeline completa dei 3 agenti.

    Args:
        vectorstore: ChromaDB ephemeral della sessione (da rag.ingest_pdf).
        lingua:      "Italiano" o "English" (default: "Italiano").

    Returns:
        Report finale in formato Markdown.

    Raises:
        Exception: Se le API Groq non sono raggiungibili.
    """
    graph = build_analysis_graph()

    initial_state: ContractAnalysisState = {
        "vectorstore":    vectorstore,
        "lingua":         lingua,
        "raw_context":    "",
        "legal_analysis": "",
        "final_report":   "",
    }

    final_state = graph.invoke(initial_state)
    return final_state["final_report"]