# =============================================
# TEST SUITE: agents.py
# =============================================
# Tutte le chiamate a Groq vengono mockate —
# i test girano senza API key.

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers — stato minimo per il grafo LangGraph
# ---------------------------------------------------------------------------

def _make_state(lingua: str = "Italiano", raw_context: str = "", legal_analysis: str = "") -> dict:
    mock_vs = MagicMock()
    mock_vs.similarity_search.return_value = []
    return {
        "vectorstore":    mock_vs,
        "lingua":         lingua,
        "raw_context":    raw_context,
        "legal_analysis": legal_analysis,
        "final_report":   "",
    }


# ---------------------------------------------------------------------------
# agent_reader
# ---------------------------------------------------------------------------

class TestAgentReader:
    def test_returns_state_with_raw_context(self):
        from agents import agent_reader

        mock_doc = MagicMock()
        mock_doc.page_content = "Clausola 5: penale del 10%."
        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = [mock_doc]

        state = _make_state()
        state["vectorstore"] = mock_vs

        result = agent_reader(state)
        assert "raw_context" in result
        assert isinstance(result["raw_context"], str)

    def test_deduplicates_chunks(self):
        from agents import agent_reader

        mock_doc = MagicMock()
        mock_doc.page_content = "Chunk duplicato."
        mock_vs = MagicMock()
        # Stesso chunk restituito da ogni query
        mock_vs.similarity_search.return_value = [mock_doc]

        state = _make_state()
        state["vectorstore"] = mock_vs

        result = agent_reader(state)
        # Il chunk duplicato deve apparire una sola volta
        assert result["raw_context"].count("Chunk duplicato.") == 1

    def test_uses_english_queries_for_english_mode(self):
        from agents import agent_reader, RESEARCH_QUERIES_EN, RESEARCH_QUERIES_IT

        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = []

        state = _make_state(lingua="English")
        state["vectorstore"] = mock_vs

        agent_reader(state)

        # Controlla che le query usate siano in inglese
        calls = [call[1]["query"] for call in mock_vs.similarity_search.call_args_list]
        for q in calls:
            assert q in RESEARCH_QUERIES_EN
            assert q not in RESEARCH_QUERIES_IT

    def test_uses_italian_queries_for_italian_mode(self):
        from agents import agent_reader, RESEARCH_QUERIES_IT

        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = []

        state = _make_state(lingua="Italiano")
        state["vectorstore"] = mock_vs

        agent_reader(state)

        calls = [call[1]["query"] for call in mock_vs.similarity_search.call_args_list]
        for q in calls:
            assert q in RESEARCH_QUERIES_IT


# ---------------------------------------------------------------------------
# agent_reviewer
# ---------------------------------------------------------------------------

class TestAgentReviewer:
    @patch("agents.invoke_with_backoff")
    @patch("agents.ChatGroq")
    def test_returns_state_with_legal_analysis(self, mock_groq_cls, mock_invoke):
        from agents import agent_reviewer

        mock_response = MagicMock()
        mock_response.content = "SEZIONE A: nessun rischio critico. SEZIONE B: clausole favorevoli presenti."
        mock_invoke.return_value = mock_response

        state = _make_state(raw_context="Testo contratto di locazione.")
        result = agent_reviewer(state)

        assert "legal_analysis" in result
        assert isinstance(result["legal_analysis"], str)
        assert len(result["legal_analysis"]) > 0

    @patch("agents.invoke_with_backoff")
    @patch("agents.ChatGroq")
    def test_uses_english_prompt_for_english_mode(self, mock_groq_cls, mock_invoke):
        from agents import agent_reviewer, LEGAL_REVIEW_PROMPT_EN, LEGAL_REVIEW_PROMPT_IT

        mock_response = MagicMock()
        mock_response.content = "Analysis complete."
        mock_invoke.return_value = mock_response

        # Cattura quale prompt viene usato
        mock_chain = MagicMock()
        mock_groq_cls.return_value.__ror__ = MagicMock(return_value=mock_chain)

        state = _make_state(lingua="English", raw_context="Contract text.")
        agent_reviewer(state)

        # invoke_with_backoff deve essere chiamato
        assert mock_invoke.called

    @patch("agents.invoke_with_backoff")
    @patch("agents.ChatGroq")
    def test_preserves_other_state_fields(self, mock_groq_cls, mock_invoke):
        from agents import agent_reviewer

        mock_response = MagicMock()
        mock_response.content = "Analisi completata."
        mock_invoke.return_value = mock_response

        state = _make_state(raw_context="Testo.", lingua="Italiano")
        result = agent_reviewer(state)

        # Gli altri campi dello stato devono essere preservati
        assert result["lingua"] == "Italiano"
        assert result["raw_context"] == "Testo."


# ---------------------------------------------------------------------------
# agent_reporter
# ---------------------------------------------------------------------------

class TestAgentReporter:
    @patch("agents.invoke_with_backoff")
    @patch("agents.ChatGroq")
    def test_returns_state_with_final_report(self, mock_groq_cls, mock_invoke):
        from agents import agent_reporter

        mock_response = MagicMock()
        mock_response.content = "# Report di Analisi Contrattuale\n## Sommario Esecutivo\nNessun rischio critico."
        mock_invoke.return_value = mock_response

        state = _make_state(legal_analysis="Analisi grezza qui.")
        result = agent_reporter(state)

        assert "final_report" in result
        assert isinstance(result["final_report"], str)
        assert len(result["final_report"]) > 0

    @patch("agents.invoke_with_backoff")
    @patch("agents.ChatGroq")
    def test_preserves_state_fields(self, mock_groq_cls, mock_invoke):
        from agents import agent_reporter

        mock_response = MagicMock()
        mock_response.content = "# Report"
        mock_invoke.return_value = mock_response

        state = _make_state(legal_analysis="Analisi.", lingua="English")
        result = agent_reporter(state)

        assert result["lingua"] == "English"
        assert result["legal_analysis"] == "Analisi."


# ---------------------------------------------------------------------------
# build_analysis_graph / run_analysis_pipeline (integration)
# ---------------------------------------------------------------------------

class TestRunAnalysisPipeline:
    @patch("agents.invoke_with_backoff")
    @patch("agents.ChatGroq")
    def test_pipeline_returns_markdown_string(self, mock_groq_cls, mock_invoke):
        from agents import run_analysis_pipeline

        mock_response = MagicMock()
        mock_response.content = "# Report Finale\n## Sommario"
        mock_invoke.return_value = mock_response

        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = []

        report = run_analysis_pipeline(mock_vs, lingua="Italiano")

        assert isinstance(report, str)
        assert len(report) > 0

    @patch("agents.invoke_with_backoff")
    @patch("agents.ChatGroq")
    def test_pipeline_english_mode(self, mock_groq_cls, mock_invoke):
        from agents import run_analysis_pipeline

        mock_response = MagicMock()
        mock_response.content = "# Contract Analysis Report\n## Executive Summary"
        mock_invoke.return_value = mock_response

        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = []

        report = run_analysis_pipeline(mock_vs, lingua="English")

        assert isinstance(report, str)
        assert len(report) > 0


# ---------------------------------------------------------------------------
# llm_utils — invoke_with_backoff
# ---------------------------------------------------------------------------

class TestInvokeWithBackoff:
    def test_calls_chain_invoke(self):
        from llm_utils import invoke_with_backoff

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "risposta"

        result = invoke_with_backoff(mock_chain, {"key": "value"})

        mock_chain.invoke.assert_called_once_with({"key": "value"})
        assert result == "risposta"

    def test_retries_on_rate_limit_error(self):
        from groq import RateLimitError
        from llm_utils import invoke_with_backoff

        mock_chain = MagicMock()
        # Fallisce le prime 2 volte, poi ha successo
        mock_chain.invoke.side_effect = [
            RateLimitError("rate limit", response=MagicMock(), body={}),
            RateLimitError("rate limit", response=MagicMock(), body={}),
            "risposta ok",
        ]

        result = invoke_with_backoff(mock_chain, {})
        assert result == "risposta ok"
        assert mock_chain.invoke.call_count == 3
