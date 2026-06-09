# =============================================
# TEST SUITE: rag.py
# =============================================
# Tutte le chiamate a Groq e HuggingFace vengono
# mockate — i test girano senza API key né GPU.

import io
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers — PDF sintetico in memoria
# ---------------------------------------------------------------------------

def _make_pdf_bytes(text: str = "CONTRATTO DI LOCAZIONE\nLe parti concordano quanto segue.") -> bytes:
    """
    Crea un PDF minimo in memoria con il testo fornito.
    Prova fpdf2 (in requirements.txt del progetto), poi reportlab come fallback.
    """
    # Tentativo 1 — fpdf2 (dipendenza ufficiale del progetto)
    try:
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        # Helvetica supporta solo Latin-1: sostituisci caratteri fuori range
        # (es. em-dash —, €, ecc.) con il carattere più vicino ASCII.
        safe_text = text.encode("latin-1", errors="replace").decode("latin-1")
        pdf.multi_cell(0, 10, safe_text)
        return bytes(pdf.output())
    except ImportError:
        pass

    # Tentativo 2 — reportlab (presente in molti ambienti Python)
    try:
        from reportlab.pdfgen import canvas as rl_canvas
        from reportlab.lib.pagesizes import A4
        import io
        buf = io.BytesIO()
        c = rl_canvas.Canvas(buf, pagesize=A4)
        y = 750
        for line in text.splitlines():
            c.drawString(72, y, line)
            y -= 15
        c.save()
        return buf.getvalue()
    except ImportError:
        pass

    pytest.skip("Nessuna libreria PDF disponibile (fpdf2 / reportlab) — test saltato")


# ---------------------------------------------------------------------------
# _extract_text_from_bytes
# ---------------------------------------------------------------------------

class TestExtractTextFromBytes:
    def test_extracts_text_from_valid_pdf(self):
        from rag import _extract_text_from_bytes
        pdf_bytes = _make_pdf_bytes("Testo contratto di prova.")
        text = _extract_text_from_bytes(pdf_bytes)
        assert isinstance(text, str)
        assert len(text.strip()) > 0

    def test_raises_on_empty_pdf(self):
        from rag import _extract_text_from_bytes
        with pytest.raises(ValueError, match="testo estraibile|estrarre testo"):
            _extract_text_from_bytes(b"PDF vuoto non valido")

    def test_raises_on_corrupt_bytes(self):
        from rag import _extract_text_from_bytes
        with pytest.raises((ValueError, Exception)):
            _extract_text_from_bytes(b"\x00\x01\x02\x03")


# ---------------------------------------------------------------------------
# _extract_first_pages_text
# ---------------------------------------------------------------------------

class TestExtractFirstPagesText:
    def test_returns_string_for_valid_pdf(self):
        from rag import _extract_first_pages_text
        pdf_bytes = _make_pdf_bytes("Prima pagina del contratto.")
        result = _extract_first_pages_text(pdf_bytes, n_pages=1)
        assert isinstance(result, str)

    def test_returns_empty_string_on_corrupt_bytes(self):
        from rag import _extract_first_pages_text
        result = _extract_first_pages_text(b"non un pdf", n_pages=2)
        assert result == ""


# ---------------------------------------------------------------------------
# check_if_contract
# ---------------------------------------------------------------------------

class TestCheckIfContract:
    @patch("rag.invoke_with_backoff")
    @patch("rag.ChatGroq")
    def test_returns_true_for_si_response(self, mock_groq_cls, mock_invoke):
        from rag import check_if_contract
        pdf_bytes = _make_pdf_bytes("CONTRATTO DI LOCAZIONE tra Parte A e Parte B.")

        mock_response = MagicMock()
        mock_response.content = "SI"
        mock_invoke.return_value = mock_response

        result = check_if_contract(pdf_bytes)
        assert result is True

    @patch("rag.invoke_with_backoff")
    @patch("rag.ChatGroq")
    def test_returns_false_for_no_response(self, mock_groq_cls, mock_invoke):
        from rag import check_if_contract
        pdf_bytes = _make_pdf_bytes("Bilancio d'esercizio 2024 — Relazione finanziaria.")

        mock_response = MagicMock()
        mock_response.content = "NO"
        mock_invoke.return_value = mock_response

        result = check_if_contract(pdf_bytes)
        assert result is False

    @patch("rag.invoke_with_backoff", side_effect=Exception("API down"))
    @patch("rag.ChatGroq")
    def test_returns_false_on_api_error(self, mock_groq_cls, mock_invoke):
        """Fail-safe: in caso di errore del classificatore, non procedere."""
        from rag import check_if_contract
        pdf_bytes = _make_pdf_bytes("Testo qualsiasi.")
        result = check_if_contract(pdf_bytes)
        assert result is False

    def test_returns_false_on_empty_pdf(self):
        """PDF senza testo estraibile → fail-safe False."""
        from rag import check_if_contract
        result = check_if_contract(b"")
        assert result is False


# ---------------------------------------------------------------------------
# query_vector_db
# ---------------------------------------------------------------------------

class TestQueryVectorDb:
    def test_returns_list_of_strings(self):
        from rag import query_vector_db

        mock_doc = MagicMock()
        mock_doc.page_content = "Clausola 5: penale del 10%."

        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = [mock_doc, mock_doc]

        results = query_vector_db(mock_vs, "penale", k=2)
        assert isinstance(results, list)
        assert all(isinstance(r, str) for r in results)
        assert len(results) == 2
        mock_vs.similarity_search.assert_called_once_with(query="penale", k=2)

    def test_returns_empty_list_when_no_results(self):
        from rag import query_vector_db

        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = []

        results = query_vector_db(mock_vs, "clausola inesistente", k=5)
        assert results == []


# ---------------------------------------------------------------------------
# ask_document
# ---------------------------------------------------------------------------

class TestAskDocument:
    @patch("rag.invoke_with_backoff")
    @patch("rag.ChatGroq")
    def test_returns_string_response(self, mock_groq_cls, mock_invoke):
        from rag import ask_document

        mock_response = MagicMock()
        mock_response.content = "La penale è pari al 10% del canone mensile."
        mock_invoke.return_value = mock_response

        mock_doc = MagicMock()
        mock_doc.page_content = "Penale: 10% del canone per ogni mese di ritardo."
        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = [mock_doc]

        result = ask_document("Qual è la penale?", mock_vs, lingua="Italiano")
        assert isinstance(result, str)
        assert len(result) > 0

    @patch("rag.invoke_with_backoff")
    @patch("rag.ChatGroq")
    def test_chat_history_passed_correctly(self, mock_groq_cls, mock_invoke):
        from rag import ask_document

        mock_response = MagicMock()
        mock_response.content = "La durata è 3 anni."
        mock_invoke.return_value = mock_response

        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = []

        history = [
            {"role": "user",      "content": "Qual è la penale?"},
            {"role": "assistant", "content": "10% del canone."},
        ]

        result = ask_document("E la durata?", mock_vs, lingua="Italiano", chat_history=history)
        assert isinstance(result, str)
        # Verifica che invoke sia stato chiamato con history non vuota
        call_payload = mock_invoke.call_args[0][1]
        assert "Utente" in call_payload["history"] or "User" in call_payload["history"]

    @patch("rag.invoke_with_backoff")
    @patch("rag.ChatGroq")
    def test_english_mode(self, mock_groq_cls, mock_invoke):
        from rag import ask_document

        mock_response = MagicMock()
        mock_response.content = "The penalty is 10% of the monthly fee."
        mock_invoke.return_value = mock_response

        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = []

        result = ask_document("What is the penalty?", mock_vs, lingua="English")
        assert isinstance(result, str)
