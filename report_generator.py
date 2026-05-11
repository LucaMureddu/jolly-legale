# =============================================
# MODULO C: Generazione Report PDF
# =============================================
# Libreria: fpdf2 — puro Python, zero dipendenze di sistema
# Font: DejaVu (Unicode completo) dalla cartella fonts/
# =============================================

import re
import os
from fpdf import FPDF
from fpdf.enums import XPos, YPos

FONTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")


class ReportPDF(FPDF):
    def header(self):
        self.set_font("DejaVu", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 8, "RISERVATO E CONFIDENZIALE — Il Jolly Legale",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(201, 168, 76)
        self.set_line_width(0.4)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("DejaVu", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Pagina {self.page_no()}", align="C")


def _parse_markdown_lines(text: str) -> list:
    blocks = []
    for line in text.splitlines():
        stripped = line.rstrip()
        if stripped.startswith("### "):
            blocks.append({"type": "h3", "text": stripped[4:]})
        elif stripped.startswith("## "):
            blocks.append({"type": "h2", "text": stripped[3:]})
        elif stripped.startswith("# "):
            blocks.append({"type": "h1", "text": stripped[2:]})
        elif stripped.startswith("---"):
            blocks.append({"type": "hr"})
        elif stripped.startswith("- ") or stripped.startswith("* "):
            blocks.append({"type": "bullet", "text": stripped[2:]})
        elif "|" in stripped and stripped.startswith("|"):
            if re.match(r"^\|[-| :]+\|$", stripped):
                blocks.append({"type": "table_separator"})
            elif blocks and blocks[-1]["type"] in ("table_header", "table_row"):
                cells = [c.strip() for c in stripped.strip("|").split("|")]
                blocks.append({"type": "table_row", "cells": cells})
            else:
                cells = [c.strip() for c in stripped.strip("|").split("|")]
                blocks.append({"type": "table_header", "cells": cells})
        elif stripped == "":
            blocks.append({"type": "blank"})
        else:
            blocks.append({"type": "normal", "text": stripped})
    return blocks


def _clean(text: str) -> str:
    """Rimuove simboli Markdown residui."""
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*",     r"\1", text)
    text = re.sub(r"`(.*?)`",       r"\1", text)
    return text.strip()


def _render_table(pdf: FPDF, header_block: dict, row_blocks: list) -> None:
    """
    Renderizza una tabella Markdown in PDF con word-wrap corretto.

    Strategia:
    - Calcola l'altezza massima di ogni riga (la cella più alta).
    - Usa multi_cell() per ogni cella, riposizionando manualmente X/Y.
    - Garantisce che nessun testo venga troncato.
    """
    NAVY  = (13,  27,  42)
    WHITE = (255, 255, 255)
    LIGHT = (245, 245, 240)
    BLACK = (26,  26,  46)

    headers = header_block["cells"]
    n_cols  = len(headers)
    usable_w = pdf.w - pdf.l_margin - pdf.r_margin
    col_w    = usable_w / n_cols
    line_h   = 5.5  # altezza base per riga di testo

    def calc_cell_height(text: str, width: float, font_size: int) -> float:
        """Stima l'altezza necessaria per il testo con wrapping."""
        pdf.set_font("DejaVu", "", font_size)
        # Conta le righe necessarie basandosi sulla larghezza
        words = text.split()
        lines = 1
        current_line = ""
        for word in words:
            test_line = f"{current_line} {word}".strip()
            if pdf.get_string_width(test_line) <= width - 2:
                current_line = test_line
            else:
                lines += 1
                current_line = word
        return lines * line_h + 2  # padding verticale

    def render_row(cells: list, bg_color: tuple, font_style: str,
                   font_size: int, text_color: tuple) -> None:
        """Renderizza una singola riga con altezza uniforme su tutte le celle."""
        # Calcola l'altezza massima tra tutte le celle della riga
        row_h = max(
            calc_cell_height(_clean(cells[ci]) if ci < len(cells) else "", col_w, font_size)
            for ci in range(n_cols)
        )
        row_h = max(row_h, line_h + 2)  # altezza minima garantita

        # Controlla se la riga sfora la pagina
        if pdf.get_y() + row_h > pdf.h - pdf.b_margin:
            pdf.add_page()

        row_y = pdf.get_y()

        for ci in range(n_cols):
            txt = _clean(cells[ci]) if ci < len(cells) else ""
            x = pdf.l_margin + ci * col_w

            # Sfondo cella
            pdf.set_fill_color(*bg_color)
            pdf.rect(x, row_y, col_w, row_h, style="F")

            # Bordo sottile tra celle
            pdf.set_draw_color(200, 200, 195)
            pdf.set_line_width(0.1)
            pdf.rect(x, row_y, col_w, row_h, style="D")

            # Testo con multi_cell per wrapping automatico
            pdf.set_font("DejaVu", font_style, font_size)
            pdf.set_text_color(*text_color)
            pdf.set_xy(x + 1.5, row_y + 1.5)
            pdf.multi_cell(
                w=col_w - 3,
                h=line_h,
                text=txt,
                align="L",
                new_x=XPos.RIGHT,
                new_y=YPos.TOP,
            )

        # Sposta Y alla fine della riga
        pdf.set_xy(pdf.l_margin, row_y + row_h)

    pdf.ln(4)

    # Header
    render_row(
        cells=headers,
        bg_color=NAVY,
        font_style="B",
        font_size=9,
        text_color=WHITE,
    )

    # Righe dati
    for row_num, row_block in enumerate(row_blocks):
        if row_block["type"] == "table_separator":
            continue
        bg = LIGHT if row_num % 2 == 0 else (255, 255, 255)
        render_row(
            cells=row_block["cells"],
            bg_color=bg,
            font_style="",
            font_size=9,
            text_color=BLACK,
        )

    pdf.ln(3)


def markdown_to_pdf_bytes(markdown_text: str) -> bytes:
    """
    Converte Markdown in PDF professionale con fpdf2 + DejaVu (Unicode).
    Restituisce bytes — mai scritto su disco.
    """
    pdf = ReportPDF()
    pdf.set_margins(left=22, top=20, right=22)
    pdf.set_auto_page_break(auto=True, margin=20)

    pdf.add_font("DejaVu",  "",   os.path.join(FONTS_DIR, "DejaVuSansCondensed.ttf"))
    pdf.add_font("DejaVu",  "B",  os.path.join(FONTS_DIR, "DejaVuSansCondensed-Bold.ttf"))
    pdf.add_font("DejaVu",  "I",  os.path.join(FONTS_DIR, "DejaVuSansCondensed-Oblique.ttf"))
    pdf.add_font("DejaVu",  "BI", os.path.join(FONTS_DIR, "DejaVuSansCondensed-BoldOblique.ttf"))

    pdf.add_page()

    NAVY  = (13,  27,  42)
    GOLD  = (201, 168, 76)
    GRAY  = (80,  80,  80)
    BLACK = (26,  26,  46)

    blocks = _parse_markdown_lines(markdown_text)
    i = 0

    while i < len(blocks):
        block = blocks[i]
        btype = block["type"]

        if btype == "h1":
            pdf.ln(4)
            pdf.set_font("DejaVu", "B", 20)
            pdf.set_text_color(*NAVY)
            pdf.multi_cell(0, 10, _clean(block["text"]), align="L")
            pdf.set_draw_color(*GOLD)
            pdf.set_line_width(0.8)
            pdf.line(pdf.l_margin, pdf.get_y() + 1, pdf.w - pdf.r_margin, pdf.get_y() + 1)
            pdf.ln(6)

        elif btype == "h2":
            pdf.ln(6)
            pdf.set_font("DejaVu", "B", 14)
            pdf.set_text_color(*NAVY)
            pdf.set_fill_color(*GOLD)
            pdf.rect(pdf.l_margin, pdf.get_y(), 1.2, 7, style="F")
            pdf.set_x(pdf.l_margin + 4)
            pdf.cell(0, 7, _clean(block["text"]),
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(3)

        elif btype == "h3":
            pdf.ln(4)
            pdf.set_font("DejaVu", "B", 12)
            pdf.set_text_color(*GRAY)
            pdf.multi_cell(0, 6, _clean(block["text"]), align="L")
            pdf.ln(2)

        elif btype == "hr":
            pdf.ln(4)
            pdf.set_draw_color(200, 200, 195)
            pdf.set_line_width(0.3)
            pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
            pdf.ln(4)

        elif btype == "bullet":
            pdf.set_font("DejaVu", "", 10)
            pdf.set_text_color(*BLACK)
            pdf.set_x(pdf.l_margin + 4)
            pdf.cell(4, 6, "•")
            pdf.multi_cell(0, 6, _clean(block["text"]))

        elif btype == "table_header":
            # Raccoglie tutte le righe della tabella
            row_blocks = []
            j = i + 1
            while j < len(blocks) and blocks[j]["type"] in ("table_row", "table_separator"):
                row_blocks.append(blocks[j])
                j += 1
            _render_table(pdf, block, row_blocks)
            i = j
            continue

        elif btype == "blank":
            pdf.ln(3)

        elif btype == "normal":
            pdf.set_font("DejaVu", "", 10)
            pdf.set_text_color(*BLACK)
            pdf.multi_cell(0, 6, _clean(block["text"]))
            pdf.ln(1)

        i += 1

    return bytes(pdf.output())