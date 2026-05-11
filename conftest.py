import sys
import os

# Aggiunge la root del progetto al path di Python
# così pytest trova rag.py, agents.py, report_generator.py
sys.path.insert(0, os.path.dirname(__file__))
