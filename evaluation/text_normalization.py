"""Single reproducible normalization of English ASR hypothesis and reference."""
import re


def normalize_text(text):
    # Preserve words and digits; punctuation ignored. CER includes inter-word spaces.
    return ' '.join(re.sub(r"[^A-Z0-9' ]+",' ',str(text).upper()).split())
