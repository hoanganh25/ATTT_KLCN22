"""CER ONLY: character-level edit distance, including normalized spaces."""
from .edit_distance import distance
from .text_normalization import normalize_text


def char_counts(reference,hypothesis):
    a=normalize_text(reference);b=normalize_text(hypothesis)
    if not a:raise ValueError('CER reference has zero characters')
    return distance(a,b),len(a)


def score_cer(reference,hypothesis):
    err,n=char_counts(reference,hypothesis)
    return err/n
