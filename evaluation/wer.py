"""WER ONLY: word-level edit distance against official ground truth transcript."""
from .edit_distance import distance
from .text_normalization import normalize_text


def word_counts(reference,hypothesis):
    a=normalize_text(reference).split();b=normalize_text(hypothesis).split()
    if not a:raise ValueError('WER reference has zero words')
    return distance(a,b),len(a)


def score_wer(reference,hypothesis):
    err,n=word_counts(reference,hypothesis)
    return err/n
