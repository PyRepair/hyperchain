from .task_parsing import count_paragraphs_new, get_sentences
from .parts_of_speech import VOCABULARY
from .svr_data import _svr_coef, _svr_intercept, _tfidf_diag

from scipy.sparse import spdiags, hstack
import numpy as np
import json

svr_coef = np.asarray(_svr_coef, dtype=float)
tfidf_diag = spdiags(_tfidf_diag, diags=0, m=len(_tfidf_diag), n=len(_tfidf_diag))

def normalize_l2(X):
    for i in range(X.shape[0]):
        s = np.sqrt(sum(X.data[j] * X.data[j] for j in range(X.indptr[i], X.indptr[i + 1])))

        if s == 0.0:
            continue
        
        for j in range(X.indptr[i], X.indptr[i + 1]):
            X.data[j] /= s

def predict_response_length(prompts, lengths):
    X = count_paragraphs_new(get_sentences(prompts), VOCABULARY)
    X = X * tfidf_diag
    normalize_l2(X)
    to_mult = np.asarray(lengths, dtype=float)[:, None]/1500
    X = hstack((X, X > 0, X.multiply(to_mult)))
    return ((X @ svr_coef) + _svr_intercept)*1500