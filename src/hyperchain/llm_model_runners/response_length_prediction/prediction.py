from .task_parsing import count_paragraphs_new, get_sentences
from .parts_of_speech import VOCABULARY
from .svr_data import _svr_dict, _tfidf_diag

from sklearn.svm import LinearSVR
from sklearn.preprocessing import normalize
from scipy.sparse import spdiags, hstack
import numpy as np
import json

svr = LinearSVR()
svr.__dict__ =  {
    k: np.asarray(v[0], dtype=v[2]) if isinstance(v, list) and v[1] == 'np.ndarray' else v 
    for k, v in _svr_dict.items()
}

tfidf_diag = spdiags(_tfidf_diag, diags=0, m=len(_tfidf_diag), n=len(_tfidf_diag))

def predict_response_length(prompts, lengths):
    X = count_paragraphs_new(get_sentences(prompts), VOCABULARY)
    X = X * tfidf_diag
    X = normalize(X, norm='l2', copy=False)
    to_mult = np.asarray(lengths, dtype=float)[:, None]/1500
    X = hstack((X, X > 0, X.multiply(to_mult)))
    return svr.predict(X)*1500