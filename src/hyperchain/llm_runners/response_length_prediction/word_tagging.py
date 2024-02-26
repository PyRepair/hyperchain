from .parts_of_speech import TAGS_ADJ, TAGS_NOUN, TAGS_VERB, TAGS_MASK

import numpy as np

ADJ_2L = set(["ed", "ny", "dy", "ty", "ky", "ly"])
ADJ_3L = set(["nal", "ial", "cal", "ful", "ous", "ish", "ing", "ive"])
ADJ_4L = set(["able", "ible", "less"])

def get_tags_int(word):
    word_int = 0
    if word in TAGS_MASK:
        word_int = TAGS_MASK[word]

    w3 = word[-3:]
    w2 = word[-2:]
    if not word_int & 128:
        w4 = word[-4:]
        word_int |= (w2 in ADJ_2L or w3 in ADJ_3L or w4 in ADJ_4L) << 7

    w_to_v = 0
    w_to_v_start = 0
    if len(word) > 3:
        if w3 == "ing":
            w_to_v = 3
        elif w2 == "ed":
            w_to_v = 2
        else: 
            if word[-1] == "s":
                w_to_v = 1
                if word[-2] == "e":
                    w_to_v += 1
                    if word[-3] == "i":
                        w_to_v += 1
            elif word[-1] == "e":
                w_to_v = 1
        
        if word[0] == "u" and word[1] == "n":
            w_to_v_start = 2
        elif word[0] == "d" and word[1] == "i" and word[2] == "s":
            w_to_v_start = 3
        
    if w_to_v:
        word_int |= (word[w_to_v_start:-w_to_v] in TAGS_VERB) << 8
    else:
        word_int |= (word[w_to_v_start:] in TAGS_VERB) << 8
    
    n_to_v = 0
    if w2 == "es":
        n_to_v = 2
    elif len(word) and (word[-1] == "s" or word[-1] == "e"):
        n_to_v = 1
    
    if n_to_v:
        word_int |= (word[:-n_to_v] in TAGS_NOUN) << 9
    else:
        word_int |= (word in TAGS_NOUN) << 9

    return word_int

def binary_tag_sentence(sent):
    return np.fromiter(map(get_tags_int, sent), dtype=int)