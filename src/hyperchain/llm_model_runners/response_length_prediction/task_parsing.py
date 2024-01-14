from .automatas import POS_TO_MASK_DICT, get_automata_list
from .task_extraction import to_task_list

from functools import partial
from scipy.sparse import csr_matrix
import numpy as np
import re

adj_matcher = lambda x: x[1]&POS_TO_MASK_DICT["a"]
verb_matcher = lambda x: x[1]&POS_TO_MASK_DICT["v"]
aux_matcher = lambda x: x[1]&POS_TO_MASK_DICT["i"]
modal_matcher = lambda x: x[1]&POS_TO_MASK_DICT["m"]
noun_matcher = lambda x: x[1]&POS_TO_MASK_DICT["n"]
and_matcher = lambda x: x[1]&POS_TO_MASK_DICT["d"]
question_matcher = lambda x: x[1]&POS_TO_MASK_DICT["q"]

# QUESTION_MODAL_LONG: {<.*q.*><.*n.*><.*(i|m).*><.*s.*>*<.*v.*>} #ex. What traits do they exhibit?  
def parse_question_modal_long(line):
    question = line[0][0]
    verb = (line[-1][0])
    noun = (line[1][0])
    return [("TM", question, verb, noun), ("QV", question, verb), ("Q", question, noun)]

#  QUESTION_MODAL_AUX: {^<.*q.*><.*(i|m).*><.*s.*>*<.*v.*><.*(s|a).*>*<.*n.*>} #ex. How would you solve the problem?  
def parse_question_modal(line):
    question = line[0][0]
    verb = (next((line[i][0] for i in range(2, len(line) - 2) if verb_matcher(line[i])), None))
    noun = (line[-1][0])
    return [("TM", question, verb, noun), ("QV", question, verb)]

#  QUESTION_MODAL_AUX_SHORT: {^<.*q.*><.*(i|m).*><.*s.*>*<.*v.*>} #ex. How would you answer?  
def parse_question_modal_short(line):
    question = line[0][0]
    verb = (line[-1][0])
    return [("QV", question, verb)]

#  QUESTION: {<.*q.*><.*s.*>?<.*(v|i).*><.*(s|a).*>*<.*n.*>} #ex. What is a long answer?
def parse_question(line):
    question = line[0][0]
    noun = (line[-1][0])
    return [("Q", question, noun)]

#  QUESTION_YES_NO_LONG_VERB: im;sa*;n;s?;v
def parse_question_yes_no_long_verb(line):
    verb = (line[-1][0])
    noun = (line[-2][0]) if noun_matcher(line[-2]) else (line[-3][0])
    return [("YN", verb, noun)]

#  QUESTION_YES_NO_LONG: {<.*(i|m).*><.*(s|a).*>*<.*n.*><.*i.*><.*(s|a).*>*<.*n.*>
def parse_question_yes_no_long(line):
    noun = (line[-1][0])
    return [("YN", noun)]

#  QUESTION_YES_NO: {^<.*(i|m).*><.*(s|a).*>*<.*n.*>} #ex. Is this answer (correct)?
def parse_question_yes_no(line):
    verb = (line[0][0])
    noun = (line[-1][0])
    return [("YN", verb, noun)]

#  QUESTION_YES_NO_VERB: {^<.*(i|m).*><.*s.*>*<.*v.*>} #ex. Is this done (correctly)?
def parse_question_yes_no_verb(line):
    verb1 = (line[0][0])
    verb2 = (line[-1][0])
    return [("YN", verb1, verb2)]

#  TASK_MODAL: {^<.*v.*><.*q.*><.*s.*>*<.*v.*><.*(s|a).*>*<.*n.*>} #ex. Explain how you solved the problem.
def parse_task_modal(line):
    verb1 = (line[0][0])
    question = line[1][0]
    adj = (next((line[i][0] for i in range(len(line) - 2, 2, -1) if adj_matcher(line[i])), None))
    verb2 = (next((line[i][0] for i in range(2, len(line) - 1) if verb_matcher(line[i])), None))
    noun = (line[-1][0])
    return [("TM", question, verb2, noun, verb1), ("QM", question, verb2, noun), ("T", verb1, noun)]

#  TASK_MODAL_INV: {^<.*v.*><.*q.*><.*(s|a).*>*<.*n.*><.*(i|m).*>*<.*v.*>} #ex. Explain how the problem was solved.
def parse_task_modal_inv(line):
    verb1 = (line[0][0])
    question = line[1][0]
    verb2 = (line[-1][0])
    noun = (next((line[i][0] for i in range(len(line) - 1, 1, -1) if noun_matcher(line[i])), None))
    return [("TM", question, verb2, noun, verb1), ("QM", question, verb2, noun), ("T", verb1, noun)]

#  TASK_MODAL_SHORT: {^<.*v.*><.*q.*><.*s.*>*<.*v.*>} #ex. Explain how you solved it.
def parse_task_modal_short(line):
    verb1 = (line[0][0])
    verb2 = (line[-1][0])
    question = line[1][0]
    return [("TM", question, verb2, verb1), ("QV", question, verb2)]

# QUESTION_SHORT: {<.*q.*><.*(s|a).*><.*n.*>} #ex. Which languege (is this)?
def parse_question_short(line):
    question = line[0][0]
    noun = (line[-1][0])
    adj = (next((line[i][0] for i in range(len(line) - 1, 0, -1) if adj_matcher(line[i])), None))
    if adj is not None:
        return [("Q", question, adj, noun),("Q", question, noun)]
    return [("Q", question, noun)]

# QUESTION_SHORT_VERB: {<.*q.*><.*s.*><.*v.*>} #ex. How to answer?
def parse_question_short_verb(line):
    question = line[0][0]
    verb = (line[-1][0])
    return [("QV", question, verb)]

#  TASK_DOUBLE: {<.*w.*><.*(s|a).*>*<.*n.*><.*d.*><.*(s|a).*>*<.*n.*>} #ex. Give me an answer and a hypothesis.
def parse_task_double(line):
    verb = (line[0][0])
    noun1 = (next((line[i-1][0] for i in range(2, len(line)) if and_matcher(line[i])), None))
    noun2 = (line[-1][0])
    return [("T2", verb, noun1, noun2), ("T", verb, noun1), ("T", verb, noun2)]

#  TASK: {^<.*v.*><.*(s|a).*>*<.*n.*>} #ex. Give me an answer.
def parse_task(line):
    adj = (next((line[i][0] for i in range(len(line) - 2, 0, -1) if adj_matcher(line[i])), None))
    verb = (line[0][0])
    noun = (line[-1][0])
    if adj is not None:
        return [("TA", verb, adj, noun), ("T", verb, noun)]
    return [("T", verb, noun)]

#  QUESTION_UNKNOWN: {^<.*q.*>} #ex. Why
def parse_question_unknown(line):
    return [("QU", line[0][0])]

#  TASK_UNKNOWN: {^<.*v.*>} #ex. Explain
def parse_task_unknown(line):
    verb = (line[0][0])
    return [("TU", verb)]

parser_map = {
    "QUESTION_MODAL_LONG": parse_question_modal_long,
    "QUESTION_MODAL_AUX": parse_question_modal,
    "QUESTION_MODAL_AUX_SHORT": parse_question_modal_short,
    "QUESTION": parse_question,
    "QUESTION_YES_NO_LONG_VERB": parse_question_yes_no_long_verb,
    "QUESTION_YES_NO_LONG": parse_question_yes_no_long,
    "QUESTION_YES_NO": parse_question_yes_no,
    "QUESTION_YES_NO_VERB": parse_question_yes_no_verb,
    "TASK_MODAL": parse_task_modal,
    "TASK_MODAL_INV": parse_task_modal_inv,
    "TASK_MODAL_SHORT": parse_task_modal_short,
    "QUESTION_SHORT": parse_question_short,
    "QUESTION_SHORT_VERB": parse_question_short_verb,
    "QUESTION_SHORT_AUX": parse_question_short,
    "TASK_DOUBLE": parse_task_double,
    "TASK": parse_task,
    "QUESTION_UNKNOWN": parse_question_unknown,
    "TASK_UNKNOWN": parse_task_unknown,
}

def parse_sentence_new(sentence):
    if len(sentence) == 0:
        return []
    
    ans = []
    line = sentence
    label = line[0]
    
    if label in parser_map:
        ans.extend([",".join(map(str,a)) for a in parser_map[label](list(zip(line[2], line[1])))])
    else:
        print(f"ERROR! NO PARSER FUNCTION MAPPED FOR LABEL {label}.")

    return ans

def count_paragraphs_new(paragraphs, vocabulary):
    j_indices= []
    indptr = [0]
    values = []

    vocabulary_dict = dict()
    for i, word in enumerate(vocabulary):
        vocabulary_dict[word] = i
    
    for paragraph in paragraphs:
        found = {}
        tasks = []

        for sentence in paragraph:
            for value in parse_sentence_new(sentence=sentence):
                if value in vocabulary_dict:
                    idx = vocabulary_dict[value]
                    if idx not in found:
                        found[idx] = 1
                    else:
                        found[idx] += 1

        j_indices.extend(found.keys())
        values.extend(found.values())
        indptr.append(len(j_indices))

    j_indices = np.asarray(j_indices, dtype=np.int64)
    indptr = np.asarray(indptr, dtype=np.int64)
    values = np.asarray(values, dtype=float)

    result = csr_matrix(
        (values, j_indices, indptr),
        shape=(len(indptr) - 1, len(vocabulary)),
        dtype=float,
    )
    result.sort_indices()
        
    return result

def get_sentences(paragraph):
    return [*map(partial(to_task_list, get_automata_list()), paragraph)]