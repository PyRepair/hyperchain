from .word_tagging import binary_tag_sentence
from .automatas import get_automata_list

import re
import numpy as np
from itertools import groupby

def remove_duplicates_stem(word):
    if len(word) < 4:
        return word
    return ''.join(char for char, _ in groupby(word))

re_punct = "\.|\!|:|;"
re_title = re.compile("(?<=mr|st|ms|dr)\.")
re_website_2 = re.compile("\.(?=com|org|net)")
re_website_3 = re.compile("\.(?=uk|jp|br|ru|de|fr|it|pl)")
re_punct_dup = re.compile(f"(?<={re_punct}|\?|,)({re_punct}|,|\?)")
re_clean = re.compile("[^a-z\?]")
def re_clean_func(word):
    return remove_duplicates_stem(re_clean.sub("", word))

def get_sentences(paragraph):
    paragraph = paragraph.lower()
    paragraph = re_punct_dup.sub("", paragraph)
    paragraph = re_title.sub("<dot>", paragraph)
    paragraph = re_website_2.sub("<dot>", paragraph)
    paragraph = re_website_3.sub("<dot>", paragraph)
    paragraph = paragraph.replace("?", " ?<new>")
    paragraph = paragraph.replace(".", "<new>")
    paragraph = paragraph.replace("!", "<new>")
    paragraph = paragraph.replace(":", "<new>")
    paragraph = paragraph.replace(";", "<new>")
    paragraph = paragraph.replace("n't", " not")
    paragraph = paragraph.replace("'s", " is")
    paragraph = paragraph.replace("'re", " are")
    paragraph = paragraph.replace("'ve", " have")
    paragraph = paragraph.replace("'d", " would")
    paragraph = paragraph.replace("<dot>", ".")
    result = []
    paragraph_split = paragraph.split("\"")
    for p in paragraph_split[1::2] + [" quote ".join(paragraph_split[::2])]:
            ps = p.split(",")
            if len(ps) < 3:
                result.extend(p.split("<new>"))
            else:
                for x in ps[1::2]:
                    result.extend(x.split("<new>"))
                result.extend(" ".join(ps[::2]).split("<new>"))

    return list(filter(None, (list(filter(None, map(re_clean_func,x))) for x in map(str.split, result))))

def to_task_list(automatas, paragraph):
    sent_list = get_sentences(paragraph.lower())
    result = []
    for sent in sent_list:
        is_question = sent[-1] == "?"
        bin_sent = binary_tag_sentence(sent)
        found_sent = np.zeros(len(bin_sent), dtype=bool)
        for name, automata, sent_type in automatas:
            if sent_type == "Q" and not is_question:
                continue
            found = []
            automata.reset()
            for i, word_bin in enumerate(bin_sent):
                if found_sent[i]:
                    automata.reset()
                else:
                    automata_value = automata.transition(word_bin)
                    if automata_value > 0:
                        found.append((automata_value, i))
            found.sort(reverse=True)
            for av, y in found:
                if not found_sent[y]:
                    x = y - av + 1
                    if not found_sent[x]:
                        found_sent[x:y+1]=True
                        result.append((name, bin_sent[x:y+1], sent[x:y+1]))

    return result