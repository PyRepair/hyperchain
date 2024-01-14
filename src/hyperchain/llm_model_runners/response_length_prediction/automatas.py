class NFAState:
    def __init__(self, label):
        self.label = label
        self._is_accept = False
        self._transitions = {}

    def get_label(self):
        return self.label

    def add_transition(self, key, other_state):
        if key in self._transitions:
            if other_state not in self._transitions[key]:
                self._transitions[key].append(other_state)
        else:
            self._transitions[key] = [other_state]
    
    def set_accept(self, accept):
        self._is_accept = accept
    
    def transition(self, key):
        res = []
        for transition_key, transition_list in self._transitions.items():
            if key & transition_key:
                res.extend(transition_list)
        return res

class ByteNFA:
    def __init__(self, regex):
        state_count = 0
        self.dist = 0
        self.start_state = NFAState(state_count)
        state_count += 1
        self.current_states = [self.start_state]
        
        states = [self.start_state]
        
        for a in regex.split(";"):
            p = int(a.strip("?").strip("*"))
            if a.endswith("*"):
                for state in states:
                    state.add_transition(p, state)
            else:
                new_state = NFAState(state_count)
                state_count += 1
                for state in states:
                    state.add_transition(p, new_state)

                if a.endswith("?"):
                    states.append(new_state)
                else:
                    for state in states:
                        state.set_accept(False)
                    states = [new_state]
                    new_state.set_accept(True)
    
    def reset(self):
        self.dist = 0
        self.current_states = [self.start_state]

    def transition(self, nr):
        new_states = []
        found_accept = False
        for state in self.current_states:
            new_states.extend(state.transition(nr))
        
        if len(new_states) == 0:
            self.dist = 0
            self.current_states = self.start_state.transition(nr)
        else:
            self.current_states = new_states
        
        if len(self.current_states) > 0:
            self.dist += 1
        
        return self.dist if any(state._is_accept for state in self.current_states) else 0

POS_TO_MASK_DICT = {
    "w":1,
    "s":2,
    "q":4,
    "m":8,
    "i":16,
    "t":32,
    "d":64,
    "a":128,
    "v":256,
    "n":512,
    "c":1024,
    "*":0,
    "?":0,
    "+":0
}

AUTOMATAS_BLUEPRINT = [
    ("QUESTION_MODAL_LONG", "q;n;im;s*;v", "Q"),
    ("QUESTION_MODAL_AUX", "q;im;s*;v;sa*;n", "Q"),
    ("QUESTION", "q;s?;vi;sa*;n", "Q"),
    ("QUESTION_MODAL_AUX_SHORT", "q;im;s*;v", "Q"),
    ("QUESTION_YES_NO_LONG_VERB", "im;sa*;n;s?;v", "Q"),
    ("QUESTION_YES_NO_LONG", "im;sa*;n;i;sa*;n", "Q"),
    ("QUESTION_YES_NO", "im;sa+;n", "Q"),
    ("QUESTION_YES_NO_VERB","im;sa+;v", "Q"),
    ("TASK_MODAL", "w;qt;s*;v;sa*;n", "A"),
    ("TASK_MODAL_INV", "w;qt;sa*;n;im*;v", "A"),
    ("TASK_MODAL_SHORT", "w;qt;s*;v", "A"),
    ("QUESTION_SHORT", "q;sa*;n", "Q"),
    ("QUESTION_SHORT_VERB", "q;s*;v", "Q"),
    ("QUESTION_SHORT_AUX", "q;mi", "Q"),
    ("TASK_DOUBLE", "w;sa*;n;d;sa*;n", "A"),
    ("TASK", "w;sa*;n", "A"),
    ("TASK_UNKNOWN", "c", "A"),
    ("QUESTION_UNKNOWN", "q", "Q")
]

MASK_TO_POS_DICT = {v:k for k, v in POS_TO_MASK_DICT.items()}

def get_automata(regex):
    MASK_REGEX = []
    for token in regex.split(";"):
        masked_token = str(sum(map(POS_TO_MASK_DICT.get, token)))
        if token[-1] == "?":
            MASK_REGEX.append(masked_token + "?")

        elif token[-1] == "*":
            MASK_REGEX.append(masked_token + "*")

        elif token[-1] == "+":
            MASK_REGEX.append(masked_token)
            MASK_REGEX.append(masked_token + "*")

        else:
            MASK_REGEX.append(masked_token)

    return ByteNFA(";".join(MASK_REGEX))

def get_automata_list():
    return [(name, get_automata(blueprint), sent_type) for name, blueprint, sent_type in AUTOMATAS_BLUEPRINT]

