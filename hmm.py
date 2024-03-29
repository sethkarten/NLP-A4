from collections import Counter
import numpy as np

def get_transition_data(sentences):
    out = []
    for sentence in sentences:
        d = ("BOS " + sentence + " EOS").split()
        out.extend([tuple(d[i:i + 2]) for i in range(len(d) - 2 + 1)])
    return out

def get_emission_data(sentences, all_tags):
    out = []
    for sentence, tags in zip(sentences, all_tags):
        sentence, tags = sentence.split(), tags.split()
        assert len(sentence) == len(tags)
        out.extend([(tags[i], sentence[i]) for i in range(len(sentence))])
    return out

def get_transition_probs():
    x1_tag, x2_tag, x3_tag  = "D N V D N", "D N V D N", "N N"
    y_data = get_transition_data([x1_tag,x2_tag,x3_tag])
    counts_h = Counter(y_data)
    sum = np.sum([v for v in counts_h.values()])
    t = {}
    for k, v in zip(counts_h.keys(), counts_h.values()):
        sum = np.sum([a[0] == k[0] for a in y_data])
        t[k] = v / sum
    # t[("BOS", 'D')] = 2/3.
    # t[("BOS", 'N')] = 1/3.
    # t[("N", 'EOS')] = 1
    return t

def get_emission_probs():
    x1, x2, x3 = "the man saw the cut", "the saw cut the man", "the saw"
    x1_tag, x2_tag, x3_tag  = "D N V D N", "D N V D N", "N N"
    data = get_emission_data([x1,x2,x3],[x1_tag,x2_tag,x3_tag])
    counts_h = Counter(data)
    sum = np.sum([v for v in counts_h.values()])
    o = {}
    for k, v in zip(counts_h.keys(), counts_h.values()):
        sum = np.sum([a[0] == k[0] for a in data])
        o[k] = v / sum
    return o

# Answer to 1.1
def get_o_t():
    t = get_transition_probs()
    for ti in t:
        print(ti, round(t[ti],5))
    print()
    o = get_emission_probs()
    for oi in o:
        print(oi, round(o[oi],5))
    print()
    return o, t

if __name__ == "__main__":
    o, t = get_o_t()
