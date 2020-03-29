tags = ('D', 'N', 'V')
sequence1 = ('the', 'saw', 'cut', 'the', 'man')
sequence2 = ('the', 'man', 'saw', 'the', 'cut')

# gotten from hmm.py
emission_probability = {
'D' : {'the': 1., 'saw': 0, 'cut': 0, 'man': 0},
'N' : {'man': 1/3., 'cut': 1/6., 'saw': 1/3., 'the': 1/6.},
'V': {'saw': 1/2., 'cut': 1/2., 'the': 0, 'man': 0}
}

# gotten from hmm.py
transition_probability = {
'BOS' : {'D': 2/3., 'N': 1/3., 'V':0, 'EOS': 0},
'D' : {'N': 1.0, 'D': 0, 'V': 0, 'EOS': 0},
'V' : {'D': 1.0, 'N': 0, 'V': 0, 'EOS': 0},
'N' : {'V': 1/3., 'EOS': 1/2., 'N':1/6., 'D': 0}
}

def forward_backward(sequence):
    # Forward
    pi = []
    prev = dict()
    for i, X in enumerate(sequence):
        cur = dict()
        for tag in tags:
            if i == 0:
                cur[tag] = transition_probability['BOS'][tag] * emission_probability[tag][X]
            else:
                cur[tag] = sum(prev[tag_prime]*transition_probability[tag_prime][tag] for tag_prime in tags) * emission_probability[tag][X]
        pi.append(cur)
        prev = cur

    # Backward
    beta = []
    prev = dict()
    for i, X in reversed(list(enumerate(sequence[1:]+('EOS',)))):
        cur = dict()
        for tag in tags:
            if i == len(sequence)-1:
                cur[tag] = transition_probability[tag]["EOS"]
            else:
                cur[tag] = sum(transition_probability[tag][tag_prime] * emission_probability[tag_prime][X] * prev[tag_prime] for tag_prime in tags)
        beta.insert(0,cur)
        prev = cur
    # Out
    mu = [{tag: pi[i][tag] * beta[i][tag] for tag in tags} for i in range(len(sequence))]
    return mu

mu1 = forward_backward(sequence1)
print(mu1[2]['V'])
mu2 = forward_backward(sequence2)
print(mu2[4]['N'])
