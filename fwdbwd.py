states = ('D', 'N', 'V')
end_state = 'EOS'

observations1 = ('the', 'saw', 'cut', 'the', 'man')
observations2 = ('the', 'man', 'saw', 'the', 'cut')
start_probability = {'D': 2/3., 'N': 1/3., "V": 0}

emission_probability = {
'D' : {'the': 1., 'saw': 0, 'cut': 0, 'man': 0},
'N' : {'man': 1/3., 'cut': 1/6., 'saw': 1/3., 'the': 1/6.},
'V': {'saw': 1/2., 'cut': 1/2., 'the': 0, 'man': 0}
}

transition_probability = {
'BOS' : {'D': 2/3., 'N': 1/3., 'V':0, 'EOS': 0},
'D' : {'N': 1.0, 'D': 0, 'V': 0, 'EOS': 0},
'V' : {'D': 1.0, 'N': 0, 'V': 0, 'EOS': 0},
'N' : {'V': 1/3., 'EOS': 1/2., 'N':1/6., 'D': 0}
}


def fwd_bkw(observations, states, start_prob, trans_prob, emm_prob, end_st):
    """Forward–backward algorithm."""
    # Forward part of the algorithm
    fwd = []
    f_prev = {}
    for i, observation_i in enumerate(observations):
        f_curr = {}
        for st in states:
            if i == 0:
                # base case for the forward part
                prev_f_sum = start_prob[st]
            else:
                prev_f_sum = sum(f_prev[k]*trans_prob[k][st] for k in states)

            f_curr[st] = emm_prob[st][observation_i] * prev_f_sum

        fwd.append(f_curr)
        f_prev = f_curr

    p_fwd = sum(f_curr[k] * trans_prob[k][end_st] for k in states)

    # Backward part of the algorithm
    bkw = []
    b_prev = {}
    for i, observation_i_plus in enumerate(reversed(observations[1:]+(None,))):
        b_curr = {}
        for st in states:
            if i == 0:
                # base case for backward part
                # b_curr[st] = 1
                b_curr[st] = trans_prob[st][end_st]
            else:
                b_curr[st] = sum(trans_prob[st][l] * emm_prob[l][observation_i_plus] * b_prev[l] for l in states)

        bkw.insert(0,b_curr)
        b_prev = b_curr

    p_bkw = sum(start_prob[l] * emm_prob[l][observations[0]] * b_curr[l] for l in states)
    # Merging the two parts
    posterior = []
    for i in range(len(observations)):
        posterior.append({st: fwd[i][st] * bkw[i][st] / p_fwd for st in states})
        # posterior.append({st: fwd[i][st] * bkw[i][st] for st in states})
    assert p_fwd == p_bkw
    return fwd, bkw, posterior

fwd, bkw, posterior = fwd_bkw(observations1,
               states,
               start_probability,
               transition_probability,
               emission_probability,
               end_state)
print(posterior[2]['V'])
print()
fwd, bkw, posterior = fwd_bkw(observations2,
               states,
               start_probability,
               transition_probability,
               emission_probability,
               end_state)
print(posterior[4]['N'])
print()
