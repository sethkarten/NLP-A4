import numpy as np
from pprint import pprint
unary = [('D', 'the'), ('D', 'a'), ('N', 'boy'), ('N', 'man'),
        ('N', 'telescope'), ('P', 'with'), ('V', 'saw')]
binary_tup = [('S', 'NP', 'VP'), ('PP', 'P', 'NP'), ('VP', 'V', 'NP'),
        ('VP', 'VP', 'PP'), ('NP', 'NP', 'PP'), ('NP', 'D', 'N')]
nonterminals = ('S', 'NP', 'VP', 'PP', 'V', 'P', 'D', 'N')
sentence = ['the','boy','saw','the','man','with','a','telescope']
'''
S -> NP VP 1.0
PP -> P NP 1.0
VP -> V NP 0.6667
VP -> VP PP 0.3333
NP -> NP PP 0.14285714285714285
NP -> D N 0.8571428571428571
D -> the 0.6667
D -> a 0.3333
N -> boy 0.3333
N -> man 0.3333
N -> telescope 0.3333
P -> with 1.0
V -> saw 1.0
'''
unary_prob = {
'D': {'the': 2/3., 'a': 1/3.},
'N': {'boy': 1/3., 'man': 1/3., 'telescope': 1/3.},
'P': {'with': 1.0},
'V': {'saw': 1.0}
}
binary_prob = {
'S': {'NP VP': 1.0},
'PP': {'P NP': 1.0},
'VP': {'V NP': 2/3., 'VP PP': 1/3.},
'NP': {'NP PP': 1/7., 'D N': 6/7.},
}
binary = {
'S': [('NP', 'VP')],
'PP': [(('P', 'NP'))],
'VP': [('V', 'NP'), ('VP', 'PP')],
'NP': [('NP', 'PP'), ('D', 'N')]
}
get_nonterminal_numb = {}
for i,t in enumerate(nonterminals):
    get_nonterminal_numb[t] = i

def get_unary_rule_prob(X,i):
    print(X,i,tuple([X,i]))
def get_binary_rule_prob(A,B,C,i,k,j):
    print(tuple([X,Y,Z,i,k,j]))

def compute_inside():
    alpha = np.zeros((8,8,len(nonterminals)))
    for i, w in enumerate(sentence):
        for k, X in enumerate(nonterminals):
            if tuple([X,w]) in unary:
                alpha[i,i,k] = unary_prob[X][w]
            else:
                alpha[i,i,k] = 0
    for l in range(len(sentence)-1):
        for i in range(len(sentence)-l):
            j = i + l
            total = 0
            for x, X in enumerate(nonterminals):
                if X not in binary: continue
                for Y,Z in binary[X]:
                    y = get_nonterminal_numb[Y]
                    z = get_nonterminal_numb[Z]
                    for k in range(i,j):
                        assert alpha[i,k,y] >= 0
                        assert alpha[k+1,j,z] >= 0
                        if alpha[k+1,j,z] > 0 and alpha[i,k,y] > 0:
                            total += binary_prob[X][Y+' '+Z] * alpha[i,k,y] * alpha[k+1,j,z]
                alpha[i,j,x] += total
    a = alpha.sum(axis=2)
    # print(alpha)
    # Z = alpha[0,:,get_nonterminal_numb['S']]
    # pprint(Z)

    beta = np.zeros((8,8,len(nonterminals)))
    beta[0,7,get_nonterminal_numb['S']] = 1
    for l in range(len(sentence)-3, -1, -1):
        for i in range(len(sentence)-l):
            j = i+l
            for k in range(j+1, len(sentence)):
                for (Z,X,Y) in binary_tup:
                    z,y,x = get_nonterminal_numb[Z],get_nonterminal_numb[Y],get_nonterminal_numb[X]
                    beta[i,j,x] +=  beta[i,k,z] * alpha[j+1,k,y] * binary_prob[Z][X+' '+Y]
            for k in range(i):
                for (Z,Y,X) in binary_tup:
                    z,y,x = get_nonterminal_numb[Z],get_nonterminal_numb[Y],get_nonterminal_numb[X]
                    beta[i,j,x] += beta[k,j,z] * alpha[k,i-1,y] * binary_prob[Z][Y+' '+X]
    b = beta.sum(axis=2)
    # pprint(beta)
    mu = alpha * beta
    pprint(mu[3][7][get_nonterminal_numb['NP']])
    pprint(mu[2][4][get_nonterminal_numb['VP']])
'''
    inside = defaultdict(float)
    for i, word in enumerate(sentence):
        w = sentence[i-1]
        print(w)
        for X in nonterminals:
            if tuple([X,w]) in unary:
                inside[X,i,i] = get_unary_rule_prob(X,i)
            else:
                inside[X,i,i] = 0.0

    for l in range(1,len(sentence)):
        for i in range(1,len(sentence)-l+1):
            j = i+l
            for(A,B,C) in binary:
                for k in range(i,j):
                    if inside[B,i,k] and inside[C,k+1,j]:
                        inside[A,i,j]+=get_binary_rule_prob(A,B,C,i,k,j)*inside[B,i,k]*inside[C,k+1,j]

    if inside['S',1,len(sentence)]:
        return inside
    else:
        print(sentence)
'''
compute_inside()
