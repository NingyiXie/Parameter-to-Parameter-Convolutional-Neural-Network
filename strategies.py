import numpy as np
from utils import normalize,recover,predict
from maxcut import FQAOA,QAOA_MaxCut


# alg1 in the paper
def PPN_1(graph,ppn,p1param,pt):
    init_value = recover(predict(normalize(p1param),ppn,pt-1))
    q = QAOA_MaxCut(graph)
    q.run(pt,init_value)
    return q.optimized_expected_value

# alg2 in the paper
def PPN_2(graph,ppn,p1param,p1exp):
    #intial 
    bestOutput=0
    p=1
    params = p1param
    tempOutput = p1exp
    while tempOutput>bestOutput:
        bestOutput = tempOutput
        p = p + 1
        params = recover(predict(normalize(params),ppn,1))
        tempOutput = FQAOA(params,graph)
    return bestOutput

# using the parameters corresponding to the maximum FQAOA (expected value)
def ParameterListStrategy(graph,paramlist,pt):
    explist = []
    for param in paramlist:
        explist.append(FQAOA(param,graph))
    init_value = paramlist[np.argmax(explist)]
    q = QAOA_MaxCut(graph)
    q.run(pt,init_value)
    return q.optimized_expected_value,q.optimized_params