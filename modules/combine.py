import numpy as np

# merge two score keeping the best value score at their items
def merge(score_one, score_two, normalization = True):
    if(normalization):
        score_one = normalize(score_one) 
        score_two = normalize(score_two)
    score_merge = np.where(score_one > score_two, score_one, score_two)
    return score_merge

# sum scores in arglist with weights
def combine(arglist, weights, normalization = True): 
    if(normalization):
        args = np.array(list(map(normalize, arglist)))
    else:
        args = arglist

    args_w = np.multiply(args, weights)
    #args_d = np.divide(args, weights.sum())
 
    sum_score = args_w.sum(axis=0)
        
    return sum_score 

def normalize(score):
    score_bias = np.abs(np.min(score))
    score_norm = np.linalg.norm(np.add(score, score_bias))
    score_normalized = np.divide(np.add(score, score_bias), score_norm)
    
    return score_normalized
