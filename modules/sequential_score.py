import numpy as np

def sequential_score(sequential_list, URM, user_id, discriminate = False, total_playlists = 20635):
    tracks = sequential_list[sequential_list[:,0] == user_id][:,1]
    size = tracks.size
    
    values = np.arange(start = size, stop = 0, step = -1)
    indexes = tracks
    score = np.zeros((total_playlists))

    np.put(score, indexes, values)
    score_b = np.add(score, 1)
    if(discriminate):
        iterations = np.array(URM.sum(axis = 0)).squeeze() 
        
        value = np.std(iterations) - np.average(iterations)
        threshold = np.argwhere(np.sort(iterations)[::-1] \
                < value).squeeze()[0]
        top_pop = np.argsort(iterations)[::-1][:threshold]
        np.put(score_b, top_pop, [1])
        
    return score_b

