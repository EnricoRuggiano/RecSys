import numpy as np
import scipy.sparse as sps

item_path = 'data/tracks.csv'
playlist_path = 'data/train.csv'
target_playlist_path = 'data/target_playlists.csv'
sequential_playlist_path = 'data/train_sequential.csv'

def track_splitrow(line):
    split = line.split(',')
    split[3].replace('\n', '')
    
    split[0] = int(split[0])
    split[1] = int(split[1])
    split[2] = int(split[2])
    split[3] = float(split[3])
    return split

def playlist_splitrow(line):
    split = line.split(',')
    split[1].replace('\n', '')

    split[0] = int(split[0])
    split[1] = int(split[1])
    return split

def import_target_playlist():
    file_target_playlist = open(target_playlist_path, 'r')
    file_target_playlist.readline() #drop header
    list_target_playlist = [int(line.strip()) for line in file_target_playlist]
    return list_target_playlist

def import_sequential_playlist():
    file_sequential_playlist = open(sequential_playlist_path, 'r')
    file_sequential_playlist.readline() #drop header
    sequential_playlist = np.array([playlist_splitrow(line) for line in file_sequential_playlist])
    return sequential_playlist 
    
# row:track_id; columns: album_id, artist_id, duration_sec  
def create_ICM():
    file_track = open(item_path, 'r')
    file_track.readline() # drop header 
    ICM_list = [track_splitrow(line) for line in file_track]
    ICM_matrix = np.array(ICM_list)[:,1:]
   
    return ICM_matrix

# row:playlist id; colums: item id_1, item id_2, ... , item id_n
def create_URM():
    file_playlist = open(playlist_path, 'r')
    file_playlist.readline() # drop header
    playlist_matrix = np.array([playlist_splitrow(line) for line in file_playlist])
    playlist_ids = set(playlist_matrix[:,0])    
    ICM_matrix = create_ICM()
    
    user_numbers = len(playlist_ids)
    item_numbers = ICM_matrix.shape[0]

    URM_matrix = np.zeros((user_numbers, item_numbers), dtype = int)
    for user in playlist_matrix:
        URM_matrix[user[0], user[1]] = 1  

    URM_matrix = sps.csr_matrix(URM_matrix)
    return URM_matrix
