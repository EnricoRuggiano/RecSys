from numpy import *

item_path = 'data/tracks.csv'
playlist_path = 'data/train.csv'
target_playlist_path = 'data/target_playlists.csv'

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

# track_id, album_id, artist_id, duration_sec  
def create_ICM():
    file_track = open(item_path, 'r')
    file_track.readline() # drop header 
    ICM_list = [track_splitrow(line) for line in file_track]
    ICM_matrix = array(ICM_list) 
    return ICM_matrix 

# playlist id, item id_1, item id_2, ... , item id_n
def create_URM():
    file_playlist = open(playlist_path, 'r')
    file_playlist.readline() # drop header
    playlist_matrix = array([playlist_splitrow(line) for line in file_playlist])
    playlist_ids = set(playlist_matrix[:,0])    
    ICM_matrix = create_ICM()
    
    user_numbers = len(playlist_ids)
    item_numbers = ICM_matrix[0:,0].size

    URM_matrix = zeros((user_numbers, item_numbers + 1), dtype = int)
    for user in playlist_matrix:
        URM_matrix[user[0], 0] = user[0]
        URM_matrix[user[0], user[1] + 1] = 1 # incremented by 1 
    return URM_matrix
