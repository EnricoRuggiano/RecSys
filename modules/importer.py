from numpy import *

item_path = 'data/tracks.csv'
playlist_path = 'data/train.csv'
target_playlist_path = 'data/target_playlists.csv'

def test(user, playlist_matrix):
    return [index[1] for index in playlist_matrix if index[0] == user]

def print_test(user, to_test, URM_matrix):
    print("Testing user: {} ".format(user))
    print("Matrix_row:\n{}".format(URM_matrix[user,:]))
    print("list fo items: \n{}".format(to_test))
    for elem in to_test:
        print("item to test: {}".format(elem))
        print("is present on matrix?")
        print(URM_matrix[user, elem + 1])


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

def create_URM():
    file_playlist = open(playlist_path, 'r')
    file_playlist.readline() # drop header
    playlist_matrix = array([playlist_splitrow(line) for line in file_playlist])
    playlist_ids = set(playlist_matrix[:,0])
    
    ICM_matrix = create_ICM()
    
    user_numbers = len(playlist_ids)
    item_numbers = ICM_matrix[0:,0].size

    #print("user_numbers: {}".format(user_numbers))
    #print("item_numbers: {}".format(item_numbers))
    
    # playlist id, item id_1, item id_2, ... , item id_n
    URM_matrix = zeros((user_numbers, item_numbers + 1), dtype = int)

    for user in playlist_matrix:
        URM_matrix[user[0], 0] = user[0]
        URM_matrix[user[0], user[1] + 1] = 1 # incremented by 1 
#    print(URM_matrix)
#    print(URM_matrix.shape)

    return URM_matrix
    ### example 
"""
    test_one = test(1, playlist_matrix)
    test_middle = test(25, playlist_matrix)
    test_last = test(30778, playlist_matrix)

    print_test(123, test_one, URM_matrix)
    print_test(25, test_middle, URM_matrix)
    print_test(30778, test_last, URM_matrix)

"""
