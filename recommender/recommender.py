class Recommender:
    
    def __init__(self, ICM, URM, target_playlists):
        self.ICM = ICM
        self.URM = URM
        self.target_playlists = target_playlists
        self.output_path = 'output/submission.csv'
        self.submission = []

    def submit_solution(self): 
        with open(self.output_path, 'w') as f:
            
            print("playlist_id,track_ids", end='\n', file=f)
            for item in self.submission:
                print(item[0], end=',', file = f)
                print(*item[1], sep=' ', end='\n', file = f)


