class Recommender:
    
    def __init__(self, ICM, URM, URM_train, URM_test, target_playlists, sequential_list, sequential_playlist_train, sequential_playlist_test):
        self.ICM = ICM
        self.URM = URM
        self.URM_train = URM_train
        self.URM_test = URM_test
        self.target_playlists = target_playlists
        self.target_sequential_playlist = sequential_list
        self.sequential_playlist_train = sequential_playlist_train
        self.sequential_playlist_test = sequential_playlist_test
        self.output_path = 'output/submission.csv'
        self.submission = []

    def fit(self):
        return 0

    def recommend(self):
        return 0
    
    def recommend_all(self):
        return 0

    def submit_solution(self, solution): 
        with open(self.output_path, 'w') as f: 
            print("playlist_id,track_ids", end='\n', file=f)
            for item in solution:
                print(item[0], end=',', file = f)
                print(*item[1], sep=' ', end='\n', file = f)

    def execute(self):
        self.fit(self.URM, self.target_sequential_playlist)
        self.recommend_all()

    def test(self):
        test_set = [item for item in range(self.URM_test.shape[0])]
        self.fit(self.URM_train, self.sequential_playlist_train) 
