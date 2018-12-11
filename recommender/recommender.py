class Recommender:
    
    def __init__(self, ICM, URM, URM_train, URM_test, target_playlists):
        self.ICM = ICM
        self.URM = URM
        self.URM_train = URM_train
        self.URM_test = URM_test
        self.target_playlists = target_playlists
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
        self.fit(self.URM)
        #solution = self.recommend_all(self.URM, self.target_playlists, at=10)
        #self.submit_solution(solution)
        self.recommend_all()

    def test(self):
        test_set = [item for item in range(self.URM_test.shape[0])]
        self.fit(self.URM_train)
        #self.submission = self.recommend_all(self.URM_train, test_set, at=10)
