# Playlist dataset Recommender System

A basic python hybrid recommender system for a playlist dataset.

## Dataset

dataset is divided in four files:

* tracks.csv: which contains a track id with its attributes [artist_id, album_id, duration]

* train.csv: which contains information about playlist_id and tracks_id added to it.

* target_playlist.csv: the target playlist to which are needed the recommendation

* train_sequential.csv: an histroical timestamp of train.csv that indicates in which order tracks were added to the playlists. (not all playlist are here)

## The Recommender

the recommender system built is a two layer hybrid recommender system.

the layers are combined then with a weighted average.

### First Layer

* Collaborative Filter Item Based with a matrix of cosine similarity. The matrix of similarity is adjusted for every user based on its historical submissions of the tracks.

* Lightfm recommender system from lightfm library.

### Second Layer

* Artist score: Tracks of the artist already in the playlists receive a score based on how many times they appear in them. 

* Album score: Tracks of the album already in the playlists receive a score based on how many times they appear in them.


### Combination
First layer and second layer scores are normalized and combined with an weigthed average.

## How it works

to test the recommender:

``` python3 main.py  --test```

to perfom the recommendations:

``` python3 main.py --execute```

### Testing
After the testing is performed an evaluation score is printed.
The evaluation score is [Precision, Recall, MAP] values.

### Execution
After the execution is performed a ```submission.csv``` will be created in `./output`.

#License
MIT license