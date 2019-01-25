# Playlist dataset Recommender System

A basic python hybrid recommender system for a playlist dataset.

## Dataset

The dataset is divided in four files:

* tracks.csv: which contains a track id with its attributes [artist_id, album_id, duration]

* train.csv: which contains information about playlist_id and tracks_id added to it.

* target_playlist.csv: the target playlist to which are needed the recommendation

* train_sequential.csv: an histroical timestamp of train.csv that indicates in which order tracks were added to the playlists. (not all playlist are here)

## The Recommender

The recommender system built is a two layer hybrid recommender system.

The layers are combined then with a weighted average.

### First Layer

* Collaborative Filter Item Based with a matrix of cosine similarity. The matrix of similarity is adjusted for every user based on its historical submissions of the tracks.

* Lightfm recommender system from [lightfm](https://github.com/lyst/lightfm) library.

### Second Layer

* Artist score: tracks of the artist already in the playlists receive a score based on how many times they appear in them. 

* Album score: tracks of the album already in the playlists receive a score based on how many times they appear in them.


### Combination
First layer and second layer scores are normalized and combined with an weigthed average.

## How it works

To test the recommender:

``` python3 main.py  --test```

To perfom the recommendations:

``` python3 main.py --execute```

### Testing
After the testing is performed an evaluation score is printed.
The evaluation score is [Precision, Recall, MAP] values.

### Execution
After the execution is performed a ```submission.csv``` will be created in `./output`.

# License
MIT license

Copyright (c) 2019 Enrico Ruggiano

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.