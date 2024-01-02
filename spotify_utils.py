# Quero analisar a playlist com as minhas liked songs
# Vou começar com uma playlist mais pequena tipo a do House4Pitex (id = 6cC2jgTKPDYJGh6Ww9F6ga) 

import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from prenda_utils import get_track_properties


def get_playlist(spotify, id):
    #my_tracks = spotify.current_user_saved_tracks(limit = 20, offset = 0)
    playlist = spotify.playlist_tracks(id)
    playlist_properties = get_track_properties(playlist= playlist)

    desired_keys = ['track_name', 'danceability', 'energy', 'key', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    features = {key: [] for key in desired_keys}

    for i, track_id in enumerate(playlist_properties['track_id']):
        features = get_track_analysis(spotify.audio_features(track_id), features, playlist_properties['track_name'][i])
    features = pd.DataFrame(features)

    analyze_dataframe(features = features)



def get_track_analysis(audio_features, features, track_name):
    try:
        if audio_features:
            for key in list(features.keys()):
                if key == 'track_name':
                    features['track_name'].append(track_name)
                else:
                    features[key].append(audio_features[0][key])
    except (TypeError, KeyError, IndexError) as e:
        # Handle the specific exception or print an error message
        print(f"Error: {e}")
        # Optionally, you can log the error or take other actions
    return features


def analyze_dataframe(features):
    print(features.describe().round(3))

    
    features_no_track_name = features.loc[:, ~features.columns.isin(['track_name'])]
    correlations = features_no_track_name.corr()
    print(correlations['danceability'])
    
    # Linear Regression
    X = features.loc[:, ~features.columns.isin(['track_name', 'danceability'])]
    Y = features.loc[:, features.columns.isin(['danceability'])]








# Maybe create a dataset with the names of the tracks and their audio features? Then idk predict what is the genre of the song according to the data

