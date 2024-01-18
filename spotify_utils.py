# Quero analisar a playlist com as minhas liked songs
# Vou começar com uma playlist mais pequena tipo a do House4Pitex (id = 6cC2jgTKPDYJGh6Ww9F6ga)

# I need more data

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import os

from prenda_utils import get_track_properties
from use_spotify_api import (
    get_saved_songs,
    get_user_playlists,
    get_token,
    run_api_request,
    get_track_information,
)
from spotify_analysis import analyze_dataframe


def get_playlist(update_dataframe=0):
    dataframe_path = "./data/audio_features.csv"
    user_ids = [
        "11127927763",
        "anaserrogomes",
        "31k27bhdk4puklqlfhgqxszgvxqy",
    ]  # Meu, Ana, Pitex
    playlist_properties = {}
    if update_dataframe == 1:
        token = get_token()
        # result = get_saved_songs(token) Nao funciona nao sei porque
        all_playlists = get_user_playlists(
            token=token, user_id="31k27bhdk4puklqlfhgqxszgvxqy"
        )
        playlists_id = get_playlists_id(all_playlists)
        playlist_properties = get_playlist_tracks(
            all_playlists=all_playlists,
            playlists_id=playlists_id,
            playlist_properties=playlist_properties,
        )
        flattened_playlist_properties = flatten_dictionary(playlist_properties)
        playlist_properties_pd = change_panda_variables(
            flattened_playlist_properties, dance_thresh=0.6
        )
        print(playlist_properties_pd)
        save_json_dataframe(
            dataframe=playlist_properties_pd, filename=dataframe_path, save=1
        )
    else:
        features = pd.read_csv(dataframe_path)
        print("else")
        analyze_dataframe(features=features, filename=dataframe_path, graph=0)


def get_playlist_tracks(all_playlists, playlists_id, playlist_properties):
    for i, idx in enumerate(playlists_id):
        print(i, idx)
        features = {
            "track_name": [],
            "track_artist": [],
            "track_id": [],
            "user_id": [],
            "playlist_id": [],
            "genres": [],
            "danceability": [],
            "energy": [],
            "key": [],
            "loudness": [],
            "speechiness": [],
            "acousticness": [],
            "instrumentalness": [],
            "liveness": [],
            "valence": [],
            "tempo": [],
        }
        playlist = run_api_request(
            all_playlists["items"][i]["tracks"]["href"], query=""
        )
        for j in range(playlist["total"]):
            if j == 50:
                playlist = run_api_request(
                    all_playlists["items"][i]["tracks"]["href"], query=f"?offset={j}"
                )
            genres = get_track_information(
                playlist["items"][j % 50]["track"]["album"]["artists"][0]["id"]
            )["genres"]
            features["track_name"].append(playlist["items"][j % 50]["track"]["name"])
            features["track_artist"].append(
                playlist["items"][j % 50]["track"]["album"]["artists"][0]["name"]
            )
            features["track_id"].append(playlist["items"][j % 50]["track"]["id"])
            features["user_id"].append(all_playlists["items"][i]["owner"]["id"])
            features["playlist_id"].append(all_playlists["items"][i]["id"])
            if genres:
                features["genres"].append(genres)
            else:
                features["genres"].append("No Genre")
        features = get_audio_features(features["track_id"], features)
        playlist_properties[idx] = features
    return playlist_properties


def get_audio_features(track_ids, features):
    # Since audio features are in a list I have to iterate through the list to get the audio features for each song.
    max_ids = 100
    nr_track_ids = len(track_ids)
    num_batches = (nr_track_ids + max_ids - 1) // max_ids
    batches = [track_ids[i * max_ids : (i + 1) * max_ids] for i in range(num_batches)]
    for i, batch in enumerate(batches, 1):
        str_track_ids = ",".join(batch)
        url = f"https://api.spotify.com/v1/audio-features?ids={str_track_ids}"
        try:
            audio_features = run_api_request(url, query="")["audio_features"]
            if audio_features:
                for j, feats in enumerate(audio_features):
                    for key in features.keys():
                        if key in feats.keys():
                            features[key].append(feats[key])
        except (TypeError, KeyError, IndexError) as e:
            # Handle the specific exception or print an error message
            print(f"Error: {e}")
            # Optionally, you can log the error or take other actions

    return features


def flatten_dictionary(dict_to_flatten):
    flattened_dict = {
        key: [] for key in list(dict_to_flatten[next(iter(dict_to_flatten))].keys())
    }  # The list(dict_to_flatten[next(iter(dict_to_flatten))].keys()) gives me the first index of the dictionary
    for i, inner_dict in enumerate(dict_to_flatten.values()):
        for idx, subkey in enumerate(inner_dict):
            flattened_dict[subkey].extend(inner_dict[subkey])
    return flattened_dict


def change_panda_variables(features, dance_thresh):
    features = pd.DataFrame(features)
    features["danceability_binary"] = features["danceability"].apply(
        lambda x: int(1) if x >= dance_thresh else int(0)
    )
    features["key"] = features["key"].apply(
        lambda x: (
            "C"
            if x == 0
            else "C#"
            if x == 1
            else "D"
            if x == 2
            else "D#"
            if x == 3
            else "E"
            if x == 4
            else "F"
            if x == 5
            else "F#"
            if x == 6
            else "G"
            if x == 7
            else "G#"
            if x == 8
            else "A"
            if x == 9
            else "A#"
            if x == 10
            else "B"
            if x == 11
            else "Unknown"
        )
    )

    return features


def save_json_dataframe(dataframe, filename, save=0):
    if save == 1:
        # Check if the folder exists
        if not os.path.exists(filename) or os.stat(filename).st_size == 0:
            dataframe.to_csv(filename, index=False)
        else:  # Caso contrario vamos adicionar à pasta
            dataframe.to_csv(filename, mode="a", index=False, header=False)


def get_playlists_id(playlists):
    playlists_id = []
    playlists_name = []
    for idx, key in enumerate(playlists["items"]):
        playlists_id.append(playlists["items"][idx]["id"])
        playlists_name.append(playlists["items"][idx]["name"])
    print(playlists_name)
    return playlists_id
