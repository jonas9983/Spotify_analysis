import numpy as np


full_dict = {
    "id_1": {
        "track_name": ["Curly Martin", "Kaa", "Quarter Master"],
        "track_artist": ["Terrace Martin", "Maisha", "Snarky Puppy"],
    },
    "id_2": {
        "track_name": ["Fire Dance", "Ola", "Eu sou o Joao"],
        "track_artist": ["Dizzy Gillespie", "Tiago Nacarato", "LCD Soundsystem"],
    },
    "id_3": {
        "track_name": ["I can love you like that", "Encontro tosco", "Rise up"],
        "track_artist": ["Angela", "Leo", "Um bro dos morangos"],
    },
}



audio_features = {
    "audio_features": [
        {
            "danceability": 0.82,
            "energy": 0.412,
            "key": 7,
            "loudness": -10.235,
            "mode": 1,
            "speechiness": 0.262,
            "acousticness": 0.509,
            "instrumentalness": 0.128,
            "liveness": 0.179,
            "valence": 0.281,
            "tempo": 77.52,
            "type": "audio_features",
            "id": "4eOEhu9eJZ0NMOo5cgK99a",
            "uri": "spotify:track:4eOEhu9eJZ0NMOo5cgK99a",
            "track_href": "https://api.spotify.com/v1/tracks/4eOEhu9eJZ0NMOo5cgK99a",
            "analysis_url": "https://api.spotify.com/v1/audio-analysis/4eOEhu9eJZ0NMOo5cgK99a",
            "duration_ms": 270013,
            "time_signature": 4,
        },
        {
            "danceability": 0.729,
            "energy": 0.536,
            "key": 0,
            "loudness": -10.397,
            "mode": 1,
            "speechiness": 0.114,
            "acousticness": 0.0695,
            "instrumentalness": 8.15e-05,
            "liveness": 0.118,
            "valence": 0.767,
            "tempo": 97.478,
            "type": "audio_features",
            "id": "4Wt9YQUTQI4Nybbw3APZt0",
            "uri": "spotify:track:4Wt9YQUTQI4Nybbw3APZt0",
            "track_href": "https://api.spotify.com/v1/tracks/4Wt9YQUTQI4Nybbw3APZt0",
            "analysis_url": "https://api.spotify.com/v1/audio-analysis/4Wt9YQUTQI4Nybbw3APZt0",
            "duration_ms": 248939,
            "time_signature": 4,
        },
        {
            "danceability": 0.698,
            "energy": 0.662,
            "key": 7,
            "loudness": -7.997,
            "mode": 0,
            "speechiness": 0.0861,
            "acousticness": 0.567,
            "instrumentalness": 1.44e-06,
            "liveness": 0.0876,
            "valence": 0.815,
            "tempo": 141.666,
            "type": "audio_features",
            "id": "6mZI2vbLv1UvlclwDQ4uvc",
            "uri": "spotify:track:6mZI2vbLv1UvlclwDQ4uvc",
            "track_href": "https://api.spotify.com/v1/tracks/6mZI2vbLv1UvlclwDQ4uvc",
            "analysis_url": "https://api.spotify.com/v1/audio-analysis/6mZI2vbLv1UvlclwDQ4uvc",
            "duration_ms": 283701,
            "time_signature": 4,
        },
    ]
}

audio_features_2 = {
    "danceability": 0.698,
    "energy": 0.662,
    "key": 7,
    "loudness": -7.997,
    "mode": 0,
    "speechiness": 0.0861,
    "acousticness": 0.567,
    "instrumentalness": 1.44e-06,
    "liveness": 0.0876,
    "valence": 0.815,
    "tempo": 141.666,
    "type": "audio_features",
    "id": "6mZI2vbLv1UvlclwDQ4uvc",
    "uri": "spotify:track:6mZI2vbLv1UvlclwDQ4uvc",
    "track_href": "https://api.spotify.com/v1/tracks/6mZI2vbLv1UvlclwDQ4uvc",
    "analysis_url": "https://api.spotify.com/v1/audio-analysis/6mZI2vbLv1UvlclwDQ4uvc",
    "duration_ms": 283701,
    "time_signature": 4,
}

features = {
    "track_name": [],
    "track_artist": [],
    "track_id": [],
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

playlist_id = [
    "6w2XVDmTtHzURLi3oLV849",
    "53Z2byTaum9E3fhu2tKtm2",
    "5sK1legJCslY7BkNShIZ2w",
]

max_ids = 50
nr_track_ids = 102
num_batches = (nr_track_ids + max_ids - 1) // max_ids

print(num_batches)

#batches = [track_ids[i * max_ids : (i + 1) * max_ids] for i in range(num_batches)]



def flatten_dictionary(dict_to_flatten):
    flattened_dict = {
        key: [] for key in list(dict_to_flatten[next(iter(dict_to_flatten))].keys())
    }  # The list(dict_to_flatten[next(iter(dict_to_flatten))].keys()) gives me the first index of the dictionary
    for i, inner_dict in enumerate(full_dict.values()):
        for idx, subkey in enumerate(inner_dict):
            flattened_dict[subkey].extend(inner_dict[subkey])

    return flattened_dict


def get_track_analysis(track_ids):
    max_ids = 100
    nr_track_ids = len(track_ids)
    num_batches = (nr_track_ids + max_ids - 1) // max_ids
    batches = [track_ids[i * max_ids : (i + 1) * max_ids] for i in range(num_batches)]

    for i, batch in enumerate(batches, 1):
        str_track_ids = "%".join(batch)
        print(str_track_ids)


def get_features(audio_features, features):
    # Quero colocar os valores que estão em audio_features em features
    for i, val in enumerate(audio_features["audio_features"]):
        print(val)
        for key in features.keys():
            if key in val.keys():
                features[key].append(val[key])
    print(features)


def iterate_playlist_id(playlist_id):
    for i, idx in enumerate(playlist_id):
        if i in playlist_id.index():
            print(i, idx)


#iterate_playlist_id(playlist_id=playlist_id)
# get_features(audio_features= audio_features, features = features)
# get_track_analysis(['4eOEhu9eJZ0NMOo5cgK99a']*101)
