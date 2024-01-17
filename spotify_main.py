# Client_ID = 5f0a6662bc1f40ea8e84f3a167681c62
# Client ID Secret = 6e73a7e65e504062bf3a1591dd07414e

# Playlist ID = 2H1UmImj2tpiz6TIPSu75M

import spotipy
import sys
from spotipy.oauth2 import SpotifyClientCredentials
import base64
from requests import post, get
import os
import json


from prenda_utils import get_track_properties, download_missing_songs, check_music_duration
from spotify_utils import get_playlist

spotify = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id= "71235600c4ce46f4ad90f99168ebf7ca", client_secret= "11540b15b6b64d7dba3e6cb500a610a1" ))



def call_functions():
    #download_missing_songs(spotify= spotify, id= "2H1UmImj2tpiz6TIPSu75M", paths = "./sofiii/annie", dl = 0)
    #check_music_duration(spotify= spotify, id= "37i9dQZF1DZ06evO0P3UNG")
    get_playlist(update_dataframe = 0) # Mix de jazz https://open.spotify.com/playlist/37i9dQZF1EQqA6klNdJvwx

call_functions()
# This was kind of a flop but I guess I did what I wanted 
# No it wasn't

# Most efficient code possible


