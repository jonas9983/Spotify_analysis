# Client_ID = 5f0a6662bc1f40ea8e84f3a167681c62
# Client ID Secret = 6e73a7e65e504062bf3a1591dd07414e

# Playlist ID = 2H1UmImj2tpiz6TIPSu75M

import spotipy
import sys
from spotipy.oauth2 import SpotifyClientCredentials
import shutil
import os
import spotdl
import datetime
import time
import pandas as pd

from prenda_utils import get_track_properties, change_playlist_order, download_missing_songs, compare_playlists, check_music_duration
from spotify_utils import get_playlist

spotify = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id= "71235600c4ce46f4ad90f99168ebf7ca", client_secret= "11540b15b6b64d7dba3e6cb500a610a1" ))

# I should check for duplicates

def call_functions():
    #download_missing_songs(spotify= spotify, id= "2H1UmImj2tpiz6TIPSu75M", paths = "./sofiii/annie", dl = 0)
    #check_music_duration(spotify= spotify, id= "2H1UmImj2tpiz6TIPSu75M")
    get_playlist(spotify= spotify, id = "6cC2jgTKPDYJGh6Ww9F6ga")

#download_missing_songs(spotify, my_id= "6w2XVDmTtHzURLi3oLV849", dl = 0)
#compare_playlists(annie_id= "2H1UmImj2tpiz6TIPSu75M", my_id= "6w2XVDmTtHzURLi3oLV849")
call_functions()


# This was kind of a flop but I guess I did what I wanted 


# Comparar a musica que falta na pasta custom order com as musicas do my folder
# depois quero passar essa musica para a pasta custom order, com a devida ordem (ou seja modification date)
