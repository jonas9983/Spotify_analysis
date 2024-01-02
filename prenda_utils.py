import shutil
import datetime
import time
import os
import pandas as pd
import math

# Code is not perfect:
# - I need to 

def move_to_file(folder_track_names, origin_path, destination_path, mv = 0):
    path_track_names = folder_track_names + ".mp3"
    if mv == 1:
        idx = os.listdir(origin_path).index(path_track_names)
        music_path = os.listdir(origin_path)[idx]
        shutil.copy(os.path.join(origin_path,music_path), destination_path)

def check_music_duration(spotify, id):
    playlist = spotify.playlist_tracks(id)
    duration = {'track_name': [], 'duration': []}
    s = 0
    for i in range(playlist['total']):
        duration_ms = playlist['items'][i]['track']['duration_ms']
        if s + duration_ms > 4400000:
            print(duration)
            print(len(duration['track_name']))
            print(playlist['items'][i]['track']['name'])
            print(sum(duration['duration']))
            
            duration['track_name'] = [playlist['items'][i]['track']['name']]
            duration['duration'] = [duration_ms]
            s = duration_ms
            
        else: 
            duration['track_name'].append(playlist['items'][i]['track']['name'])
            duration['duration'].append(duration_ms)
            s += duration_ms
    print(duration)
    print(len(duration['track_name']))
    if duration['track_name']:
        print(duration['track_name'][-1])
    print(sum(duration['duration']))


def change_playlist_order(playlist_track_names, folder_track_names, paths, rename = 0):
    # I still have to order according to the spotify playlist. I should find a way to this automatically

    destination_path = paths + "_custom_order"

    # Nao preciso de correr duas vezes a função compare_playlists

    missing_track = compare_playlists(playlist_1= "./sofiii/annie", playlist_2= "./sofiii/annie_custom_order")

    if missing_track != []:
        for i, name in enumerate(missing_track['name']):
            move_to_file(folder_track_names = name, origin_path = paths, destination_path= destination_path, mv = 0)

    for i, name in enumerate(os.listdir(destination_path)):
        _, _ , track_name = get_track_properties(name, playlist= str())
        if rename == 1:
            idx = playlist_track_names.index(track_name)
            print(idx)
            old_path = os.path.join(destination_path, name)
            new_path = os.path.join(destination_path, str(idx + 1) + "_" + name)
            os.rename(old_path, new_path)

def download_missing_songs(spotify, id, paths, dl = 0):
    # Ler o id da playlist, comparar as musicas que estão na minha pasta e verificar qual é que esta a faltar
    folder_names = []
    music_dict = {'full_music_name': [], 'track_name': []}

    my_playlist = spotify.playlist_tracks(id)    
    playlist_properties = get_track_properties(track = str(), playlist = my_playlist)


    for file in os.listdir(paths):
        full_music_name, _ , track_name =  get_track_properties(track = file, playlist = str())
        music_dict['full_music_name'].append(full_music_name)
        music_dict['track_name'].append(track_name)
    folder_names = music_dict['track_name']

    missing_tracks, idx = compare_playlists(playlist_1 = playlist_properties['track_name'], playlist_2 = folder_names)
    print(missing_tracks)

    if dl == 1:
        os.chdir(paths)
        for i, track in enumerate(missing_tracks):
            os.system('spotdl %s ' % playlist_properties['url'][idx[i]])
    elif dl == 2:
        os.chdir(paths)
        os.system('spotdl download https://open.spotify.com/playlist/2H1UmImj2tpiz6TIPSu75M')

    change_playlist_order(playlist_track_names = playlist_properties['track_name'], folder_track_names = music_dict, paths = paths, rename = 0)


def compare_playlists(playlist_1, playlist_2):
    # Compare two playlists from folders
    # Compare two playlists from folder and spotify
    # Compare two playlists from spotify 

    if isinstance(playlist_1, str) and isinstance(playlist_2, str):
        missing_tracks = {'full_file_name': [], 'name': [], 'artist_name': [], 'track_name': []}
        top = os.listdir(playlist_1) if len(os.listdir(playlist_1)) > len(os.listdir(playlist_2)) else os.listdir(playlist_2)
        bot = os.listdir(playlist_2) if len(os.listdir(playlist_1)) > len(os.listdir(playlist_2)) else os.listdir(playlist_1)
        for track in top:
            if track not in bot:
                full_music_name, artist_name, track_name = get_track_properties(track, playlist = str())
                missing_tracks['full_file_name'].append(track)
                missing_tracks['name'].append(full_music_name)
                missing_tracks['artist_name'].append(artist_name)
                missing_tracks['track_name'].append(track_name)
        return missing_tracks

    elif type(playlist_1) == list and type(playlist_2) == list:
        top = playlist_1 if len(playlist_1) > len(playlist_2) else playlist_2
        bot = playlist_2 if len(playlist_1) > len(playlist_2) else playlist_1
        missing_tracks = []
        idx = []

        for i, track in enumerate(top):
            if track not in bot:
                missing_tracks.append(track)
                idx.append(i)
        return(missing_tracks, idx)
    
def get_track_properties(track = str(), playlist = str()):
    if track != str():
        full_music_name = os.path.splitext(track)[0]
        artist_name = full_music_name.split("-", 1)[0].strip()        
        track_name = full_music_name.split("-", 1)[1].strip()
        return(full_music_name, artist_name, track_name)   
    elif playlist != str():
        track_dict = {'track_name': [], 'track_id': [], 'url': []}
        for i in range(playlist['total']):
            track_dict['track_name'].append(playlist['items'][i]['track']['name'])
            track_dict['track_id'].append(playlist['items'][i]['track']['id'])
            track_dict['url'].append(playlist['items'][i]['track']['external_urls']['spotify'])

        return(track_dict)


