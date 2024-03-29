import base64
from requests import post, get
import os
import json


client_id = "7ffc843a14b947d28f600dcc67c6c1d4"
client_secret = ""


def get_token():
    auth_string = client_id + ":" + client_secret
    auth_bytes = auth_string.encode("utf-8")
    auth_base64 = str(base64.b64encode(auth_bytes), "utf-8")

    url = "https://accounts.spotify.com/api/token"
    headers = {
        "Authorization": "Basic " + auth_base64,
        "Content-Type": "application/x-www-form-urlencoded",
    }
    data = {"grant_type": "client_credentials"}

    result = post(url, headers=headers, data=data)
    json_result = json.loads(result.content)
    token = json_result["access_token"]
    return token


def get_auth_header(token):
    return {"Authorization": "Bearer " + token}


def search_for_artist(token, artist_name):
    url = "https://api.spotify.com/v1/search"
    headers = get_auth_header(token)
    query = f"?q={artist_name}&type=artist&limit=1"

    query_url = url + query
    result = get(query_url, headers=headers)
    json_result = json.loads(result.content)["artists"]["items"]
    if len(json_result) == 0:
        print("No artist with this name exists")
        return None
    return json_result[0]


def get_songs_by_artist(token, artist_id):
    url = f"https://api.spotify.com/v1/artists/{artist_id}/top-tracks?country=US"
    headers = get_auth_header(token)
    result = get(url, headers=headers)
    json_result = json.loads(result.content)["tracks"]
    return json_result


def get_track_information(artist_id):
    token = get_token()
    url = f"https://api.spotify.com/v1/artists/{artist_id}"
    headers = get_auth_header(token)
    result = get(url, headers=headers)
    json_result = json.loads(result.content)
    return json_result


def get_saved_songs(token):
    url = f"https://api.spotify.com/v1/me/tracks?market=US&offset=0&limit=50"
    headers = get_auth_header(token)
    result = get(url, headers=headers)
    json_result = json.loads(result.content)
    return json_result


def get_user_playlists(token, user_id, offset = 0, limit=50):
    if offset == "":
        url = f"https://api.spotify.com/v1/users/{user_id}/playlists?limit=50"
    else:
        url = f"https://api.spotify.com/v1/users/{user_id}/playlists?limit=50&offset={offset}" 
    headers = get_auth_header(token)
    result = get(url, headers=headers)
    json_result = json.loads(result.content)
    return json_result


def get_playlist_items(playlist_id, query):
    token = get_token()
    url = f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks{query}"
    headers = get_auth_header(token)
    result = get(url, headers=headers)
    json_result = json.loads(result.content)
    return json_result


def run_api_request(url, query):
    if query == "":
        query_url = url
    else:
        query_url = url + query
    token = get_token()
    headers = get_auth_header(token)
    result = get(query_url, headers=headers)
    json_result = json.loads(result.content)

    return json_result


token = get_token()

"""token = get_token()
result = search_for_artist(token, "ACDC")
print(result)
artist_id = result['id']
songs = get_songs_by_artist(token, artist_id= artist_id)

for idx, song in enumerate(songs):
    print(f"{idx + 1}. {song['name']}")"""
