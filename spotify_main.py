from prenda_utils import (
    get_track_properties,
    download_missing_songs,
    check_music_duration,
)
from spotify_utils import get_playlist


def call_functions():
    # download_missing_songs(spotify= spotify, id= "2H1UmImj2tpiz6TIPSu75M", paths = "./sofiii/annie", dl = 0)
    # check_music_duration(spotify= spotify, id= "37i9dQZF1DZ06evO0P3UNG")
    get_playlist(
        update_dataframe=1
    )  # Mix de jazz https://open.spotify.com/playlist/37i9dQZF1EQqA6klNdJvwx


call_functions()
# This was kind of a flop but I guess I did what I wanted
# No it wasn't

# Most efficient code possible
