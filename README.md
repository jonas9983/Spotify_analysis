# IMPORTANT
To call the SPOTIFY API a secret client ID is needed 

# Analyze data from the Spotify API

### spotify_main.py 
This script is just to call other functions, needs a change
The get_playlists function carries a parameter = {0,1} that decides whether to update or not the dataframe 

### spotify_utils.py 

 If update_dataframe = 1:
    # connect with the spotify API and collect track properties (track names, track ids, genres,...). 
    # then it either creates or updates the dataframe stored on the data folder


    # PS: talvez possa mudar a change_pandas_variable, nao preciso de chama-la aqui

# else:
    # read the audio_features.csv file on the data folder 
    # analyze this data 

### spotify_analysis.py 
call a function first describes the data and serves to clear the data of outliers, etc
then call a function that shows plots done so far


### use_spotify_api.py 
Kinda messy funcation
It serves to call the spotify API but each function has its output
I don't know if it makes sense to condense all the functions into one

But it needs some work

### Prenda_utils.py

era para aquela cena dos CDs mas ja nao pego nisto ha algum tempo

### dict_tests.py

I usually do some tests and experiments here



