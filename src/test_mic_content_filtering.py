
from models.mic_filtering_class import  mic_content_filtering
from input_variables_setter import input_variables_setter
import pandas as pd
import time
import os
import sys
sys.path.append('../DataSelection')
from features.mic_data_selection import mic_data_selection

beep = lambda x: os.system("echo -n '\a';sleep 0.2;" * x)
beep(1)
start = time.time()
myFilter = mic_content_filtering()
###############################

input_path = "../Data/"
file_name = "data.csv"
input_variables_setter(file_name,myFilter)
myFilter.set_data(input_path + file_name)
myFilter.normalize_data()
myFilter.print_vars()
#exit()
song_name = "put a s"
artist_name = "nina sim"


song_name = "give it"
artist_name = "red hot"

#artist_list = myFilter.get_artist_closedto(artist_name)

#print(artist_list)
#exit()

ret = myFilter.get_tracks_with_name(song_name)
print(myFilter.get_tracks_of_artist(artist_name))

songnum = myFilter.get_track_num_of_artist(artist_name,song_name)
print("song num = ",songnum)

#####################"""
"""input_path = "../Data/"
file_name = "final_all.csv"
myFilter.set_data(input_path + file_name)
#definition des variables
input_variables_setter(file_name,myFilter)
#on enlève les hashtag 'nowplaying', surmajoritaires
myFilter.data_frame = myFilter.data_frame[myFilter.data_frame[myFilter.item_key_name] != 'nowplaying']
myFilter.drop_dupicated_tracks() ##pb valeur index
myFilter.normalize_data()
songnum = myFilter.get_random_track_id()
###############################"""

print(myFilter.get_item_description(songnum))
number_to_get = 20
SongId , SongName , ArtistName , Distance = myFilter.content_filter_music_recommender(songnum, number_to_get)

outputdf = pd.DataFrame()
outputdf["song_id"] = SongId
outputdf["artists"] = ArtistName
outputdf["song_name"] = SongName
outputdf["distance"] = Distance


print("OUTPUT ------------------> Artiste et chanson les plus proches : \n")
print(outputdf.head(number_to_get))
outcsv_path = '../Data/CsvResults/'
outputdf.to_csv(outcsv_path+'content_filtering'+'.csv')

stop = time.time()

print("Process took ",stop-start," sec")
beep(5)

