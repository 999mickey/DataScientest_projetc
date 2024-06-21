from models.mic_filtering_class import mic_hybrid_filtering
import time
import os
import pandas as pd
from input_variables_setter import input_variables_setter
beep = lambda x: os.system("echo -n '\a';sleep 0.2;" * x)
beep(1)

start = time.time()

input_path = '../Data/'

file_name = 'twitter_track_sentiments.csv'
file_name = 'user_and_track_sentiment.csv'
file_name = 'spotify_user_track_norm.csv'
#file_name = "simulation5-10.csv"
file_name = 'spotify_user_track_reduced10.01.csv'

myfilter = mic_hybrid_filtering()
myfilter.set_data(input_path+file_name)
input_variables_setter(file_name,myfilter)
myfilter.clean_columns()

######################spotify_user_track_reduced10.01.csv  171.8 M
#myfilter.data_frame = myfilter.data_frame.sample(frac=0.01,random_state=666)
#"""   
###############################'user_and_track_sentiment.csv' 428.6 M
#on enlève les hashtag majoritaires
"""myfilter.data_frame = myfilter.data_frame[myfilter.data_frame['hashtag'] != 'nowplaying']
myfilter.data_frame.info()
print("hashtag les plus courants => ")
print(myfilter.data_frame["hashtag"].unique())
myfilter.clean_columns()
myfilter.data_frame.info()
global_hashtag = 'punk'
ret = myfilter.get_random_row_from('hashtag',global_hashtag)
##"""
############################'simulation5-10.csv'
"""

"""
##################################################
myfilter. display_tracks_and_users_num()
userId1 = myfilter.get_random_user_id()
outcsv_path = '../Data/CsvResults/'
number_of_line_todump = 10;

from surprise import NormalPredictor
predictor = NormalPredictor()
pred = myfilter.predict(userId1,predictor) 
ret = myfilter.get_full_data(pred,myfilter.item_key_name)
#ret = pred
filename = str(userId1)+"_predNormal_user.csv"
ret.head(number_of_line_todump).to_csv(outcsv_path+filename)
print(ret)
print("Fin des predictions pour l 'utilisateur [",userId1,"]")

"""param_grid ={}
ret = myfilter.predictor_ajustement(predictor,param_grid)
print("paramètres optimaux pour Normal Pred")
print(ret)
print(type(ret))
"""
##########

from surprise import SVD
predictor = SVD()
pred = myfilter.predict(userId1,predictor) 
print("longeur des predictions SVD : ",len(pred))
print(pred)
ret = myfilter.get_full_data(pred,myfilter.item_key_name)
#ret = pred
filename = str(userId1)+"_predSVD_user.csv"
ret.head(number_of_line_todump).to_csv(outcsv_path+filename)
print(ret)

"""
param_grid = {'n_factors': [100,150],
              'n_epochs': [20,25,30],
              'lr_all':[0.005,0.01,0.1],
              'reg_all':[0.02,0.05,0.1]}

ret = myfilter.predictor_ajustement(predictor,param_grid)
print("paramètres optimaux pour Normal Pred")
print(ret)
print(type(ret))"""


stop = time.time()

print("Process took ",stop-start," sec")
#beep permet d avoir un signal sonore à la fin de calculs très longs
beep(5)