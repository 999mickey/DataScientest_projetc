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
file_name = "simulation.csv"
file_name = "simulationcurrent.csv"
#file_name = "simulationbig.csv"
#file_name = "merge.csv"

#file_name = "simulation5-10.csv"
#file_name = "simulation3-5.csv"
file_name = "ratings_flrd.csv"
myfilter = mic_hybrid_filtering()
input_variables_setter(file_name,myfilter)
myfilter.set_data(input_path+file_name)

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
##################################################
myfilter. display_tracks_and_users_num()
#choix 
# de l'utilisateur 
userId1 = myfilter.get_random_user_id()
#userId1 = 'user0'
userId1 = myfilter.default_user

outcsv_path = '../Data/CsvResults/'
number_of_line_todump = 10

print(myfilter.get_user_description(userId1))
##################################### NormalPredictor
from surprise import NormalPredictor
"""predictor = NormalPredictor()
param_grid ={}
ret = myfilter.predictor_ajustement(predictor,param_grid)
print("paramètres optimaux pour Normal Pred")
print(ret)"""
#print("myfilter.best_params_predictor = \n",myfilter.best_params_predictor)
"""print("++++++++++++++Predictions NormalPredictor() pour l 'utilisateur [",userId1,"]")
predictor = NormalPredictor(**myfilter.best_params_predictor)

pred = myfilter.predict(userId1,predictor,5) 
print(pred)


pred = myfilter.predict_with_train_split(userId1,predictor,5) 
print(pred)

print("++++++++++++++Fin des predictions NormalPredictor() pour l 'utilisateur [",userId1,"]")

#"""
#######################################SVD
from surprise import SVD
predictor = SVD()
"""param_grid = {'n_factors': [100,150],
              'n_epochs': [20,25,30],
              'lr_all':[0.005,0.01,0.1],
              'reg_all':[0.02,0.05,0.1]}
"""
param_grid = {'n_factors': [10,15],
                        'n_epochs': [20,25],
                        'lr_all':[0.005,0.1],
                        'reg_all':[0.02,0.1]}

ret = myfilter.predictor_ajustement(predictor,param_grid)
print("paramètres optimaux pour Normal Pred")
print(ret)

nfactor = ret["n_factors"]
nepochs = ret["n_epochs"]
lrall = ret["lr_all"]
regall = ret["reg_all"]


print("++++++++++++++Predictions SVD() pour l 'utilisateur [",userId1,"]  ==>")
#predictor = SVD()
predictor = SVD(**myfilter.best_params_predictor)
pred = myfilter.predict(userId1,predictor,5) 
ret = pred
print(ret)

predictor = SVD(**myfilter.best_params_predictor)
pred = myfilter.predict_with_train_split(userId1,predictor,5) 
ret = pred
print(ret)

print("++++++++++++++Fin des predictions SVD() pour l 'utilisateur [",userId1,"]")
exit()
##############################KNNBasic
"""from surprise.prediction_algorithms.knns import KNNBasic
sim_options = {'name': 'cosine',
               'user_based': False
               }
predictor = KNNBasic(sim_options=sim_options)
pred = myfilter.predict(userId1,predictor) 
ret = pred
print(ret)
print("Fin des predictions KNN() pour l 'utilisateur [",userId1,"]")
"""
####################################BaselineOnly
####################################KNNBaseline
####################################Co-clustering
"""from sklearn.cluster import SpectralCoclustering 
predictor = SpectralCoclustering()
pred = myfilter.predict(userId1,predictor) 
ret = pred
print(ret)
print("Fin des predictions SpectralCoclustering() pour l 'utilisateur [",userId1,"]")
"""

stop = time.time()

print("Process took ",stop-start," sec")
#beep permet d avoir un signal sonore à la fin de calculs très longs
beep(5)