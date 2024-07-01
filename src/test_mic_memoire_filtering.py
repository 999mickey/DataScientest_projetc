from models.mic_filtering_class import mic_collaborativ_filtering
import time
import os
import pandas as pd
from input_variables_setter import input_variables_setter ,getstat_of_object

beep = lambda x: os.system("echo -n '\a';sleep 0.2;" * x)
beep(1)

start = time.time()
outcsv_path = '../Data/CsvResults/'
input_path = '../Data/'
file_name = 'user_and_track_sentiment.csv'
file_name = 'spotify_user_track_reduced10.01.csv'
#file_name = "simulation.csv"
#file_name = "simulation5-10.csv"
file_name = "simulation3-5.csv"
file_name = "simulationcurrent.csv"
#file_name = "ratings_flrd.csv"
#file_name = "merge.csv"

myfilter = mic_collaborativ_filtering()
input_variables_setter(file_name,myfilter)
myfilter.set_data(input_path+file_name)
myfilter.print_vars()
#myfilter.data_frame = myfilter.data_frame.sample(frac=0.01,random_state=666)

myfilter.clean_columns()

###################################################################"#'user_and_track_sentiment.csv'
#Le plus d'occurrence ( 5765350 ) est pour :  ['nowplaying'] !!! on les vire (6053259)
"""myfilter.set_data(input_path+file_name)
getstat_of_object(myfilter.data_frame,'hashtag')
myfilter.data_frame = myfilter.data_frame[myfilter.data_frame['hashtag'] != 'nowplaying']
print(myfilter.data_frame.shape)
print(myfilter.data_frame["hashtag"].unique())
#deathrock' 'postpunk' 'punk' ... 'romanticlove' 'confused' , 'criminallyunderrated'
global_hashtag = 'punk'
ret = myfilter.get_random_row_from('hashtag',global_hashtag)
print("user Id = ",ret)
userId = ret[myfilter.user_key_name]
print(userId)
#"""
#########################################################################""spotify_user_track_reduced0.01.csv'
#file_name = 'spotify_user_track_reduced10.01.csv'
#myfilter.data_frame = myfilter.data_frame.sample(frac=0.01,random_state=666)
#input_path = ''
#file_name = 'currentDfDumped.csv'
#myfilter.set_data(input_path+file_name)
#myfilter.set_target_key_name('sentiment_score')
#myfilter.set_user_key_name('user_id')
#myfilter.set_item_key_name('track_name')
#myfilter.set_item_key_name('name')
#myfilter.set_visual_key_name('playlist')

#myfilter.data_frame.head(10).to_csv('currentDf10Dumped.csv')
#myfilter.set_visual_key_name('name')#pas besoin de visal key name est un la somme du nom de l artist et du ttre
#myfilter.data_frame = myfilter.data_frame.sample(frac=0.01,random_state=666)
#myfilter.clean_columns()
#myfilter.data_frame.info()

###"""
#########################################################################""file_name = 'user_and_track_sentiment.csv'
"""file_name = 'user_and_track_sentiment.csv'
myfilter.set_data(input_path+file_name)
myfilter.set_target_key_name('sentiment_score')
myfilter.set_user_key_name('user_id')
myfilter.set_item_key_name('track_id')
#myfilter.set_item_key_name('name')
myfilter.set_visual_key_name('hashtag')
oldshape = myfilter.data_frame.shape
myfilter.data_frame = myfilter.data_frame[myfilter.data_frame['hashtag'] != 'nowplaying']
#myfilter.data_frame = myfilter.data_frame.sample(frac=0.01,random_state=666)
print("Nouvelle forme après avoir enlever les hashtag 'nowplaying' on passe de ",oldshape," à ",myfilter.data_frame.shape,"")
###"""
###############################"simulation.csv"
#myfilter.set_data(input_path+"simulation.csv")
"""file_name = "simulation5-10.csv"
myfilter.set_data(input_path+file_name)

myfilter.set_target_key_name('sentiment_score')
myfilter.set_user_key_name('user_id')
myfilter.set_item_key_name('track_id')
############################################"""

userIds = myfilter.get_best_voter_users(10)

print("Le user le plus actif est : ")

print(userIds[myfilter.user_key_name].iloc[0])

userId = userIds[myfilter.user_key_name].iloc[0]

userId = myfilter.default_user
#userId = 'user7'
#userId = 'user1'
#userId = '004d5e96c8a318aeb006af50f8cc949c'

myfilter.generate_notation_mattrix()

k_close_param = 3

#myfilter.matrice_pivot.to_csv('CsvResults/'+file_name+"matrice_pivot.csv")
number_of_line_todump = 10
score_seuil = 0

#userId = myfilter.get_random_user_id()

simlilar_users = myfilter.get_similar_users(k_close_param,userId)
#on prend le plus proche
#print("************myfilter.get_similar_users() =>",simlilar_users)
print("Chosen user : ",userId)

print(myfilter.get_user_description(userId))


print("simlilar_users = \n",simlilar_users)


print(myfilter.get_user_description(userId))
user_preferences = myfilter.get_preferences(userId,score_seuil,5)
#print("longeur des preferences : ",len(user_preferences))

#top10 =user_preferences.sort_values(myfilter.target_key_name, ascending=False).head(10)
top = user_preferences.sort_values(myfilter.target_key_name, ascending=False)

print("10 Preferences user [",userId,"] : \n",top)
filename = str(userId)+"_10preference.csv"
print("-----------------ganna save to ",outcsv_path+filename)
top.head(number_of_line_todump).to_csv(outcsv_path+filename)

#on calcule ici une recommandation faite à partir des autres utilisateurs les plus proches 
# dont le nombre est k_close_param

reco_user = myfilter.pred_user(k_close_param,userId,5)
print("userId[",userId,"] reco_user======================",reco_user)
filename = str(userId)+"_pred_user.csv"
#reco_user.head(number_of_line_todump).to_csv(outcsv_path+filename)
ret = myfilter.get_full_data(reco_user,myfilter.item_key_name)
ret.head(number_of_line_todump).to_csv(outcsv_path+filename)

reco_item = myfilter.pred_item( k_close_param,userId,5).sort_values(ascending=False).head(10)
print("userId[",userId,"] reco_item======================",reco_item)
filename = str(userId) + "_pred_item.csv"
#reco_item.head(number_of_line_todump).to_csv('CsvResults/'+filename)
ret = myfilter.get_full_data(reco_item,myfilter.item_key_name)
ret.head(number_of_line_todump).to_csv(outcsv_path+filename)

#################################################
print("User proches :\n",simlilar_users)
closestUserId = simlilar_users.index[0]
print("closestUserId = \n",closestUserId)
print(myfilter.get_user_description(closestUserId))

print(myfilter.get_user_description(closestUserId))
user_preferences = myfilter.get_preferences(closestUserId,score_seuil,100)
top10 = user_preferences.sort_values(myfilter.target_key_name, ascending=False)
print("10 Preferences closest user [",closestUserId,"] : \n",top10)
filename = str(closestUserId)+"_10preference.csv"
top10.head(number_of_line_todump).to_csv(outcsv_path+filename)

reco_user = myfilter.pred_user(k_close_param,closestUserId)
print("userId[",closestUserId,"] reco_user======================",reco_user)
filename = str(closestUserId) + "_pred_user.csv"
reco_user.head(number_of_line_todump).to_csv(outcsv_path+filename)
ret = myfilter.get_full_data(reco_user,myfilter.item_key_name)
ret.head(number_of_line_todump).to_csv(outcsv_path+filename)

reco_item = myfilter.pred_item( k_close_param,closestUserId).sort_values(ascending=False).head(10)
print("userId[",closestUserId,"] reco_item======================",reco_item)
filename = str(closestUserId) + "_pred_item.csv"
reco_item.head(number_of_line_todump).to_csv(outcsv_path+filename)
ret = myfilter.get_full_data(reco_item,myfilter.item_key_name)
ret.head(number_of_line_todump).to_csv(outcsv_path+filename)

#####################"""

pref_1 = myfilter.matrice_pivot.loc[userId, :].values

pref_2 = myfilter.matrice_pivot.loc[closestUserId, :].values

similarity = myfilter.sim_cos(pref_1, pref_2)

print("La similarité entre les deux utilisateurs user1[",userId,"] user2[",closestUserId,"] est ", similarity)

stop = time.time()

print("Process took ",stop-start," sec")
beep(5)