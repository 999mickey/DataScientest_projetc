
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import io
import time
import sys,os
import json

#sys.path.append(os.path.realpath('..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..',''))

#from models.mic_filtering_class import  mic_content_filtering ,mic_base_filter , mic_hybrid_filtering, mic_collaborativ_filtering
from mic_filtering_class import  mic_content_filtering ,mic_base_filter , mic_hybrid_filtering, mic_collaborativ_filtering
from input_variables_setter import input_variables_setter , list_files_recursive 
from features.mic_data_selection import mic_data_selection
from visualization.visualize import mic_vizualizer
from fire_state import create_store, form_update , set_state , get_state

from surprise import SVD
from surprise import NormalPredictor



###spéciifc display
from display_user_selection import display_user_selection , display_nb_pres_selection,display_predictors , display_nb_pres

print("os.getcwd() =-> ",os.getcwd())


#
# content_path = '../../Data'
content_path = './Data/'


slot = "home_page"

def get_session_state(key):
    if key in st.session_state:
        return st.session_state[key]
    return None

#load_count , file_name, song_name , artist_name , user_id ,my_data , my_visualizer , my_content_filter = create_store(slot, [        
create_store(slot, [        
    ("load_count", 0),        
    ("file_name", ''),        
    ("close_song_name",''),
    ("close_artist_name",''),
    ("true_song_name",''),
    ("true_artist_name",''),
    ("user_id",''),

    ("mic_data", None)  ,  
    ("mic_viz",None)    ,
    ("mic_content_filtering",None),
    ("mic_collaborativ_filtering",None),
    ("mic_hybrid_filtering",None),
    ("num_user_to_present",5)   , 
    ("predictor",None)
])
def init():
    set_state(slot, ("file_name", ''))
    set_state(slot, ("close_song_name", ''))
    set_state(slot, ("close_artist_name",''))
    set_state(slot, ("true_song_name",''))
    set_state(slot, ("true_artist_name",''))
    set_state(slot, ("user_id", ''))
    set_state(slot, ("mic_data", None))
    set_state(slot, ("mic_viz", None))
    set_state(slot, ("mic_content_filtering", None))
    set_state(slot, ("mic_collaborativ_filtering", None))
    set_state(slot, ("mic_hybrid_filtering", None))
    set_state(slot, ("num_user_to_present", 5))
    set_state(slot, ("predictor", None))

load_count = get_state(slot,"load_count")
current_filename = get_state(slot,"file_name")

user_id = get_state(slot,"user_id")
my_data = get_state(slot,"mic_data")
my_visualizer = get_state(slot,"mic_viz")
my_content_filter = get_state(slot,"mic_content_filtering")
my_collaborativ_filtering = get_state(slot,"mic_collaborativ_filtering") 
my_hybrid_filtering = get_state(slot,"mic_hybrid_filtering")
def printvar():
    print('true_artist  = ',true_artist_name)
    print('true_song  = ',true_song_name)
    print('close_artist  = ',close_artist_name)
    print('close_song  = ',close_song_name)


print("-------------->load_count => ", load_count)        

print("--------------------------------file_name >> ", current_filename)        

load_count = get_state(slot,"load_count")

load_count += 1
set_state(slot ,("load_count",load_count))

#micdata = mic_data_selection()
if get_state(slot,'mic_data') == None :
    print("mic_base_filter allocation")
    my_data = mic_base_filter()
    set_state(slot, ("mic_data", my_data))

if get_state(slot,'mic_viz') == None :
    print("mic_vizualizer_filter allocation")
    my_visualizer = mic_vizualizer()
    set_state(slot, ("mic_viz", my_visualizer))

if get_state(slot,'mic_content_filtering') == None :
    print("mic_content_filtering_filter allocation")
    my_content_filter = mic_content_filtering()    
    set_state(slot, ("mic_content_filtering", my_content_filter))

if get_state(slot,'mic_collaborativ_filtering') == None :
    print("mic_collaborativ_filtering allocation")
    my_collaborativ_filtering = mic_collaborativ_filtering()    
    set_state(slot, ("mic_collaborativ_filtering", my_collaborativ_filtering))
    
if get_state(slot,'mic_hybrid_filtering') == None :
    print("mic_hybrid_filtering allocation")
    my_hybrid_filtering = mic_hybrid_filtering()    
    set_state(slot, ("mic_hybrid_filtering", my_hybrid_filtering))
       
@st.cache_data
def load_data(name):    
    print("------------------------------------------------------------------<load_data(",name,") called")
    input_variables_setter(name,my_data)
    input_variables_setter(name,my_visualizer)
    input_variables_setter(name,my_content_filter)
    input_variables_setter(name,my_collaborativ_filtering)
    input_variables_setter(name,my_hybrid_filtering)
    
    with st.spinner('lecture des données...'):
        df = pd.read_csv(content_path + name)
        my_data.data_frame = df
       
        my_content_filter.data_frame = df

        if len(my_content_filter.features ):
            my_content_filter.normalize_data()
           
        my_visualizer.data_frame = df
        my_collaborativ_filtering.data_frame = df
        my_hybrid_filtering.data_frame = df        

        st.success('Chargement effectué!')                
        #init()
        set_state(slot, ("file_name", name))

    
    return df

current_filename = get_state(slot,"file_name")
df = pd.DataFrame()

print("Projet de Recommandation8 Musicale avec ["+str(current_filename)+"]")

pages=["Intro","Choisir les données", "Visualization des données", "Filtrage Contenu","Filtrage Memoire ","Filtrage Hybride "]
st.title("Projet de Recommandation8 Musicale avec ["+str(current_filename)+"]")
st.sidebar.title("Fichier selectionné : "+current_filename)

page=st.sidebar.radio("Aller vers", pages)

print("---page",page)

if page == pages[0] : 
    st.write("### Introduction mic666 test de Recommantions Musicales")
    st.write("     ---  Le Filtrage Contenu nécessite les valeurs intrinsèques des morceaux musiques \
    telles que 'liveness', 'speechiness', 'danceability', 'valence', 'loudness', 'tempo', 'acousticness','energy', 'mode', 'key', 'instrumentalness'  ")
    st.write("     ---  Le Filtrage Mémoire et Hybride nécessite des identifiants d'utilisateurs et de chasons ainsi que des scores attribués aux morceaux ")
            
    
if page == pages[1] : 
    st.write("### Choisir des données")
    print("### Choisir des données")
    datas=["merge.csv","data.csv","simulation.csv","simulationcurrent.csv"]
    #datas = list_files_recursive(content_path)
    datas.insert(0,'')
    
    option = st.selectbox('Choix des données', datas,key="data_name")
        
    if current_filename != option and option != '': 

        df = load_data(option)        
        current_filename = option
        st.write('Le fichier choisi est :', option)                    
        st.write("Forme :",df.shape)
        buffer = io.StringIO()
        dfinfo = pd.DataFrame(df.info())
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text("Info()")
        st.text(s)
        #st.text(dfinfo.head())
        st.text("Head()")
        st.dataframe(df.head())
        st.text("Describe()")
        st.dataframe(df.describe())        
       
    else :
        df = my_data.data_frame    

if page == pages[2] :
    st.write("### Visualisation des données ",str(current_filename))
    if current_filename != '':
        print("### Data Vizualisation current_filename = ",current_filename)
        input_variables_setter(current_filename,my_visualizer)
        my_visualizer.set_data(content_path +  current_filename)
        #myfilter.data_frame = micdata.data_frame
        print('target are : ',my_visualizer.target_key_name)
        print('features are : ',my_visualizer.features)                
        
        if st.checkbox("Présenter la Heatmap",key="check_heatmap") :
            with st.spinner('Calcul en cours...'):
                fig = my_visualizer.plot_heatmap()
                st.pyplot(fig)

        checktitle = "Présenter le Distribution de "+my_visualizer.target_key_name
        if st.checkbox(checktitle,key="dist_target") :
            with st.spinner('Calcul en cours...'):
                plt.title("Distribution de "+my_visualizer.target_key_name)
                st.pyplot(my_visualizer.plot_repartion_count_target(my_visualizer.target_key_name))

        if st.checkbox("Cercle de corrélation",key="corr_target") :
            with st.spinner('Calcul en cours...'):
                plt.title("Cercle de corrélation ")
                st.pyplot(my_visualizer.plot_corrrelation_circle())

        title = "répartition par "+ my_visualizer.target_key_name
        
        if st.checkbox(title,key="rep_target") :
            with st.spinner('Calcul en cours...'):
                plt.title(title)
                st.pyplot(my_visualizer.plot_groupement_from_key(my_visualizer.target_key_name))            

    else:
        print("no filename selectionned")

if page == pages[3]:

    if len(my_content_filter.features) == 0:
        st.write("### Filtrage contenu avec ",str(current_filename), "impossible (valeurs musicales intrinsèques absentes )" )
    elif current_filename != '':
        st.write("### Filtrage contenu avec ",str(current_filename))
        if st.checkbox("Nettoyer les noms",key="clean_name") :
            true_artist_name = ''
            true_song_name = ''
            close_artist_name = ''
            close_song_name = ''
            set_state(slot, ("close_song_name", ''))
            set_state(slot, ("close_artist_name",''))
            set_state(slot, ("true_song_name",''))
            set_state(slot, ("true_artist_name",''))
            #st.session_state['clean_name'] = False

        else:    
            true_artist_name = get_state(slot,'true_artist_name')
            true_song_name = get_state(slot,'true_song_name')
            close_artist_name = get_state(slot,'close_artist_name')
            close_song_name = get_state(slot,'close_song_name')
      
            #choix approximamtif
            if close_artist_name == '' and close_song_name == '':    
                with  st.form("close artist and song"):                
                    close_artist_name = st.text_area('choisir le nom d atiste approximativement',key='close_artist_name')    
                    close_song_name = st.text_area('choisir le nom de chanson approximativement',key='close_song_name')    
                    st.form_submit_button(label="Valider", on_click=form_update, args=(slot,))        
            #choix de l'artiste
            if close_artist_name != '':
                with  st.form("artist "):                
                    artist_list = my_content_filter.get_artist_closedto(close_artist_name)                
                    artist_list.insert(0,'')
                    true_artist_name = st.selectbox('Choix de l artiste', artist_list,key="artist_name")
                    st.form_submit_button(label="Valider artist", on_click=form_update, args=(slot,))        
            #choix du morceau
            if true_artist_name != '':
                with  st.form("song "):                                               
                    song_list = my_content_filter.get_artist_song_closedto(true_artist_name,close_song_name)
                    song_list.insert(0,'')                
                    true_song_name = st.selectbox('Choix de la chanson', song_list,key="song_name")
                    st.form_submit_button(label="Valider song", on_click=form_update, args=(slot,))        
            #caclul des mtoceaux les plus proches
            if true_artist_name != '' and true_song_name != '':         

                ltext = 'Choix du nom d atiste : '+ true_artist_name
                st.text(str(ltext))    
                ltext = 'Choix du nom de chanson : ' + true_song_name
                st.text(str(ltext))    
                #print("my_content_filter.artist_key_name =",my_content_filter.artist_key_name)
                #print(my_content_filter.data_frame[my_content_filter.artist_key_name].head(10))
                #num = display_nb_pres(10)
                num = 10

                        
                songnum = my_content_filter.get_track_num_of_artist(true_artist_name,true_song_name)                
                
                print('--------------- songnum = ',songnum)
                with st.spinner('Wait for compute tracks...'):
                    SongId , SongName , ArtistName , Distance = my_content_filter.content_filter_music_recommender(songnum, num)
                st.success('compute tracks Done!')                

                outputdf = pd.DataFrame()
                outputdf["song_id"] = SongId
                outputdf["artists"] = ArtistName
                outputdf["song_name"] = SongName
                outputdf["distance"] = Distance
                print(outputdf.head(num))
                st.write('Chanson le splus proches :')                    
                st.dataframe(outputdf.head(10))    
                            
    else :
        print('no filename selected')

if page == pages[4] : 
    if my_content_filter.user_key_name     == '':
        st.write("### Filtrage mémoire avec ",str(current_filename), "impossible (valeurs utilisateur absentes )" )
    else :
        st.write("### Filtrage mémoire ",str(current_filename))
        
        if current_filename == '':
            st.text("Il faut selectionner un fichier !")    
        else :    

            my_collaborativ_filtering.data_frame.info()    
            my_collaborativ_filtering.clean_columns()
            userId =  display_user_selection(my_collaborativ_filtering)

            print("Utilisateur selectionné => ",userId)        
            #end of utlisateur
            if userId != '':
                k_close_param , npref , npreditem , npreduser = display_nb_pres_selection()

                st.text("Utilsateur "+userId)
                #print(my_collaborativ_filtering.get_user_description(userId))
                st.text(my_collaborativ_filtering.get_user_description(userId))

                print(my_collaborativ_filtering.get_user_description(userId))
                
                score_seuil = 0
                my_collaborativ_filtering.generate_notation_mattrix()
                simlilar_users = my_collaborativ_filtering.get_similar_users(k_close_param,userId)
                st.text("Utilsateurs similaires ")
                st.dataframe(simlilar_users)    

                st.text("Préférences ")            
                user_preferences = my_collaborativ_filtering.get_preferences(userId,score_seuil,npref)
                top = user_preferences.sort_values(my_collaborativ_filtering.target_key_name, ascending=False)
                st.dataframe(top)    

                st.text("Prédiction utillisateur :")
                reco_user = my_collaborativ_filtering.pred_user(k_close_param,userId,npreduser)
                st.dataframe(reco_user)    

                st.text("Prédiction item :")
                reco_item = my_collaborativ_filtering.pred_item( k_close_param,userId,npreditem).sort_values(ascending=False).head(npreditem)
                st.dataframe(reco_item)    

if page == pages[5] : 
    if my_content_filter.user_key_name     == '':
        st.write("### Filtrage hybride avec ",str(current_filename), "impossible (valeurs utilisateur absentes )" )
    else :
        st.write("### Filtrage hybride ",str(current_filename))
        #my_hybrid_filtering
        my_hybrid_filtering.generate_dateset_autofold()
        my_hybrid_filtering.data_frame.info()    
        my_hybrid_filtering.clean_columns()
        ispredictor_trained = False
        predictor , predictor_name = display_predictors()

        if predictor != None:
            print("predictor = ",predictor_name)
            ltext = "Predicteur choisi :"+predictor_name
            st.text(ltext)
            if predictor != None :
        
                if st.button("Entrainer le modèle"):
                    if predictor_name == 'SVD()':
                        #param_grid = {'n_factors': [100,150],
                        #    'n_epochs': [20,25,30],
                        #    'lr_all':[0.005,0.01,0.1],
                        #    'reg_all':[0.02,0.05,0.1]}
                        #param_grid = {'n_factors': [10,15],
                        #    'n_epochs': [20,25],
                        #    'lr_all':[0.005,0.1],
                        #    'reg_all':[0.02,0.1]}
                        param_grid = {}
                        with st.spinner('Calcul en cours...'):
                            ret = my_hybrid_filtering.predictor_ajustement(predictor,param_grid)
                            #nfactor = ret["n_factors"]
                            #nepochs = ret["n_epochs"]
                            #lrall = ret["lr_all"]
                            #regall = ret["reg_all"]

                            st.text("Les meilleurs paramères sont: ")                                            
                            st.text(json.dumps(ret))
            
                            predictor = SVD(**my_hybrid_filtering.best_params_predictor)
                            
                        set_state(slot, ("predictor", predictor))
            
                            #SVD(**myfilter.best_params_predictor
                        ispredictor_trained = True

                    if predictor_name == 'NormalPredictor()':
                        
                        with st.spinner('Calcul en cours...'):
                            param_grid = {}
                            ret = my_hybrid_filtering.predictor_ajustement(predictor,param_grid)
                            st.text("Les meilleurs paramères sont: ")                                            
                            st.text(json.dumps(ret))
                        ispredictor_trained = True
                        set_state(slot, ("predictor", predictor))

        
        if get_state(slot,'predictor') != None :        
            #default user
            if st.checkbox("Utilisateur pa rdefaut",key="default_user") :
                with st.spinner('Calcul en cours...'):
                    npreditem = 10
                    pred = my_hybrid_filtering.predict(my_hybrid_filtering.default_user,get_state(slot,'predictor'),npreditem) 

                    st.text("Préférences ")            
                    user_preferences = my_collaborativ_filtering.get_preferences(my_hybrid_filtering.default_user,score_seuil,10)
                    top = user_preferences.sort_values(my_collaborativ_filtering.target_key_name, ascending=False)
                    st.dataframe(top)    

                    #pred = myfilter.predict_with_train_split(userId1,predictor,5) 
                    st.text("Prédiction utillisateur :")
                    st.dataframe(pred)    

            #chosen user
            userId =  display_user_selection(my_hybrid_filtering)
            if userId != '':
                #k_close_param , npref , npreditem , npreduser = display_nb_pres_selection()
                npreditem = 10
                pred = my_hybrid_filtering.predict(userId,get_state(slot,'predictor'),npreditem) 
                #pred = myfilter.predict_with_train_split(userId1,predictor,5) 
                st.text("Prédiction utillisateur :")
                st.dataframe(pred)    
    


