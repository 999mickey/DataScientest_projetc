
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import io
import time
import sys,os

sys.path.append(os.path.realpath('..'))

from models.mic_filtering_class import  mic_content_filtering ,mic_base_filter , mic_hybrid_filtering, mic_collaborativ_filtering
from input_variables_setter import input_variables_setter
from features.mic_data_selection import mic_data_selection
from visualization.visualize import mic_vizualizer
from fire_state import create_store, form_update , set_state , get_state

content_path = '../../Data/'

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
    ("user_id",''),
    ("mic_data", None)  ,  
    ("mic_viz",None)    ,
    ("mic_content_filtering",None),
    ("mic_collaborativ_filtering",None),
    ("mic_hybrid_filtering",None),
    
])
load_count = get_state(slot,"load_count")
file_name = get_state(slot,"file_name")
close_song_name = get_state(slot,"close_song_name")
close_artist_name = get_state(slot,"close_artist_name")

if close_artist_name == '' and get_session_state('close_artist_name'):
    close_artist_name = get_session_state('close_artist_name')
    set_state(slot, ("close_artist_name", close_artist_name))


user_id = get_state(slot,"user_id")
my_data = get_state(slot,"mic_data")
my_visualizer = get_state(slot,"mic_viz")
my_content_filter = get_state(slot,"mic_content_filtering")
my_collaborativ_filtering = get_state(slot,"mic_collaborativ_filtering") 
my_hybrid_filtering = get_state(slot,"mic_hybrid_filtering")
print("--------------app_content_filtering---------->IN \nst.session_state = ",st.session_state)

print("load_count => ", load_count)        
print("file_name => ", file_name)        
print("close_song_name => ",close_song_name)
print("close_artist_name => ",close_artist_name)
print("user_get_stateid => ",user_id)
print("mic_data => ", my_data)   
print("mic_viz",my_visualizer)    
print("mic_collaborativ_filtering",my_collaborativ_filtering)        
print("mic_content_filtering",my_content_filter)        
print("mic_hybrid_filtering",my_hybrid_filtering)        

print("------------------app_content_filtering.py----------------<IN \n ")

load_count = get_state(slot,"load_count")

load_count += 1
set_state(slot ,("load_count",load_count))

current_filename = get_state(slot,'file_name')

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
       
#@st.cache_data
def load_data(name):    
    print("------------------------------------------------------------------<load_data(",name,") called")
    input_variables_setter(name,my_data)
    input_variables_setter(name,my_visualizer)
    input_variables_setter(name,my_content_filter)
    input_variables_setter(name,my_collaborativ_filtering)
    input_variables_setter(name,my_hybrid_filtering)

    with st.spinner('Wait for load data...'):
        df = pd.read_csv(content_path + name)
        my_data.data_frame = df
       
        my_content_filter.data_frame = df

        if len(my_content_filter.features ):
            my_content_filter.normalize_data()
       
        print("load_data my_content_filter.artist_key_name = ",my_content_filter.artist_key_name)
        my_visualizer.data_frame = df
        my_collaborativ_filtering.data_frame = df
        my_hybrid_filtering.data_frame = df
       
        #set_state(slot, ("mic_data", my_data))
        #set_state(slot, ("mic_content_filtering", my_content_filter))
        #set_state(slot, ("mic_viz", my_visualizer))
       
       #my_data.set_data(content_path + name)    
    st.success('Load Data Done!')                
    set_state(slot, ("file_name", name))
    
    return df


df = pd.DataFrame()


st.title("Projet de Recommandation Musicale avec ["+str(current_filename)+"]")
st.sidebar.title(current_filename)
pages=["Intro","Choisir les données", "DataVizualization", "Content Filter","Memoire filtering","Hybrid filtering"]

page=st.sidebar.radio("Aller vers", pages)

print("---page",page)

if page == pages[0] : 
    st.write("### Introduction mic666 test de Recommantions Musicales")
    print("### Introduction mic666 test -----------> page 0")
    #test
    #file_name = "simulationcurrent.csv"
    #load_data(file_name)

    
if page == pages[1] : 
    st.write("### Choisir des données")
    print("### Choisir des données")
    datas=["","merge.csv","data.csv","simulation.csv","simulationcurrent.csv"]

    option = st.selectbox('Choix des données', datas,key="data_name")
        
    if current_filename != option and option != '': 

        df = load_data(option)        
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
            with st.spinner('Computing...'):
                fig = my_visualizer.plot_heatmap()
                st.pyplot(fig)

        checktitle = "Présenter le Distribution de "+my_visualizer.target_key_name
        if st.checkbox(checktitle,key="dist_target") :
            with st.spinner('Computing...'):
                plt.title("Distribution de "+my_visualizer.target_key_name)
                st.pyplot(my_visualizer.plot_repartion_count_target(my_visualizer.target_key_name))

        if st.checkbox("Cercle de corrélation",key="corr_target") :
            with st.spinner('Computing...'):
                plt.title("Cercle de corrélation ")
                st.pyplot(my_visualizer.plot_corrrelation_circle())

        title = "répartition par "+ my_visualizer.target_key_name
        
        if st.checkbox(title,key="rep_target") :
            with st.spinner('Computing...'):
                plt.title(title)
                st.pyplot(my_visualizer.plot_groupement_from_key(my_visualizer.target_key_name))            

    else:
        print("no filename selectionned")

if page == pages[3]:

    st.write("### Filtrage contenu avec ",str(current_filename))
    #st.write("### Content Filtering")
    #myFilter = mic_content_filtering()
    if current_filename != '':
        #input_variables_setter(current_filename,my_content_filter)
        #myFilter.set_data(content_path +  current_filename)
    
        def artist():
            print("-------------------------> artist")
        def songs():
            print("-------------------------> song")
        def song():
            print("-------------------------> song")

        #close_artist_name = st.text_area('choisir le nom d atiste',key='close_artist_name')    
        #close_song_name = st.text_area('choisir le nom de chanson',key='close_song_name')    
        print("close_song_name = ",close_song_name)
        print("close_artist_name = ",close_artist_name)
        print("my_content_filter.artist_key_name =",my_content_filter.artist_key_name)
        
        #my_content_filter.data_frame.info()

        if get_session_state("close_artist_name") != None and close_artist_name == '':
            close_artist_name =  get_session_state("close_artist_name")

        if get_session_state("close_song_name") != None and close_song_name == '':
            close_song_name =  get_session_state("close_song_name")

        if  close_artist_name == '':
            print(" choose artist")        
            close_artist_name = st.text_area('choisir le nom d atiste',key='close_artist_name')    
            
            set_state(slot, ("close_artist_name", close_artist_name))
        elif close_song_name == '':
            print(" choose chosse song")
            close_song_name = st.text_area('choisir le nom de chanson',key='close_song_name')    
            set_state(slot, ("close_song_name", close_song_name))
            
        else:            
            print("3333333333333 current_filename  = ",current_filename )
            print("close_artist_name = ",close_artist_name)
            print("close_song_name = ",close_song_name)

            ltext = 'choix du nom d atiste : '+close_artist_name
            st.text(str(ltext))    
            ltext = 'choix  du nom de chanson :'  +close_song_name
            st.text(str(ltext))    
            print("my_content_filter.artist_key_name =",my_content_filter.artist_key_name)
            print(my_content_filter.data_frame[my_content_filter.artist_key_name].head(10))
                    
            songnum = my_content_filter.get_track_num_of_artist(close_artist_name,close_song_name)                
            
            print('--------------- songnum = ',songnum)
            with st.spinner('Wait for compute tracks...'):
                SongId , SongName , ArtistName , Distance = my_content_filter.content_filter_music_recommender(songnum, 10)
            st.success('compute tracks Done!')                

            outputdf = pd.DataFrame()
            outputdf["song_id"] = SongId
            outputdf["artists"] = ArtistName
            outputdf["song_name"] = SongName
            outputdf["distance"] = Distance
            print(outputdf.head(10))
            st.write('Chanson le splus proches :')                    
            st.dataframe(outputdf.head(10))    
                        
    else :
        print('no filename selected')

if page == pages[4] : 
    st.write("### Filtrage mémoire ",str(current_filename))
    
    if current_filename == '':
        st.text("Il faut selectionner un fichier !")    
    else :    
        # my_collaborativ_filtering
        #file_name = "simulationcurrent.csv"
        #my_collaborativ_filtering = mic_collaborativ_filtering()
        #my_collaborativ_filtering.set_data(content_path+file_name)
        #input_variables_setter(file_name,my_collaborativ_filtering)

        my_collaborativ_filtering.data_frame.info()    
        my_collaborativ_filtering.clean_columns()

        ##utilisateur
        #userId = my_collaborativ_filtering.default_user
        userIds =[]
        userIds = my_collaborativ_filtering.get_best_voter_users(10)
        st.dataframe(userIds)    
        
        userIds = userIds[my_collaborativ_filtering.user_key_name]
        userId = ''
        userId = st.selectbox('Choix de l utilisateur', userIds,key="user_id")

        print("Utilsateur from selectbox => ",userId)
        exit()
        st.text("Utilsateur "+userId)
        #end of utlisateur
        if userId != '':
            k_close_param = 3

            number_of_line_todump = 10
            score_seuil = 0
            my_collaborativ_filtering.generate_notation_mattrix()
            simlilar_users = my_collaborativ_filtering.get_similar_users(k_close_param,userId)
            st.text("Utilsateurs similaires ")
            st.dataframe(simlilar_users)    

            st.text("Préférences ")
            user_preferences = my_collaborativ_filtering.get_preferences(userId,score_seuil,5)
            top = user_preferences.sort_values(my_collaborativ_filtering.target_key_name, ascending=False)
            st.dataframe(top)    

if page == pages[5] : 
    st.write("### Filtrage hybride ",str(current_filename))
    #my_hybrid_filtering
    


