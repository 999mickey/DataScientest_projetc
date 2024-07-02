import os

def input_variables_setter(file_name, filter):
    #default 
    filter.init_vars()
    filter.set_target_key_name('sentiment_score')
    filter.set_user_key_name('user_id')
    filter.set_item_key_name('track_id')
    
    if file_name.find("user_and_track_sentiment") != -1:
        filter.set_target_key_name('sentiment_score')
        filter.set_user_key_name('user_id')
        filter.set_item_key_name('track_id')
        filter.set_visual_key_name('hashtag')

    elif file_name == 'data.csv':
        varlist = ['liveness', 'speechiness', 'danceability', 'valence', 'loudness', 'tempo', 'acousticness','energy', 'mode', 'key', 'instrumentalness']
        artist_key = 'artists'
        songname_key = 'name'
        filter.set_artist_key_name(artist_key)
        filter.set_item_key_name(songname_key)
        filter.set_feature_list(varlist)
        filter.set_target_key_name('popularity')

    elif file_name == 'final_all.csv':
        varlist = ['liveness', 'speechiness', 'danceability', 'valence', 'loudness', 'tempo', 'acousticness','energy', 'mode', 'key', 'instrumentalness']
        artist_key = 'user_id'
        track_id_key = 'track_id'
        songname_key = 'hashtag'
        filter.set_artist_key_name(songname_key)
        filter.set_item_key_name(track_id_key)
        filter.set_visual_key_name(songname_key)
        filter.set_feature_list(varlist)

    elif file_name.find("simulation") != -1:
        filter.set_target_key_name('sentiment_score')
        filter.set_user_key_name('user_id')
        filter.set_item_key_name('track_id')
        filter.set_visual_key_name('user_likes')
        filter.default_user  = 'user0'

    elif file_name.find("merge") != -1:
        filter.set_target_key_name('sentiment_score')
        filter.set_user_key_name('user_id')
        filter.set_item_key_name('track_id')
        filter.set_visual_key_name('genre')
        filter.default_user  = 'user1'

    elif file_name.find("spotify_user_track") != -1:
        filter.set_target_key_name('sentiment_score')
        filter.set_user_key_name('user_id')
        filter.set_item_key_name('name')

    elif file_name.find("ratings") != -1:
        filter.set_target_key_name('rating')
        filter.set_user_key_name('user_id')
        filter.set_item_key_name('title')
        filter.default_user  = '004d5e96c8a318aeb006af50f8cc949c'
    #print("input_variables_setter(",file_name,", filter):")        
    #filter.print_vars()

def getstat_of_object(df, key):        
    print("Analyse de la variable object ",key)    
    print("nunique() ",df[key].nunique())    
    max= df[key].value_counts().max()    
    min= df[key].value_counts().min()
    counts = df[key].value_counts()
    print("nombre de ",key," différents = ",len(counts))    
    print("Le plus d'occurrences (",max,") est pour : ",df[df[key].isin(counts[counts == max].index)][key].unique())
    print("Le moins d'occurrences (",min,") est pour : ",df[df[key].isin(counts[counts == min].index)][key].unique())


def get_only_file_name(path):
    retval = ''
    index =  path.rindex('/')
    index += 1
    
    retval = path[index:]
    
    return retval

def list_files_recursive(path='.'):
    print("list_files_recursivepath = ",path)
    print("os.path.abspath(__file__) = ",os.path.abspath(__file__))
    listFiles = []
    if os.path.isdir(path):
        print("path = ",path)
        for entry in os.listdir(path):
            full_path = os.path.join(path, entry)
            if os.path.isdir(full_path):
                ret = list_files_recursive(full_path)
                if len(ret):
                    listFiles.append(ret)
            else:
            
                if entry.endswith(".csv"):
                #					
                    lname = get_only_file_name(full_path)
                    listFiles.append(lname)

                else :
                    print("bad file path Nothing to do, file shall be with .csv")	
    else:   
        if path.endswith(".csv"):
            lname = get_only_file_name(path)
            listFiles.append(lname)
        
        else :
            print("bad file path Nothing to do, file shall be with .csv")
     
    return listFiles			

















