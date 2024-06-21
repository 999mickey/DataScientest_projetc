

def input_variables_setter(file_name, filter):
    
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
    elif file_name.find("spotify_user_track") != -1:
        filter.set_target_key_name('sentiment_score')
        filter.set_user_key_name('user_id')
        filter.set_item_key_name('name')

def getstat_of_object(df, key):        
        print("Analyse de la variable object ",key)    
        print("nunique() ",df[key].nunique())    
        max= df[key].value_counts().max()    
        min= df[key].value_counts().min()
        counts = df[key].value_counts()
        print("nombre de ",key," différents = ",len(counts))    
        print("Le plus d'occurrence (",max,") est pour : ",df[df[key].isin(counts[counts == max].index)][key].unique())
        print("Le moins d'occurrence (",min,") est pour : ",df[df[key].isin(counts[counts == min].index)][key].unique())

