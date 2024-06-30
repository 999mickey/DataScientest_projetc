
import pandas as pd
import numpy as np
import random 
from sklearn.metrics import euclidean_distances
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

class mic_data_selection():
    def __init__(self):
        self.data_frame = pd.DataFrame()
        self.features = []
        self.target_name = ""
        self.file_name=""

    def set_feature_list(self, list):
        self.features = list

    def set_target_name(self,target_name):
        self.target_name = target_name

    def set_data(self,file_name):
        print("************************* set_data(",file_name,") **********************")
        self.file_name = file_name
        self.data_frame = pd.read_csv(file_name)
        #print(self.data_frame.shape)
        #self.data_frame.info()
        #print(self.data_frame.head(5))
    
    def reduce_data(self,frac):
        print("*********************reduce_data **************************** In")
        print("before self.data_frame.shape = ",self.data_frame.shape)
        self.data_frame  = self.data_frame .sample(frac=frac, random_state=42)
        print("after self.data_frame.shape = ",self.data_frame.shape)
        print("*********************reduce_data **************************** Out")

    def log_info(self):
        df = self.data_frame
        df.info()
        print(df.head(2))
        print(df.describe())

    def getstat_of_object(self , key):
        df = self.data_frame
        print(self.file_name,"*****************************  getstat_of_object ")
        print("Analyse de la variable object ",key)    
        print("nunique() ",df[key].nunique())    
        max= df[key].value_counts().max()    
        min= df[key].value_counts().min()
        counts = df[key].value_counts()
        print("nombre de ",key," différents = ",len(counts))    
        print("Le plus d'occurrence (",max,") est pour : ",df[df[key].isin(counts[counts == max].index)][key].unique())
        print("Le moins d'occurrence (",min,") est pour : ",df[df[key].isin(counts[counts == min].index)][key].unique())

    def getstat_of_numeric(self,key):
        df = self.data_frame
        print(self.file_name,"***************************** getstat_of_numeric")
        print("Analyse de la variable numérique ",key)    
        print("moyenne = ",df[key].mean())
        print("max = ",df[key].max())
        print("min = ",df[key].min())
        print("écart type = ",df[key].std())

    def describe_key(self , key):    
        df = self.data_frame
        if np.issubdtype(df[key].dtype, np.number) == True:
            self.getstat_of_numeric(key)
        else:
            self.getstat_of_object(key)     
        
    def count(self,key):
        df = self.data_frame
        count = df.groupby([key]).count() 
        return count
        #print(count)
    
    def mean(self,key)    :
        df = self.data_frame
        counts = df[key].value_counts()
        return df[key].mean()


    def count_diff_vals(self,key) :   
        df = self.data_frame
        ret = df.nunique().value_counts()
        print("********** count_diff_vals ( ", key,") = ",ret)
        return ret

    def get_diff_vals(self,key) :       
        df = self.data_frame
        counts = df[key].value_counts()
        print("********** get_diff_vals ( ", key,") = ",counts)
        counts.mean()
        return counts

    def get_max_number_of_occurence(self,key ):
        df = self.data_frame
        max= df[key].value_counts().max()    
        counts = df[key].value_counts()
        ret = df[df[key].isin(counts[counts == max].index)][key].unique()
        #print("la chanson la plus présente est : ",ret)
        return ret

    def get_number_Of_occurance(self,key):
        df = self.data_frame       
        ret = df[key].value_counts()
        #print(df[key].value_counts())    
        return ret
    
    def clean(self):
        df = self.data_frame
        print("********************** clean *********************************************** In ")
        print("before df shape = ",df.shape)
        df = df.dropna(axis = 0, how = 'all', subset = df.columns)
        print("after df shape = ",df.shape)
        print("********************** clean *********************************************** Out ")

    def limit_df_to_oneuser(self,user_id):                
        self.data_frame = self.data_frame[self.data_frame['user_id'] == user_id]

    def get_duplicated(self,key):
        df = self.data_frame
        ret =  df.duplicated(subset=[key])
        return ret
        
        #######################user
    def max_voter_is(self):
        return self.get_max_number_of_occurence("user_id")


    def get_user_vals(self,user_id):
        df = self.data_frame 
        ret = df[df['user_id']==user_id]
        return ret            
    
    def get_not_user_vals(self,user_id):
        df = self.data_frame 
        ret = df[df['user_id']!=user_id]
        return ret            
    

    def get_user_hashtags(self,user_id):
        df = self.data_frame 
        df = df[df['user_id']==user_id]
        ret  = df['hashtag']
        return ret            
    
    def get_user_songs(self,user_id):
        #df = self.data_frame
        df = self.get_user_vals(user_id)        
        #df = df.drop_duplicates(subset=['track_id'])
        #ret = None
        #ret = df[['track_id']+self.features+[self.target_name]] 
        return df
    
    def get_not_user_songs(self,user_id):
        df = self.get_not_user_vals(user_id)
        return df
    
    def get_random_row(self,df):
        print('******************** get_random_row ************************')                
        df = df.sample(n=1)
        return df
    
    def get_row_at(self,df,index):
        print('******************** get_row_at ************************')
        print(df.head())
        
        df = df.iloc[index]
        return df


    def get_positive_user_songs(self,user_id):
        df = self.get_user_songs(user_id)
        #print(df.shape)
        #print(df["track_id"].nunique())
        df.drop_duplicates(subset=['track_id'],inplace=True)
        #print(df.shape)
        #print(df["track_id"].nunique())
        
        ret = df[df[self.target_name] == 1][['track_id']+self.features+[self.target_name]]
        return ret

    def get_negative_user_songs(self,user_id):
        df = self.get_user_songs(user_id)
        df.drop_duplicates(subset=['track_id'],inplace=True)
        ret = df[df[self.target_name] == 0][['track_id']+self.features+[self.target_name]]
        return ret
    
    def get_negative_user_song(self,user_id,track_id):
        df = self.get_user_songs(user_id)
        df = df[df[self.target_name] == 0] [['track_id']+self.features+[self.target_name]]
        ret = df[df["track_id"] == track_id]
        return ret

    def get_positive_user_song(self,user_id,track_id):
        df = self.get_user_songs(user_id)
        df = df[df[self.target_name] == 1] [['track_id']+self.features+[self.target_name]]
        ret = df[df["track_id"] == track_id]
        return ret

    def get_from_user_and_hashtag(self,user_id,hashtag):
        df = self.data_frame        
        df = df[df['user_id']== user_id ]
        df = df[df['hashtag']== hashtag ]
        return df

    def get_user_songs_from_hashtag(self,user_id,hashtag):         
        df = self.get_from_user_and_hashtag(user_id,hashtag)
        ret = df["track_id"] 
        print(ret)
        return ret
    
    def get_incoherent_sentiment(self,user_id,track_id)    :

        retNo = self.get_positive_user_song(user_id,track_id)        

        retOk = self.get_negative_user_song(user_id,track_id)
        #print("get_incoherent_sentiment()  Ok=>",retOk.shape," NOk=>",retNo.shape)
        print("get_incoherent_sentiment()  len(Ok)=>",len(retOk)," len(NOk)=>",len(retNo))

    def get_incoherent_sentiments(self,user_id):
        dftracks = self.get_user_songs(user_id)

        print(" get_incoherent_sentiments dftracks => ",dftracks.shape)
        
        for i , trackid  in enumerate (dftracks["track_id"]):            
            print(" trackid[",i,"]  = ",trackid)
            self.get_incoherent_sentiment(user_id,trackid)

            
    def get_blobal_sentiment(self,trackid):
        df = self.data_frame       
        df = df[df["trackid"]]         

            
    def get_song_sentiment(self,track_id):
        df = self.data_frame       
        gb = df.groupby('track_id').agg({'sentiment':[lambda x:tuple(x),'mean']})
        gb.columns = ['sentiments','mean']
        
        print("gb head()=> ",gb.head())
        print("gb.tail()=> ",gb.tail())
        
        #exit()

        gb['n'] = gb['sentiments'].str.len()
        print(gb['n'])
        
        gb = gb.sort_values(by='n',ascending=False).reset_index()
        gb['sentiment'] = (gb['mean']>0.1).astype(int)
        gb['sentiment'] = (gb['mean']>0.001).astype(int)
        print(gb['sentiment'].value_counts())
        testdf = gb[gb['n']>20]
        testdf = gb['n']
        #print(testdf.shape)
    
        #print(" ")
    ##sentiment

    def compute_diff_with_userplaylist(self, user_id,track_id):
        df =  self.data_frame[self.features]              
        list_to_use = self.get_user_vals[self.features]              
        song = df[df['track_id']==track_id]
        song_caracter = song[self.features]              
        closest = np.argmin(euclidean_distances(list_to_use, song_caracter), axis=1)

    def select(self,key,listvals):
        print("****************** select ***************************")
        df = self.data_frame

    def clean_data(self):
        df = self.data_frame       
        print("****************** clean_data ************************************* In")
        print("original df.shape => ",df.shape)
        gb = df.groupby(['track_id','user_id']).agg({'sentiment':[lambda x:tuple(x),'mean']})

        print("après regroupement par user et track gb.shape => ",gb.shape)
            
        gb.columns = ['sentiments','mean']
        gb['user_votes_for_track'] = gb['sentiments'].str.len()

        gb = gb.sort_values(by='user_votes_for_track',ascending=False).reset_index()
        
        gb['sentiment'] = (gb['mean'] > 0.5).astype(int)

        
        gb = pd.merge(gb,df[['track_id','sentiment_score']+self.features],on='track_id',how='left').drop_duplicates()
        gb = gb.drop("sentiments",axis=1)
        gb = gb.drop("mean",axis=1)
        
        print("après  merge df.shape => ",gb.shape)
        df = gb
        df = df.drop_duplicates(subset=['user_id','track_id'],keep = 'first')    

        print("après suppression duplicats df.shape => ",gb.shape)
                
        #enlever les utilisateurs qui ont moins de 20 track id par exemple
        #print(df['user_id'].unique().count())
        #drop avec condition
        min= 20
        counts = df['user_id'].value_counts()
        df = df[~df['user_id'].isin(counts[counts < min].index)]
        df.info()
        exit()
        df.to_csv("testcleanover20.csv")
        self.data_frame = df
        
        print("après suppression des user dont le nombre d 'occurence est < ",min," df.shape => ",df.shape)
        print("****************** clean_data ************************************* Out")
        return df
        
    def get_user_tracks(self,user_id):
        df = self.data_frame
        df = df[df['user_id'] == user_id]
        return df
    
    def get_users_with_different_sentiment(self):
        print("****************** clean_data ************************************* Out")
        df = self.data_frame
        df = df[df.sentiment == 1 & df.sentiment == 0]
        return df

    def add(self , key1 , key2):
        df = self.data_frame
        df[key1+key2] = df[key1] + df[key2]

    """def setsentiment(self):
        df = self.data_frame
        df["sentiment"] = df[~df['user_id'].isin(counts[counts < min].index)]"""
        

    def test(self):
        print("test")
        df = self.data_frame
        name_list_to_keep = []
        for val in df.columns:
            print('check column ',val)
            if is_numeric_dtype(df[val]) :
                print(" colonne[",val,"] est numerique")
                name_list_to_keep . append(val)

            
        """df = self.data_frame       
        print("before = ",df.shape)
        #df = df[df['user_id'] ==  	811851805]
        df = df[['user_id','track_id','sentiment','hashtag']+self.features].drop_duplicates()
        print("after = ",df.shape)
        grouped_df = df.groupby(['user_id'], as_index = False)['track_id'].count()
        grouped_df = grouped_df.rename({"track_id":"track_count"},axis=1)
        print(grouped_df.shape)

        fusion = df.merge(grouped_df,how="left",on='user_id')
        print(fusion.shape)
        fusion.drop(fusion[fusion["track_count"] < 10].index,inplace=True)
        print(fusion['user_id'].nunique())
        print(fusion['track_id'].nunique())
        print(fusion['sentiment'].value_counts())
        print(fusion.shape)
        fusion.to_csv("test.csv")
        exit()

        grouped_df.drop(grouped_df[grouped_df["track_count"] < 20].index,inplace=True)

        #mon_dataframe.loc[ condition sur les lignes ,  colonne(s) ]

        grouped_df.to_csv("test.csv")
        print(grouped_df.shape)

        for val in grouped_df["user_id"]:
            df.loc()
            print(val)
        exit()

        print("after = ",df.shape)

        df.drop( df[ df['Sex'] == 1 ].index, inplace=True)

        print(df[df["user_id"]==212913 ])
        test = df[df["user_id"]==212913 ]
        test["track_count"] = 2
        print("====================",test["track_count"])
        exit()
        df.loc()
        print(df[df["user_id"]==212913 ]["track_count"] )
        exit()
        #df_user = df_user[['user_id','track_id','sentiment','hashtag']+feature_cols].drop_duplicates()
        #df_user

        #print(df["track_id"].count())
        print(df.shape)
        df["nombre_track"] = df["track_id"]
        #print(resultat)
        #gb = df.groupby('user_id').agg({'track_id':[lambda x:len(x)]})
        grouped_df = df.groupby(['user_id'], as_index = False)['track_id'].count()
        grouped_df = grouped_df.rename({"track_id":"track_count"},axis=1)
        print(grouped_df)
        #for val in df["user_id"]:
        #    print(grouped_df[val]["track_count"])
            
        exit()
        #df["nombre_track"] = df.groupby(['user_id']).agg({'nombre_track':[lambda x:len(x)]})

        #df["nombre_track"] = df[df["user_id"] == gb["user_id"]][]

        #gb['n'] = gb['track_id'].len()
        #gb.to_frame() 

        print(len(gb))
        print(type(gb))
        print(type(gb.columns))
        exit()
        for i in range(len(gb)):
            print(gb[i])    
        #for val in  gb:
        #    print(val)
            #print(val.inde)

        exit()
        for i,val in enumerate( df["user_id"]):
            print(i)
            #print(val)
            print("val[",val,"]  gb.index[",gb.index[i],"] => " ,gb.values[i])
            
        #    print(gb[val])
            #df[val]["nombre_track"] = gb[val]
        
        print("------------------")
        print(gb)

        exit()
        
        
        resultat = df.groupby('user_id')['track_id'].nunique()

        resultat = resultat.sort_values(ascending=False)
        print("1   len(resultat) => ",len(resultat))

        #resultat > 20
        
        #resultat[(resultat > 20) & (s < 8)]
        resultat["user_id"] = resultat[(resultat > 10) ]

        #resultat = resultat[resultat[""]]
        
        #print(resultat > 20)
        
        print("2   len(resultat) => ",len(resultat))
        print(resultat)
        """

        


        



