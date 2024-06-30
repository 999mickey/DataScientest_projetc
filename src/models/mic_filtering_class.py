
import pandas as pd
import numpy as np

from sklearn.decomposition import TruncatedSVD
import sklearn.metrics.pairwise as dist

from scipy.sparse import csr_matrix

from surprise import Reader
from surprise import Dataset

from surprise.model_selection import GridSearchCV
from surprise import SVD

from scipy.spatial.distance import cosine, euclidean, hamming
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler

import seaborn as sns 
import matplotlib.pyplot as plt
import random 

from surprise import SVD
from surprise import NormalPredictor
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split
from pandas.api.types import is_numeric_dtype

class mic_base_filter():
    def __init__(self):
        self.data_frame = pd.DataFrame()
        self.df_features = pd.DataFrame()        
        self.artist_key_name = ''        
        self.item_key_name = ''
        self.user_key_name = ''        
        self.visual_key_name = ''
        self.default_user = ''
        self.features = []
        self.target_key_name = ""
        self.file_name = ""
        
    #on utilise cette variable pour identifier humainement les mrorceaux
    def init_vars(self):
        self.artist_key_name = ''        
        self.item_key_name = ''
        self.user_key_name = ''        
        self.visual_key_name = ''
        self.default_user = ''
        self.features = []
        self.target_key_name = ""
        
        
    def print_vars(self):
        print("self.artist_key_name = ",self.artist_key_name )
        print("self.item_key_name = ",self.item_key_name )
        print("self.user_key_name = ",self.user_key_name )
        print("self.visual_key_name = ",self.visual_key_name )
        print("self.target_key_name = ",self.target_key_name )
        print("self.features = ",self.features)
        
    def set_visual_key_name(self,key):
        self.visual_key_name = key

    def set_target_key_name(self,key):        
        self.target_key_name = key        
    
    def set_artist_key_name(self,key):
        self.artist_key_name = key

    #def set_track_key_name(self,key):
    def set_item_key_name(self,key):
        self.item_key_name = key

    def set_user_key_name(self,key):
        self.user_key_name = key

    def set_feature_list(self, list):        
        self.features = list

    def get_tracks_with_name(self,name):
        print("************************* get_tracks_of_artist(",name,") **********************")
        
        song_name_lower = name.lower()
        
        df = self.data_frame[self.item_key_name].str.lower() 
        #condition = (df.str.startswith(song_name_lower) )#ok            
        condition = (df.str.contains(song_name_lower) )#ok            
        
        return self.data_frame[condition][[self.artist_key_name,self.item_key_name]]

    def get_artist_closedto(self,artist_name):
        print("type(artist_name) = ",type(artist_name) ,"  artist_name = ",artist_name )
        artist_lower = artist_name.lower()
        df = self.data_frame
        #df = pd.DataFrame()
        df[self.artist_key_name] = self.data_frame[self.artist_key_name].str.lower() 
        print("-----------------------------------")
        #df.info()
        #print(df.head())
        
        #toberemvoed
        print("artist_lower = ",artist_lower)
        #condition = (df.str.startswith(artist_lower) )#ok              
        condition = (df[self.artist_key_name].str.contains(artist_lower) )#ok                    
        print(condition)
        

        ldf = df[condition][[self.artist_key_name]]
        print("-------------",len(ldf))
        
        #ldf.to_csv("dumptorem.csv")
        #ldf = df[condition][[self.artist_key_name]]
        retval = []
        
        for val in ldf[self.artist_key_name].unique():
            print("type(val)=",type(val) ,"  val=====",val)
            val = val.replace("['", '')
            val = val.replace("']", '')
            retval.append(val)
        print("len(retval) = ",len(retval))
        print(retval)
        exit()
        #return ldf[self.artist_key_name].unique()
        return retval
        #return self.data_frame[condition][[self.artist_key_name]].nunique()
        #return self.data_frame[condition][[self.artist_key_name]].unique()

    def get_tracks_of_artist(self,artist_name):
        print("************************* get_tracks_of_artist(",artist_name,") **********************")
        """artist_lower = artist_name.lower()
        df = self.data_frame[self.artist_key_name].str.lower()                     
        condition = (df.str.contains(artist_lower) )#ok                    
        return self.data_frame[condition][[self.artist_key_name,self.item_key_name]]"""
        artist_lower = artist_name.lower()
        ret= pd.DataFrame()
        ret["lower"] = self.data_frame[self.artist_key_name].str.lower() 
        condition = (ret["lower"].str.contains(artist_lower) )#ok            

        ret = ret[condition]    
        return ret
    
    def get_random_track_num_of_artist(self,artist_name):
        ret = self.get_tracks_of_artist(artist_name)        
        n = random.randint(0,len(ret)-1) 
        ret = ret.iloc[n,:]
        return ret

    def get_track_num_of_artist(self,artist_name,song_name):
        print("************************* get_track_num_of_artist(",artist_name,",",song_name,") **********************")
        ret = self.get_tracks_of_artist(artist_name)
        
        song_name_lower = song_name.lower()
        
        ret["lower"] = self.data_frame[self.item_key_name].str.lower() 
        #df = [self.data_frame[self.item_key_name]].str.lower() 
        #condition = (df.str.startswith(song_name_lower) )#ok            
        condition = (ret["lower"].str.contains(song_name_lower) )#ok            

        ret = ret[condition]    
        return ret.index.tolist()[0]
    
    def set_data(self,file_name):
        #print("************************* set_data(",file_name,") **********************")
        self.file_name = file_name
        self.data_frame = pd.read_csv(file_name)
        
        self.data_frame = self.data_frame.drop_duplicates()
        ##25.06.2024
        #self.data_frame = self.data_frame.dropna(axis=0)
        ##
        

        self.data_frame.style.set_properties(**{'text-align': 'left'})

        
        #self.data_frame.dropna()
        #self.data_frame.info()
    
    def get_item_description(self,num):
        return self.data_frame.iloc[num]
    
    def select_by_key_val(self,key , val):
        df = self.data_frame
        df = df[df[key] == val]
        #return self.filter_util_vals(df)
        return df
    
    def select_by_key_approximativ_val(self,key , val):
        df = self.data_frame
        val = val.lower()
        
        df = self.data_frame[key].str.lower() 
        condition = (df.str.contains(val) )#ok            

        return self.filter_util_vals(self.data_frame[condition])
        
        df = df[df[key] == val]
    
        return df
    
    def filter_util_vals(self,df):

        if self.visual_key_name != '':
            ret = df[[self.user_key_name,self.item_key_name,self.visual_key_name,self.target_key_name]]
        else :
            ret = df[[self.user_key_name,self.item_key_name,self.target_key_name]]

        return ret    
    

    def filter_util_vals_for_présentation(self,df):

        if self.visual_key_name != '':
            ret = df[[self.item_key_name,self.visual_key_name,self.target_key_name]]
        else :
            ret = df[[self.item_key_name,self.target_key_name]]

        return ret    
        
         
    def get_random_row_from(self,key , val):
        print("*****************get_random_row_from ***************")
        df = self.select_by_key_val(key,val)
        print('len(self.select_by_key_val(',key,',',val,'))',len(df))
        n = random.randint(0,len(df)-1) 
        df = df.iloc[n,:]
        print('self.select_by_key_val(',key,',',val,'))=>\n',df.head())
        
        return self.filter_util_vals(df)
        #return df
    
    def get_random_user_id(self):
        print("**********************get_random_user_id ************************")
        
        n = random.randint(0,len(self.data_frame)-1) 
        #print(n)
        
        df = self.data_frame.iloc[n,:]
        return df[self.user_key_name]
    
    def clean_columns(self):
        print('*****************clean_data() ******************')
        df = self.data_frame
        
        varlist = [self.user_key_name,self.item_key_name,self.target_key_name]
        if self.visual_key_name != '':
            varlist.append( self.visual_key_name)
        #df.info()    
        print(varlist)
        df  = df[varlist]        

        df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
        df.reindex()        
        self.data_frame = df
                
    def get_user_df(self, userid):
        return self.data_frame[self.data_frame[self.user_key_name] == userid]
    
    def get_user_description(self,userid):
        #print("*************get_user_description(self,userid)***************")
        descrition = 'Description de l utisateur '
        descrition += str(userid)
        df = self.get_user_df(userid)
        if self.visual_key_name != '':
            descrition += '\n'
            descrition += '     Il aime :'
            descrition += df.iloc[0][self.visual_key_name ]

        descrition += '\n'
        descrition += '     Il est est pésent ' + str(len(df)) + ' fois dans le dataset'
        return descrition



    def get_random_track_id(self):
        n = random.randint(0,len(self.data_frame)-1) 
        return n

    def get_top_rated_songs(self,num):
        print("**********************get_top_rated_songs ************************")
        df = self.data_frame
        aggregated_data = df.groupby(self.item_key_name )[self.target_key_name].count().reset_index()
        #aggregated_data = df.groupby(self.item_key_name )[self.target_key_name].count()

        # Tri du DataFrame agrégée par note en ordre décroissant
        sorted_aggregated_data = aggregated_data.sort_values(by=self.target_key_name, ascending=False)

        # Sélection des 10 premiers morceaux les mieux notés
        top_rated_tracks = sorted_aggregated_data.head(num)
        return top_rated_tracks
    
    def get_top_rater_users(self,num):
        df = self.data_frame
        aggregated_data = df.groupby(self.user_key_name )[self.target_key_name].count().reset_index()
        
        # Tri du DataFrame agrégée par note en ordre décroissant
        sorted_aggregated_data = aggregated_data.sort_values(by=self.target_key_name, ascending=False)

        # Sélection des 10 premiers morceaux les mieux notés
        top_rater_users = sorted_aggregated_data.head(num)
        return top_rater_users

    #normalisation des donées de sortie
    def get_full_data(self,indf,key ):
        #print("***************** get_full_data ******************")
        ret = self.data_frame.merge(indf,how='right',on=key )
        if self.visual_key_name == '':
            ret = ret[self.item_key_name]
        else:
            ret = ret[[self.item_key_name,self.visual_key_name]]
        
        return ret
    
    def get_util_data_for_presentation(self,df):
        #print("***************** get_util_data_for_presentation ******************")
        ret = any
        if self.visual_key_name != '':
            ret = df[[self.item_key_name,self.visual_key_name,self.target_key_name]]
        else :
            ret = df[[self.item_key_name,self.target_key_name]]
        
        return ret
        
    def drop_dupicated_tracks(self):
        print("********************* drop_dupicated_tracks *************")
        df = self.data_frame
        print(" before drop_duplicates shape = ",df.shape)
        print(df.head(10))
        if self.user_key_name == '':
              df = df.drop_duplicates(subset=[self.item_key_name,self.artist_key_name])  
        else:
              df = df.drop_duplicates(subset=[self.item_key_name,self.artist_key_name,self.user_key_name])  
        df = df.reset_index(drop=True)
        self.data_frame = df
        print(" after drop_duplicates shape = ",df.shape)
        print(df.head(10))
    
    
    def get_best_voted_songs(self,num):
        
        df = self.data_frame
        df.info()
        # On regroupe les données par titre et calcule le nombre dde notes ('count') et la note moyenne ('mean') pour chaque livre.
        item_stats = df.groupby(self.item_key_name)[self.target_key_name].agg(['count', 'mean']).reset_index()
        item_stats.info()
        print("self.target_key_name = ",self.target_key_name)
        
        # Affichage du graphique
        df = self.data_frame
        aggregated_data = df.groupby(self.item_key_name)[self.target_key_name].mean().reset_index()

        # Tri du DataFrame agrégée par note en ordre décroissant
        sorted_aggregated_data = aggregated_data.sort_values(by=self.target_key_name, ascending=False)

        # Sélection des 10 premiers tracks les mieux notés
        print("Morceaux les plus mieux notés : ")
        best_rated_tracks = sorted_aggregated_data.head(10)
        print(best_rated_tracks)
        return best_rated_tracks
        #"""


    def get_best_voter_users(self,num):    
        df = self.data_frame
        df = self.data_frame
        agg_dict = {self.item_key_name:lambda x:list(x)}

        df = df[df[self.target_key_name] != np.nan]

        big_user = df.groupby(self.user_key_name).agg(agg_dict).reset_index()
        big_user['count'] = big_user[self.item_key_name].str.len()
        
        big_user = big_user.drop([self.item_key_name],axis=1)        

        big_user = big_user.sort_values(by='count',ascending=False)
        user = big_user[self.user_key_name].iloc[0]

        return big_user.head(num)

    def display_tracks_and_users_num(self):
        n_users = self.data_frame[self.user_key_name].nunique()

        n_tracks = self.data_frame[self.item_key_name].nunique()

        print("Nombre d 'utilisateurs : ",n_users)
        print("Nombre de morceaux : ",n_tracks)

    #def get_df_for_heatmap(self):
    def get_numerical_var(self):
        
        df = self.data_frame
        name_list_to_keep = []
        for val in df.columns:
            #print('get_df_for_heatmap check column ',val)
            if is_numeric_dtype(df[val]) :
                #print(" colonne[",val,"] est numerique")
                name_list_to_keep . append(val)
        #print(name_list_to_keep)
        #df.info()
        return df[name_list_to_keep]

#########################################################""    
        
class mic_content_filtering( mic_base_filter):
    """
    mic_content_filtering
    """
    def __init__(self):
        mic_base_filter.__init__(self)
        self.df_normalized = pd.DataFrame()

    def normalize_data(self)    :        
        print('*********************** normalize_data ******************')
        
        self.data_frame[self.artist_key_name] = self.data_frame[self.artist_key_name].map(lambda x: x.lstrip('[').rstrip(']'))

        self.data_frame[self.artist_key_name] = self.data_frame[self.artist_key_name].map(lambda x: x[1:-1])
        
        self.data_frame['song_id']=self.data_frame.index

        self.df_normalized = self.data_frame[self.features]
        
        #self.df_normalized.index = self.data_frame['song_id']
        self.df_normalized = pd.DataFrame(normalize(self.df_normalized, axis=1))

        self.df_normalized.index = self.data_frame['song_id']


    def content_filter_music_recommender(self,songidp, N):
        print("***************** content_filter_music_recommender(songidp = ",songidp,") *************************")
        
        distance_method = euclidean
        #distance_method =  hamming
        
        print(self.data_frame[self.artist_key_name].head(10))
                
        allSongs = pd.DataFrame(self.df_normalized.index)
        allSongs["distance"] = allSongs["song_id"].apply(lambda x: distance_method(self.df_normalized.loc[songidp], self.df_normalized.loc[x]))
        # sort by distance then recipe id, the smaller value of recipe id will be picked. 
        TopNRecommendation = allSongs.sort_values(["distance"]).head(N).sort_values(by=['distance', 'song_id'])
        Recommendation = pd.merge(TopNRecommendation , self.data_frame, how='inner', on='song_id')        
        
        TopNUnRecommendation = allSongs.sort_values(["distance"]).tail(N).sort_values(by=['distance', 'song_id'])
        #UnRecommendation = pd.merge(TopNUnRecommendation , self.data_frame, how='inner', on='song_id')
        #print("les plus éloignés....", UnRecommendation.tail(N))
        
        SongName = Recommendation[self.item_key_name ]  
        ArtisName = Recommendation[self.artist_key_name ]
        Distance = Recommendation["distance"]
        SongId  = Recommendation["song_id"]

        return SongId , SongName , ArtisName , Distance


########################################

class mic_collaborativ_filtering(mic_base_filter ):
    """
    mic_collaborativ_filtering
    """
    def __init__(self):
        print("***************** __init()__ *************************")
        mic_base_filter.__init__(self)
        #self.data_frame = pd.DataFrame()
        self.matrice_pivot = pd.DataFrame()
        self.sparse_ratings = any
        self.track_ids = list()
        self.user_ids = list()
        

    def generate_notation_mattrix(self):
        print("*************** generate_notation_mattrix ******************")
        df = self.data_frame

        #df = df.dropna()
        """print(df.isna() )
        print(df.head())
        df = df.dropna()
        print(df.isna() )
        exit()"""
                
        n_users = df[self.user_key_name].nunique()

        #n_tracks = df[self.item_key].nunique()
        n_tracks = df[self.item_key_name].nunique()

        print("Nombre d 'utilisateurs : ",n_users)
        print("Nombre de morceaux : ",n_tracks)
        #la matrice de notations 
        #               chaque ligne représente les notes données par un utilisateur 
        #              chaque colonne les notes attribuées à un contenu. 
        #self.matrice_pivot = df.pivot_table(columns=self.item_key, index=self.user_key_name, values=self.target_key_name)
        
        #data = np.array(df)
        #data = np.asmatrix(df,colu)
        #data = np.as_matrix(df)
        #print(data)
        self.matrice_pivot = df.pivot_table(columns=self.item_key_name
                                            , index=self.user_key_name
                                            , values=self.target_key_name)

        #print(self.matrice_pivot)
        print("Nombre de notes manquantes : ",self.matrice_pivot.isna().value_counts().sum())

        #25.06.2024
        self.matrice_pivot = self.matrice_pivot +1
        
        self.matrice_pivot.fillna(0, inplace=True)
        #

        print("Matrice de notation (shape = ",self.matrice_pivot.shape,") \n\
              ---les colonnes contiennent les notes données par un utilisateur\n\
              ---les lignes contiennent les notes attribuées à un contenu")
        
        # Convertir la matrice de notations 'self.matrice_pivot' en une matrice creuse 'sparse_ratings'.
        self.sparse_ratings = csr_matrix(self.matrice_pivot)        
        print(self.sparse_ratings)
        """
        note
            content--        c1                  c2             c3              ... 
        user    
          |

        u1                  note[u1,c1]         none            note[u1,c3]     ...
        u2                  none                note[u2,c2]     none            ...
        u3                  note[u3,c1]         note[u3,c2]     note[u3,c3]     ...
        ...
        """

        # Extraire les identifiants des utilisateurs et les track_id à partir de la matrice de notations.
        self.user_ids = self.matrice_pivot.index.tolist()  
        #print("Nombre de users dans la matrice pivot : ",len(self.user_ids) )
        self.track_ids = self.matrice_pivot.columns.tolist()  
        #print("Nombre de morceaux dans la matrice pivot : ",len(self.track_ids) )

        self.user_similarity = self.get_user_similarity()
        self.item_similarity = self.get_item_similarity()

        # Afficher la matrice creuse 'sparse_ratings'.
        #print(sparse_ratings)

    def get_preference(self,user_id):
        return self.matrice_pivot.loc[user_id, :].values    

    def sim_cos(self,x, y):
        # Calcul du produit scalaire entre les vecteurs 'x' et 'y'.
        dot_product = np.dot(x, y)
        
        # Calcul des normes euclidiennes de 'x' et 'y'.
        norm_x = np.sqrt(np.sum(x ** 2))
        norm_y = np.sqrt(np.sum(y ** 2))
        
        # Vérification si l'une des normes est nulle pour éviter une division par zéro.
        if norm_x == 0 or norm_y == 0:
            return 0
        
        # Calcul de la similarité cosinus en utilisant la formule.
        similarity = dot_product / (norm_x * norm_y)
        return similarity
    
    def  get_preferences(self,userid,score_seuil,number_of_ligns = None) -> pd.DataFrame :                

        user_preferences = self.data_frame[(self.data_frame[self.user_key_name] == userid) 
                                           & (self.data_frame[self.target_key_name] >= score_seuil)
                                           & (self.data_frame[self.target_key_name] != np.nan)]        
        
        user_preferences = user_preferences.sort_values(self.target_key_name, ascending=False).drop_duplicates(subset=[self.item_key_name])        
                
        num =  10
        if number_of_ligns != None:
            user_preferences = user_preferences.head(number_of_ligns)            

        ret = user_preferences
        if self.visual_key_name != '':
            ret = ret[[self.item_key_name,self.visual_key_name,self.target_key_name]]
        else :    
            ret = ret[[self.item_key_name,self.target_key_name]]    
        ret = ret.sort_values(self.target_key_name, ascending=False)
        

        #print("**************************** get_preferences(score_seuil = ",score_seuil,") ****************************>")
        return ret
    
    def get_item_similarity(self):
        #print("************** get_item_similarity *************************")
        item_similarity = dist.cosine_similarity(self.sparse_ratings.T)

        # Création d'un DataFrame pandas à partir de la matrice de similarité entre utilisateurs.
        # Les index et les colonnes du DataFrame sont les identifiants des utilisateurs.
        item_similarity = pd.DataFrame(item_similarity, index=self.track_ids, columns=self.track_ids)
        
        return item_similarity

    def get_user_similarity(self):
        #print("************** get_user_similarity *************************")   
        # Utilisation de la fonction 'cosine_similarity' du module 'dist' pour calculer la similarité cosinus entre les utilisateurs.    
        user_similarity = dist.cosine_similarity(self.sparse_ratings)

        # Création d'un DataFrame pandas à partir de la matrice de similarité entre utilisateurs.
        # Les index et les colonnes du DataFrame sont les identifiants des utilisateurs.
        user_similarity = pd.DataFrame(user_similarity, index=self.user_ids, columns=self.user_ids)
        
        return user_similarity
    
    def get_similar_users(self,  nearest_count, user_id):
        
        #user_similarity = self.get_user_similarity()
        # Utilisation de la fonction 'cosine_similarity' du module 'dist' pour calculer la similarité cosinus entre les utilisateurs.
        user_similarity = dist.cosine_similarity(self.sparse_ratings)

        # Création d'un DataFrame pandas à partir de la matrice de similarité entre utilisateurs.
        # Les index et les colonnes du DataFrame sont les identifiants des utilisateurs.
        user_similarity = pd.DataFrame(user_similarity, index=self.user_ids, columns=self.user_ids)
        
        # Sélectionner dans matrice pivot les morceaux qui n'ont pas été encore écouté par le user        

        # Sélectionner les k users les plus similaires en excluant le user lui-même
        similar_users = user_similarity.loc[user_id].sort_values(ascending=False)[1:nearest_count+1]
        return similar_users
    
    def pred_user(self,  nearest_count, user_id,number_of_predictions=None):
        print("*************** pred_user *****************")
        # Sélectionner dans mat_ratings les contenus qui n'ont pas été encore écouté par le user
        to_predict = self.matrice_pivot.loc[user_id][self.matrice_pivot.loc[user_id]==0]                
            
        # Sélectionner les k users les plus similaires en excluant le user lui-même
        #similar_users = user_similarity.loc[user_id].sort_values(ascending=False)[1:nearest_count+1]
        similar_users = self.get_similar_users(nearest_count,user_id)
                
        # Calcul du dénominateur
        norm = np.sum(np.abs(similar_users))
        #print("len(to_predict.index) = ",len(to_predict.index))
        
        for i in to_predict.index:
            # Récupérer les notes des users similaires associées au morceau i
            ratings = self.matrice_pivot[i].loc[similar_users.index]
            # Calculer le produit scalaire entre ratings et similar_users
            scalar_prod = np.dot(ratings,similar_users)            
            #Calculer la note prédite pour le film i
            pred = scalar_prod / norm

            # Remplacer par la prédiction
            to_predict[i] = pred
        retpredict = to_predict
        
        #to_predict = pd.merge(to_predict,self.data_frame, on=[self.item_key], how='inner')
        #to_predict = to_predict.sort_values(by=self.target_key_name,ascending =False)
        if number_of_predictions != None:
            retpredict  = retpredict.head(number_of_predictions )

        retpredict = retpredict[retpredict != 0]    
                        
        retpredict = retpredict.sort_values(ascending =False)

        return retpredict

    #def pred_item(mat_ratings, item_similarity, k, user_id):
    def pred_item(self, nearest_count, user_id,number_of_predictions=None):
        print("*************** pred_item user_id[",user_id,"]*****************")
        item_similarity = self.get_item_similarity()
        
        # Sélectionner dans la self.matrice_pivot les morcecaux qui n'ont pas été encore écouté par l utilisateur
        #print(self.matrice_pivot.loc[user_id])
        
        to_predict = self.matrice_pivot.loc[user_id][self.matrice_pivot.loc[user_id]==0]
        #to_predict.rename("score moyen", inplace=True)
        """
        for i in to_predict.index:            
            similar_items = item_similarity.loc[i].sort_values(ascending=False)[1:nearest_count+1]            
            norm = np.sum(np.abs(similar_items))            
            ratings = self.matrice_pivot[similar_items.index].loc[user_id]
            scalar_prod = np.dot(ratings,similar_items)                        
            pred = scalar_prod / norm                    
            to_predict[i] = pred
        """
        
        # Itérer sur tous ces morceaux 
        for i in to_predict.index:

            #Trouver les k morceaux les plus similaires en excluant le morceau lui-même
            similar_items = item_similarity.loc[i].sort_values(ascending=False)[1:nearest_count+1]

            # Calcul de la norme du vecteur similar_items
            norm = np.sum(np.abs(similar_items))

            # Récupérer les notes données par l'utilisateur aux k plus proches voisins
            ratings = self.matrice_pivot[similar_items.index].loc[user_id]

            # Calculer le produit scalaire entre ratings et similar_items
            scalar_prod = np.dot(ratings,similar_items)
            
            #Calculer la note prédite pour le morceau i
            pred = scalar_prod / norm
        
            # Remplacer par la prédiction
            if pred != 0:
                to_predict[i] = pred
        
        #to_predict = pd.merge(to_predict,self.data_frame, on=[self.item_key], how='inner')
        if number_of_predictions != None:
            to_predict  = to_predict.head(number_of_predictions )
        to_predict = to_predict[to_predict != 0]
        to_predict = to_predict.sort_values(ascending =False)
        print(to_predict)
        
        return to_predict

#################################################"modéle hybride"
    
class mic_hybrid_filtering(mic_base_filter):
    def __init__(self):
        print('*****************mic_hybrid_filtering __init()__ *************************')
        mic_base_filter.__init__(self)
        self.best_params_predictor = {}
        self.data = any
        
    def set_data(self, file_name):
        super().set_data(file_name)
        self.print_vars()
        
        min_rating = self.data_frame[self.target_key_name].min()
        max_rating = self.data_frame[self.target_key_name].max()

        print("min_rating = ",min_rating)
        print("max_rating = ",max_rating)

        reader = Reader(rating_scale=(min_rating, max_rating))
        self.data = Dataset.load_from_df(self.data_frame[[self.user_key_name,
                                                                 self.item_key_name,
                                                                  self.target_key_name]], reader)
        print(self.data_frame.dtypes)
        #exit()

    def predictor_ajustement(self,predictor,param_search):    
        results = cross_validate(predictor, self.data, measures=['RMSE', 'MAE'], cv=10, verbose=True)

        print("predictor_ajustement() Average MAE: ", np.average(results["test_mae"]))
        print("predictor_ajustement() Average RMSE: ", np.average(results["test_rmse"]))

        
        gs = GridSearchCV(SVD, param_search, measures=['rmse', 'mae'], cv=10)
        gs.fit(self.data)
        
        print(gs.best_score['rmse'])
        print(gs.best_params['rmse'])        
        
        # best hyperparameters
        #self.best_params_predictor = gs.best_params['rmse']['n_factors']
        #self.best_params_predictor = gs.best_params['rmse']['n_epochs']
        self.best_params_predictor = gs.best_params['rmse']
        return self.best_params_predictor
    def compute_antitraintest(self,user_id):
                # Récupération de la liste des morceaux
        track_ids = self.data_frame[self.item_key_name].unique()
        
        #Récupération des morceaux écoutés par l'utilisateur identifié par user_id
        #track_ids_user = self.data_frame.loc[self.data_frame[self.user_key_name] == user_id, self.item_key_name]
        track_ids_user = self.data_frame.loc[self.data_frame[self.user_key_name] == user_id]

        #On considère que les morceaux sans note n'on pas étét écoutés
        #track_ids_user = track_ids_user.dropna()

        track_ids_user = track_ids_user.loc[self.data_frame[self.user_key_name] == user_id, self.item_key_name]
        #track_ids_user = track_ids_user
        print("l'utilisateur [",user_id,"] a écouté ",len(track_ids_user)," morceaux")
        
        # Récupération des morceaux non écoutés par l'utilisateur identifié par user_id
        track_ids_to_pred = np.setdiff1d(track_ids, track_ids_user)        
            #track_ids_to_pred = self.data_frame.loc[self.data_frame[self.user_key_name] != user_id, self.item_key_name]

        print("l'utilisateur [",user_id,"] n a pas écouté ",len(track_ids_to_pred)," morceaux")
        #print(track_ids_to_pred[0])
                    
        list_out = [[user_id, track_id, 0] for track_id in track_ids_to_pred]
        return list_out

    def compute_antitraintest_with_threshold(self,user_id,seuil):
        df = self.data_frame
        min_rating = self.data_frame[self.target_key_name].min()
        max_rating = self.data_frame[self.target_key_name].max()

        reader = Reader(rating_scale=(min_rating, max_rating))
        df_surprise = Dataset.load_from_df(df[[self.user_key_name, self.item_key_name, self.target_key_name]], reader=reader)
        
        train_set = df_surprise.build_full_trainset()

        targetUser = train_set.to_inner_uid(user_id)        
        
        # Obtenir la valeur de remplissage à utiliser (moyenne globale des notes du jeu d'entraînement)
        moyenne = train_set.global_mean
        print("moyenne = ",moyenne)
        
        # Obtenir les évaluations de l'utilisateur cible pour les trackss
        user_note = train_set.ur[targetUser]
        #print(user_note)
        #on enlève les morceau non  notés, si pas noté pas écouté...
        from math import isnan
        #suppression des nan de la liste de tuple
        #user_note = [t for t in user_note if not any(isinstance(n, float) and isnan(n) for n in t)]        
        user_note = [t for t in user_note 
                     if not any( isnan(n)  for n in t) 
                     #and any(isinstance(n, float) and n > seuil for n in t)]        
                     and any(isinstance(n, float) and n > seuil for n in t)]        
        
        #user_note = [t for t in user_note if not any(isinstance(n, float) and n > 7 and isnan(n) for n in t)]        
        #print(user_note)
        
        
        # Extraire la liste des morceaux notés par l'utilisateur
        user_tracks = [item for (item,_) in (user_note)]

        # Obtenir toutes les notations du jeu d'entraînement
        ratings = train_set.all_ratings()
        #for el in ratings:
        #    print(el)
        
        list_out = []
        # Boucle sur tous les items du jeu d'entraînement
        for track in train_set.all_items():
            # Si l'item n'a pas été noté par l'utilisateur
            if track not in user_tracks:
                # Ajouter la paire (utilisateur, morceau, valeur de remplissage) à la liste "anti-testset"
                list_out.append((user_id, train_set.to_raw_iid(track), moyenne))

        return list_out
    

    def predict(self,user_id,predictor,number_of_predictions=None) :        
        print("************predict *********************")      
        trainset = self.data.build_full_trainset()
        print('trainset.n_users = ',trainset.n_users)
        print('trainset.n_items = ',trainset.n_items)
        print('trainset.n_ratings = ',trainset.n_ratings)
        # Entrainement de  aglorithme sur le trainset
        #testset = self.compute_antitraintest(user_id)
        testset = self.compute_antitraintest_with_threshold(user_id,4)
        #print('len(testset) = ',len(testset))
        ##################
        predictor.fit(trainset)
        #list_out = self.compute_antitraintest(user_id)
        
        # Prediction des scores and generations des recommendations
        predictions = predictor.test(testset)
        
        #n_items = 10
        #print("Top {0} item recommendations for user {1}:".format(n_items, user_id))
        # on garde les plus conseillées
        #self.data_frame[self.data_frame[self.item_key_name]==track_id][self.item_key_name].values[0], pred_ratings[i]    
        predictions = pd.DataFrame(predictions)

        predictions.sort_values(by=['est'], inplace=True, ascending=False)

        ldict ={'uid':self.user_key_name,'iid':self.item_key_name}
        predictions = predictions.rename(ldict,axis=1)
        if number_of_predictions != None:
            predictions  = predictions.head(number_of_predictions )
        return predictions

    def predict_with_train_split(self,user_id,predictor,number_of_predictions=None) :      
        print("************predict_with_train_split *********************")              
        trainset, testset = train_test_split(self.data, test_size=.50)
        
        print('trainset.n_users = ',trainset.n_users)
        print('trainset.n_items = ',trainset.n_items)
        print('trainset.n_ratings = ',trainset.n_ratings)

        #testset = self.compute_antitraintest(user_id)
        testset = self.compute_antitraintest_with_threshold(user_id,4)
        #build_testset
        print('len(testset) = ',len(testset))
        #############################    
        predictor.fit(trainset)
        
        # Construire le jeu d'entraînement complet à partir du DataFrame df_surprise        
        
        predictions = predictor.test(testset)  
        #
        predictions = pd.DataFrame(predictions)

        # Trier les prédictions par estmiations décroissantes
        predictions.sort_values(by=['est'], inplace=True, ascending=False)
        
        ldict ={'uid':self.user_key_name,'iid':self.item_key_name}
        predictions = predictions.rename(ldict,axis=1)
        if number_of_predictions != None:
            predictions  = predictions.head(number_of_predictions )
        return predictions 

    