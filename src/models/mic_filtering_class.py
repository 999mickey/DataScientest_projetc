
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


class mic_base_filter():
    def __init__(self):
        self.data_frame = pd.DataFrame()
        self.features = []
        self.target_key_name = ""
        self.file_name = ""
        self.df_features = pd.DataFrame()        
        self.artist_key_name = ''        
        self.item_key_name = ''
        self.user_key_name = ''        
        self.visual_key_name = ''
    #on utilise cette variable pour identifier humainement les mrorceaux
    def print_vars(self):
        print("self.artist_key_name = ",self.artist_key_name )
        print("self.item_key_name = ",self.item_key_name )
        print("self.user_key_name = ",self.user_key_name )
        print("self.visual_key_name = ",self.visual_key_name )
        print("self.target_key_name = ",self.target_key_name )
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
            
    def get_tracks_of_artist(self,artist_name):
        print("************************* get_tracks_of_artist(",artist_name,") **********************")
        artist_lower = artist_name.lower()
        df = self.data_frame[self.artist_key_name].str.lower() 
                
        #condition = (df.str.startswith(artist_lower) )#ok            
        condition = (df.str.contains(artist_lower) )#ok            
        
        return self.data_frame[condition][[self.artist_key_name,self.item_key_name]]
    
    def get_random_track_num_of_artist(self,artist_name):
        ret = self.get_tracks_of_artist(artist_name)        
        n = random.randint(0,len(ret)-1) 
        ret = ret.iloc[n,:]
        return ret

    def get_track_num_of_artist(self,artist_name,song_name):
        print("************************* get_track_num_of_artist(",artist_name,",",song_name,") **********************")
        ret = self.get_tracks_of_artist(artist_name)
        
        song_name_lower = song_name.lower()
        
        df = self.data_frame[self.item_key_name].str.lower() 
        #condition = (df.str.startswith(song_name_lower) )#ok            
        condition = (df.str.contains(song_name_lower) )#ok            

        ret = ret[condition]    
        return ret.index.tolist()[0]
    
    def set_data(self,file_name):
        print("************************* set_data(",file_name,") **********************")
        self.file_name = file_name
        self.data_frame = pd.read_csv(file_name)
        
        self.data_frame = self.data_frame.drop_duplicates()
        self.data_frame.style.set_properties(**{'text-align': 'left'})
        #self.data_frame.dropna()
        self.data_frame.info()
    
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
        df.info()    
        print(varlist)
        df  = df[varlist]        

        df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
        df.reindex()        
        self.data_frame = df
                
    def get_user_df(self, userid):
        return self.data_frame[self.data_frame[self.user_key_name] == userid]

    def get_random_track_id(self):
        n = random.randint(0,len(self.data_frame)-1) 
        return n

        
    def get_top_rated_songs(self,num):
        print("**********************get_top_rated_songs ************************")
        df = self.data_frame
        aggregated_data = df.groupby(self.item_key_name )[self.target].count().reset_index()
        #aggregated_data = df.groupby(self.item_key_name )[self.target].count()

        # Tri du DataFrame agrégée par note en ordre décroissant
        sorted_aggregated_data = aggregated_data.sort_values(by=self.target, ascending=False)

        # Sélection des 10 premiers morceaux les mieux notés
        top_rated_tracks = sorted_aggregated_data.head(num)
        return top_rated_tracks
    
    def get_top_rater_users(self,num):
        df = self.data_frame
        aggregated_data = df.groupby(self.user_key_name )[self.target].count().reset_index()
        
        # Tri du DataFrame agrégée par note en ordre décroissant
        sorted_aggregated_data = aggregated_data.sort_values(by=self.target, ascending=False)

        # Sélection des 10 premiers morceaux les mieux notés
        top_rater_users = sorted_aggregated_data.head(num)
        return top_rater_users

    #normalisation des donées de sortie
    def get_full_data(self,indf,key ):
        print("***************** get_full_data ******************")
        ret = self.data_frame.merge(indf,how='right',on=key )
        if self.visual_key_name == '':
            ret = ret[self.item_key_name]
        else:
            ret = ret[[self.item_key_name,self.visual_key_name]]
        
        return ret
    
    def get_util_data_for_presentation(self,df):
        print("***************** get_util_data_for_presentation ******************")
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
        aggregated_data = df.groupby(self.item_key_name)[self.target_key_name].mean().reset_index()

        # Tri du DataFrame agrégée par note en ordre décroissant
        sorted_aggregated_data = aggregated_data.sort_values(by=self.target, ascending=False)

        # Sélection des 10 premiers tracks les mieux notés
        print("Morceaux les plus mieux notés : ")
        best_rated_tracks = sorted_aggregated_data.head(10)
        print(best_rated_tracks)
        return best_rated_tracks

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
        
        self.data_frame['song_id'] = self.df_normalized.index
        
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
        
        n_users = df[self.user_key_name].nunique()

        #n_tracks = df[self.item_key].nunique()
        n_tracks = df[self.item_key_name].nunique()

        print("Nombre d 'utilisateurs : ",n_users)
        print("Nombre de morceaux : ",n_tracks)
        #la matrice de notations 
        #               chaque ligne représente les notes données par un utilisateur 
        # e             chaque colonne les notes attribuées à un contenu. 
        #self.matrice_pivot = df.pivot_table(columns=self.item_key, index=self.user_key_name, values=self.target)
        
        #data = np.array(df)
        #data = np.asmatrix(df,colu)
        #data = np.as_matrix(df)
        #print(data)
        
        #exit()
        """test = df.pivot_table(columns=self.item_key_name
                                            , index=self.user_key_name
                                            , values=self.target_key_name
                                            ,dropna=False)
        test.to_csv("savetemp.csv")"""


        self.matrice_pivot = df.pivot_table(columns=self.item_key_name
                                            , index=self.user_key_name
                                            , values=self.target_key_name)

        #toberemoved
        #self.matrice_pivot.to_csv("matrice_pivot.csv")    
        #
        #exit()
        self.matrice_pivot = self.matrice_pivot +1

        #print(self.matrice_pivot)

        self.matrice_pivot.fillna(0, inplace=True)
        print("Matrice de notation (shape = ",self.matrice_pivot.shape,") \n\
              ---les colonnes contiennent les notes données par un utilisateur\n\
              ---les lignes contiennent les notes attribuées à un contenu")
        
        # Convertir la matrice de notations 'self.matrice_pivot' en une matrice creuse 'sparse_ratings'.
        self.sparse_ratings = csr_matrix(self.matrice_pivot)        
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
        """
        Anvant d appeler cette fonction il faut générer la matrice pivot avec generate_notation_mattrix()
        """
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
        print("**************************** get_preferences(score_seuil = ",score_seuil,") ****************************<")
        #user_preferences = self.data_frame[(self.data_frame[self.user_key_name] == userid) & (self.data_frame[self.target] >= score_seuil)]        
        

        user_preferences = self.data_frame[(self.data_frame[self.user_key_name] == userid) 
                                           & (self.data_frame[self.target_key_name] >= score_seuil)
                                           & (self.data_frame[self.target_key_name] != np.nan)]        
        
        user_preferences = user_preferences.sort_values(self.target_key_name, ascending=False).drop_duplicates(subset=[self.item_key_name])
        print(user_preferences)
                
        num =  10
        if number_of_ligns != None:
            num = number_of_ligns

        #ret = user_preferences.sort_values(self.target_key_name, ascending=False).drop_duplicates().head(num)
        ret = user_preferences
        if self.visual_key_name != '':
            ret = ret[[self.item_key_name,self.visual_key_name,self.target_key_name]]
        else :    
            ret = ret[[self.item_key_name,self.target_key_name]]    
        #print("toberemoved len(user_preferences) = ",user_preferences)
        ret = ret.sort_values(self.target_key_name, ascending=False)
        print("toberemoved ret = ",ret)

        print("**************************** get_preferences(score_seuil = ",score_seuil,") ****************************>")
        return ret
    
    def get_item_similarity(self):
        print("************** get_item_similarity *************************")
        item_similarity = dist.cosine_similarity(self.sparse_ratings.T)

        # Création d'un DataFrame pandas à partir de la matrice de similarité entre utilisateurs.
        # Les index et les colonnes du DataFrame sont les identifiants des utilisateurs.
        item_similarity = pd.DataFrame(item_similarity, index=self.track_ids, columns=self.track_ids)
        
        return item_similarity

    def get_user_similarity(self):
        print("************** get_item_similarity *************************")   
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
    
    def pred_user(self,  nearest_count, user_id):
        print("*************** pred_user *****************")
        # Sélectionner dans mat_ratings les contenus qui n'ont pas été encore écouté par le user
        to_predict = self.matrice_pivot.loc[user_id][self.matrice_pivot.loc[user_id]==0]        
        to_predict.rename("score moyen", inplace=True)
    
        
        # Sélectionner les k users les plus similaires en excluant le user lui-même
        #similar_users = user_similarity.loc[user_id].sort_values(ascending=False)[1:nearest_count+1]
        similar_users = self.get_similar_users(nearest_count,user_id)
                
        # Calcul du dénominateur
        norm = np.sum(np.abs(similar_users))
        #print("len(to_predict.index) = ",len(to_predict.index))
        
        for i in to_predict.index:
            # Récupérer les notes des users similaires associées au film i
            ratings = self.matrice_pivot[i].loc[similar_users.index]
            
            # Calculer le produit scalaire entre ratings et similar_users
            scalar_prod = np.dot(ratings,similar_users)
            
            #Calculer la note prédite pour le film i
            pred = scalar_prod / norm

            # Remplacer par la prédiction
            to_predict[i] = pred
        #to_predict = pd.merge(to_predict,self.data_frame, on=[self.item_key], how='inner')
        return to_predict

    #def pred_item(mat_ratings, item_similarity, k, user_id):
    def pred_item(self, nearest_count, user_id):
        print("*************** pred_item *****************")
        item_similarity = self.get_item_similarity()
        
        # Sélectionner dans la self.matrice_pivot les morcecaux qui n'ont pas été encore écouté par l utilisateur
        to_predict = self.matrice_pivot.loc[user_id][self.matrice_pivot.loc[user_id]==0]
        to_predict.rename("score moyen", inplace=True)
        
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
            to_predict[i] = pred

        #to_predict = pd.merge(to_predict,self.data_frame, on=[self.item_key], how='inner')
        return to_predict

    #################################################"modéle hybride"
    def pred_user_with_similarity(self,mat_ratings, user_similarity, k, user_id):
        mat_ratings = self.matrice_pivot #ou pas

        # Sélectionner dans mat_ratings les items qui n'ont pas été encore écoutés par le user
        to_predict = mat_ratings.loc[user_id][mat_ratings.loc[user_id]==0]

        # Sélectionner les k users les plus similaires en excluant le user lui-même
        similar_users = user_similarity.loc[user_id].sort_values(ascending=False)[1:k+1]
        
        # Calcul du dénominateur
        norm = np.sum(np.abs(similar_users))
        #print("len(to_predict.index) = ",len(to_predict.index))
        for i in to_predict.index:
            # Récupérer les notes des users similaires associées au film i
            ratings = mat_ratings[i].loc[similar_users.index]
            
            # Calculer le produit scalaire entre ratings et similar_users
            scalar_prod = np.dot(ratings,similar_users)
            
            #Calculer la note prédite pour le film i
            pred = scalar_prod / norm

            # Remplacer par la prédiction
            to_predict[i] = pred

        return to_predict


    #def pred_item_with_similarity(self,mat_ratings,item_similarity ,nearest_count, user_id):
    def pred_item_with_similarity(self,item_similarity ,nearest_count, user_id):
        print("*************** pred_item *****************")
        mat_ratings = self.matrice_pivot #ou pas                    
        # Sélectionner dans la self.matrice_pivot les morcecaux qui n'ont pas été encore lu par le user
        #to_predict = self.matrice_pivot.loc[user_id][self.matrice_pivot.loc[user_id]==0]
        to_predict = mat_ratings.loc[user_id][mat_ratings.loc[user_id]==0]
        
        # Itérer sur tous ces morceaux 
        for i in to_predict.index:

            #Trouver les k morceaux les plus similaires en excluant le morceau lui-même
            similar_items = item_similarity.loc[i].sort_values(ascending=False)[1:nearest_count+1]

            # Calcul de la norme du vecteur similar_items
            norm = np.sum(np.abs(similar_items))

            # Récupérer les notes données par l'utilisateur aux k plus proches voisins
            #ratings = self.matrice_pivot[similar_items.index].loc[user_id]
            ratings = mat_ratings[similar_items.index].loc[user_id]

            # Calculer le produit scalaire entre ratings et similar_items
            scalar_prod = np.dot(ratings,similar_items)
            
            #Calculer la note prédite pour le morceau i
            pred = scalar_prod / norm

            # Remplacer par la prédiction
            to_predict[i] = pred

        return to_predict
    
    def get_item_similarity_transformed(self):
        print("************** get_item_similarity_transformed *************************")
        svd = TruncatedSVD(n_components=12)

        ratings = svd.fit_transform(self.sparse_ratings.T)

        print(ratings.shape)

        item_similarity = dist.cosine_similarity(ratings)

        # Création d'un DataFrame pandas à partir de la matrice de similarité entre utilisateurs.
        # Les index et les colonnes du DataFrame sont les identifiants des utilisateurs.
        item_similarity = pd.DataFrame(item_similarity, index=self.track_ids, columns=self.track_ids)
        
        return item_similarity
    
    def get_item_similarity_invtransformed(self):
        print("************** get_item_similarity_invtransformed *************************")
        svd = TruncatedSVD(n_components=12)

        ratings = svd.fit_transform(self.sparse_ratings.T)
        new_ratings = svd.inverse_transform(ratings)
        
        item_similarity = dist.cosine_similarity(new_ratings)

        # Création d'un DataFrame pandas à partir de la matrice de similarité entre utilisateurs.
        # Les index et les colonnes du DataFrame sont les identifiants des utilisateurs.
        item_similarity = pd.DataFrame(item_similarity, index=self.track_ids, columns=self.track_ids)
        
        return item_similarity

    #prdictor = SVD ou NormalPredictor

class mic_hybrid_filtering(mic_base_filter):
    def __init__(self):
        print("*****************mic_hybrid_filtering __init()__ *************************")
        mic_base_filter.__init__(self)
    def compute_antitraintest(self,user_id):
        print("******************  compute_antitraintest ****************************")
        df = self.data_frame
        reader = Reader(rating_scale=(0, 5))
        df_surprise = Dataset.load_from_df(df[[self.user_key_name, self.item_key_name, self.target_key_name]], reader=reader)
        train_set = df_surprise.build_full_trainset()

        targetUser = train_set.to_inner_uid(user_id)        
        
        # Obtenir la valeur de remplissage à utiliser (moyenne globale des notes du jeu d'entraînement)
        moyenne = train_set.global_mean

        # Obtenir les évaluations de l'utilisateur cible pour les trackss
        user_note = train_set.ur[targetUser]

        # Extraire la liste des morceaux notés par l'utilisateur
        user_tracks = [item for (item,_) in (user_note)]

        # Obtenir toutes les notations du jeu d'entraînement
        ratings = train_set.all_ratings()

        print(ratings)

        list_out = []
        # Boucle sur tous les items du jeu d'entraînement
        for track in train_set.all_items():
            # Si l'item n'a pas été noté par l'utilisateur
            if track not in user_tracks:
                # Ajouter la paire (utilisateur, morceau, valeur de remplissage) à la liste "anti-testset"
                list_out.append((user_id, train_set.to_raw_iid(track), moyenne))

        return list_out

    def predict(self,user_id,predictor) :
        print("******************  predict(",user_id,",",predictor," **************************<")
        df = self.data_frame
        reader = Reader(rating_scale=(0, 5))

        df_surprise = Dataset.load_from_df(df[[self.user_key_name, self.item_key_name, self.target_key_name]], reader=reader)
        cross_validate(predictor, df_surprise, measures=['RMSE', 'MAE'], cv=5, verbose=True)
        # Construire le jeu d'entraînement complet à partir du DataFrame df_surprise        
        list_out = self.compute_antitraintest(user_id)
        svd = predictor

        #cross_validate(svd, df_surprise, measures=['RMSE', 'MAE'], cv=5, verbose=True)
        
                # test sur l'ensemble des iem auxquels le user n a aps participé
        predictions = svd.test(list_out)  

        predictions = pd.DataFrame(predictions)

        # Trier les prédictions par estmiations décroissantes
        predictions.sort_values(by=['est'], inplace=True, ascending=False)
        
        print("******************  predict(self,user_id,predictor) **************************<")
        ldict ={'uid':self.user_key_name,'iid':self.item_key_name}
        predictions = predictions.rename(ldict,axis=1)
        return predictions 
        
    def predictor_ajustement(self,predictor,param_search):
        print("***************** predictor_ajustement() ***************************<")
        df = self.data_frame
        reader = Reader(rating_scale=(0, 5))                
        df_surprise = Dataset.load_from_df(df[[self.user_key_name, self.item_key_name, self.target_key_name]], reader=reader)        

        """param_grid = {'n_factors': [100,150],
              'n_epochs': [20,25,30],
              'lr_all':[0.005,0.01,0.1],
              'reg_all':[0.02,0.05,0.1]}"""
        #grid_search = GridSearchCV(SVD, param_grid, measures=['rmse','mae'], cv=3)
        #grid_search = GridSearchCV(type(predictor), param_grid, measures=['rmse','mae'], cv=3)
        grid_search = GridSearchCV(type(predictor), param_search, measures=['rmse','mae'], cv=3)
        grid_search.fit(df_surprise)     

        print(grid_search.best_score['rmse'])

        print(grid_search.best_score['mae'])

        print(grid_search.best_params['rmse'])

        tunedParams= grid_search.best_estimator['rmse']
        print("cross_validate() => ")
        print(cross_validate(tunedParams, df_surprise, measures=['RMSE', 'MAE'], cv=5, verbose=True))

        print("***************** prdedictor_ajustement() ***************************>")
        #ret = pd.DataFrame(tunedParams.items())
        return tunedParams

    