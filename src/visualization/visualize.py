#print("visualize.py =>__package__ ",__package__ )
#print("visualize.py =>__name__ ",__name__ )
#print("visualize.py =>__file__ ",__file__ )

from models.mic_filtering_class import mic_base_filter
import seaborn as sns
import matplotlib.pyplot as plt
class mic_vizualizer(mic_base_filter ):
    """
    mic_vizualizer
    objet de vizualisation de dataset
    """
    def __init__(self):
         mic_base_filter.__init__(self)


    def plot_repartion_target(self,key_name,dump):
        print("************** plot_repartion_target **********************")
        df = self.data_frame        
        
        max =  df[key_name].nunique()
        print("             max = ",max)
        sns.countplot(x=key_name, data=df, palette="viridis")
        #fig,ax = plt.subplots(figsize=(12,4))
        #ax.set_ylim(0, max)

        #axes.set_xticks(axes.get_xticks()[::2]
        if dump == True:
            plt.savefig('hist_'+key_name+'1.png')
        plt.show()
        plt.close()
    
    """
    def plot_repartion_target(self):
        print("************** plot_repartion_target **********************")
        df = self.data_frame        
        sns.countplot(x=self.target_key_name, data=df, palette="viridis")
        plt.title("Distribution des sentimentss", fontsize=14)
        
        #axes.set_xticks(axes.get_xticks()[::2]
        plt.savefig('matrice_de_notation_user_track_sentiment.png')
        plt.show()
        plt.close()
    """
    def plot_most_popular_tracks(self,num):
        print("***************** plot_most_popular_tracks ********************")
                
        sorted_aggregated_data = self.get_top_rated_songs(num)

        # Sélection des 10 premiers morceaux les plus populaires
        print("Morceaux les plus populaires : ")
        top_rated_tracks = sorted_aggregated_data.head(num)
        print(top_rated_tracks)
        df = df.set_index(self.item_key_name).T
        sns.barplot(y=self.item_key_name, x=self.target_key_name, data=top_rated_tracks, orient = 'h')
        
        plt.yticks(fontsize=5)
        plt.xlabel('Nombre de votes')
        plt.ylabel('Track')
        plt.title(f'Top 10 track les plus Populaires')
        plt.savefig('top10_track_popularity.png')

        plt.show()
        plt.close()


    def plot_best_noted_tracks(self,num):
        print("***************** plot_best_noted_tracks ********************")
            
        best_rated_tracks = self.get_best_voted_songs(10)        

        # Affichage du graphique
        
        sns.barplot(y=self.item_key_name, x=self.target_key_name, data=best_rated_tracks, orient = 'h')
        plt.yticks(fontsize=5)
        plt.title(f'Top 10 tracks les mieux notés')
        plt.savefig('top10_track_bestnoted.png')
        plt.xlabel("Note moyenne")
        plt.show()
        plt.close()

    def plot_most_voter_user(self):
        print("***************** plot_most_voter_user ********************")
        
        gb_user = self.get_best_voter_users(10)
        print(gb_user)

        # Affichage du graphique
        sns.barplot(y=self.user_key_name, x='count', data=gb_user, orient = 'h')
        plt.xlabel('Nombre de votes')
        plt.ylabel('User')
        plt.title(f'Top 10 des user les plus participatifs')
        plt.savefig('top10_user_voter_count.png')
        plt.show()
        plt.close()


