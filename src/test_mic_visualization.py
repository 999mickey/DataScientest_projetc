#import sys
#sys.path.append('../models')
#print("test_mic_visualization.py =>__package__ ",__package__ )
#print("test_mic_visualization.py =>__name__ ",__name__ )
#print("test_mic_visualization.py =>__file__ ",__file__ )
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

file = Path(__file__).resolve()
package_root_directory = file.parents[1]
sys.path.append(str(package_root_directory))

from visualization.visualize import mic_vizualizer
from input_variables_setter import input_variables_setter 

input_path = '../Data/'
file_name = 'user_and_track_sentiment.csv'
file_name = "simulationbig.csv"
file_name = "ratings_flrd.csv"
file_name = "data.csv"

myfilter = mic_vizualizer()
input_variables_setter(file_name,myfilter)
myfilter.set_data(input_path + file_name)
myfilter.data_frame.info()

#ret = myfilter.plot_repartion_count_target(myfilter.target_key_name)
#plt.show()

#plt.show()
#df = pd.read_csv(input_path + file_name)
#fig = sns.displot(x = 'popularity', data = df)
#plt.show()
#plt.title("Distribution de "+myfilter.target_key_name)
#plt.show()
#fig = sns.displot(x = myfilter.target_key_name, data = myfilter.data_frame)
#plt.title("Distribution de ",myfilter.target_key_name)
#myfilter.plot_best_noted_tracks(10)
#myfilter.plot_most_popular_tracks(10)
#myfilter. plot_repartion_cat_target('genre',False)
#myfilter. plot_repartion_count_target('sentiment_score',False)
#myfilter. plot_repartion_cat_target('track_id',False)
#myfilter. plot_repartion_cat_target('user_id',False)

#ret = myfilter.plot_principal_components()
#plt.show()
#myfilter.plot_pie_of_principal_comp()
#plt.show()
#myfilter.plot_corrrelation_circle()
#plt.show()
myfilter.plot_groupement_from_key(myfilter.target_key_name)
plt.show()

