#import sys
#sys.path.append('../models')
#print("test_mic_visualization.py =>__package__ ",__package__ )
#print("test_mic_visualization.py =>__name__ ",__name__ )
#print("test_mic_visualization.py =>__file__ ",__file__ )
import sys
from pathlib import Path

file = Path(__file__).resolve()
package_root_directory = file.parents[1]
sys.path.append(str(package_root_directory))

from visualization.visualize import mic_vizualizer


input_path = '../Data/'
file_name = 'user_and_track_sentiment.csv'

myfilter = mic_vizualizer()

myfilter.set_data(input_path + file_name)
myfilter. plot_repartion_target('sentiment_score',False)