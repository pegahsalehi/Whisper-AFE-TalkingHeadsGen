
'''
The AUE evaluation is based on OpenFace (https://github.com/TadasBaltrusaitis/OpenFace). 
First, use OpenFace's FeatureExtraction to process the reconstructed video ("A_generated.mp4" for example) 
and the corresponding GT ("A_GT.mp4" for example) respectively.
Then, run "python auerror.py A_generated A_GT"

Default directory structure is:

|--- auerror.py
|--- OpenFace_2.2.0_win_x64
     |--- proceessed
     |--- ...

'''

import pandas as pd
import numpy as np

# List of Action Units (AU) items
AUitems = [' AU01_r',' AU02_r', ' AU04_r', ' AU05_r', ' AU06_r', ' AU07_r', ' AU09_r', ' AU10_r', ' AU12_r', ' AU14_r', ' AU15_r', ' AU17_r', ' AU20_r', ' AU23_r', ' AU25_r', ' AU26_r', ' AU45_r']

# Paths to your CSV files
df_1_path = '/home/host/evl/RAD-NeRF/Ground_Trouth/Shaheen.csv'
df_2_path = '/home/host/evl/ER-NeRF/whisper/Shaheen.csv'

# Read the CSV files
df_1 = pd.read_csv(df_1_path)[AUitems]
df_2 = pd.read_csv(df_2_path)[AUitems]

# Calculate the error (squared differences)
error = (df_1 - df_2) ** 2
print(error.mean().sum())

# Separate AU items into lower and upper categories
AUitems_lower = [' AU10_r', ' AU12_r', ' AU14_r', ' AU15_r', ' AU17_r', ' AU20_r', ' AU23_r', ' AU25_r', ' AU26_r']
AUitems_upper = [' AU01_r',' AU02_r', ' AU04_r', ' AU05_r', ' AU06_r', ' AU07_r', ' AU09_r', ' AU45_r']

# Calculate the error for lower and upper AUs
error_l = (df_1[AUitems_lower] - df_2[AUitems_lower]) ** 2
error_u = (df_1[AUitems_upper] - df_2[AUitems_upper]) ** 2

# Print the results
print('Lower AUs error:', error_l.mean().sum(), 'Upper AUs error:', error_u.mean().sum())
