# Steps on how the data was downloaded/created.
## METR_LA.csv
How the METR-LA.csv dataset can be downloaded/created
```
import h5py
import numpy as np
import pandas as pd

#change to h5 file path
#download it using one of the following sources
#https://www.kaggle.com/datasets/annnnguyen/metr-la-dataset
#https://www.kaggle.com/code/xiaohualu/mapvisualization-metrla-pydeck/input?select=metr-la.h5
#https://data.mendeley.com/datasets/s42kkc5hsw/1
filename = metr-la.h5'

#read h5 file
dataset = h5py.File(filename, 'r')

#print the first unknown key in the h5 file
print(dataset.keys()) #returns df


#save the h5 file to csv using the first key df
with pd.HDFStore(filename, 'r') as d:
    df = d.get('df')
    print(df)
    df.to_csv('./METR-LA.csv')
```
## graph_sensor_locations.csv
For re-running the map with the sensor locations, one needs the locations file for the sensors.
This can be received under one of the following resources:
https://www.kaggle.com/code/xiaohualu/mapvisualization-metrla-pydeck/input?select=graph_sensor_locations.csv
https://github.com/deepkashiwa20/DL-Traff-Graph/tree/main/METRLA/graph_sensor_locations.csv
https://github.com/chnsh/DCRNN_PyTorch/tree/pytorch_scratch/data/sensor_graph/graph_sensor_locations.csv
