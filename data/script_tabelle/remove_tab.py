import pandas as pd
import numpy
from IPython.display import display

df = pd.read_csv("Training_2_357picchi.csv",
                    delimiter=';', index_col='ID Strain')
df2 = pd.read_csv("Testing_2.csv",
                      delimiter=';', index_col='ID Strain')

index = df2.index
df_restrict = df[~df.index.isin(index)] 
display(df_restrict)
df_restrict.to_csv('Training_4_357picchi_NoTest2.csv', sep=';')
    