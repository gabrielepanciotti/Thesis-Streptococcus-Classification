import pandas as pd
import numpy
from IPython.display import display

start = 9
n_antibiotici = 9
n_geni = 27
n_virulenza = 18
n = 306

n2 = 56
n_antibiotici = 9
n_geni = 27
n_virulenza = 18

picco_max = 20000

df = pd.read_csv("Training_1_306picchi.csv",
                    delimiter=';', index_col='ID Strain')
df2 = pd.read_csv("Testing_1_0%.csv",
                      delimiter=';', index_col='ID Strain')

feat = df.iloc[:,0:9]
display(feat)
maldi = df.iloc[:,start:start+n]
targets = df.iloc[:,start+n:start+n+n_antibiotici+n_geni+n_virulenza]


#df_feat = pd.concat([feat_agg, subspecies], axis=1)
#df_targets = pd.concat([antibiotici, geni_antibiotici, virulenza], axis=1)

maldi.fillna(0, inplace=True)
maldi = maldi.replace(',', '.', regex=True)
columns = maldi.columns
for column in columns:
    maldi[column] = maldi[column].astype(float)

col = maldi.columns.to_list()
col = [i.replace(',', '.') for i in col]
col = [int(float(i)) for i in col]
maldi.columns = col

feat2 = df2.iloc[:,0:9]
maldi2 = df2.iloc[:,start:start+n2]
targets2 = df2.iloc[:,start+n2:start+n2+n_antibiotici+n_geni+n_virulenza]

#df2_feat = pd.concat([feat_agg2,  subspecies2], axis=1)
#df2_targets = pd.concat([antibiotici2, geni_antibiotici2, virulenza2], axis=1)

maldi2.fillna(0, inplace=True)
maldi2 = maldi2.replace(',', '.', regex=True)
columns = maldi2.columns
for column in columns:
    maldi2[column] = maldi2[column].astype(float)

col = maldi2.columns.to_list()
col = [i.replace(',', '.') for i in col]
col = [int(float(i)) for i in col]
maldi2.columns = col

maldi_all = pd.concat([maldi, maldi2], axis=0)
maldi_all.fillna(0, inplace = True)
display(maldi_all)

uni_feat = pd.concat([feat, feat2], axis=0)
display(uni_feat)
uni_targets = pd.concat([targets, targets2], axis=0)
display(uni_targets)
df_unified = pd.concat([uni_feat, maldi_all, uni_targets], axis=1)

df_unified.dropna(subset=['Clindamicina'], inplace=True)
display(df_unified)
df_unified.to_csv('Training_2_357picchi.csv', sep=';')
    