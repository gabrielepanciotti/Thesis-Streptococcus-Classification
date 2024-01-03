import pandas as pd
from IPython.display import display
import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 46
N_JOBS = -1

class_names = ["Canis", "Dysg. Equisimilis", "Dysg. Dysgalactiae"]

map_target = {
    "Streptococcus canis": 2,
    "Streptococcus dysgalactiae subsp. dysgalactiae": 1,
    "Streptococcus dysgalactiae subsp. equisimilis": 0
}

map_target_inv = {
    2: "Canis",
    1: "Dysgalactiae",
    0: "Equisimilis"

}

list_agg = ['Animal species of origin_Bovine', 'Animal species of origin_Cat',
       'Animal species of origin_Dog', 'Animal species of origin_Donkey',
       'Animal species of origin_Goat', 'Animal species of origin_Hedgehog',
       'Animal species of origin_Horse', 'Animal species of origin_Ovine',
       'Animal species of origin_Reference strain (CCUG)',
       'Animal species of origin_Swine',
       'Animal species of origin_Water buffalo',
       'Animal species of origin_Wolf',
       'Animal species of origin_Wild boar',
       #'LANCEFIELD GROUP_A', 'LANCEFIELD GROUP_C', 'LANCEFIELD GROUP_G',
       'Haemolysis_a', 'Haemolysis_b']

start = 9
n_antibiotici = 9
n_geni = 27
n_virulenza = 18
n = 306

df = pd.read_csv("../data/Training_1_"+str(n)+"picchi.csv",
                    delimiter=';', index_col='ID Strain')

feat_agg = df.iloc[:,[2,8]]

subspecies = df[['Putative Subspecies']]
maldi = df.iloc[:,start:start+n]
display(feat_agg)
display(maldi)

'''antibiotici = df.iloc[:,start+n:start+n+n_antibiotici]
geni_antibiotici = df.iloc[:,start+n+n_antibiotici:start+n+n_antibiotici+n_geni]
virulenza = df.iloc[:,start+n+n_antibiotici+n_geni:start+n+n_antibiotici+n_geni+n_virulenza]'''

maldi.fillna(0, inplace=True)
maldi = maldi.replace(',', '.', regex=True)
columns = maldi.columns
for column in columns:
    maldi[column] = maldi[column].astype(float)

picchi_test = 56
df_test = pd.read_csv('../data/Testing_1_0%.csv',
                      delimiter=';', index_col='ID Strain')

df_test['Haemolysis'] = df_test['Haemolysis'].str.replace(" ", "")
feat_agg_test = df_test.iloc[:,[2,8]]
subspecies_test = df_test[['Putative Subspecies']]
maldi_test = df_test.iloc[:,start:start+picchi_test]
display(feat_agg_test)
display(maldi_test)

'''antibiotici_test = df_test.iloc[:,start+n_test:start+n_test+n_antibiotici]
geni_antibiotici_test = df_test.iloc[:,start+n_test+n_antibiotici:start+n_test+n_antibiotici+n_geni]
virulenza_test = df_test.iloc[:,start+n_test+n_antibiotici+n_geni:start+n_test+n_antibiotici+n_geni+n_virulenza]'''

maldi_test.fillna(0, inplace=True)
maldi_test = maldi_test.replace(',', '.', regex=True)
columns = maldi_test.columns
for column in columns:
    maldi_test[column] = maldi_test[column].astype(float)

'''feat_agg_dummies = pd.DataFrame.from_dict(pd.get_dummies(feat_agg_test))
feat_agg_dummies = feat_agg_dummies.astype(int)
missing_cols_agg = set(list_agg) - set(feat_agg_dummies.columns)

for c in missing_cols_agg:
    feat_agg_dummies[str(c)] = 0
feat_agg_dummies = feat_agg_dummies[list_agg]'''

col = maldi.columns.to_list()
col = [i.replace(',', '.') for i in col]
col = [int(float(i)) for i in col]

col_test = maldi_test.columns.to_list()
col_test = [i.replace(',', '.') for i in col_test]
col_test = [int(float(i)) for i in col_test]

maldi.columns = col
maldi_test.columns = col_test

tab1 = pd.concat([feat_agg, subspecies, maldi], axis=1)
tab2 = pd.concat([feat_agg_test, subspecies_test, maldi_test], axis=1)

tab = pd.concat([tab1, tab2], axis=0)
tab.fillna(0, inplace = True)
display(tab)
tab.to_csv('../data/Merge_tab_training.csv', index = True)