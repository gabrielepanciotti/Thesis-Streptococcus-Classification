import pandas as pd
import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score, recall_score, f1_score
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
start = 2
n_antibiotici = 9
n_geni = 27
n_virulenza = 18
n = 306
picco_max = 20000

tutti_picchi = 'tutti_picchi_'
scaled = ''
scaler = ''
reduction = 'pca_'

def dimensionality_reduction_cluster(X,n_components):
    X.columns = X.columns.astype(str)
    pca = PCA(n_components)
    X_pca = pca.fit_transform(X)
    X_pca = pd.DataFrame(X_pca, index=X.index.to_list())
    X_pca.columns = X_pca.columns.astype(str)
    return X_pca

'''def dimensionality_reduction_cluster(X,n_components=0):
    X.columns = X.columns.astype(str)
    if n_components != 0:
        pca = PCA(n_components)
        X_pca = pca.fit_transform(X)
    else:
        pca = pickle.load(open('../models/pca_cluster_'+scaler+tutti_picchi+'npicchi'+str(n)+'_npicchimax'+str(picco_max)+'.pkl',"rb"))
        X_pca = pca.transform(X)
    X_pca = pd.DataFrame(X_pca, index=X.index.to_list())
    X_pca.columns = X_pca.columns.astype(str)
    return X_pca
'''

def makeScore(y_test, y_pred):
    score = {}

    score['acc'] = accuracy_score(y_test, y_pred)
    score['b_acc'] = balanced_accuracy_score(y_test, y_pred)
    score['st'] = score['acc'].std()
    score['prec'] = precision_score(y_test, y_pred, average='weighted')
    score['rec'] = recall_score(y_test, y_pred, average='weighted')
    score['f1'] = f1_score(y_test, y_pred, average='weighted')

    return score

N_CLUSTERS = 3
list_animals = ['Dog', 'Cat', 'Bovine', 'Swine', 'Ovine', 'Goat', 'Hedgehog',
       'Horse', 'Donkey', 'Wolf', 'Reference strain (CCUG)',
       'Water buffalo','Wild boar']
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
list_subs = ["K-means_Canis", "K-means_Dysgalactiae", "K-means_Equisimilis"]
models_cluster = [
  'K-means'
]

file_path = '../data/tabella_input.csv'
picchi_test = 114

df = pd.read_csv("../data/Merge_tab_training_prediction.csv",
                    delimiter=',', index_col='ID Strain')
df['Haemolysis'] = df['Haemolysis'].str.replace(" ", "")
feat_agg = df.iloc[:,[0,1]]
maldi = df.iloc[:,start:start+n]
display(feat_agg)
display(maldi)

'''maldi.fillna(0, inplace=True)
maldi = maldi.replace(',', '.', regex=True)
columns = maldi.columns
for column in columns:
    maldi[column] = maldi[column].astype(float)'''

'''
antibiotici = df.iloc[:,start+n:start+n+n_antibiotici]
geni_antibiotici = df.iloc[:,start+n+n_antibiotici:start+n+n_antibiotici+n_geni]
virulenza = df.iloc[:,start+n+n_antibiotici+n_geni:start+n+n_antibiotici+n_geni+n_virulenza]

targets = {}
targets = {'antibiotici' : antibiotici,
            'geni_antibiotici' : geni_antibiotici,
            'virulenza' : virulenza}

for str_target,target in targets.items():
    columns = target.columns
    for column in columns:
        if str_target == 'antibiotici':
            target[column] = df[column].map(map_target_antibiotici)
        rapporto = (target[column] == 0).sum() / target.shape[0]
        #if (antibiotici[column] == 0).all() or (antibiotici[column] == 1).all():
        print(column+" : "+str(rapporto))
        if rapporto < 0.15 or rapporto > 0.85:
            target.drop([column], axis=1, inplace=True)

    #display(target)
'''

targets = ['subspecies']

try:
    df_test = pd.read_csv(file_path,
                      delimiter=',', index_col='ID Strain')
except Exception as e:
    print(f"Errore durante il caricamento dei dati: {e}")

df_test['Haemolysis'] = df_test['Haemolysis'].str.replace(" ", "")
feat_agg_test = df_test.iloc[:,[0,1]]
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

feat_agg_dummies = pd.DataFrame.from_dict(pd.get_dummies(feat_agg_test))
feat_agg_dummies = feat_agg_dummies.astype(int)
missing_cols_agg = set(list_agg) - set(feat_agg_dummies.columns)

for c in missing_cols_agg:
    feat_agg_dummies[str(c)] = 0
feat_agg_dummies = feat_agg_dummies[list_agg]


col = maldi.columns.to_list()
#col = [i.replace(',', '.') for i in col]
col = [int(i) for i in col]
maldi.columns = col

col_test = maldi_test.columns.to_list()
col_test = [i.replace(',', '.') for i in col_test]
col_test = [int(float(i)) for i in col_test]


maldi_test.columns = col_test

maldi_all = pd.concat([maldi, maldi_test], axis=0)
maldi_all.fillna(0, inplace = True)
maldi_all

if tutti_picchi != '':
    for i in range(2000,picco_max):
        if i not in maldi_all.columns:
            maldi_all[i] = 0
    maldi_all = maldi_all.reindex(sorted(maldi_all.columns), axis=1)

if reduction != '':
    if scaled == '':
      n_components = 62
    else :
      n_components = 145
    maldi_all = dimensionality_reduction_cluster(maldi_all, n_components)
else:
    col = maldi_all.columns.to_list()
    col = [str(i) for i in col]
    maldi_all.columns = col

maldi = maldi_all[maldi.shape[0]:]

df_agg = pd.concat([maldi, feat_agg_dummies], axis=1)

#Dataframe per confronto predizioni
pred_ensemble_cluster = {}
dfs_cluster = {
       'agg' : df_agg}

#Dataframe con risultati metriche per ogni modello
metrics_df_cluster = pd.DataFrame(columns=['Target', 'Dataframe', 'Model', 'Accuracy', 'St. Dev.',
                            'Precision', 'Recall', 'F1-Score', 'Bal. Accuracy',
                            'Silhouette', 'Calinski', 'Davies'])

for str_df, X in dfs_cluster.items():
    pred_ensemble_cluster[str_df] = pd.DataFrame()
    #Scorre i modelli nel dizionario dei modelli utilizzati
    for name in models_cluster:
        model = pickle.load(open('../models/cluster_'+tutti_picchi+reduction+scaled+scaler+str_df+'_npicchi'+str(n)+'_npicchimax'+str(picco_max)+'_'+name+'_TrainExt.pkl', "rb"))
        y_pred = model.predict(X)
        pred_ensemble_cluster[str_df][name] = y_pred
    
    pred_ensemble_cluster[str_df].index = maldi.index
display(pred_ensemble_cluster[str_df])

clusters = {}
clusters_str = {}
clusters_dummies = {}
for dfs in dfs_cluster:
  clusters[dfs] = pred_ensemble_cluster[dfs]
  clusters_str[dfs] = clusters[dfs].applymap(map_target_inv.get)
  clusters_dummies[dfs] = pd.DataFrame.from_dict(pd.get_dummies(clusters_str[dfs]))
  missing_cols_cluster = set(list_subs) - set(clusters_dummies[dfs].columns)

  # Add a missing column in test set with default value equal to 0
  for c in missing_cols_cluster:
      clusters_dummies[dfs][str(c)] = 0
  # Ensure the order of column in the test set is in the same order than in train set
  clusters_dummies[dfs] = clusters_dummies[dfs][list_subs]
  clusters_dummies[dfs] = clusters_dummies[dfs].astype(int)

df_cluster_agg = pd.concat([df_agg, clusters_dummies['agg']], axis=1)

dfs_test = {
      'clusters+agg' : df_cluster_agg}

metrics_test = pd.DataFrame(columns=['Target', 'Dataframe', 'Model', 'Accuracy','Bal. Accuracy',
                                     'St. Dev.', 'Precision', 'Recall','F1-Score',])
prediction = pd.DataFrame(index = df_cluster_agg.index)
for str_df,X in dfs_test.items():
    for columns in targets:
      #Per ogni tipologia di target scorre tutti i target
      for column in columns:
        if (columns == 'subspecies'):
           column = 'subspecies'
        model = pickle.load(open('../models/stack_'+tutti_picchi+reduction+scaled+scaler+column+'_'+str_df+'_npicchi'+str(n)+'_npicchimax'+str(picco_max)+'_TrainExt.pkl', "rb"))
        y_pred = model.predict(X)
        prediction[column+'_pred'] = y_pred
    prediction['subspecies_pred'] = prediction['subspecies_pred'].map(map_target_inv)
prediction.to_csv('results_TrainExt.csv', index = True)
display(prediction)