# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
import joblib
import sys

from scipy.spatial import distance
from sklearn.utils.multiclass import unique_labels
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.cluster import AgglomerativeClustering

from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, make_scorer
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, StratifiedKFold, KFold, GridSearchCV, RandomizedSearchCV, ParameterGrid
from sklearn.linear_model import LogisticRegression, RidgeClassifier, LassoCV, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.base import clone
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn import model_selection
from sklearn.semi_supervised import LabelPropagation, LabelSpreading

from mlxtend.classifier import StackingCVClassifier
from sklearn.ensemble import StackingClassifier

from IPython.display import display

import warnings

warnings.filterwarnings("ignore")
np.set_printoptions(precision=5, suppress=True)


RANDOM_STATE = 46
N_JOBS = -1

f = open('Output_Test.txt','w')
sys.stdout = f
e = open('Error_Test.txt','w')
sys.stderr = e

start = 9
n_antibiotici = 9
n_geni = 27
n_virulenza = 18

'''n = 306
n_test = 56
cartella_train = 'train_1'
cartella_test = 'test_1'
file_train = "Training_1_"+str(n)+"picchi.csv"
file_test = 'Testing_1.csv'
'''
'''n = 306
n_test = 114
cartella_train = 'train_1'
cartella_test = 'test_2'
file_train = "Training_1_"+str(n)+"picchi.csv"
file_test = 'Testing_2_with_Antibiotics.csv'
'''
'''n = 357
n_test = 56
cartella_train = 'train_2'
cartella_test = 'test_1'
file_train = "Training_2_"+str(n)+"picchi.csv"
file_test = 'Testing_1.csv'
'''
n = 357
n_test = 114
cartella_train = 'train_2'
cartella_test = 'test_2'
file_train = "Training_2_"+str(n)+"picchi.csv"
file_test = 'Testing_2_with_Antibiotics.csv'

'''n = 306
n_test = 56
cartella_train = 'train_3'
cartella_test = 'test_1'
file_train = "Training_3_"+str(n)+"picchi_NoTest2.csv"
file_test = 'Testing_1.csv'
'''
'''n = 306
n_test = 114
cartella_train = 'train_3'
cartella_test = 'test_2'
file_train = "Training_3_"+str(n)+"picchi_NoTest2.csv"
file_test = 'Testing_2_with_Antibiotics.csv'
'''
'''n = 357
n_test = 56
cartella_train = 'train_4'
cartella_test = 'test_1'
file_train = "Training_4_"+str(n)+"picchi_NoTest2.csv"
file_test = 'Testing_1.csv'
'''
'''n = 357
n_test = 114
cartella_train = 'train_4'
cartella_test = 'test_2'
file_train = "Training_4_"+str(n)+"picchi_NoTest2.csv"
file_test = 'Testing_2_with_Antibiotics.csv'
'''
scaled = ''
scaler = ''
tutti_picchi = 'tutti_picchi_'
reduction = 'pca_'
tuning = 'tuning_'
picco_max = 20000

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
map_target_antibiotici = {
    "S" : 1,
    "NS" : 0
}
map_target_antibiotici_inv = {
    1 : "S",
    0 : "NS"
}

map_cluster = [
    {
        2 : 2,
        1 : 1,
        0 : 0
    },
    {
        2 : 2,
        1 : 0,
        0 : 1
    },
    {
        2 : 1,
        1 : 0,
        0 : 2
    },
    {
        2 : 1,
        1 : 2,
        0 : 0
    },
    {
        2 : 0,
        1 : 2,
        0 : 1
    },
    {
        2 : 0,
        1 : 1,
        0 : 2
    }
]

metrics = ['accuracy', 'recall_weighted', 'precision_weighted','f1_weighted']
metrics_cluster = ['Silhouette', 'Calinski', 'Davies']
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

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

models = {
  'LogisticRegression': LogisticRegression(random_state=RANDOM_STATE),
  'Ridge' : RidgeClassifier(random_state=RANDOM_STATE),
  'DecisionTree': DecisionTreeClassifier(random_state=RANDOM_STATE),
  'K-nn': KNeighborsClassifier(),
  'RandomForest': RandomForestClassifier(random_state=RANDOM_STATE),
  'BernoulliNB': BernoulliNB(),
  'GaussianNB': GaussianNB(),
  #'NearestCentroid': NearestCentroid(),
  'SVC' : SVC(),
  'LinearSVC' : LinearSVC(),
  'LabelPropagation' : LabelPropagation(),
  'LabelSpreading' : LabelSpreading(),
  'SGDClassifier' : SGDClassifier()
}

models_cluster = [
  'K-means'
]

# Define a function for standard scaling
def standard_scaler(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# Define a function for dimensionality reduction using PCA
def dimensionality_reduction(X_train, X_test, n_components):
    X_train.columns = X_train.columns.astype(str)
    X_test.columns = X_test.columns.astype(str)
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    #save model
    pickle.dump(pca, open('../'+cartella_train+'/models/pca_'+scaler+tutti_picchi+'npicchi'+str(n)+'_npicchimax'+str(picco_max)+'.pkl',"wb"))

    X_train_pca = pd.DataFrame(X_train_pca)
    X_test_pca = pd.DataFrame(X_test_pca)
    #print(X_train_pca.shape)
    return X_train_pca, X_test_pca

def dimensionality_reduction_cluster(X, n_components):
    X.columns = X.columns.astype(str)
    print(X.shape)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    #save model
    pickle.dump(pca, open('../'+cartella_train+'/models/pca_cluster_'+scaler+tutti_picchi+'npicchi'+str(n)+'_npicchimax'+str(picco_max)+'.pkl',"wb"))
    X_pca = pd.DataFrame(X_pca, index=X.index.to_list())
    print(X_pca.shape)
    X_pca.columns = X_pca.columns.astype(str)
    return X_pca

def makeScoreMeanWithoutNaN(metrics):
    for name, metrica in metrics.items():
        print(name)
        print(metrics[name])
        metrics[name] = metrics[name][~np.isnan(metrics[name])]
        print(metrics[name])
        metrics[name] = np.mean(metrics[name])
        print(metrics[name])
    print(metrics)
    return metrics

def makeScore(y_test, y_pred):
    score = {}

    score['acc'] = accuracy_score(y_test, y_pred)
    score['b_acc'] = balanced_accuracy_score(y_test, y_pred)
    score['st'] = score['acc'].std()
    score['prec'] = precision_score(y_test, y_pred, average='weighted')
    score['rec'] = recall_score(y_test, y_pred, average='weighted')
    score['f1'] = f1_score(y_test, y_pred, average='weighted')

    return score

def makeCrossValidation(model, X_train, y_train):
    score = {}
    cv = cross_validate(estimator=model, X=X_train, y=y_train,
                        scoring=metrics, cv=skfold,
                        n_jobs=N_JOBS, verbose=0)

    score['acc'] = cv.get('test_accuracy').mean()
    score['st'] = cv.get('test_accuracy').std()
    score['prec'] = cv.get('test_precision_weighted').mean()
    score['rec'] = cv.get('test_recall_weighted').mean()
    score['f1'] = cv.get('test_f1_weighted').mean()

    return score

def makeCrossValidationCluster(model, X):
    model.fit(X)
    labels = model.labels_
    pca = PCA(n_components = 2)
    pca.fit(X)
    X_pca = pca.transform(X)
    avg_silhouette = silhouette_score(X_pca, labels)
    avg_calinski_harabasz = calinski_harabasz_score(X_pca, labels)
    avg_davies_bouldin = davies_bouldin_score(X_pca, labels)

    score = {}
    score['Silhouette'] = avg_silhouette
    score['Calinski'] = avg_calinski_harabasz
    score['Davies'] = avg_davies_bouldin

    return score

# Load the data
df = pd.read_csv("../data/"+file_train,
                    delimiter=';', index_col='ID Strain')
df['subspecies'] = df["Putative Subspecies"].map(map_target)

animals = df.iloc[:,2]
feat_agg = df.iloc[:,[2,8]]
display(feat_agg)
st = df.iloc[:,[4]]
display(st)
subspecies = df[['subspecies']]
maldi = df.iloc[:,start:start+n]
antibiotici = df.iloc[:,start+n:start+n+n_antibiotici]
geni_antibiotici = df.iloc[:,start+n+n_antibiotici:start+n+n_antibiotici+n_geni]
virulenza = df.iloc[:,start+n+n_antibiotici+n_geni:start+n+n_antibiotici+n_geni+n_virulenza]

maldi.fillna(0, inplace=True)
maldi = maldi.replace(',', '.', regex=True)
columns = maldi.columns
for column in columns:
    maldi[column] = maldi[column].astype(float)
display(maldi)

#targets = {}
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
        '''if column == 'ErmB' or column == 'tetO' :
            target.drop([column], axis=1, inplace=True)'''

    display(target)

targets['subspecies'] = subspecies


df_test = pd.read_csv("../data/"+file_test,
                      delimiter=';', index_col='ID Strain')

df_test['subspecies'] = df_test["Putative Subspecies"].map(map_target)
df_test['Haemolysis'] = df_test['Haemolysis'].str.replace(" ", "")
animals_test = df_test.iloc[:,2]
feat_agg_test = df_test.iloc[:,[2,8]]
display(feat_agg_test)
st_test = df_test.iloc[:,[4]]
display(st_test)
subspecies_test = df_test[['subspecies']]
maldi_test = df_test.iloc[:,start:start+n_test]
antibiotici_test = df_test.iloc[:,start+n_test:start+n_test+n_antibiotici]
geni_antibiotici_test = df_test.iloc[:,start+n_test+n_antibiotici:start+n_test+n_antibiotici+n_geni]
virulenza_test = df_test.iloc[:,start+n_test+n_antibiotici+n_geni:start+n_test+n_antibiotici+n_geni+n_virulenza]

maldi_test.fillna(0, inplace=True)
maldi_test = maldi_test.replace(',', '.', regex=True)
columns = maldi_test.columns
for column in columns:
    maldi_test[column] = maldi_test[column].astype(float)
display(maldi_test)

targets_test = {}
targets_test = {'antibiotici' : antibiotici_test,
            'geni_antibiotici' : geni_antibiotici_test,
            'virulenza' : virulenza_test}

for str_target,target in targets_test.items():
    columns = target.columns
    for column in columns:
        if str_target == 'antibiotici':
            target[column] = df_test[column].map(map_target_antibiotici)
        rapporto = (target[column] == 0).sum() / target.shape[0]
        #if (antibiotici[column] == 0).all() or (antibiotici[column] == 1).all():
        print(column+" : "+str(rapporto))
        '''if rapporto < 0.15 or rapporto > 0.85:
            target.drop([column], axis=1, inplace=True)'''
        target.fillna(0, inplace = True)
    display(target)

targets_test['subspecies'] = subspecies_test
print(targets_test)
animals_dummies = pd.DataFrame.from_dict(pd.get_dummies(animals_test))
feat_agg_dummies = pd.DataFrame.from_dict(pd.get_dummies(feat_agg_test))

animals_dummies = animals_dummies.astype(int)
feat_agg_dummies = feat_agg_dummies.astype(int)

missing_cols_animals = set(list_animals) - set(animals_dummies.columns)
print(len(missing_cols_animals))

missing_cols_agg = set(list_agg) - set(feat_agg_dummies.columns)
print(len(missing_cols_agg))

# Add a missing column in test set with default value equal to 0
for c in missing_cols_animals:
    animals_dummies[str(c)] = 0
# Ensure the order of column in the test set is in the same order than in train set
animals_dummies = animals_dummies[list_animals]

for c in missing_cols_agg:
    feat_agg_dummies[str(c)] = 0
feat_agg_dummies = feat_agg_dummies[list_agg]

display(animals_dummies)
display(feat_agg_dummies)

col = maldi.columns.to_list()
col = [i.replace(',', '.') for i in col]
col = [int(float(i)) for i in col]

col_test = maldi_test.columns.to_list()
col_test = [i.replace(',', '.') for i in col_test]
col_test = [int(float(i)) for i in col_test]

maldi.columns = col
maldi_test.columns = col_test

maldi_all = pd.concat([maldi, maldi_test], axis=0)
maldi_all.fillna(0, inplace = True)
display(maldi_all)

if tutti_picchi != '':
  for i in range(2000,picco_max):
    if i not in maldi_all.columns:
      maldi_all[i] = 0
  maldi_all = maldi_all.reindex(sorted(maldi_all.columns), axis=1)
  display(maldi_all)

if scaled != '':
  scaler_filename = '../'+cartella_train+'/models/scaler_'+scaler+tutti_picchi+'npicchi'+str(n)+'_npicchimax'+str(picco_max)+'.save'
  sc = joblib.load(scaler_filename)
  maldi_all = pd.DataFrame(sc.transform(maldi_all), index = maldi_all.index, columns = maldi_all.columns)

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

df_animals = pd.concat([maldi, animals_dummies], axis=1)
display(df_animals)

df_agg = pd.concat([maldi, feat_agg_dummies], axis=1)
display(df_agg)

#Dataframe per confronto predizioni
pred_ensemble_cluster = {}
dfs_cluster = {'maldi' : maldi,
       'animals' : df_animals,
       'agg' : df_agg}

#Dataframe con risultati metriche per ogni modello
metrics_df_cluster = pd.DataFrame(columns=['Target', 'Dataframe', 'Model', 'Accuracy', 'St. Dev.',
                            'Precision', 'Recall', 'F1-Score', 'Bal. Accuracy',
                            'Silhouette', 'Calinski', 'Davies'])
column = 'subspecies'
y = subspecies_test.values
for str_df, X in dfs_cluster.items():
  print('Dataframe: '+str_df)
  pred_ensemble_cluster[str_df] = pd.DataFrame()
  display(X)
  #Scorre i modelli nel dizionario dei modelli utilizzati
  for name in models_cluster:
    print("Modello "+name)
    model = pickle.load(open('../'+cartella_train+'/models/cluster_'+tutti_picchi+reduction+scaled+scaler+str_df+'_npicchi'+str(n)+'_npicchimax'+str(picco_max)+'_'+name+'.pkl', "rb"))
    y_pred = model.predict(X)
    y_pred = pd.DataFrame(y_pred,X.index)

    max = 0
    for mapping in map_cluster:
      print(mapping)
      y_pred_mapped = y_pred.iloc[:,0].map(mapping)
      score_mapped = makeScore(y, y_pred_mapped)
      if score_mapped['acc'] > max:
        max = score_mapped['acc']
        print(max)
        score = score_mapped
        y_pred_def = y_pred_mapped

    pred_ensemble_cluster[str_df][name] = y_pred_def
    score_cluster = makeCrossValidationCluster(model, X)
    print(y)
    print(y_pred_def.values)
    score = makeScore(y, y_pred_def.values)
    ris = {'Target': column,
              'Dataframe' : str_df,
              'Model': name,
              'Accuracy' : score['acc'],
              'St. Dev.' : score['st'],
              'Precision' : score['prec'],
              'Recall' : score['rec'],
              'F1-Score' : score['f1'],
              'Bal. Accuracy' : score['b_acc'],
              'Silhouette' : score_cluster['Silhouette'],
              'Calinski' : score_cluster['Calinski'],
              'Davies' : score_cluster['Davies']}
    #display(ris)
    metrics_df_cluster.loc[len(metrics_df_cluster)] = ris
  pred_ensemble_cluster[str_df].index = maldi.index
  pred_ensemble_cluster[str_df].to_csv('../'+cartella_test+'/predictions/cluster_'+cartella_train+'_'+tutti_picchi+reduction+scaled+scaler+str_df+'_npicchi'+str(n)+'_npicchimax'+str(picco_max)+'.csv', index = True)
  #Aggiunge i valori del target nei dizionari
metrics_df_cluster.to_csv('../'+cartella_test+'/results/cluster_'+cartella_train+'_'+tutti_picchi+reduction+scaled+scaler+'npicchi'+str(n)+'_npicchimax'+str(picco_max)+'.csv', index = False)
display(metrics_df_cluster)

clusters = {}
clusters_str = {}
clusters_dummies = {}
for dfs in dfs_cluster:
  print(dfs)
  clusters[dfs] = pred_ensemble_cluster[dfs]
  display(clusters[dfs])
  clusters_str[dfs] = clusters[dfs].applymap(map_target_inv.get)
  display(clusters_str[dfs])
  clusters_dummies[dfs] = pd.DataFrame.from_dict(pd.get_dummies(clusters_str[dfs]))
  display(clusters_dummies[dfs])
  missing_cols_cluster = set(list_subs) - set(clusters_dummies[dfs].columns)
  print(len(missing_cols_cluster))

  # Add a missing column in test set with default value equal to 0
  for c in missing_cols_cluster:
      clusters_dummies[dfs][str(c)] = 0
  # Ensure the order of column in the test set is in the same order than in train set
  clusters_dummies[dfs] = clusters_dummies[dfs][list_subs]
  clusters_dummies[dfs] = clusters_dummies[dfs].astype(int)
  display(clusters_dummies[dfs])

df_clusters = pd.concat([maldi, clusters_dummies['maldi']], axis=1)
display(df_clusters)

df_cluster_agg = pd.concat([df_agg, clusters_dummies['agg']], axis=1)
display(df_cluster_agg)

#Dataframe per confronto predizioni
score_target = {}

dfs = {'maldi' : maldi,
       'clusters' : df_clusters,
       'animals' : df_animals,
       'agg' : df_agg,
       'clusters+agg' : df_cluster_agg}

model_obj = {}
#Dataframe con risultati metriche per ogni modello
metrics_test = pd.DataFrame(columns=['Target', 'Dataframe', 'Model', 'Accuracy','Bal. Accuracy',
                                     'St. Dev.', 'Precision', 'Recall','F1-Score',])
prediction = {}
#Per ogni tipologia di target del problema (sottospecie, antibiotici, geni, fattori, st)
for str_target, target in targets.items():
  columns = target.columns
  #Per ogni tipologia di target scorre tutti i target
  target = targets_test[str_target]
  for column in columns:
    prediction[column] = pd.DataFrame()
    #print("Colonna:"+column)
    y = target[column]
    for str_df, X in dfs.items():
      #print('Dataframe: '+str_df)

      #Scorre i modelli nel dizionario dei modelli utilizzati
      for name in models:
          #print("Modello "+name)
          model = pickle.load(open('../'+cartella_train+'/models/models_base/'+name+'_'+tutti_picchi+reduction+scaled+scaler+tuning+column+'_'+str_df+'_npicchi'+str(n)+'_npicchimax'+str(picco_max)+'.pkl', 'rb'))

          y_pred = model.predict(X)
          score = makeScore(y, y_pred)

          ris = {'Target': column,
              'Dataframe' : str_df,
              'Model': name,
              'Accuracy' : score['acc'],
              'Bal. Accuracy' : score['b_acc'],
              'St. Dev.' : score['st'],
              'Precision' : score['prec'],
              'Recall' : score['rec'],
              'F1-Score' : score['f1']}

          #display(ris)
          metrics_test.loc[len(metrics_test)] = ris

    prediction[column].index = X.index
    #Aggiunge i valori del target nei dizionari
    prediction[column]['Target'] = y.values
    prediction[column].to_csv('../'+cartella_test+'/predictions/'+column+'_'+cartella_train+'_'+tutti_picchi+reduction+scaled+scaler+tuning+'_basemodel_'+name+'_'+str_df+'_npicchi'+str(n)+'_npicchimax'+str(picco_max)+'.csv', index = True)
metrics_test.to_csv('../'+cartella_test+'/results/models_base_'+cartella_train+'_'+tutti_picchi+reduction+scaled+scaler+tuning+'npicchi'+str(n)+'_npicchimax'+str(picco_max)+'.csv', index = False)

display(metrics_test)

metrics_test.loc[metrics_test['Target'] == 'subspecies']

dfs_test = {'maldi' : maldi,
      'clusters' : df_clusters,
      'animals' : df_animals,
      'agg' : df_agg,
      'clusters+agg' : df_cluster_agg}

metrics_test = pd.DataFrame(columns=['Target', 'Dataframe', 'Model', 'Accuracy','Bal. Accuracy',
                                     'St. Dev.', 'Precision', 'Recall','F1-Score',])
prediction = {}

for str_df,X in dfs_test.items():
    print('Dataframe: '+str_df)
    display(X)
    prediction[str_df] = pd.DataFrame(index = df_cluster_agg.index)
    for str_target, target in targets.items():
      columns = target.columns
      #Per ogni tipologia di target scorre tutti i target
      for column in columns:
        #print("Colonna:"+column)
        y = targets_test[str_target][column]
        #display(y)
        #print('Model: '+column+'_'+str_df+'_Stack')
        model = pickle.load(open('../'+cartella_train+'/models/stack_'+tutti_picchi+reduction+scaled+scaler+tuning+column+'_'+str_df+'_npicchi'+str(n)+'_npicchimax'+str(picco_max)+'.pkl', "rb"))
        print(model)
        y_pred = model.predict(X)
        #display(y_pred)
        score = makeScore(y, y_pred)
        prediction[str_df][column+'_pred'] = y_pred
        if (str_target == 'antibiotici'):
          prediction[str_df][column+'_pred'] = prediction[str_df][column+'_pred'].map(map_target_antibiotici_inv)
        prediction[str_df][column] = df_test[column]
        ris = {'Target': column,
              'Dataframe' : str_df,
              'Model': 'Stack',
              'Accuracy' : score['acc'],
              'Bal. Accuracy' : score['b_acc'],
              'St. Dev.' : score['st'],
              'Precision' : score['prec'],
              'Recall' : score['rec'],
              'F1-Score' : score['f1']}

        #display(ris)
        metrics_test.loc[len(metrics_test)] = ris
    prediction[str_df]['subspecies'] = prediction[str_df]['subspecies'].map(map_target_inv)
    prediction[str_df]['subspecies_pred'] = prediction[str_df]['subspecies_pred'].map(map_target_inv)
    prediction[str_df].to_csv('../'+cartella_test+'/predictions/stack_'+cartella_train+'_'+tutti_picchi+reduction+scaled+scaler+tuning+str_df+'_npicchi'+str(n)+'_npicchimax'+str(picco_max)+'.csv', index = True)
    display(prediction[str_df])
metrics_test.to_csv('../'+cartella_test+'/results/stack_'+cartella_train+'_'+tutti_picchi+reduction+scaled+scaler+tuning+'npicchi'+str(n)+'_npicchimax'+str(picco_max)+'.csv', index = True)
display(metrics_test)

name_best = list()
score_best = list()
target_list = list()
for str_target, target in targets.items():
  columns = target.columns
  for column in columns:
    filter = metrics_test["Target"]==column
    subs_df = metrics_test.where(filter, inplace = False).dropna()
    name = subs_df['Accuracy'].idxmax()
    score = subs_df['Accuracy'].max()
    target_list.append(column)
    name_best.append(name)
    score_best.append(score)

print(target_list)
print(name_best)
print(score_best)

# bars are by default width 0.8, so we'll add 0.1 to the left coordinates
# so that each bar is centered
y_pos = np.arange(len(target_list))

# plot bars with left x-coordinates [xs], heights [num_oscars]
plt.barh(y_pos, score_best, align='center')
# label x-axis with movie names at bar centers
plt.yticks(y_pos, target_list)
plt.xlabel("% of Accuracy")
plt.title("Risultati di Balanced Accuracy sul miglior modello ensemble sui targets")
plt.show()