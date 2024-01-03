# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import joblib
import sys

from sklearn.metrics import silhouette_samples, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans, AgglomerativeClustering

from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold, RandomizedSearchCV, ParameterGrid
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.base import clone
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.semi_supervised import LabelPropagation, LabelSpreading

from mlxtend.classifier import StackingCVClassifier
from sklearn.ensemble import StackingClassifier

from IPython.display import display

import warnings

# warnings -> to silence warnings
warnings.filterwarnings(action="ignore")

f = open('Output_Train.txt','w')
sys.stdout = f
e = open('Error_Train.txt','w')
sys.stderr = e

np.set_printoptions(precision=5, suppress=True)

RANDOM_STATE = 46
N_JOBS = -1

class_names = ["Canis", "Dysg. Equisimilis", "Dysg. Dysgalactiae"]

start = 9
n_antibiotici = 9
n_geni = 27
n_virulenza = 18

'''n = 306
cartella = 'train_1'
file_train = "Training_1_"+str(n)+"picchi.csv"
'''

'''n = 357
cartella = 'train_2'
file_train = "Training_2_"+str(n)+"picchi.csv"
'''

n = 306
cartella = 'train_3'
file_train = "Training_3_"+str(n)+"picchi_NoTest2.csv"


'''n = 357
cartella = 'train_4'
file_train = "Training_4_"+str(n)+"picchi_NoTest2.csv"
'''
scaled = ''
scaler = ''
tutti_picchi = ''
reduction = ''
tuning = 'tuning_'
picco_max = 20000

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

maps_cluster = [
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

# define the models
models = {
  'LogisticRegression': LogisticRegression(random_state=RANDOM_STATE),
  'Ridge' : RidgeClassifier(random_state=RANDOM_STATE),
  'DecisionTree': DecisionTreeClassifier(random_state=RANDOM_STATE),
  'K-nn': KNeighborsClassifier(),
  'RandomForest': RandomForestClassifier(random_state=RANDOM_STATE),
  'BernoulliNB': BernoulliNB(),
  'GaussianNB': GaussianNB(),
  'NearestCentroid': NearestCentroid(),
  'SVC' : SVC(),
  'LinearSVC' : LinearSVC(dual=True),
  'LabelPropagation' : LabelPropagation(),
  'LabelSpreading' : LabelSpreading(),
  'SGDClassifier' : SGDClassifier()
}

models_stack = {
  'LogisticRegression': LogisticRegression(random_state=RANDOM_STATE),
  'Ridge' : RidgeClassifier(random_state=RANDOM_STATE),
  'DecisionTree': DecisionTreeClassifier(random_state=RANDOM_STATE),
  'K-nn': KNeighborsClassifier(),
  #'RandomForest': RandomForestClassifier(random_state=RANDOM_STATE),
  'GaussianNB': GaussianNB(),
  'SVC' : SVC()
}

# define the models for clustering
models_cluster = {
  'K-means' : KMeans(n_clusters = N_CLUSTERS, random_state=RANDOM_STATE),
  #'AgglomerativeClustering' : AgglomerativeClustering(n_clusters = N_CLUSTERS),
  #'DBSCAN' : DBSCAN(),
  #'GaussianMixture' : GaussianMixture(n_components=N_CLUSTERS, random_state=RANDOM_STATE),
  #'OPTICS' : OPTICS()
}

param_grid = {}
param_grid_cluster = {}
# Hyperparameter tuning using RandomizedSearchCV
param_grid = {
    'LogisticRegression': {
        'penalty': ['l1', 'l2'],
        'C': [0.001, 0.01, 0.1, 1, 10, 100]
    },
    'Ridge': {
        'alpha': [0.1, 1, 10, 100, 1000]
    },
    'DecisionTree': {
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'KNeighbors': {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance']
    },
    'RandomForest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'BernoulliNB': {
        'alpha': [0.001, 0.01, 0.1, 1]
    },
    'GaussianNB': {
    },
    'SVC': {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    },
    'LinearSVC': {
        'penalty': ['l1', 'l2'],
        'C': [0.01, 0.1, 1, 10]
    },
    'LabelPropagation': {
        'gamma': [0.01, 0.1, 1, 10],
        'n_neighbors': [3, 5, 7, 9]
    },
    'LabelSpreading': {
        'gamma': [0.01, 0.1, 1, 10],
        'n_neighbors': [3, 5, 7, 9]
    },
    'SGDClassifier': {
        'penalty': ['l2', 'l1', 'elasticnet'],
        'alpha': [0.0001, 0.001, 0.01, 0.1],
        'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive']
    }
}

# Hyperparameter tuning using RandomizedSearchCV NON USATO PER CLUSTERING
param_grid_cluster = {
  'K-means' : list(ParameterGrid({
    'n_clusters': range(2, 6),  # Range of number of clusters
    'init': ['k-means++', 'random'],  # Initialization methods
    'max_iter': range(100, 1001, 100),  # Maximum number of iterations
    'tol': [1e-3, 1e-4, 1e-5],  # Tolerance for convergence
    'algorithm': ['lloyd', 'elkan'],  # K-means algorithm type
    'random_state': [None, RANDOM_STATE, 100]  # Random seed for centroid initialization
    })),
  'AgglomerativeClustering' : list(ParameterGrid({
    'n_clusters': range(2, 8),  # Number of clusters
    'linkage': ['ward', 'complete', 'average', 'single'],  # Linkage type
    })),
  'DBSCAN' : list(ParameterGrid({
    'eps': [0.1, 0.5, 1.0],  # Maximum distance between two samples
    'min_samples': [2, 5, 10]  # Minimum number of samples in a neighborhood
    })),
  'GaussianMixture' : list(ParameterGrid({
    'n_components': range(2, 7),  # Number of mixture components
    'covariance_type': ['full', 'tied', 'diag', 'spherical'],  # Covariance type
    'tol': [1e-4, 1e-3, 1e-2]  # Tolerance for convergence
    }))
}

def dimensionality_reduction_cluster(X, n_components):
    X.columns = X.columns.astype(str)
    #print(X.shape)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    #save model
    pickle.dump(pca, open('../'+cartella+'/models/pca_cluster_'+scaler+tutti_picchi+'npicchi'+str(n)+'_npicchimax'+str(picco_max)+'.pkl',"wb"))
    X_pca = pd.DataFrame(X_pca, index=X.index.to_list())
    #print(X_pca.shape)
    X_pca.columns = X_pca.columns.astype(str)
    return X_pca

def makeScore(y_test, y_pred):
    score = {}

    score['acc'] = accuracy_score(y_test, y_pred)
    score['b_acc'] = balanced_accuracy_score(y_test, y_pred)
    score['st'] = score['acc'].std()
    score['prec'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    score['rec'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    score['f1'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)

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

def makeTuning(model, X_train, y_train, name):
    score = {}
    params = param_grid[name]
    rs = RandomizedSearchCV(estimator=model, param_distributions=params,
                            scoring=metrics, refit="accuracy", cv=skfold,
                            n_jobs=N_JOBS, random_state=RANDOM_STATE, verbose=0)
    rs.fit(X_train, y_train)

    model_best = rs.best_estimator_

    score = makeCrossValidation(model_best, X_train, y_train)

    return clone(model_best), score

df = pd.read_csv("../data/"+file_train,
                    delimiter=';', index_col='ID Strain')

df['subspecies'] = df["Putative Subspecies"].map(map_target)
df['Haemolysis'] = df['Haemolysis'].str.replace(" ", "")
animals = df.iloc[:,2]
feat_agg = df.iloc[:,[2,8]]
st = df.iloc[:,[4]]
subspecies = df[['subspecies']]
maldi = df.iloc[:,start:start+n]
display(maldi)
antibiotici = df.iloc[:,start+n:start+n+n_antibiotici]
display(antibiotici)
geni_antibiotici = df.iloc[:,start+n+n_antibiotici:start+n+n_antibiotici+n_geni]
virulenza = df.iloc[:,start+n+n_antibiotici+n_geni:start+n+n_antibiotici+n_geni+n_virulenza]

maldi.fillna(0, inplace=True)
maldi = maldi.replace(',', '.', regex=True)
columns = maldi.columns
for column in columns:
    maldi[column] = maldi[column].astype(float)
display(maldi)

#Togliere commenti per includere altri target oltre subspecies
#targets = {}
targets = {'antibiotici' : antibiotici,
            'geni' : geni_antibiotici,
            'virulenza' : virulenza}

for str_target,target in targets.items():
  columns = target.columns
  for column in columns:
    if str_target == 'antibiotici':
      target[column] = df[column].map(map_target_antibiotici)
    rapporto = (target[column] == 0).sum() / target.shape[0]
    print(column+" : "+str(rapporto))
    if rapporto < 0.15 or rapporto > 0.85:
      target.drop([column], axis=1, inplace=True)
    '''if column == 'ErmB' or column == 'tetO' :
      target.drop([column], axis=1, inplace=True)'''
  display(target)

targets['subspecies'] = subspecies

animals_dummies = pd.DataFrame.from_dict(pd.get_dummies(animals))
feat_agg_dummies = pd.DataFrame.from_dict(pd.get_dummies(feat_agg))

animals_dummies = animals_dummies.astype(int)
feat_agg_dummies = feat_agg_dummies.astype(int)

missing_cols_animals = set(list_animals) - set(animals_dummies.columns)
#print(len(missing_cols_animals))

missing_cols_agg = set(list_agg) - set(feat_agg_dummies.columns)
#print(len(missing_cols_agg))

# Add a missing column in test set with default value equal to 0
for c in missing_cols_animals:
    animals_dummies[str(c)] = 0
# Ensure the order of column in the test set is in the same order than in train set
animals_dummies = animals_dummies[list_animals]

for c in missing_cols_agg:
    feat_agg_dummies[str(c)] = 0
feat_agg_dummies = feat_agg_dummies[list_agg]

'''display(animals_dummies)
display(feat_agg_dummies)'''

col = maldi.columns.to_list()
col = [i.replace(',', '.') for i in col]
col = [int(float(i)) for i in col]
maldi.columns = col
#print(maldi.columns)

tutti_picchi = 'tutti_picchi_'
if tutti_picchi != '':
  for i in range(2000,picco_max):
    if i not in maldi.columns:
      maldi[i] = 0
  maldi = maldi.reindex(sorted(maldi.columns), axis=1)
  display(maldi)
  maldi = maldi.fillna(0)

#scaled = 'scaled_'
#scaler = 'ss_'
if scaled != '':
  if scaler == 'mm_':
    sc = MinMaxScaler()
  elif scaler == 'ss_':
    sc = StandardScaler()
  maldi = pd.DataFrame(sc.fit_transform(maldi), index = maldi.index, columns = maldi.columns)
  #save model
  scaler_filename = '../'+cartella+'/models/scaler_'+scaler+tutti_picchi+'npicchi'+str(n)+'_npicchimax'+str(picco_max)+'.save'
  joblib.dump(sc, scaler_filename)

reduction = 'pca_'
if reduction != '':
  if scaled == '':
    n_components = 62
  else :
    n_components = 145
  maldi = dimensionality_reduction_cluster(maldi,n_components)
else:
  col = maldi.columns.to_list()
  col = [str(i) for i in col]
  maldi.columns = col

df_animals = pd.concat([maldi, animals_dummies], axis=1)
#display(df_animals)

df_agg = pd.concat([maldi, feat_agg_dummies], axis=1)
#display(df_agg)

#Dataframe per confronto predizioni
pred_ensemble_cluster = {}

dfs_cluster = {'maldi' : maldi,
       'animals' : df_animals,
       'agg' : df_agg}

model_obj_cluster = {}
#Dataframe con risultati metriche per ogni modello
metrics_df_cluster = pd.DataFrame(columns=['Target', 'Dataframe', 'Model', 'Accuracy', 'St. Dev.',
                            'Precision', 'Recall', 'F1-Score', 'Bal. Accuracy',
                            'Silhouette', 'Calinski', 'Davies'])
column = 'subspecies'
y = subspecies
display(y)

print("Colonna:"+column)
for str_df, X in dfs_cluster.items():
  print('Dataframe: '+str_df)
  pred_ensemble_cluster[str_df] = pd.DataFrame()
  #display(X)
  #Scorre i modelli nel dizionario dei modelli utilizzati
  for name, model in models_cluster.items():
    #print("Modello "+name)
    #model = clone(model_cluster)
    #print(model.get_params())
    #Modello base: cross validation with score, fit, predict
    score_cluster = makeCrossValidationCluster(model, X)

    y_pred = model.fit_predict(X)
    y_pred = pd.DataFrame(y_pred,X.index, columns=['subspecies'])
    max = 0

    #display(y_pred['subspecies'].values)

    for mapping in maps_cluster:
      #print(mapping)
      y_pred_mapped = y_pred.iloc[:,0].map(mapping)
      score_mapped = makeScore(y, y_pred_mapped)
      if score_mapped['acc'] > max:
        max = score_mapped['acc']
        print(max)
        score = score_mapped
        y_pred_def = y_pred_mapped

    pred_ensemble_cluster[str_df][name] = y_pred_def
    
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
    model_obj_cluster[column+'_'+str_df+'_'+name] = model
    pickle.dump(model, open('../'+cartella+'/models/cluster_'+tutti_picchi+reduction+scaled+scaler+str_df+'_npicchi'+str(n)+'_npicchimax'+str(picco_max)+'_'+name+'.pkl', 'wb'))
    
  #Aggiunge i valori del target nei dizionari
  pred_ensemble_cluster[str_df].index = maldi.index
  pred_ensemble_cluster[str_df].to_csv('../'+cartella+'/predictions/cluster_'+tutti_picchi+reduction+scaled+scaler+str_df+'_npicchi'+str(n)+'_npicchimax'+str(picco_max)+'.csv', index = True)
metrics_df_cluster.to_csv('../'+cartella+'/results/cluster_'+tutti_picchi+reduction+scaled+scaler+'npicchi'+str(n)+'_npicchimax'+str(picco_max)+'.csv', index = False)
display(metrics_df_cluster)
#print("Modelli:")
#print(model_obj_cluster)

clusters = {}
clusters_str = {}
clusters_dummies = {}
for dfs in dfs_cluster:
  clusters[dfs] = pred_ensemble_cluster[dfs]
  #display(clusters[dfs])
  clusters_str[dfs] = clusters[dfs].applymap(map_target_inv.get)
  #display(clusters_str[dfs])
  clusters_dummies[dfs] = pd.DataFrame.from_dict(pd.get_dummies(clusters_str[dfs]))
  clusters_dummies[dfs] = clusters_dummies[dfs].astype(int)
  #display(clusters_dummies[dfs])

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
metrics_df = pd.DataFrame(columns=['Target', 'Dataframe', 'Model', 'Accuracy CV', 'St. Dev. CV',
                            'Precision CV', 'Recall CV','F1-Score CV','Accuracy','Bal. Accuracy'])
prediction = {}
#Per ogni tipologia di target del problema (sottospecie, antibiotici, geni, fattori, st)
for str_target, target in targets.items():
  columns = target.columns
  #Per ogni tipologia di target scorre tutti i target
  for column in columns:
    prediction[column] = pd.DataFrame()
    print("Colonna:"+column)
    y = target[column]
    for str_df, X in dfs.items():
      print('Dataframe: '+str_df)

      #parameter range for BernoulliNB in base of the number of classes in the target
      n_classes = np.unique(y)
      #param_grid['BernoulliNB']['class_prior'] = [None, [0.1,]* len(n_classes)]

      # split the data into training and testing sets
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

      #Modello di stacking della libreria mlxtend
      stack = StackingCVClassifier(classifiers = list(models_stack.values()),
                                  shuffle = False,
                                  use_probas = False,
                                  cv = 5,
                                  meta_classifier = models_stack['SVC'])
      models['stack'] = stack

      #Scorre i modelli nel dizionario dei modelli utilizzati
      for name, model in models.items():
        print("Modello "+name)
        model_base = model
        model_best = model

        #Modello base: cross validation with score, fit, predict
        score_cv = makeCrossValidation(model_base, X_train, y_train)
        model_base.fit(X_train, y_train)
        y_pred = model_base.predict(X_test)

        score = makeScore(y_test, y_pred)

        ris = {'Target': column,
                'Dataframe' : str_df,
                'Model': name,
                'Accuracy CV' : score_cv['acc'],
                'St. Dev. CV' : score_cv['st'],
                'Precision CV' : score_cv['prec'],
                'Recall CV' : score_cv['rec'],
                'F1-Score CV' : score_cv['f1'],
                'Accuracy' : score['acc'],
                'Bal. Accuracy' : score['b_acc']}

        #display(ris)
        metrics_df.loc[len(metrics_df)] = ris
        model_obj[str_target+'_'+column+'_'+str_df+'_'+name] = model_base
        
        if name in param_grid and column == 'subspecies' and tuning!='':
          #TUNING MODEL
          print('TUNING')
          model_best, score_cv = makeTuning(model_best, X_train, y_train, name)
          
          model_best.fit(X_train, y_train)
          #display(X_test)
          y_pred = model_best.predict(X_test)
          score = makeScore(y_test, y_pred)

          prediction[column][str_df+name] = y_pred

          ris = {'Target': column,
                  'Dataframe' : str_df,
                  'Model': name+'_Best',
                  'Accuracy CV' : score_cv['acc'],
                  'St. Dev. CV' : score_cv['st'],
                  'Precision CV' : score_cv['prec'],
                  'Recall CV' : score_cv['rec'],
                  'F1-Score CV' : score_cv['f1'],
                  'Accuracy' : score['acc'],
                  'Bal. Accuracy' : score['b_acc']}
          #display(ris)
          metrics_df.loc[len(metrics_df)] = ris
          model_obj[str_target+'_'+column+'_'+str_df+'_'+name+'_Best'] = model_best
          pickle.dump(model_best, open('../'+cartella+'/models/models_base/'+name+'_'+tutti_picchi+reduction+scaled+scaler+tuning+column+'_'+str_df+'_npicchi'+str(n)+'_npicchimax'+str(picco_max)+'.pkl', 'wb'))
          #print("Tuning fatto")
        else:
          pickle.dump(model_base, open('../'+cartella+'/models/models_base/'+name+'_'+tutti_picchi+reduction+scaled+scaler+tuning+column+'_'+str_df+'_npicchi'+str(n)+'_npicchimax'+str(picco_max)+'.pkl', 'wb'))
        
  #Aggiunge i valori del target nei dizionari
  prediction[column].index = X_test.index
  prediction[column]['Target'] = y_test.values
  prediction[column].to_csv('../'+cartella+'/predictions/models_base_'+column+'_'+tutti_picchi+reduction+scaled+scaler+tuning+name+'_'+str_df+'_npicchi'+str(n)+'_npicchimax'+str(picco_max)+'.csv', index = True)
metrics_df.to_csv('../'+cartella+'/results/models_base_'+tutti_picchi+reduction+scaled+scaler+tuning+'npicchi'+str(n)+'_npicchimax'+str(picco_max)+'.csv', index = False)

display(metrics_df)
#print("Modelli:")
#print(model_obj)

#Per ogni target prende il migliore in base alla metrica utilizzata,
#una volta che un modello è stato selezionato come migliore per un target data una specifica metrica,
#viene rimosso in modo che non si ripeta lo stesso modello per tutte le metriche
metriche = ['Accuracy CV', 'Bal. Accuracy', 'F1-Score CV', 'Recall CV', 'Precision CV']
best_models = {}
for str_df, X in dfs.items():
  best_models[str_df] = {}
  for str_target, target in targets.items():
    columns = target.columns
    for column in columns:
      filter = metrics_df["Target"]==column
      filter_df = metrics_df["Dataframe"]==str_df
      # filtering data
      subs_df = metrics_df.where(filter & filter_df, inplace = False).dropna()
      list_models = []
      #display(subs_df)
      for metric in metriche:
        #Prende l'id del modello con valore massimo per lo speficico target e metrica
        id = subs_df[metric].idxmax()
        model = subs_df.loc[id,'Model']
        max = subs_df[metric].max()
        list_models.append(str_target+'_'+column+'_'+str_df+'_'+model)
        subs_df.drop(id, inplace = True)

        print(column+' : '+str_df+' : '+model+' : '+metric+' : '+str(max))
      best_models[str_df][column] = list_models
#display(best_models)

metrics_stack = pd.DataFrame(columns=['Target', 'Dataframe', 'Model', 'Accuracy CV', 'St. Dev. CV',
                            'Precision CV', 'Recall CV','F1-Score CV','Accuracy','Bal. Accuracy'])
dfs = {'maldi' : maldi,
       'clusters' : df_clusters,
       'animals' : df_animals,
       'agg' : df_agg,
       'clusters+agg' : df_cluster_agg}

final_models = {}
for str_df,targets_best in best_models.items():
  #display(targets_best)
  prediction[str_df] = pd.DataFrame()
  for str_target,list_models in targets_best.items():
    #print(str_target)
    #print(list_models)
    stack_models = list()
    
    for str_model in list_models:
      model = model_obj[str_model]
      #print(model)
      stack_models.append((str_model, clone(model)))

    final_model = LogisticRegression()
    stack = StackingClassifier(estimators=stack_models, final_estimator=final_model)
    model_base = stack
    model_best = stack

    s = str_model.split('_')
    #print(s)
    trg = targets[s[0]]
    y = trg.loc[:,s[1]]
    X = dfs[str_df]
    #display(X)
    #display(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    #display(X_train)
    X_train.dropna(inplace=True)
    prediction[str_df].index = X_test.index
    
    model_base.fit(X_train, y_train)
    final_models[str_model] = model_base
    y_pred = model_base.predict(X_test)

    prediction[str_df][str_target+'_pred'] = y_pred
    
    if (str_target == 'antibiotici'):
      prediction[str_df][column+'_pred'] = prediction[str_df][column+'_pred'].map(map_target_antibiotici_inv)
    prediction[str_df][column] = y

    #Modello base: cross validation with score, fit, predict
    score_cv = makeCrossValidation(model_base, X_train, y_train)
    score = makeScore(y_test, y_pred)

    ris = {'Target': str_target,
          'Dataframe' : str_df,
          'Model': 'Stack',
          'Accuracy CV' : score_cv['acc'],
          'St. Dev. CV' : score_cv['st'],
          'Precision CV' : score_cv['prec'],
          'Recall CV' : score_cv['rec'],
          'F1-Score CV' : score_cv['f1'],
          'Accuracy' : score['acc'],
          'Bal. Accuracy' : score['b_acc']}

    #display(ris)
    metrics_stack.loc[len(metrics_stack)] = ris
    
    pickle.dump(model_base, open('../'+cartella+'/models/stack_'+tutti_picchi+reduction+scaled+scaler+tuning+str_target+'_'+str_df+'_npicchi'+str(n)+'_npicchimax'+str(picco_max)+'.pkl', 'wb'))

  prediction[str_df]['subspecies'] = prediction[str_df]['subspecies'].map(map_target_inv)
  prediction[str_df]['subspecies_pred'] = prediction[str_df]['subspecies_pred'].map(map_target_inv)
  prediction[str_df].to_csv('../'+cartella+'/predictions/stack_'+tutti_picchi+reduction+scaled+scaler+tuning+str_df+'_npicchi'+str(n)+'_npicchimax'+str(picco_max)+'.csv', index = True)
  #display(prediction[str_df])
print('\n')
metrics_stack.to_csv('../'+cartella+'/results/stack_'+tutti_picchi+reduction+scaled+scaler+tuning+'npicchi'+str(n)+'_npicchimax'+str(picco_max)+'.csv', index = False)
display(metrics_stack)

name_best = list()
score_best = list()
target_list = list()
for str_target, target in targets.items():
  columns = target.columns
  for column in columns:
    filter = metrics_stack["Target"]==column
    subs_df = metrics_stack.where(filter, inplace = False).dropna()
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
plt.rcParams["figure.figsize"] = (16,6)
# plot bars with left x-coordinates [xs], heights [num_oscars]
plt.barh(y_pos, score_best, align='center',)
# label x-axis with movie names at bar centers
plt.yticks(y_pos, target_list)
plt.xlabel("% of Accuracy")
plt.title("Risultati di Balanced Accuracy sul miglior modello ensemble sui targets")
plt.show()