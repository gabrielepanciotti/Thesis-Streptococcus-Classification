from lazypredict.Supervised import LazyClassifier
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn.inspection import DecisionBoundaryDisplay

from scipy import stats
from pca import pca
from IPython.display import display
import dataframe_image as dfi

import warnings
    
# warnings -> to silence warnings

warnings.filterwarnings("ignore")
np.set_printoptions(precision=5, suppress=True)


RANDOM_STATE = 42
N_JOBS = -1
class_names = ["Canis", "Dysg. Equisimilis", "Dysg. Dysgalactiae"]

map_target = {
    "Streptococcus canis": 0,
    "Streptococcus dysgalactiae subsp. equisimilis": 1,
    "Streptococcus dysgalactiae subsp. dysgalactiae": 2
}

map_target_inv = {
    0: "Strept. canis",
    1: "Strept. dysg. equisimilis",
    2: "Strept. dysg. dysgalactiae"
}

map_target_antibiotici = {
    "S" : 1,
    "NS" : 0
}
start = 9
n_antibiotici = 9
n_geni = 27
n_virulenza = 18
n_picchi = ['46','306']
#n_picchi = ['46']
N_BEST = 3

def makeDictBest(models, modelli, n_best = 5):
    for i in range(n_best):
        best = models.index[i]
        score = modelli.get(best)
        if score != None:
            modelli[best] = score + (n_best-i)
        else:
            modelli[best] = (n_best-i)
    return modelli

def dimensionality_reduction(X_tr, X_te, n_components):
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_tr)
    X_test_pca = pca.transform(X_te)
    X_train_pca = pd.DataFrame(X_train_pca)
    X_test_pca = pd.DataFrame(X_test_pca)
    
    return X_train_pca, X_test_pca

for n in n_picchi:
    print('DATAFRAME CON '+n+' PICCHI')
    df = pd.read_csv("data/Dati_Matemaldomics_"+n+"picchi.csv",
                    delimiter=';', index_col='ID Strain')
    n = int(n)
    animal  = df[['Animal species of origin']]
    lancefield = df[['LANCEFIELD GROUP']]
    haemolysis = df[['Haemolysis']]
    subspecies = df[['Putative Subspecies']]

    st = df[[df.columns[4]]]
    maldi = df[df.columns[start:start+n]]
    antibiotici = df[df.columns[start+n:start+n+n_antibiotici]]
    geni_antibiotici = df[df.columns[start+n+n_antibiotici:start+n+n_antibiotici+n_geni]]
    virulenza = df[df.columns[start+n+n_antibiotici+n_geni:start+n+n_antibiotici+n_geni+n_virulenza]]
    
    maldi.fillna(0, inplace=True)
    maldi = maldi.replace(',', '.', regex=True)
    columns = maldi.columns
    for column in columns:
        maldi[column] = maldi[column].astype(float)
    display(maldi)
    
    targets = {'antibiotici' : antibiotici,
                'geni_antibiotici' : geni_antibiotici,
                'virulenza' : virulenza}
    
    feats_agg = {'lancefield' : lancefield,
                'haemolysis' : haemolysis,
                'subspecies' : subspecies,        
                'animal' : animal
                }
    
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
        
        display(target)
        
    metrics_df = pd.DataFrame(columns=['Target','Model','Accuracy','Precision','Recall','F1-Score','CV'])
    metrics_df_agg = pd.DataFrame(columns=['Target','Model','Accuracy','Precision','Recall','F1-Score','CV'])
    skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    targets['st'] = st
    
    
    for redu in ['','_pca']:
        X = maldi
        targets['subspecies'] = subspecies
        modelli = {}
        for str_target, target in targets.items():
            columns = target.columns
            for column in columns:    
                y = target[column]
                n_classes = np.unique(y)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
                if redu == '_pca':
                    X_train, X_test = dimensionality_reduction(X_train, X_test, n_components=0.95)
                clf = LazyClassifier(predictions=True)
                models, predictions = clf.fit(X_train, X_test, y_train, y_test)
                print("Colonna:"+column)
                display(models)
                print("\n")
                modelli = makeDictBest(models, modelli)
                
                models.to_csv('Risultati/LazyPredictor/model_'+str(n)+column+redu+'.csv')
        if redu == '':
            del targets['subspecies']
            for str_feat, feat_agg in feats_agg.items():
                display(feat_agg)
                X = pd.concat([X, feat_agg], axis=1)
                for str_target, target in targets.items():
                    columns = target.columns
                    for column in columns:    
                        y = target[column]
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
                        if redu == '_pca':
                            X_train, X_test = dimensionality_reduction(X_train, X_test, n_components=0.95)
                        clf = LazyClassifier(predictions=True)
                        models, predictions = clf.fit(X_train, X_test, y_train, y_test)
                        print("Colonna: "+column+" con feat agg: "+str_feat)
                        display(models)
                        print("\n")
                        modelli = makeDictBest(models, modelli)
                        
                        models.to_csv('Risultati/LazyPredictor/model_'+str(n)+column+'_morefeat'+str_feat+redu+'.csv')
        modelli = sorted(modelli.items(), key=lambda x:x[1])
        print(modelli)