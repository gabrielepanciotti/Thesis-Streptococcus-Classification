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

from sklearn.metrics import accuracy_score
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

from src.visualization import feature_importances_plot

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

with open('output_LazyModel_antibiotici.txt', 'w') as sys.stdout:
    #df_46 = pd.read_csv("data/Dati_Matemaldomics_46picchi.csv", delimiter=';', index_col='ID Strain')
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
        
        feats_agg = {'haemolysis' : haemolysis,
                    'subspecies' : subspecies,
                    'lancefield' : lancefield,
                    'animal' : animal}
        
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
        
        
        
        targets['st'] = st
        X = maldi
        for str_target, target in targets.items():
            columns = target.columns
            for column in columns:    
                y = target[column]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
                clf = LazyClassifier(predictions=True)
                models, predictions = clf.fit(X_train, X_test, y_train, y_test)
                print("Colonna:"+column)
                print(models)
                print("\n")
                models.to_csv('Risultati/model_'+str(n)+column+'.csv')
        
        
        
        display(X)
        for str_feat, feat_agg in feats_agg.items():
            display(feat_agg)
            X = pd.concat([X, feat_agg], axis=1)
            for str_target, target in targets.items():
                columns = target.columns
                for column in columns:    
                    y = target[column]
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
                    clf = LazyClassifier(predictions=True)
                    models, predictions = clf.fit(X_train, X_test, y_train, y_test)
                    print("Colonna: "+column+" con feat agg: "+str_feat)
                    print(models)
                    print("\n")
                    models.to_csv('Risultati/model_'+str(n)+column+'_morefeat'+str_feat+'.csv')
                
        #print(y_test)
        #display(predictions)