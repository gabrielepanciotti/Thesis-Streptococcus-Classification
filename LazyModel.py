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

map_reduce = {
    "Streptococcus canis" : "Canis",
    "Streptococcus dysgalactiae subsp. equisimilis" : "Equis",
    "Streptococcus dysgalactiae subsp. dysgalactiae" : "Dysg"
}

with open('output_LazyModel_Subspecies.txt', 'w') as sys.stdout:
    df_46 = pd.read_csv("data/Dati_Matemaldomics_46picchi.csv",
                    delimiter=';', index_col='ID Strain')
    df_306 = pd.read_csv("data/Dati_Matemaldomics_306picchi.csv",
                    delimiter=';', index_col='ID Strain')
    
    maldi_46 = df_46[df_46.columns[9:55]]
    maldi_306 = df_306[df_306.columns[9:315]]
    
    dfs = {'46picchi' : maldi_46,
           '306picchi' : maldi_306}
    
    target = df_46["Putative Subspecies"].map(map_reduce)
    
    for str_df,df in dfs.items():
        df.fillna(0, inplace=True)
        df = df.replace(',', '.', regex=True)
        columns = df.columns
        for column in columns:
            df[column] = df[column].astype(float)
        display(df)
        
        X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.2, random_state=RANDOM_STATE)
        clf = LazyClassifier(predictions=True)
        models, predictions = clf.fit(X_train, X_test, y_train, y_test)
        display(models)
        predictions.index = y_test.index
        predictions['target'] =  y_test
        display(predictions[[models.iloc[0].name,models.iloc[1].name,models.iloc[2].name,'target']])
