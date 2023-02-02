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

with open('output.txt', 'w') as sys.stdout:
    df_46 = pd.read_csv("data/Dati_Matemaldomics_46picchi.csv",
                    delimiter=';', index_col='ID Strain')
    df_306 = pd.read_csv("data/Dati_Matemaldomics_306picchi.csv",
                    delimiter=';', index_col='ID Strain')

    maldi_46 = df_46[df_46.columns[9:55]]
    df_46['target'] = df_46["Putative Subspecies"].map(map_target)

    maldi_306 = df_306[df_306.columns[9:315]]
    df_306['target'] = df_306["Putative Subspecies"].map(map_target)

    maldi_46.fillna(0, inplace=True)
    maldi_46 = maldi_46.replace(',', '.', regex=True)
    columns = maldi_46.columns
    for column in columns:
        maldi_46[column] = maldi_46[column].astype(float)
    display(maldi_46)

    maldi_306.fillna(0, inplace=True)
    maldi_306 = maldi_306.replace(',', '.', regex=True)
    columns = maldi_306.columns
    for column in columns:
        maldi_306[column] = maldi_306[column].astype(float)
    display(maldi_306)

    mm = MinMaxScaler()
    ss = StandardScaler()
    logr = LogisticRegression(random_state=RANDOM_STATE, solver='lbfgs', max_iter=1000)
    ridge = Ridge(random_state=RANDOM_STATE)
    lasso = Lasso(random_state=RANDOM_STATE)
    dtc = DecisionTreeClassifier(random_state=RANDOM_STATE)
    knn = KNeighborsClassifier()
    rm = RandomForestClassifier(oob_score=False, n_jobs=N_JOBS, random_state=RANDOM_STATE)
            
    dataframes = [['maldi_46' , maldi_46 , df_46['target']],['maldi_306', maldi_306, df_306['target']]]
    scalers = {'MinMax' : mm, 'SS' : ss}
    models = {'Logistic Regression' : logr,'Ridge' : ridge, 'Lasso' : lasso, 'Decision Trees' : dtc, 'K-nn' : knn, 'Random Forest' : rm}
    type_features = ["all_features", "pca"]
    cv_scores = list()
    ac_scores = list()
    data_cv = list()
    data_ac = list()
    data_cv_base = list()
    data_ac_base = list()

    alphas = np.linspace(0, 0.2, 21)
    #alphas = np.logspace(-5, 0, 100)
    
    skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    x = 0
    t = 0
    dataframe = list()
    for type_feature in type_features:
        cv_scores.append(list())
        ac_scores.append(list())
        d=0
        for dataframe in dataframes:
            str_df = dataframe[0]
            X = dataframe[1]
            y = dataframe[2]
            cv_scores[t].append(list())
            ac_scores[t].append(list())
            s=0
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y)
            for str_scaler, scaler in scalers.items():
                cv_scores[t][d].append(list())
                ac_scores[t][d].append(list())
                #Scala i dati
                X_train = scaler.fit_transform(X=X_train)
                X_test = scaler.transform(X=X_test)
                    
                if type_feature == "pca":
                    model_pca = PCA(n_components = 0.95)
                    X_train = model_pca.fit_transform(X_train)
                    X_test = model_pca.transform(X_test)
                    print("Feature con PCA:", X_train.shape)
                
                data_cv.append(list())
                data_ac.append(list())
                data_cv[x].append("Best_"+str_df+"_"+type_feature+"_"+str_scaler)
                data_ac[x].append("Best_"+str_df+"_"+type_feature+"_"+str_scaler)
                m=0
                for str_model, model in models.items():
                    model.fit(X_train, y_train)
                    
                    if str_model == 'Logistic Regression':
                        model = LogisticRegression(random_state=RANDOM_STATE, solver='lbfgs', max_iter=1000)
                        params = {
                            "penalty": ["l2", "l1"],
                            "C": stats.loguniform(1e0, 1e2),
                            "class_weight": [None, "balanced"]
                        }
                    if str_model == 'Ridge':
                        model = Ridge(random_state=RANDOM_STATE)
                        
                        params = {
                            "alpha": alphas,
                        }
                    if str_model == 'Lasso':
                        model = Lasso(random_state=RANDOM_STATE)
                        params = {
                            "alpha": alphas,
                        }
                    if str_model == 'Decision Trees':
                        print('Altezza albero base'+str(model.get_depth()))
                        model = DecisionTreeClassifier(random_state=RANDOM_STATE)
                        path = model.cost_complexity_pruning_path(X=X_train, y=y_train)
                        ccp_alphas, impurities = path.ccp_alphas, path.impurities
                        
                        params = {
                            'ccp_alpha': ccp_alphas,
                            'max_depth': [None,3,5,7,10,15],
                            'min_samples_leaf': [1,3,5,10,15,20],
                            'min_samples_split': [2,4,6,8,10,12,14,16,18,20],
                            'criterion': ['gini','entropy']
                        }
                    if str_model == 'K-nn':
                        model = KNeighborsClassifier()
                        params = {
                            'n_neighbors' : [1,3,5,7,9,11,13,15],
                            'weights' : ['uniform','distance'],
                            'metric' : ['minkowski','euclidean','manhattan']
                        }
                    if str_model == 'Random Forest':
                        model = RandomForestClassifier(oob_score=True, n_jobs=N_JOBS, random_state=RANDOM_STATE)
                        params = {
                            "n_estimators": [25, 50, 100, 200, 250, 500],
                            "criterion": ["gini", "entropy"],
                            "max_depth": [None, 1, 2, 5, 10, 20],
                            "max_features": ["sqrt", "log2"],
                            "class_weight": [None, "balanced", "balanced_subsample"]
                        }
                    rs = RandomizedSearchCV(estimator=model, param_distributions=params,
                                        scoring="accuracy", n_jobs=-1, cv=skfold, verbose=1)
                    rs.fit(X_train, y_train)
                    print(str_model+" Best \
                        \nScaler: "+str_scaler+
                        "\nDataframe: "+str_df+
                        "\nFeatures: "+type_feature)
                    print(rs.best_params_)
                    print('Random '+str_model+': '+str(rs.best_score_))
                    parametri = rs.best_params_
                    model.set_params(**parametri)
                    print(model.get_params())
                    model.fit(X_train, y_train)
                    cv_scores[t][d][s].append(cross_val_score(estimator=model, X=X_train, y=y_train,
                                                            scoring="accuracy", cv=skfold, n_jobs=N_JOBS, verbose=0))
                    print(f"Mean CV accuracy: {cv_scores[t][d][s][m].mean():.4f} +/- {cv_scores[t][d][s][m].std():.4f}")
                    print(cv_scores[t][d][s][m])
                    
                    y_pred = model.predict(X_test)
                    ac_scores[t][d][s].append(model.score(X_test, y_test))
                    #report = classification_report(y_true=y_test, y_pred=y_pred)
                    #print(report)
                    
                    data_cv[x].append(str(cv_scores[t][d][s][m].mean()))
                    data_ac[x].append(str(ac_scores[t][d][s][m]))
                    m += 1
                    
                x += 1           
                s += 1
            d += 1
        t += 1

    feature_index = ['Feature','Logistic Regression', 'Ridge', 'Lasso', 'Decision Tree', 'Knn', 'Random Forest']
    df_ac_best = pd.DataFrame(data_ac, columns=feature_index)
    df_ac_best.to_csv('accuracy_maldi2.csv', index=False, mode='w')
    df_cv_best = pd.DataFrame(data_cv, columns=feature_index)
    df_cv_best.to_csv('cv_maldi2.csv', index=False, mode='w')

    print("AC SCORES BEST")
    display(df_ac_best)
    print("CV SCORES BEST")
    display(df_cv_best)