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

import sys
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
    dtc = DecisionTreeClassifier(random_state=RANDOM_STATE)
    knn = KNeighborsClassifier()
    rm = RandomForestClassifier(oob_score=False, n_jobs=N_JOBS, random_state=RANDOM_STATE)
            
    dataframes = {'maldi_46' : maldi_46, 'maldi_306' : maldi_306}
    scalers = {'MinMax' : mm, 'SS' : ss}
    models = {'Logistic Regression' : logr, 'Decision Trees' : dtc, 'K-nn' : knn, 'Random Forest' : rm}
    type_features = ["all_features", "pca"]
    cv_scores = list()
    ac_scores = list()
    cv_scores_base = list()
    ac_scores_base = list()
    data_cv = list()
    data_ac = list()
    data_cv_base = list()
    data_ac_base = list()

    y = df_46['target']
    skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    x = 0
    t = 0
    for type_feature in type_features:
        cv_scores.append(list())
        ac_scores.append(list())
        cv_scores_base.append(list())
        ac_scores_base.append(list())
        d=0
        for str_df, X in dataframes.items():
            cv_scores[t].append(list())
            ac_scores[t].append(list())
            cv_scores_base[t].append(list())
            ac_scores_base[t].append(list())
            s=0
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y)
            for str_scaler, scaler in scalers.items():
                cv_scores[t][d].append(list())
                ac_scores[t][d].append(list())
                cv_scores_base[t][d].append(list())
                ac_scores_base[t][d].append(list())
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
                data_cv_base.append("Base_"+str_df+"_"+type_feature+"_"+str_scaler)
                data_ac_base.append("Base_"+str_df+"_"+type_feature+"_"+str_scaler)
                data_cv[x].append("Best_"+str_df+"_"+type_feature+"_"+str_scaler)
                data_ac[x].append("Best_"+str_df+"_"+type_feature+"_"+str_scaler)
                m=0
                for str_model, model in models.items():
                    model.fit(X_train, y_train)
                    cv_scores_base[t][d][s].append(cross_val_score(estimator=model, X=X_train, y=y_train,
                                                            scoring="accuracy", cv=skfold, n_jobs=N_JOBS, verbose=1))
                    print(str_model+" Base"+
                        "\nScaler: "+str_scaler+
                        "\nDataframe: "+str_df+
                        "\nFeatures: "+type_feature)
                    print(model.get_params())
                    print(f"Mean CV accuracy: {cv_scores_base[t][d][s][m].mean():.4f} +/- {cv_scores_base[t][d][s][m].std():.4f}")
                    print(cv_scores_base[t][d][s][m])

                    y_pred = model.predict(X_test)
                    ac_scores[t][d][s].append(model.score(X_test, y_test))
                    report = classification_report(y_true=y_test, y_pred=y_pred)
                    print(report)
                    
                    if str_model == 'Logistic Regression':
                        model = LogisticRegression(random_state=RANDOM_STATE, solver='lbfgs', max_iter=1000)
                        params = {
                            "penalty": ["l2", "l1"],
                            "C": stats.loguniform(1e0, 1e2),
                            "class_weight": [None, "balanced"]
                        }
                    if str_model == 'Decision Trees':
                        print('Altezza albero base'+str(model.get_depth()))
                        path = model.cost_complexity_pruning_path(X=X_train, y=y_train)
                        ccp_alphas, impurities = path.ccp_alphas, path.impurities
                        model = DecisionTreeClassifier(random_state=RANDOM_STATE)
                        params = {
                            'ccp_alpha': ccp_alphas,
                            'max_depth': [3,5,7,10,15],
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
                    report = classification_report(y_true=y_test, y_pred=y_pred)
                    print(report)
                    
                    data_cv[x].append(str(cv_scores[t][d][s][m].mean()))
                    data_ac[x].append(str(ac_scores[t][d][s][m]))
                    m += 1
                    if str_model == 'Logistic Regression' and type_feature != 'pca':
                        alphas = np.logspace(-5, 0, 100)
                        coefs_ridge = list()
                        coefs_lasso = list()
                        acc_ridge = list()
                        acc_lasso = list()
                        cv_ridge = list()
                        cv_lasso = list()
                            
                        #Rifa ridge e lasso ma questa volta direttamente su le feature, senza polynomial features
                        for alpha in alphas:
                            ridge = Ridge(alpha=alpha)
                            ridge.fit(X_train, y_train)
                            coefs_ridge.append(ridge.coef_)
                            acc_ridge[alpha] = ridge.score(X_test, y_test)
                            lasso = Lasso(alpha=alpha)
                            lasso.fit(X_train, y_train)
                            coefs_lasso.append(lasso.coef_)
                            acc_lasso[alpha] = lasso.score(X_test, y_test)
                            r_cv_score = cross_val_score(estimator=ridge, X=X_train, y=y_train,
                                                            scoring="accuracy", cv=skfold, n_jobs=N_JOBS, verbose=0).mean()
                            cv_ridge.append(r_cv_score)
                            l_cv_score = cross_val_score(estimator=lasso, X=X_train, y=y_train,
                                                            scoring="accuracy", cv=skfold, n_jobs=N_JOBS, verbose=0).mean()
                            cv_lasso.append(l_cv_score)
                            
                        max_r_score = max(acc_ridge, key=acc_ridge.get)
                        max_l_score = max(acc_lasso, key=acc_lasso.get)
                        max_r_cv_score = max(cv_ridge, key=cv_ridge.get)
                        max_l_cv_score = max(cv_lasso, key=cv_lasso.get)
                         
                        ac_scores[t][d][s].append(max_r_score)
                        cv_scores[t][d][s].append(max_r_cv_score)
                        print("RIDGE AC / CV")
                        print(ac_scores[t][d][s][m])
                        print(cv_scores[t][d][s][m])
                        m += 1
                        ac_scores[t][d][s].append(max_l_score)
                        cv_scores[t][d][s].append(max_l_cv_score)
                        print("LASSO AC / CV")
                        print(ac_scores[t][d][s][m])
                        print(cv_scores[t][d][s][m])
                        m += 1

                x += 1           
                s += 1
            d += 1
        t += 1

    feature_index = ['Feature','Logistic Regression', 'Ridge', 'Lasso' 'Decision Tree', 'Knn', 'Random Forest']
    df_ac_best = pd.DataFrame(data_ac, columns=feature_index)
    df_ac_best.to_csv('accuracy_maldi.csv', index=False, mode='w')
    df_cv_best = pd.DataFrame(data_cv, columns=feature_index)
    df_cv_best.to_csv('cv_maldi.csv', index=False, mode='w')
    df_ac_base = pd.DataFrame(data_ac_base, columns=feature_index)
    df_ac_base.to_csv('accuracy_maldi_base.csv', index=False, mode='w')
    df_cv_base = pd.DataFrame(data_cv_base, columns=feature_index)
    df_cv_base.to_csv('cv_maldi_base.csv', index=False, mode='w')

    print("AC SCORES BASE")
    display(df_ac_base)
    print("CV SCORES BASE")
    print(df_cv_base)
        
    print("AC SCORES BEST")
    display(df_ac_best)
    print("CV SCORES BEST")
    print(df_cv_best)