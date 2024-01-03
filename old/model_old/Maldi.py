import pandas as pd
import sys
import numpy as np
from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, make_scorer
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression, RidgeClassifier, LassoCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.base import clone
from sklearn.naive_bayes import BernoulliNB, GaussianNB

import warnings
from IPython.display import display
    
warnings.filterwarnings("ignore")
np.set_printoptions(precision=5, suppress=True)


RANDOM_STATE = 1
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

# Load the data
n_picchi = ['46','306']

# Define a function for standard scaling
def standard_scaler(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# Define a function for dimensionality reduction using PCA
def dimensionality_reduction(X_train, X_test, n_components):
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    X_train_pca = pd.DataFrame(X_train_pca)
    X_test_pca = pd.DataFrame(X_test_pca)
    
    return X_train_pca, X_test_pca

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

# define the models
models = {'LogisticRegression': LogisticRegression(random_state=RANDOM_STATE),
          'Ridge' : RidgeClassifier(random_state=RANDOM_STATE),
          'DecisionTree': DecisionTreeClassifier(random_state=RANDOM_STATE),
          'K-nn': KNeighborsClassifier(),
          'RandomForest': RandomForestClassifier(random_state=RANDOM_STATE),
          'BernoulliNB': BernoulliNB(),
          'GaussianNB': GaussianNB(),
          #'NearestCentroid': NearestCentroid()
          }

n_classes = [0,1]
# Hyperparameter tuning using RandomizedSearchCV
param_grid = {'LogisticRegression': {'C': np.logspace(-4, 4, 25), 
                                    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                                    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                                    'fit_intercept': [True, False],
                                    'intercept_scaling': [0.5, 1, 2],
                                    'class_weight': [None, 'balanced']
                                    },
              'Ridge' : {'alpha': np.logspace(-5, 5, 75)},
              'DecisionTree': {'ccp_alpha': [0.0] + list(np.logspace(-3, 1, 25)),
                                'class_weight': [None, 'balanced'],
                                'criterion': ['gini', 'entropy', 'log_loss'],
                                'max_depth': [None] + list(range(1, 20)),
                                'max_features': [None, 'auto', 'sqrt', 'log2'],
                                'min_samples_leaf': range(1, 10),
                                'min_samples_split': range(2, 10),
                                'splitter': ['best', 'random']
                                },
              'K-nn': {'n_neighbors': list(range(1, 20, 1)),
                        'weights': ['uniform', 'distance'],
                        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                        'p': [1,2]
                        },
              'RandomForest': {'ccp_alpha': [0] + list(np.logspace(-3, 1, 25)),
                                'class_weight': [None, 'balanced'],
                                'n_estimators': range(50,500,50),
                                'max_features': [None, 'auto', 'sqrt', 'log2'],
                                'max_depth' : [None,4,6,8,10],
                                'criterion' :['gini', 'entropy']
                                },
              'BernoulliNB': {'alpha': np.logspace(-2, 1, 10),
                            'fit_prior': [True, False],
                            'class_prior': [None, [0.1,]* len(n_classes)],
                            'binarize': [None, -5, -2, 0.0, 2, 5, 10.0]
                            },
              'GaussianNB': {'var_smoothing': np.logspace(0,-9, num=20)
                             },
              'NearestCentroid': {'shrink_threshold': np.logspace(0, 1, 20),
                                'metric': ['euclidean', 'manhattan']
                                },
              'SVC': {'C': np.logspace(-4, 4, 25),
                      'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                      'degree': range(2,5),
                      'gamma': np.logspace(-3, 1, 25)},
              'LabelPropagation': {'n_neighbors': [7, 21, 41, 81, 121, 181, 241],
                                   'gamma': [0.1, 1, 5, 10, 20, 30, 50]},
              'LabelSpreading': {'n_neighbors': [7, 21, 41, 81, 121, 181, 241],
                                'gamma': [0.1, 1, 5, 10, 20, 30, 50],
                                'alpha': [0.15, 0.2, 0.35, 0.55, 0.75, 0.95]},
              'SGDClassifier': {'loss' : ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
                            'penalty' : ['l1', 'l2', 'elasticnet'],
                            'alpha' : np.logspace(-4, 4, 25),
                            'learning_rate' : ['constant', 'optimal', 'invscaling', 'adaptive'],
                            'class_weight' : [{1:0.5, 0:0.5}, {1:0.4, 0:0.6}, {1:0.6, 0:0.4}, {1:0.7, 0:0.3}],
                            'eta0' : [1, 10, 100]},
              'LinearSVC': {'penalty': ['l1', 'l2'],
                            'loss': ['hinge', 'squared_hinge'],
                            'class_weight': [None, 'balanced']}
}

with open('output_Maldi.txt', 'w') as sys.stdout:
    for n in n_picchi:
        df = pd.read_csv("data/Dati_Matemaldomics_"+n+"picchi.csv",
                            delimiter=';', index_col='ID Strain')
        df['subspecies'] = df["Putative Subspecies"].map(map_target)

        n = int(n)
        feat_agg = df.iloc[:,[7,8]]
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
                    
            display(target)
            
        targets['st'] = st
        target['subspecies'] = subspecies
        
        # create an empty dataframe to store the metrics
        #Tutte le metriche in cv e con st, la metrica nella tesi è quella
        score_target = {}
        metrics_df = pd.DataFrame(columns=['Target', 'Model', 'Accuracy CV', 'St. Dev. CV', 
                                        'Precision CV', 'Recall CV','F1-Score CV','Accuracy'])
        #metrics_pca_df = pd.DataFrame(columns=['Target', 'Model', 'Accuracy CV', 'STD CV', 'Precision CV','Recall CV','F1-Score CV','Accuracy'])

        metrics = {}
        scori = ['accuracy', 'recall_weighted', 'precision_weighted','f1_weighted']
        skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        X = maldi
        for str_target, target in targets.items():
            columns = target.columns
            for column in columns:    
                y = target[column]
                n_classes = np.unique(y)
                param_grid['BernoulliNB']['class_prior'] = [None, [0.1,]* len(n_classes)]
                # split the data into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                X_pca_train, X_pca_test = dimensionality_reduction(X_train, X_test, n_components=0.95)
                #print("Colonna:"+column)
                dataframes = {'' : (X_train, X_test),
                                '_PCA' : (X_pca_train, X_pca_test)}
                # evaluate the models on the original dataset
                #pca = '_PCA'
                
                for name, model in models.items():
                        for pca, dataframe in dataframes.items():
                                X_train = dataframe[0]
                                X_test = dataframe[1]
                                model_base = model
                                model_best = model
                                
                                #Modello base
                                #print("Modello:"+name)
                                
                                cv = cross_validate(estimator=model_base, X=X_train, y=y_train,
                                                    scoring=scori, cv=skfold, n_jobs=N_JOBS, verbose=0)
                
                                metrics['acc'] = cv.get('test_accuracy').mean()  
                                metrics['st'] = cv.get('test_accuracy').std()
                                metrics['prec'] = cv.get('test_precision_weighted').mean()
                                metrics['rec'] = cv.get('test_recall_weighted').mean()
                                metrics['f1'] = cv.get('test_f1_weighted').mean()
                                
                                model_base.fit(X_train, y_train)
                                y_pred = model_base.predict(X_test)
                                
                                acc = accuracy_score(y_test, y_pred)
                                #prec = precision_score(y_test, y_pred, average='weighted')
                                #rec = recall_score(y_test, y_pred, average='weighted')
                                #f1 = f1_score(y_test, y_pred, average='weighted')          
                                
                                ris = {'Target': column,
                                        'Model': name+pca,
                                        'Accuracy CV' : metrics['acc'],
                                        'St. Dev. CV' : metrics['st'],
                                        'Precision CV' : metrics['prec'],
                                        'Recall CV' : metrics['rec'],
                                        'F1-Score CV' : metrics['f1'],
                                        'Accuracy' : acc} 
                                #display(ris)
                                metrics_df = metrics_df.append(ris, ignore_index=True)  
                                '''
                                if name == 'DecisionTree' or name == 'RandomForest':
                                        print(name)
                                        parametri = model_base.get_params()
                                        print('Old Parametri:')
                                        print(parametri)
                                '''
                                #Tuning iperparametri
                                params = param_grid[name]
                                rs = RandomizedSearchCV(estimator=model_best, param_distributions=params,
                                                    scoring=scori, refit="accuracy", cv=skfold, 
                                                    n_jobs=-1, random_state=RANDOM_STATE, verbose=0)
                                rs.fit(X_train, y_train)
                                
                                results = rs.cv_results_
                                model_best = rs.best_estimator_
                                parametri = model_best.get_params()
                                cv_best = rs.best_score_
                                '''
                                if name == 'DecisionTree' or name == 'RandomForest':
                                        print('Parametri possibili:')
                                        print(params)
                                        print('New Parametri:')
                                        print(parametri)
                                        print(cv_best)
                                '''    
                                cv = cross_validate(estimator=model_best, X=X_train, y=y_train,
                                                    scoring=scori, cv=skfold, n_jobs=N_JOBS, verbose=0)
                
                                metrics['acc'] = cv.get('test_accuracy').mean()  
                                metrics['st'] = cv.get('test_accuracy').std()
                                metrics['prec'] = cv.get('test_precision_weighted').mean()
                                metrics['rec'] = cv.get('test_recall_weighted').mean()
                                metrics['f1'] = cv.get('test_f1_weighted').mean()
                                
                                model_best.fit(X_train, y_train)
                                y_pred = model_best.predict(X_test)
                                
                                acc = accuracy_score(y_test, y_pred)
                                #prec = precision_score(y_test, y_pred, average='weighted')
                                #rec = recall_score(y_test, y_pred, average='weighted')
                                #f1 = f1_score(y_test, y_pred, average='weighted')
                                
                                ris = {'Target': column,
                                        'Model': name+'_Best'+pca,
                                        'Accuracy CV' : metrics['acc'],
                                        'St. Dev. CV' : metrics['st'],
                                        'Precision CV' : metrics['prec'],
                                        'Recall CV' : metrics['rec'],
                                        'F1-Score CV' : metrics['f1'],
                                        'Accuracy' : acc} 
                                #display(ris)
                                metrics_df = metrics_df.append(ris, ignore_index=True)  
                print('\n')
                score_target[column] = metrics_df
                metrics_df.to_csv('Risultati\Results_Def_'+str(n)+'picchi\Results_'+column+'_'+str(n)+'.csv', index = False)
                
                metrics_df = pd.DataFrame(columns=['Target', 'Model', 'Accuracy CV', 'St. Dev. CV', 
                                            'Precision CV', 'Recall CV','F1-Score CV','Accuracy'])

                display(score_target[column])