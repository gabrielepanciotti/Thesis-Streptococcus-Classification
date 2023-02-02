
from IPython.display import display
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

from src.visualization import feature_importances_plot

# warnings -> to silence warnings
import warnings
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
        
dataframes = {'maldi_46' : maldi_46, 'maldi_306' : maldi_306}
scalers = {'MinMax' : mm, 'SS' : ss}
type_features = ["all_features", "pca"]
cv_scores = list()
ac_scores = list()

y = df_46['target']

for type_feature in type_features:
    cv_scores[type_feature] = list()
    ac_scores[type_feature] = list()
    for str_dataframe, X in dataframes:
        cv_scores[type_feature][str_dataframe] = list()
        ac_scores[type_feature][str_dataframe] = list()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        alphas = np.logspace(-5, 0, 100)
        coefs_ridge = []
        metrics_ridge = []
        coefs_lasso = []
        metrics_lasso = []

        #Crea feature polinomiali di grado 2
        #Prova usando alpha che varia tra 10^-5 a 10^0, la ridge e la lasso e confronta i risultati 
        for alpha in alphas:
            po_ridge = Pipeline(
            steps=[
                ("Poly", PolynomialFeatures(degree=2)),
                ("Scaler", StandardScaler()),
                ("Model", Ridge(alpha=alpha))
            ])
            po_ridge.fit(X_train, y_train)
            #print(po_ridge.named_steps["Model"].coef_)
            coefs_ridge.append(po_ridge.named_steps["Model"].coef_)
            metrics_ridge.append({"lambda": alpha, "r2_train": po_ridge.score(X_train, y_train), "r2_test": po_ridge.score(X_test, y_test)})
            po_lasso = Pipeline(
            steps=[
                ("Poly", PolynomialFeatures(degree=2)),
                ("Scaler", StandardScaler()),
                ("Model", Lasso(alpha=alpha))
            ])
            po_lasso.fit(X_train, y_train)
            #print(po_ridge.named_steps["Model"].coef_)
            coefs_lasso.append(po_lasso.named_steps["Model"].coef_)
            metrics_lasso.append({"lambda": alpha, "r2_train": po_lasso.score(X_train, y_train), "r2_test": po_lasso.score(X_test, y_test)})
            
        plt.plot(alphas, coefs_ridge)
        plt.xscale("log")
        plt.title("Ridge with polynomial features")
        plt.tight_layout()
        plt.grid()
        plt.show()

        plt.plot(alphas, coefs_lasso)
        plt.xscale("log")
        plt.title("Lasso with polynomial features")
        plt.tight_layout()
        plt.grid()
        plt.show()

        df_metrics_ridge = pd.DataFrame(metrics_ridge)
        df_metrics_ridge["difference"] = df_metrics_ridge["r2_train"] - df_metrics_ridge["r2_test"]

        df_metrics_lasso = pd.DataFrame(metrics_lasso)
        df_metrics_lasso["difference"] = df_metrics_lasso["r2_train"] - df_metrics_lasso["r2_test"]

        sns.lineplot(data=df_metrics_ridge, x="lambda", y="r2_train", label="train scores ridge")
        sns.lineplot(data=df_metrics_ridge, x="lambda", y="r2_test", label="test scores ridge")
        sns.lineplot(data=df_metrics_lasso, x="lambda", y="r2_train", label="train scores lasso")
        sns.lineplot(data=df_metrics_lasso, x="lambda", y="r2_test", label="test scores lasso")
        plt.xscale("log")
        plt.title("Score ridge/lasso with polynomial features")
        plt.grid()
        plt.tight_layout()
        plt.show()

        sns.lineplot(data=df_metrics_ridge, x="lambda", y="difference", label="difference scores ridge")
        sns.lineplot(data=df_metrics_lasso, x="lambda", y="difference", label="difference scores lasso")
        plt.xscale("log")
        plt.title("Difference between train and test score for ridge/lasso with polynomial features")
        plt.grid()
        plt.tight_layout()
        plt.show()
        {'algorithm': 'auto', 'leaf_size': 30, 'metric': 'minkowski', 'metric_params': None, 'n_jobs': None, 'n_neighbors': 5, 'p': 2, 'weights': 'uniform'}
        {'algorithm': 'auto', 'leaf_size': 30, 'metric': 'minkowski', 'metric_params': None, 'n_jobs': None, 'n_neighbors': 11, 'p': 2, 'weights': 'distance'}
        
        for str_scaler, scaler in scalers:
            cv_scores[type_feature][str_dataframe][str_scaler] = list()
            ac_scores[type_feature][str_dataframe][str_scaler] = list()
            #Scala i dati
            X_train = scaler.fit_transform(X=X_train)
            X_test = scaler.transform(X=X_test)
            
            if type_feature == "pca":
                model = pca(n_components=0.95)
                X_pca = model.fit_transform(X_train)
                fig, ax = model.plot()
                fig.savefig('pca_ncomponents_'+str_dataframe+'_95_'+str_scaler+'.png')
                
                model = PCA(n_components = 0.95)
                X_train = model.fit_transform(X_train)
                X_test = model.transform(X_test)
                print("Feature con PCA:", X_train.shape)
                
            #Prova regressione logistica non modificando gli iperparametri e misura la learning curve
            logr = LogisticRegression()
            train_sizes = np.linspace(0.01, 1, 20)

            train_sizes, train_scores, valid_scores = learning_curve(estimator=logr, X=X_train, y=y_train,
                                                                    train_sizes=train_sizes, cv=5,
                                                                    scoring="accuracy", n_jobs=-1, shuffle=True, random_state=42)

            train_mean = train_scores.mean(axis=1)
            valid_mean = valid_scores.mean(axis=1)

            plt.style.use('seaborn')
            plt.figure(figsize=(8, 5))
            plt.plot(train_sizes, train_mean, label='Training error')
            plt.plot(train_sizes, valid_mean, label='Validation error')
            plt.ylabel('MSE', fontsize=14)
            plt.xlabel('Training set size', fontsize=14)
            plt.title('Learning curves for basic linear regression model',
                    fontsize=18, y=1.03)
            plt.ylim(0, 2)
            plt.legend()
            plt.show()

            #per alpha tra -4 e 100 calcola il validation curve al variare del parametro c
            alphas = np.logspace(start=-4, stop=3, num=30)

            train_scores, valid_scores = validation_curve(estimator=logr, X=X_train, y=y_train,
                                                        param_name="C", param_range=alphas, cv=5,
                                                        scoring="accuracy", n_jobs=-1)

            train_mean = train_scores.mean(axis=1)
            valid_mean = valid_scores.mean(axis=1)

            sns.lineplot(x=alphas, y=train_mean, marker="o", label="train accuray")
            sns.lineplot(x=alphas, y=valid_mean,
                        marker="o", label="valid accuracy")
            plt.xscale("log")
            plt.grid()
            plt.title("Validation curve logr for linear regression with changing of hypermater C (alphas)")
            plt.show()

            skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            #REGRESSIONE LOGISTICA CON BRIDGE E LASSO
            
            #Con RandomSearch misura parametri migliori per regressione logistica
            params = {
                "penalty": ["l2", "l1"],
                "C": stats.loguniform(1e0, 1e2),
                "class_weight": [None, "balanced"]
            }

            rs = RandomizedSearchCV(estimator=logr, param_distributions=params,
                                    scoring="accuracy", n_jobs=-1, cv=skfold, verbose=1)
            rs.fit(X_train, y_train)

            #utilizza i parametri trovati per creare un nuovo modello di reg log
            logr_best = LogisticRegression(**rs.best_params_)
            logr_best.fit(X_train, y_train)

            cv_scores[type_feature][str_dataframe][str_scaler]['logr'] = cross_val_score(estimator=logr_best, X=X_train, y=y_train,
                                            scoring="accuracy", cv=skfold, n_jobs=N_JOBS, verbose=1)
            print(f"Mean CV accuracy: {cv_scores[type_feature][str_dataframe][str_scaler]['logr'].mean():.4f} +/- {cv_scores[type_feature][str_dataframe][str_scaler]['logr'].std():.4f}")
            print(cv_scores[type_feature][str_dataframe][str_scaler]['logr'])

            plt.style.use('default')
            fig, ax = plt.subplots(figsize=(8, 5))
            plot_confusion_matrix(estimator=logr_best, X=X_test, y_true=y_test,
                                cmap='Blues', display_labels=class_names, ax=ax)
            plt.title("Logistic regression with best params")
            plt.tight_layout()
            plt.show()

            y_pred = logr_best.predict(X_test)
            ac_scores[type_feature][str_dataframe][str_scaler]['logr'] = logr_best.score(X_test, y_test)
            report = classification_report(y_true=y_test, y_pred=y_pred)
            print(report)

            #Misura il learning curve del nuovo modello
            train_sizes, train_scores, valid_scores = learning_curve(estimator=logr_best, X=X_train, y=y_train,
                                                                    train_sizes=train_sizes, cv=5,
                                                                    scoring="accuracy", n_jobs=-1, shuffle=True, random_state=42)

            train_mean = train_scores.mean(axis=1)
            valid_mean = valid_scores.mean(axis=1)

            plt.style.use('seaborn')
            plt.figure(figsize=(8, 5))
            plt.plot(train_sizes, train_mean, label='Training error')
            plt.plot(train_sizes, valid_mean, label='Validation error')
            plt.ylabel('MSE', fontsize=14)
            plt.xlabel('Training set size', fontsize=14)
            plt.title('Learning curves for linear regression with best parameter',
                    fontsize=18, y=1.03)
            plt.ylim(0, 2)
            plt.legend()
            plt.show()

            alphas = np.logspace(-5, 0, 100)
            coefs_ridge = []
            metrics_ridge = []
            coefs_lasso = []
            metrics_lasso = []

            #Rifa ridge e lasso ma questa volta direttamente su le feature, senza polynomial features
            for alpha in alphas:
                ridge = Ridge(alpha=alpha)
                ridge.fit(X_train, y_train)
                coefs_ridge.append(ridge.coef_)
                metrics_ridge.append({"lambda": alpha, "r2_train": ridge.score(X_train, y_train), "r2_test": ridge.score(X_test, y_test)})
                lasso = Lasso(alpha=alpha)
                lasso.fit(X_train, y_train)
                coefs_lasso.append(lasso.coef_)
                metrics_lasso.append({"lambda": alpha, "r2_train": lasso.score(X_train, y_train), "r2_test": lasso.score(X_test, y_test)})
                
            plt.plot(alphas, coefs_ridge)
            plt.xscale("log")
            plt.title("Ridge without Polynomial Features")
            plt.tight_layout()
            plt.grid()
            plt.show()

            plt.plot(alphas, coefs_lasso)
            plt.xscale("log")
            plt.title("Lasso without Polynomial Features")
            plt.tight_layout()
            plt.grid()
            plt.show()

            df_metrics_ridge = pd.DataFrame(metrics_ridge)
            df_metrics_ridge["difference"] = df_metrics_ridge["r2_train"] - df_metrics_ridge["r2_test"]

            df_metrics_lasso = pd.DataFrame(metrics_lasso)
            df_metrics_lasso["difference"] = df_metrics_lasso["r2_train"] - df_metrics_lasso["r2_test"]

            sns.lineplot(data=df_metrics_ridge, x="lambda", y="r2_train", label="train scores ridge")
            sns.lineplot(data=df_metrics_ridge, x="lambda", y="r2_test", label="test scores ridge")
            sns.lineplot(data=df_metrics_lasso, x="lambda", y="r2_train", label="train scores lasso")
            sns.lineplot(data=df_metrics_lasso, x="lambda", y="r2_test", label="test scores lasso")
            plt.xscale("log")
            plt.title("Score ridge/lasso without polynomial features")
            plt.grid()
            plt.tight_layout()
            plt.show()

            sns.lineplot(data=df_metrics_ridge, x="lambda", y="difference", label="difference scores ridge")
            sns.lineplot(data=df_metrics_lasso, x="lambda", y="difference", label="difference scores lasso")
            plt.xscale("log")
            plt.title("Difference between train and test score for ridge/lasso without polynomial features")
            plt.grid()
            plt.tight_layout()
            plt.show()
            
            #ALBERO DELLE DECISIONI
            
            dtc = DecisionTreeClassifier(random_state=RANDOM_STATE)
            dtc = dtc.fit(X=X_train, y=y_train)
            print(f"Profonfità albero: {dtc.get_depth()}")

            #Dizionario con importanza feature
            feature_importances = dtc.feature_importances_
            feature_index = X.columns
            myDict = dict(zip(feature_index, feature_importances))
            myDict = dict(sorted(myDict.items(), key=lambda item: item[1], reverse = False))

            #Plot delle 15 feature con più importanza
            series = pd.Series(data=myDict.values(), index=myDict.keys()).tail(10)
            series.plot(kind="barh", figsize=(8, 5), title=f"Feature importances for basic Decision Tree", legend=None)
            plt.tight_layout()
            plt.show()

            cv_scores[type_feature][str_dataframe][str_scaler]['dtc'] = cross_val_score(estimator=dtc, X=X_train, y=y_train, 
                                            scoring="accuracy", cv=skfold, n_jobs=N_JOBS, verbose=1)
            print(f"Mean CV accuracy: {cv_scores[type_feature][str_dataframe][str_scaler]['dtc'].mean():.4f} +/- {cv_scores[type_feature][str_dataframe][str_scaler]['dtc'].std():.4f}")
            print(cv_scores[type_feature][str_dataframe][str_scaler]['dtc'])

            fig, ax = plt.subplots(figsize=(8, 5))
            plot_confusion_matrix(estimator=dtc, X=X_test, y_true=y_test, cmap='Greens', display_labels=class_names, ax=ax)
            plt.title("Basic Decision Tree")
            plt.tight_layout()
            plt.show()

            feature_names = feature_index

            plt.figure(figsize=(15, 15))
            plot_tree(decision_tree=dtc, 
                    feature_names=feature_names, 
                    class_names=class_names, 
                    filled=True, fontsize=8)
            plt.title("Basic Decision Tree")
            plt.show()

            y_pred = dtc.predict(X=X_test)
            ac_scores[type_feature][str_dataframe][str_scaler]['dtc'] = accuracy_score(y_true=y_test, y_pred=y_pred)

            y_pred_train = dtc.predict(X=X_train)
            accuracy_train = accuracy_score(y_true=y_train, y_pred=y_pred_train)
            print(f"train accuracy score: {accuracy_train:.4f}")

            report = classification_report(y_true=y_test, y_pred=y_pred)
            print(report)

            path = dtc.cost_complexity_pruning_path(X=X_train, y=y_train)
            ccp_alphas, impurities = path.ccp_alphas, path.impurities

            dtcs = []
            #Tuning per iperparametro ccp(alphas)
            for ccp_alpha in ccp_alphas:
                dtc = DecisionTreeClassifier(random_state=RANDOM_STATE, ccp_alpha=ccp_alpha)
                dtc.fit(X=X_train, y=y_train)
                dtcs.append(dtc)
                
            train_scores = [dtc.score(X=X_train, y=y_train) for dtc in dtcs]
            test_scores = [dtc.score(X=X_test, y=y_test) for dtc in dtcs]

            alpha_scores_data = {"ccp": ccp_alphas.astype(str), "train": train_scores, "test": test_scores}

            df_alpha_scores = pd.DataFrame(data=alpha_scores_data)
            display(df_alpha_scores)

            plt.figure(figsize=(7, 5))
            sns.lineplot(x=ccp_alphas, y=train_scores, marker='o', label="train", drawstyle="steps-post")
            sns.lineplot(x=ccp_alphas, y=test_scores, marker='o', label="test", drawstyle="steps-post")
            plt.title("Accuracy vs alpha for training and testing sets of decision tree")
            plt.xlabel("alpha")
            plt.ylabel("accuracy")
            plt.grid(linestyle='-.', linewidth=0.5)
            plt.tight_layout()
            plt.show()

            dtc_bigccp = DecisionTreeClassifier(ccp_alpha=0.021465639876340106	, random_state=RANDOM_STATE)
            dtc_bigccp = dtc_bigccp.fit(X=X_train, y=y_train)
            print(f"Profonfità albero con ccp=0.0214: {dtc_bigccp.get_depth()}")

            #Dizionario con importanza feature
            feature_importances = dtc_bigccp.feature_importances_
            feature_index = X.columns
            myDict = dict(zip(feature_index, feature_importances))
            myDict = dict(sorted(myDict.items(), key=lambda item: item[1], reverse = False))

            #Plot delle 15 feature con più importanza
            series = pd.Series(data=myDict.values(), index=myDict.keys()).tail(10)
            series.plot(kind="barh", figsize=(8, 5), title=f"Feature importances con ccp=0.0214", legend=None)
            plt.tight_layout()
            plt.show()

            dtc_cv_scores_bigccp = cross_val_score(estimator=dtc_bigccp, X=X_train, y=y_train, 
                                            scoring="accuracy", cv=skfold, n_jobs=N_JOBS, verbose=1)
            print(f"Mean CV accuracy: {dtc_cv_scores_bigccp.mean():.4f} +/- {dtc_cv_scores_bigccp.std():.4f}")
            print(dtc_cv_scores_bigccp)

            fig, ax = plt.subplots(figsize=(8, 5))
            plot_confusion_matrix(estimator=dtc_bigccp, X=X_test, y_true=y_test, cmap='Greens', display_labels=class_names, ax=ax)
            plt.title("Decision Tree with ccp=0.0214")
            plt.tight_layout()
            plt.show()

            feature_names = feature_index

            plt.figure(figsize=(15, 15))
            plot_tree(decision_tree=dtc_bigccp, 
                    feature_names=feature_names, 
                    class_names=class_names, 
                    filled=True, fontsize=8)
            plt.title("Decision Tree with ccp=0.0214")
            plt.show()

            y_pred = dtc_bigccp.predict(X=X_test)
            accuracy_tree_bigccp = accuracy_score(y_true=y_test, y_pred=y_pred)

            y_pred_train = dtc_bigccp.predict(X=X_train)
            accuracy_tree_train_bigccp = accuracy_score(y_true=y_train, y_pred=y_pred_train)
            print(f"train accuracy score: {accuracy_tree_train_bigccp:.4f}")

            report = classification_report(y_true=y_test, y_pred=y_pred)
            print(report)
            #Tuning per iperparametro altezza
            dtcs = []
            depths = range(1, 10)
            for depth in depths:
                dtc_vardepth = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE)
                dtc_vardepth.fit(X=X_train, y=y_train)
                dtcs.append(dtc_vardepth)
                
            train_scores = [dtc_vardepth.score(X=X_train, y=y_train) for dtc_vardepth in dtcs]
            test_scores = [dtc_vardepth.score(X=X_test, y=y_test) for dtc_vardepth in dtcs]

            alpha_scores_data = {"depth": depths, "train": train_scores, "test": test_scores}
            df_alpha_scores = pd.DataFrame(data=alpha_scores_data)
            display(df_alpha_scores)

            X_dtc = maldi[['6889,619768','3354,28405']].values
            y_dtc = df[['target']].values
            X_train_dtc, X_test_dtc, y_train_dtc, y_test_dtc = train_test_split(
                X_dtc, y_dtc, test_size=0.2, random_state=42, stratify=y)

            dtc_small = DecisionTreeClassifier(max_depth=1)
            dtc_small.fit(X=X_train_dtc, y=y_train_dtc)
            print(f"Profonfità albero: {dtc_small.get_depth()}")

            dtc_cv_scores_small_2feat = cross_val_score(estimator=dtc_small, X=X_train_dtc, y=y_train_dtc, 
                                            scoring="accuracy", cv=skfold, n_jobs=N_JOBS, verbose=1)
            print(f"Mean CV accuracy: {dtc_cv_scores_small_2feat.mean():.4f} +/- {dtc_cv_scores_small_2feat.std():.4f}")
            print(dtc_cv_scores_small_2feat)

            dtc_big = DecisionTreeClassifier()
            dtc_big.fit(X=X_train_dtc, y=y_train_dtc)
            print(f"Profonfità albero: {dtc_big.get_depth()}")

            dtc_cv_scores_big_2feat = cross_val_score(estimator=dtc_big, X=X_train_dtc, y=y_train_dtc, 
                                                scoring="accuracy", cv=skfold, n_jobs=N_JOBS, verbose=1)
            print(f"Mean CV accuracy: {dtc_cv_scores_big_2feat.mean():.4f} +/- {dtc_cv_scores_big_2feat.std():.4f}")
            print(dtc_cv_scores_big_2feat)

            fig = plt.figure(figsize=(15, 5))
            ax = fig.subplots(nrows=1, ncols=2)
            db_small = DecisionBoundaryDisplay.from_estimator(estimator=dtc_small, X=X_dtc, grid_resolution=200, alpha=0.75, ax=ax[0])
            db_big = DecisionBoundaryDisplay.from_estimator(estimator=dtc_big, X=X_dtc, grid_resolution=200, alpha=0.75, ax=ax[1])
            db_small.ax_.scatter(X_dtc[:, 0], X_dtc[:, 1], c=y, edgecolor="k")
            db_big.ax_.scatter(X_dtc[:, 0], X_dtc[:, 1], c=y, edgecolor="k")
            ax[0].set_title("Decision tree with max_depth=1")
            ax[1].set_title(f"Decision tree with max depth={dtc_big.get_depth()}")
            plt.tight_layout()
            plt.show()
            
            #K-NEARERST-NEIGHBORS
            knn_scores = []

            for k in range(1, X_train.shape[0], 2):
                # model definition and training
                knn =  KNeighborsClassifier(n_neighbors=k)
                knn.fit(X=X_train, y=y_train)
                # compute accuracy on test set
                accuracy = knn.score(X_test, y_test)
                # store the results on a list of dictionaries
                metrics = {"# neighbors": k, "accuracy": accuracy}
                knn_scores.append(metrics)

            # convert the list of dictionaries to pandas dataframe
            df_knn_scores = pd.DataFrame(data=knn_scores)    #molto facile eveloce da dizionario a dataframe 
            display(df_knn_scores)

            mask = df_knn_scores["accuracy"] == df_knn_scores["accuracy"].max()
            knn_k = df_knn_scores['accuracy'].idxmax()
            n = df_knn_scores['# neighbors'][knn_k]

            plt.figure(figsize=(7, 5))
            plt.title("KNN accuracy as function of the number of neighbors")
            sns.lineplot(x="# neighbors", y="accuracy", data=df_knn_scores)
            plt.grid(linestyle='-.', linewidth=0.5)
            plt.tight_layout()
            plt.show()

            display(df_knn_scores[mask])

            knn = KNeighborsClassifier(n_neighbors=n)
            knn.fit(X=X_train, y=y_train)

            cv_scores[type_feature][str_dataframe][str_scaler]['knn'] = cross_val_score(estimator=knn, X=X_train, y=y_train, 
                                            scoring="accuracy", cv=skfold, n_jobs=N_JOBS, verbose=1)
            print(f"Mean CV KNN accuracy: {cv_scores[type_feature][str_dataframe][str_scaler]['knn'].mean():.4f} +/- {cv_scores[type_feature][str_dataframe][str_scaler]['knn'].std():.4f}")
            print(cv_scores[type_feature][str_dataframe][str_scaler]['knn'])

            fig, ax = plt.subplots(figsize=(8, 5))
            plot_confusion_matrix(estimator=knn, X=X_test, y_true=y_test, cmap='Oranges', display_labels=class_names, ax=ax)
            plt.title("Knn with best K")
            plt.tight_layout()
            plt.show()

            y_pred = knn.predict(X=X_test)
            ac_scores[type_feature][str_dataframe][str_scaler]['knn'] = accuracy_score(y_true=y_test, y_pred=y_pred)
            report = classification_report(y_true=y_test, y_pred=y_pred)
            print(report)
            
            #RANDOM FOREST
            
            rm = RandomForestClassifier(oob_score=True, n_jobs=N_JOBS, random_state=RANDOM_STATE)
            rm_cv_scores = cross_val_score(estimator=rm, X=X_train, y=y_train, 
                                            scoring="accuracy", cv=skfold, n_jobs=N_JOBS, verbose=1)
            print(f"Mean CV Random Forest accuracy: {rm_cv_scores.mean():.4f} +/- {rm_cv_scores.std():.4f}")
            print(rm_cv_scores)

            params = {
                "n_estimators": [25, 50, 100, 200, 250, 500],
                "criterion": ["gini", "entropy"],
                "max_depth": [None, 1, 2, 5, 10, 20],
                "max_features": ["sqrt", "log2"],
                "class_weight": [None, "balanced", "balanced_subsample"]
            }
            #
            random_search = RandomizedSearchCV(estimator=rm, param_distributions=params, n_iter=20, 
                                            scoring="accuracy", n_jobs=N_JOBS, cv=skfold, verbose=1, random_state=RANDOM_STATE)
            random_search.fit(X=X_train, y=y_train)
            print(random_search.best_params_)
            print('Random Forest Score: ', random_search.best_score_)

            #n_estimators': 200, 'max_features': 'sqrt', 'max_depth': None, 'criterion': 'entropy', 'class_weight': None}
            n_estimator = 200
            max_features = 'sqrt'
            max_depth = None
            criterion = 'entropy'
            class_weight = None
            best_rm = RandomForestClassifier(n_estimators=n_estimator, max_features=max_features, max_depth=max_depth, class_weight=class_weight, criterion=criterion, oob_score=True, n_jobs=N_JOBS, random_state=RANDOM_STATE)
            best_rm.fit(X=X_train, y=y_train)
            cv_scores[type_feature][str_dataframe][str_scaler]['rm'] = cross_val_score(estimator=best_rm, X=X_train, y=y_train, 
                                            scoring="accuracy", cv=skfold, n_jobs=N_JOBS, verbose=1)
            print(f"Mean CV Best Random Forest accuracy: {cv_scores[type_feature][str_dataframe][str_scaler]['rm'].mean():.4f} +/- {cv_scores[type_feature][str_dataframe][str_scaler]['rm'].std():.4f}")
            print(cv_scores[type_feature][str_dataframe][str_scaler]['rm'])

            plt.figure(figsize=(6, 4))
            plt.title("CV Random forest model scores")
            plt.plot(rm_cv_scores, marker="o")
            plt.plot(best_rm_cv_scores, marker="o")
            plt.xticks(ticks=range(1, 11))
            plt.xlabel("fold")
            plt.ylabel("accuracy")
            plt.legend(["Base model", "Best model"])
            plt.tight_layout()
            plt.grid()
            plt.show()

            feature_importances_plot(model=best_rm, labels=X.columns)
            print("Score: ", best_rm.oob_score_)

            fig, ax = plt.subplots(figsize=(8, 5))
            plot_confusion_matrix(estimator=best_rm, X=X_test, y_true=y_test, cmap='Purples', display_labels=class_names, ax=ax)
            plt.title("Random forest with best parameter")
            plt.tight_layout()
            plt.show()

            y_pred = best_rm.predict(X_test)
            ac_scores[type_feature][str_dataframe][str_scaler]['rm'] = accuracy_score(y_true=y_test, y_pred=y_pred)

            report = classification_report(y_true=y_test, y_pred=y_pred)
            print(report)

            r2_rf = r2_score(y_test, y_pred)
            print('R2 score of random forest classifier on test set: {:.3f}'.format(r2_rf))

            rmse_rf = mean_squared_error(y_test, y_pred, squared=False)
            print('Mean squared error of random forest classifier on test set: {:.3f}'.format(rmse_rf))
            
