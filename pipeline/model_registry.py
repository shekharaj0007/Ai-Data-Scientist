"""
Model registry — broad catalog of classical ML models trainable in this pipeline.
Optional packages (LightGBM, CatBoost) are included when installed.
"""
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.naive_bayes import ComplementNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC, LinearSVR, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier, XGBRegressor

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:
    LGBMClassifier = LGBMRegressor = None

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except ImportError:
    CatBoostClassifier = CatBoostRegressor = None


def get_classification_models() -> dict:
    models = {
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=42),
        "ridge_classifier": LogisticRegression(penalty="l2", C=0.5, max_iter=2000, random_state=42),
        "linear_svc": LinearSVC(max_iter=3000, random_state=42),
        "svm_rbf": SVC(kernel="rbf", probability=False, random_state=42),
        "knn": KNeighborsClassifier(n_neighbors=7),
        "gaussian_nb": GaussianNB(),
        "decision_tree": DecisionTreeClassifier(max_depth=12, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "extra_trees": ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
        "ada_boost": AdaBoostClassifier(random_state=42),
        "xgboost": XGBClassifier(eval_metric="logloss", random_state=42, verbosity=0),
        "mlp_ann": Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400, random_state=42)),
        ]),
    }
    if LGBMClassifier:
        models["lightgbm"] = LGBMClassifier(random_state=42, verbosity=-1, n_estimators=200)
    if CatBoostClassifier:
        models["catboost"] = CatBoostClassifier(random_state=42, verbose=0, iterations=200)
    return models


def get_regression_models() -> dict:
    models = {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(random_state=42),
        "lasso": Lasso(random_state=42, max_iter=5000),
        "elastic_net": ElasticNet(random_state=42, max_iter=5000),
        "linear_svr": LinearSVR(max_iter=5000),
        "svr_rbf": SVR(kernel="rbf"),
        "knn": KNeighborsRegressor(n_neighbors=7),
        "decision_tree": DecisionTreeRegressor(max_depth=12, random_state=42),
        "random_forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "extra_trees": ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "gradient_boosting": GradientBoostingRegressor(random_state=42),
        "ada_boost": AdaBoostRegressor(random_state=42),
        "xgboost": XGBRegressor(random_state=42, verbosity=0),
        "mlp_ann": Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=400, random_state=42)),
        ]),
    }
    if LGBMRegressor:
        models["lightgbm"] = LGBMRegressor(random_state=42, verbosity=-1, n_estimators=200)
    if CatBoostRegressor:
        models["catboost"] = CatBoostRegressor(random_state=42, verbose=0, iterations=200)
    return models


def get_text_classification_models() -> dict:
    return {
        "tfidf_logistic": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=8000, ngram_range=(1, 2), min_df=2)),
            ("clf", LogisticRegression(max_iter=2000, random_state=42)),
        ]),
        "tfidf_linear_svc": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=8000, ngram_range=(1, 2), min_df=2)),
            ("clf", LinearSVC(max_iter=3000, random_state=42)),
        ]),
        "tfidf_complement_nb": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=8000, ngram_range=(1, 2), min_df=2)),
            ("clf", ComplementNB()),
        ]),
        "tfidf_random_forest": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)),
            ("clf", RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)),
        ]),
    }


def get_clustering_models() -> dict:
    from sklearn.cluster import AgglomerativeClustering, Birch, DBSCAN
    from sklearn.mixture import GaussianMixture

    return {
        "agglomerative": AgglomerativeClustering(n_clusters=3),
        "gaussian_mixture": GaussianMixture(n_components=3, random_state=42),
        "birch": Birch(n_clusters=3),
        "dbscan": DBSCAN(eps=0.8, min_samples=5),
    }
