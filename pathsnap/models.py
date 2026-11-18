"""Small model registry: add independently motivated model variants here."""

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

MODEL_SPECS = {
    "logistic": {"family": "Linear", "parameters": {"C": 1.0, "max_iter": 800}},
    "gaussian_nb": {"family": "Probabilistic", "parameters": {"var_smoothing": 1e-7}},
    "decision_tree": {"family": "Tree", "parameters": {"max_depth": 10, "min_samples_leaf": 20}},
    "random_forest": {"family": "Bagged ensemble", "parameters": {"n_estimators": 80, "max_depth": 14, "min_samples_leaf": 8}},
    "hist_gradient_boosting": {"family": "Boosting", "parameters": {"max_iter": 120, "max_leaf_nodes": 15,
                                                                                "learning_rate": 0.1, "l2_regularization": 1.0,
                                                                                "min_samples_leaf": 30}},
}


def make_model(name: str, seed: int, parameters: dict | None = None):
    p = dict(MODEL_SPECS[name]["parameters"])
    p.update(parameters or {})
    if name == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(random_state=seed, **p))
    if name == "gaussian_nb":
        return make_pipeline(StandardScaler(), GaussianNB(**p))
    if name == "decision_tree":
        return DecisionTreeClassifier(random_state=seed, **p)
    if name == "random_forest":
        return RandomForestClassifier(random_state=seed, n_jobs=4, **p)
    if name == "hist_gradient_boosting":
        # sklearn's automatic early stopping randomly divides candidate rows;
        # disabling it avoids internal trip leakage and keeps fold comparisons identical.
        return HistGradientBoostingClassifier(random_state=seed, early_stopping=False, **p)
    raise ValueError(f"Unknown model: {name}")


def search_parameters(trial, name: str) -> dict:
    if name == "logistic":
        return {"C": trial.suggest_float("C", 0.01, 10.0, log=True)}
    if name == "gaussian_nb":
        return {"var_smoothing": trial.suggest_float("var_smoothing", 1e-10, 1e-2, log=True)}
    if name == "decision_tree":
        return {"max_depth": trial.suggest_int("max_depth", 4, 16),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 80)}
    if name == "random_forest":
        return {"max_depth": trial.suggest_int("max_depth", 6, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 3, 40),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", 0.7, 1.0])}
    if name == "hist_gradient_boosting":
        return {"learning_rate": trial.suggest_float("learning_rate", 0.025, 0.2, log=True),
                "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 7, 31),
                "l2_regularization": trial.suggest_float("l2_regularization", 0.01, 10.0, log=True),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 15, 100)}
    raise ValueError(name)
