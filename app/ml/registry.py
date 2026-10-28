from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ModelSpec:
    id: str
    name: str
    task_types: List[str]  # e.g. ["binary_classification", "multiclass_classification"]
    category: str  # "Classification", "Regression", "Clustering", "Anomaly Detection"
    family: str  # "linear", "tree", "boosting", "kernel", "probabilistic", "neural_net", "discriminant", etc.
    factory: Callable[[int], Any]  # Takes random_state -> instantiated estimator
    requires_scaling: bool = False
    max_recommended_rows: int = 100_000
    supports_sparse: bool = True
    check_package: Optional[str] = None
    default_enabled: bool = True
    description: str = ""


# =====================================================================
# MODEL FACTORIES (Lazy imports with graceful fallbacks)
# =====================================================================

def _is_package_installed(pkg_name: str) -> bool:
    try:
        return importlib.util.find_spec(pkg_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


# --- Classification Factories ---
def _make_logistic_regression(rs: int):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=1000, random_state=rs)


def _make_sgd_classifier(rs: int):
    from sklearn.linear_model import SGDClassifier
    return SGDClassifier(loss="log_loss", max_iter=1000, random_state=rs)


def _make_perceptron(rs: int):
    from sklearn.linear_model import Perceptron
    return Perceptron(max_iter=1000, random_state=rs)


def _make_decision_tree_classifier(rs: int):
    from sklearn.tree import DecisionTreeClassifier
    return DecisionTreeClassifier(max_depth=12, random_state=rs)


def _make_random_forest_classifier(rs: int):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=100, max_depth=15, random_state=rs, n_jobs=-1)


def _make_extra_trees_classifier(rs: int):
    from sklearn.ensemble import ExtraTreesClassifier
    return ExtraTreesClassifier(n_estimators=100, max_depth=15, random_state=rs, n_jobs=-1)


def _make_gradient_boosting_classifier(rs: int):
    from sklearn.ensemble import GradientBoostingClassifier
    return GradientBoostingClassifier(n_estimators=100, random_state=rs)


def _make_hist_gradient_boosting_classifier(rs: int):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=100, random_state=rs)


def _make_adaboost_classifier(rs: int):
    from sklearn.ensemble import AdaBoostClassifier
    return AdaBoostClassifier(n_estimators=50, random_state=rs)


def _make_svc(rs: int):
    from sklearn.svm import SVC
    return SVC(kernel="rbf", probability=True, random_state=rs)


def _make_linear_svc(rs: int):
    from sklearn.svm import LinearSVC
    return LinearSVC(max_iter=2000, random_state=rs)


def _make_knn_classifier(rs: int):
    from sklearn.neighbors import KNeighborsClassifier
    return KNeighborsClassifier(n_neighbors=5, n_jobs=-1)


def _make_gaussian_nb(rs: int):
    from sklearn.naive_bayes import GaussianNB
    return GaussianNB()


def _make_bernoulli_nb(rs: int):
    from sklearn.naive_bayes import BernoulliNB
    return BernoulliNB()


def _make_lda(rs: int):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    return LinearDiscriminantAnalysis()


def _make_qda(rs: int):
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    return QuadraticDiscriminantAnalysis()


def _make_mlp_classifier(rs: int):
    from sklearn.neural_network import MLPClassifier
    return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400, random_state=rs)


def _make_torch_mlp_classifier(rs: int):
    from app.ml.neural_estimators import TorchMLPClassifier
    return TorchMLPClassifier(random_state=rs)


def _make_tensorflow_mlp_classifier(rs: int):
    from app.ml.neural_estimators import TensorFlowMLPClassifier
    return TensorFlowMLPClassifier(random_state=rs)


# Modern Boosting (Classification)
def _make_xgboost_classifier(rs: int):
    import xgboost as xgb
    return xgb.XGBClassifier(n_estimators=100, random_state=rs, eval_metric="logloss", verbosity=0)


def _make_lightgbm_classifier(rs: int):
    import lightgbm as lgb
    return lgb.LGBMClassifier(n_estimators=100, random_state=rs, verbose=-1)


def _make_catboost_classifier(rs: int):
    import catboost as cb
    return cb.CatBoostClassifier(iterations=100, random_seed=rs, verbose=0)


# --- Regression Factories ---
def _make_linear_regression(rs: int):
    from sklearn.linear_model import LinearRegression
    return LinearRegression()


def _make_ridge_regression(rs: int):
    from sklearn.linear_model import Ridge
    return Ridge(random_state=rs)


def _make_lasso_regression(rs: int):
    from sklearn.linear_model import Lasso
    return Lasso(random_state=rs)


def _make_elasticnet_regression(rs: int):
    from sklearn.linear_model import ElasticNet
    return ElasticNet(random_state=rs)


def _make_sgd_regressor(rs: int):
    from sklearn.linear_model import SGDRegressor
    return SGDRegressor(max_iter=1000, random_state=rs)


def _make_huber_regressor(rs: int):
    from sklearn.linear_model import HuberRegressor
    return HuberRegressor(max_iter=500)


def _make_ransac_regressor(rs: int):
    from sklearn.linear_model import RANSACRegressor
    return RANSACRegressor(random_state=rs)


def _make_decision_tree_regressor(rs: int):
    from sklearn.tree import DecisionTreeRegressor
    return DecisionTreeRegressor(max_depth=12, random_state=rs)


def _make_random_forest_regressor(rs: int):
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=100, max_depth=15, random_state=rs, n_jobs=-1)


def _make_extra_trees_regressor(rs: int):
    from sklearn.ensemble import ExtraTreesRegressor
    return ExtraTreesRegressor(n_estimators=100, max_depth=15, random_state=rs, n_jobs=-1)


def _make_gradient_boosting_regressor(rs: int):
    from sklearn.ensemble import GradientBoostingRegressor
    return GradientBoostingRegressor(n_estimators=100, random_state=rs)


def _make_hist_gradient_boosting_regressor(rs: int):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=100, random_state=rs)


def _make_adaboost_regressor(rs: int):
    from sklearn.ensemble import AdaBoostRegressor
    return AdaBoostRegressor(n_estimators=50, random_state=rs)


def _make_svr(rs: int):
    from sklearn.svm import SVR
    return SVR(kernel="rbf")


def _make_knn_regressor(rs: int):
    from sklearn.neighbors import KNeighborsRegressor
    return KNeighborsRegressor(n_neighbors=5, n_jobs=-1)


def _make_mlp_regressor(rs: int):
    from sklearn.neural_network import MLPRegressor
    return MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=400, random_state=rs)


def _make_torch_mlp_regressor(rs: int):
    from app.ml.neural_estimators import TorchMLPRegressor
    return TorchMLPRegressor(random_state=rs)


def _make_tensorflow_mlp_regressor(rs: int):
    from app.ml.neural_estimators import TensorFlowMLPRegressor
    return TensorFlowMLPRegressor(random_state=rs)


# Modern Boosting (Regression)
def _make_xgboost_regressor(rs: int):
    import xgboost as xgb
    return xgb.XGBRegressor(n_estimators=100, random_state=rs, verbosity=0)


def _make_lightgbm_regressor(rs: int):
    import lightgbm as lgb
    return lgb.LGBMRegressor(n_estimators=100, random_state=rs, verbose=-1)


def _make_catboost_regressor(rs: int):
    import catboost as cb
    return cb.CatBoostRegressor(iterations=100, random_seed=rs, verbose=0)


# --- Clustering Factories ---
def _make_kmeans(rs: int):
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=3, random_state=rs, n_init="auto")


def _make_minibatch_kmeans(rs: int):
    from sklearn.cluster import MiniBatchKMeans
    return MiniBatchKMeans(n_clusters=3, random_state=rs, n_init="auto")


def _make_agglomerative(rs: int):
    from sklearn.cluster import AgglomerativeClustering
    return AgglomerativeClustering(n_clusters=3)


def _make_dbscan(rs: int):
    from sklearn.cluster import DBSCAN
    return DBSCAN(eps=0.5, min_samples=5)


def _make_gmm(rs: int):
    from sklearn.mixture import GaussianMixture
    return GaussianMixture(n_components=3, random_state=rs)


def _make_birch(rs: int):
    from sklearn.cluster import Birch
    return Birch(n_clusters=3)


def _make_spectral(rs: int):
    from sklearn.cluster import SpectralClustering
    return SpectralClustering(n_clusters=3, random_state=rs, assign_labels="discretize")


def _make_meanshift(rs: int):
    from sklearn.cluster import MeanShift
    return MeanShift()


# --- Anomaly Detection Factories ---
def _make_isolation_forest(rs: int):
    from sklearn.ensemble import IsolationForest
    return IsolationForest(random_state=rs, n_estimators=100, n_jobs=-1)


def _make_local_outlier_factor(rs: int):
    from sklearn.neighbors import LocalOutlierFactor
    # novelty=True allows predict() on test samples
    return LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=-1)


def _make_one_class_svm(rs: int):
    from sklearn.svm import OneClassSVM
    return OneClassSVM(gamma="scale")


def _make_elliptic_envelope(rs: int):
    from sklearn.covariance import EllipticEnvelope
    return EllipticEnvelope(random_state=rs)


# =====================================================================
# MASTER MODEL REGISTRY
# =====================================================================

MODEL_REGISTRY: Dict[str, ModelSpec] = {
    # ----------------- CLASSIFICATION -----------------
    "logistic_regression": ModelSpec(
        id="logistic_regression",
        name="Logistic Regression",
        task_types=["binary_classification", "multiclass_classification"],
        category="Classification",
        family="linear",
        factory=_make_logistic_regression,
        requires_scaling=True,
        description="Linear model for classification with clear interpretability.",
    ),
    "sgd_classifier": ModelSpec(
        id="sgd_classifier",
        name="SGD Classifier",
        task_types=["binary_classification", "multiclass_classification"],
        category="Classification",
        family="linear",
        factory=_make_sgd_classifier,
        requires_scaling=True,
        description="Linear classifier optimized via Stochastic Gradient Descent.",
    ),
    "decision_tree_classifier": ModelSpec(
        id="decision_tree_classifier",
        name="Decision Tree Classifier",
        task_types=["binary_classification", "multiclass_classification"],
        category="Classification",
        family="tree",
        factory=_make_decision_tree_classifier,
        requires_scaling=False,
        description="Non-linear tree-based rule classifier.",
    ),
    "random_forest_classifier": ModelSpec(
        id="random_forest_classifier",
        name="Random Forest Classifier",
        task_types=["binary_classification", "multiclass_classification"],
        category="Classification",
        family="tree",
        factory=_make_random_forest_classifier,
        requires_scaling=False,
        description="Ensemble of decision trees with bagging.",
    ),
    "extra_trees_classifier": ModelSpec(
        id="extra_trees_classifier",
        name="Extra Trees Classifier",
        task_types=["binary_classification", "multiclass_classification"],
        category="Classification",
        family="tree",
        factory=_make_extra_trees_classifier,
        requires_scaling=False,
        description="Extremely randomized trees ensemble.",
    ),
    "gradient_boosting_classifier": ModelSpec(
        id="gradient_boosting_classifier",
        name="Gradient Boosting Classifier",
        task_types=["binary_classification", "multiclass_classification"],
        category="Classification",
        family="boosting",
        factory=_make_gradient_boosting_classifier,
        requires_scaling=False,
        description="Iterative gradient boosting trees.",
    ),
    "hist_gradient_boosting_classifier": ModelSpec(
        id="hist_gradient_boosting_classifier",
        name="HistGradientBoosting Classifier",
        task_types=["binary_classification", "multiclass_classification"],
        category="Classification",
        family="boosting",
        factory=_make_hist_gradient_boosting_classifier,
        requires_scaling=False,
        description="Histogram-based gradient boosting optimized for speed.",
    ),
    "adaboost_classifier": ModelSpec(
        id="adaboost_classifier",
        name="AdaBoost Classifier",
        task_types=["binary_classification", "multiclass_classification"],
        category="Classification",
        family="boosting",
        factory=_make_adaboost_classifier,
        requires_scaling=False,
        description="Adaptive boosting ensemble focusing on hard samples.",
    ),
    "svm_classifier": ModelSpec(
        id="svm_classifier",
        name="Support Vector Classifier (SVC)",
        task_types=["binary_classification", "multiclass_classification"],
        category="Classification",
        family="kernel",
        factory=_make_svc,
        requires_scaling=True,
        max_recommended_rows=10_000,
        description="Kernel support vector machine for non-linear boundaries.",
    ),
    "knn_classifier": ModelSpec(
        id="knn_classifier",
        name="K-Nearest Neighbors Classifier",
        task_types=["binary_classification", "multiclass_classification"],
        category="Classification",
        family="distance",
        factory=_make_knn_classifier,
        requires_scaling=True,
        max_recommended_rows=25_000,
        description="Instance-based nearest neighbor classifier.",
    ),
    "gaussian_nb": ModelSpec(
        id="gaussian_nb",
        name="Gaussian Naive Bayes",
        task_types=["binary_classification", "multiclass_classification"],
        category="Classification",
        family="probabilistic",
        factory=_make_gaussian_nb,
        requires_scaling=False,
        description="Probabilistic classifier assuming Gaussian feature distributions.",
    ),
    "linear_discriminant_analysis": ModelSpec(
        id="linear_discriminant_analysis",
        name="Linear Discriminant Analysis (LDA)",
        task_types=["binary_classification", "multiclass_classification"],
        category="Classification",
        family="discriminant",
        factory=_make_lda,
        requires_scaling=True,
        description="Linear decision boundary maximizing class separability.",
    ),
    "mlp_classifier": ModelSpec(
        id="mlp_classifier",
        name="Multi-Layer Perceptron (MLP)",
        task_types=["binary_classification", "multiclass_classification"],
        category="Classification",
        family="neural_net",
        factory=_make_mlp_classifier,
        requires_scaling=True,
        max_recommended_rows=30_000,
        description="Feedforward neural network for tabular data.",
    ),
    "pytorch_mlp_classifier": ModelSpec(
        id="pytorch_mlp_classifier",
        name="PyTorch MLP Classifier (CPU)",
        task_types=["binary_classification", "multiclass_classification"],
        category="Classification",
        family="neural_net",
        factory=_make_torch_mlp_classifier,
        requires_scaling=True,
        max_recommended_rows=20_000,
        supports_sparse=False,
        check_package="torch",
        default_enabled=False,
        description="Optional feedforward classifier powered by PyTorch.",
    ),
    "tensorflow_mlp_classifier": ModelSpec(
        id="tensorflow_mlp_classifier",
        name="TensorFlow MLP Classifier (CPU)",
        task_types=["binary_classification", "multiclass_classification"],
        category="Classification",
        family="neural_net",
        factory=_make_tensorflow_mlp_classifier,
        requires_scaling=True,
        max_recommended_rows=20_000,
        supports_sparse=False,
        check_package="tensorflow",
        default_enabled=False,
        description="Optional feedforward classifier powered by TensorFlow.",
    ),
    # Modern Boosting (Classification)
    "xgboost_classifier": ModelSpec(
        id="xgboost_classifier",
        name="XGBoost Classifier",
        task_types=["binary_classification", "multiclass_classification"],
        category="Classification",
        family="boosting",
        factory=_make_xgboost_classifier,
        check_package="xgboost",
        description="Extreme Gradient Boosting library.",
    ),
    "lightgbm_classifier": ModelSpec(
        id="lightgbm_classifier",
        name="LightGBM Classifier",
        task_types=["binary_classification", "multiclass_classification"],
        category="Classification",
        family="boosting",
        factory=_make_lightgbm_classifier,
        check_package="lightgbm",
        description="Fast, distributed gradient boosting framework.",
    ),
    "catboost_classifier": ModelSpec(
        id="catboost_classifier",
        name="CatBoost Classifier",
        task_types=["binary_classification", "multiclass_classification"],
        category="Classification",
        family="boosting",
        factory=_make_catboost_classifier,
        check_package="catboost",
        description="Gradient boosting with high-performance categorical handling.",
    ),

    # ----------------- REGRESSION -----------------
    "linear_regression": ModelSpec(
        id="linear_regression",
        name="Linear Regression",
        task_types=["regression"],
        category="Regression",
        family="linear",
        factory=_make_linear_regression,
        requires_scaling=True,
        description="Ordinary least squares linear regression.",
    ),
    "ridge_regression": ModelSpec(
        id="ridge_regression",
        name="Ridge Regression",
        task_types=["regression"],
        category="Regression",
        family="linear",
        factory=_make_ridge_regression,
        requires_scaling=True,
        description="L2 regularized linear regression.",
    ),
    "lasso_regression": ModelSpec(
        id="lasso_regression",
        name="Lasso Regression",
        task_types=["regression"],
        category="Regression",
        family="linear",
        factory=_make_lasso_regression,
        requires_scaling=True,
        description="L1 regularized sparse linear regression.",
    ),
    "elasticnet_regression": ModelSpec(
        id="elasticnet_regression",
        name="ElasticNet Regression",
        task_types=["regression"],
        category="Regression",
        family="linear",
        factory=_make_elasticnet_regression,
        requires_scaling=True,
        description="Combined L1 and L2 regularized regression.",
    ),
    "sgd_regressor": ModelSpec(
        id="sgd_regressor",
        name="SGD Regressor",
        task_types=["regression"],
        category="Regression",
        family="linear",
        factory=_make_sgd_regressor,
        requires_scaling=True,
        description="Linear model fitted by SGD.",
    ),
    "huber_regressor": ModelSpec(
        id="huber_regressor",
        name="Huber Regressor",
        task_types=["regression"],
        category="Regression",
        family="robust",
        factory=_make_huber_regressor,
        requires_scaling=True,
        description="Linear regression robust to outliers.",
    ),
    "decision_tree_regressor": ModelSpec(
        id="decision_tree_regressor",
        name="Decision Tree Regressor",
        task_types=["regression"],
        category="Regression",
        family="tree",
        factory=_make_decision_tree_regressor,
        requires_scaling=False,
        description="Non-linear tree-based regression.",
    ),
    "random_forest_regressor": ModelSpec(
        id="random_forest_regressor",
        name="Random Forest Regressor",
        task_types=["regression"],
        category="Regression",
        family="tree",
        factory=_make_random_forest_regressor,
        requires_scaling=False,
        description="Ensemble of regression trees with bagging.",
    ),
    "extra_trees_regressor": ModelSpec(
        id="extra_trees_regressor",
        name="Extra Trees Regressor",
        task_types=["regression"],
        category="Regression",
        family="tree",
        factory=_make_extra_trees_regressor,
        requires_scaling=False,
        description="Extremely randomized regression trees.",
    ),
    "gradient_boosting_regressor": ModelSpec(
        id="gradient_boosting_regressor",
        name="Gradient Boosting Regressor",
        task_types=["regression"],
        category="Regression",
        family="boosting",
        factory=_make_gradient_boosting_regressor,
        requires_scaling=False,
        description="Iterative boosting regressor.",
    ),
    "hist_gradient_boosting_regressor": ModelSpec(
        id="hist_gradient_boosting_regressor",
        name="HistGradientBoosting Regressor",
        task_types=["regression"],
        category="Regression",
        family="boosting",
        factory=_make_hist_gradient_boosting_regressor,
        requires_scaling=False,
        description="Histogram-based gradient boosting regressor.",
    ),
    "adaboost_regressor": ModelSpec(
        id="adaboost_regressor",
        name="AdaBoost Regressor",
        task_types=["regression"],
        category="Regression",
        family="boosting",
        factory=_make_adaboost_regressor,
        requires_scaling=False,
        description="Adaptive boosting regressor.",
    ),
    "svr": ModelSpec(
        id="svr",
        name="Support Vector Regressor (SVR)",
        task_types=["regression"],
        category="Regression",
        family="kernel",
        factory=_make_svr,
        requires_scaling=True,
        max_recommended_rows=10_000,
        description="Support vector regression with RBF kernel.",
    ),
    "knn_regressor": ModelSpec(
        id="knn_regressor",
        name="K-Nearest Neighbors Regressor",
        task_types=["regression"],
        category="Regression",
        family="distance",
        factory=_make_knn_regressor,
        requires_scaling=True,
        max_recommended_rows=25_000,
        description="Instance-based nearest neighbor regression.",
    ),
    "mlp_regressor": ModelSpec(
        id="mlp_regressor",
        name="Multi-Layer Perceptron Regressor",
        task_types=["regression"],
        category="Regression",
        family="neural_net",
        factory=_make_mlp_regressor,
        requires_scaling=True,
        max_recommended_rows=30_000,
        description="Neural network regressor.",
    ),
    "pytorch_mlp_regressor": ModelSpec(
        id="pytorch_mlp_regressor",
        name="PyTorch MLP Regressor (CPU)",
        task_types=["regression"],
        category="Regression",
        family="neural_net",
        factory=_make_torch_mlp_regressor,
        requires_scaling=True,
        max_recommended_rows=20_000,
        supports_sparse=False,
        check_package="torch",
        default_enabled=False,
        description="Optional feedforward regressor powered by PyTorch.",
    ),
    "tensorflow_mlp_regressor": ModelSpec(
        id="tensorflow_mlp_regressor",
        name="TensorFlow MLP Regressor (CPU)",
        task_types=["regression"],
        category="Regression",
        family="neural_net",
        factory=_make_tensorflow_mlp_regressor,
        requires_scaling=True,
        max_recommended_rows=20_000,
        supports_sparse=False,
        check_package="tensorflow",
        default_enabled=False,
        description="Optional feedforward regressor powered by TensorFlow.",
    ),
    # Modern Boosting (Regression)
    "xgboost_regressor": ModelSpec(
        id="xgboost_regressor",
        name="XGBoost Regressor",
        task_types=["regression"],
        category="Regression",
        family="boosting",
        factory=_make_xgboost_regressor,
        check_package="xgboost",
        description="Extreme Gradient Boosting regressor.",
    ),
    "lightgbm_regressor": ModelSpec(
        id="lightgbm_regressor",
        name="LightGBM Regressor",
        task_types=["regression"],
        category="Regression",
        family="boosting",
        factory=_make_lightgbm_regressor,
        check_package="lightgbm",
        description="LightGBM gradient boosting regressor.",
    ),
    "catboost_regressor": ModelSpec(
        id="catboost_regressor",
        name="CatBoost Regressor",
        task_types=["regression"],
        category="Regression",
        family="boosting",
        factory=_make_catboost_regressor,
        check_package="catboost",
        description="CatBoost regressor.",
    ),

    # ----------------- CLUSTERING -----------------
    "kmeans": ModelSpec(
        id="kmeans",
        name="K-Means",
        task_types=["clustering"],
        category="Clustering",
        family="centroid",
        factory=_make_kmeans,
        requires_scaling=True,
        description="Partition data into k centroid-based clusters.",
    ),
    "minibatch_kmeans": ModelSpec(
        id="minibatch_kmeans",
        name="Mini-Batch K-Means",
        task_types=["clustering"],
        category="Clustering",
        family="centroid",
        factory=_make_minibatch_kmeans,
        requires_scaling=True,
        description="Scalable batch-based K-Means for larger datasets.",
    ),
    "agglomerative_clustering": ModelSpec(
        id="agglomerative_clustering",
        name="Agglomerative Clustering",
        task_types=["clustering"],
        category="Clustering",
        family="hierarchical",
        factory=_make_agglomerative,
        requires_scaling=True,
        max_recommended_rows=8_000,
        description="Hierarchical bottom-up clustering.",
    ),
    "dbscan": ModelSpec(
        id="dbscan",
        name="DBSCAN",
        task_types=["clustering"],
        category="Clustering",
        family="density",
        factory=_make_dbscan,
        requires_scaling=True,
        max_recommended_rows=15_000,
        description="Density-based clustering finding non-linear shapes and noise.",
    ),
    "gaussian_mixture": ModelSpec(
        id="gaussian_mixture",
        name="Gaussian Mixture Model",
        task_types=["clustering"],
        category="Clustering",
        family="probabilistic",
        factory=_make_gmm,
        requires_scaling=True,
        description="Probabilistic soft-clustering with Gaussian distributions.",
    ),
    "birch": ModelSpec(
        id="birch",
        name="BIRCH",
        task_types=["clustering"],
        category="Clustering",
        family="hierarchical",
        factory=_make_birch,
        requires_scaling=True,
        description="Memory-efficient tree-structured clustering.",
    ),
    "spectral_clustering": ModelSpec(
        id="spectral_clustering",
        name="Spectral Clustering",
        task_types=["clustering"],
        category="Clustering",
        family="graph",
        factory=_make_spectral,
        requires_scaling=True,
        max_recommended_rows=3_000,
        description="Graph-based clustering for complex manifolds.",
    ),

    # ----------------- ANOMALY DETECTION -----------------
    "isolation_forest": ModelSpec(
        id="isolation_forest",
        name="Isolation Forest",
        task_types=["anomaly_detection"],
        category="Anomaly Detection",
        family="tree",
        factory=_make_isolation_forest,
        requires_scaling=False,
        description="Tree-based outlier isolation for tabular data.",
    ),
    "local_outlier_factor": ModelSpec(
        id="local_outlier_factor",
        name="Local Outlier Factor (LOF)",
        task_types=["anomaly_detection"],
        category="Anomaly Detection",
        family="density",
        factory=_make_local_outlier_factor,
        requires_scaling=True,
        max_recommended_rows=10_000,
        description="Density-based anomaly detection using local densities.",
    ),
    "one_class_svm": ModelSpec(
        id="one_class_svm",
        name="One-Class SVM",
        task_types=["anomaly_detection"],
        category="Anomaly Detection",
        family="kernel",
        factory=_make_one_class_svm,
        requires_scaling=True,
        max_recommended_rows=8_000,
        description="Boundary-based novelty and outlier detection.",
    ),
    "elliptic_envelope": ModelSpec(
        id="elliptic_envelope",
        name="Elliptic Envelope",
        task_types=["anomaly_detection"],
        category="Anomaly Detection",
        family="covariance",
        factory=_make_elliptic_envelope,
        requires_scaling=True,
        description="Gaussian covariance estimation for outlier detection.",
    ),
}


class ModelRegistry:
    """Central registry and factory for empirical ML models."""

    @staticmethod
    def get_spec(method_id: str) -> Optional[ModelSpec]:
        return MODEL_REGISTRY.get(method_id)

    @staticmethod
    def is_available(spec: ModelSpec) -> tuple[bool, str]:
        if spec.check_package:
            if not _is_package_installed(spec.check_package):
                return False, f"Library '{spec.check_package}' is not installed."
        return True, "Available"

    @classmethod
    def get_models_for_task(
        cls,
        task: str,
        available_only: bool = True,
    ) -> List[ModelSpec]:
        matched: List[ModelSpec] = []
        for spec in MODEL_REGISTRY.values():
            if task in spec.task_types or task == spec.category.lower().replace(" ", "_"):
                if available_only:
                    avail, _ = cls.is_available(spec)
                    if not avail:
                        continue
                matched.append(spec)
        return matched

    @classmethod
    def instantiate(cls, method_id: str, random_state: int = 42) -> Optional[Any]:
        spec = cls.get_spec(method_id)
        if not spec:
            return None
        avail, _ = cls.is_available(spec)
        if not avail:
            return None
        try:
            return spec.factory(random_state)
        except Exception:
            return None


class CandidateModelSelector:
    """
    Intelligently select suitable model candidates based on dataset profile.

    Avoids blindly running computationally prohibitive models (e.g. kernel SVM on 100k rows)
    and ensures balanced coverage across model families (linear, trees, boosting, distance).
    """

    @classmethod
    def select_candidates(
        cls,
        task: str,
        n_rows: int,
        n_cols: int,
        requested_methods: Optional[List[str]] = None,
        max_candidates: int = 8,
    ) -> List[str]:
        # If user explicitly requested specific methods, validate their availability
        if requested_methods:
            valid_requested = []
            for mid in requested_methods:
                spec = ModelRegistry.get_spec(mid)
                if spec and task in spec.task_types:
                    avail, _ = ModelRegistry.is_available(spec)
                    if avail:
                        valid_requested.append(mid)
            return valid_requested

        available_specs = ModelRegistry.get_models_for_task(task, available_only=True)
        if not available_specs:
            return []

        # Filter out models where dataset size exceeds recommended limit
        viable: List[ModelSpec] = []
        for spec in available_specs:
            if n_rows > spec.max_recommended_rows:
                continue
            viable.append(spec)

        if not viable:
            viable = available_specs

        # Prioritize diverse families:
        # We want: 1 linear/baseline, 1-2 trees, 1-2 boosting, 1 distance/probabilistic/kernel, 1 modern
        selected: List[str] = []
        selected_families: set[str] = set()

        # Preferred anchor models per task
        anchors = {
            "binary_classification": [
                "random_forest_classifier",
                "logistic_regression",
                "gradient_boosting_classifier",
                "hist_gradient_boosting_classifier",
                "decision_tree_classifier",
                "knn_classifier",
                "extra_trees_classifier",
                "gaussian_nb",
                "xgboost_classifier",
                "lightgbm_classifier",
                "catboost_classifier",
                "svm_classifier",
                "mlp_classifier",
            ],
            "multiclass_classification": [
                "random_forest_classifier",
                "logistic_regression",
                "gradient_boosting_classifier",
                "hist_gradient_boosting_classifier",
                "decision_tree_classifier",
                "knn_classifier",
                "extra_trees_classifier",
                "xgboost_classifier",
                "lightgbm_classifier",
                "catboost_classifier",
                "svm_classifier",
            ],
            "regression": [
                "random_forest_regressor",
                "linear_regression",
                "gradient_boosting_regressor",
                "ridge_regression",
                "hist_gradient_boosting_regressor",
                "decision_tree_regressor",
                "lasso_regression",
                "extra_trees_regressor",
                "huber_regressor",
                "xgboost_regressor",
                "lightgbm_regressor",
                "catboost_regressor",
                "knn_regressor",
            ],
            "clustering": [
                "kmeans",
                "agglomerative_clustering",
                "dbscan",
                "gaussian_mixture",
                "minibatch_kmeans",
                "birch",
            ],
            "anomaly_detection": [
                "isolation_forest",
                "local_outlier_factor",
                "one_class_svm",
                "elliptic_envelope",
            ],
        }

        preferred_order = anchors.get(task, [s.id for s in viable])
        viable_map = {s.id: s for s in viable}

        for mid in preferred_order:
            if mid in viable_map:
                spec = viable_map[mid]
                selected.append(mid)
                selected_families.add(spec.family)
                if len(selected) >= max_candidates:
                    break

        return selected
