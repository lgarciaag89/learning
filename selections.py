import numpy as np
import pandas as pd
from numpy.core.numeric import indices
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, RFE, SequentialFeatureSelector, RFECV
from sklearn.linear_model import LogisticRegression
from scipy.stats import pearsonr

# 1. Filtro basado en Importancia Media de Disminución (MDI)
def mdi_feature_importance(data_x, data_y, top_n=None):
    rf = RandomForestClassifier()
    rf.fit(data_x, data_y)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    if top_n:
        indices = indices[:top_n]
    return indices, importances[indices]

# 2. Filtro basado en Información Mutua
def mutual_info_selection(x_data, y_data, top_n=None):
    mi = mutual_info_classif(x_data, y_data)
    indices = np.argsort(mi)[::-1]
    if top_n:
        indices = indices[:top_n]
    return indices, mi[indices]

# 3. Wrapper: Recursive Feature Elimination (RFE)
def rfe_selection(x_data, y_data):
    estimator = RandomForestClassifier()
    selector = RFECV(estimator)
    selector = selector.fit(x_data, y_data)
    return np.nonzero(selector.support_)[0]

# 4. Wrapper: Sequential Feature Selector (SFS)
def sfs_selection(x_data, y_data,  direction='forward', random_state=42):
    estimator = LogisticRegression(max_iter=1000, random_state=random_state)
    sfs = SequentialFeatureSelector(estimator,direction=direction)
    sfs.fit(x_data, y_data)
    return np.nonzero(sfs.support_)[0]

# 5. BestFirst con CFS (Correlation-based Feature Selection)
def cfs(x_data, y_data):
    # CFS: Selecciona subconjuntos de rasgos con alta correlación con la clase y baja entre ellos
    n_features = x_data.shape[1]
    selected = []
    remaining = list(range(n_features))
    best_score = -np.inf
    improved = True
    while improved and remaining:
        improved = False
        best_candidate = None
        for feat in remaining:
            candidate = selected + [feat]
            merit = cfs_merit(x_data[:, candidate], y_data)
            if merit > best_score:
                best_score = merit
                best_candidate = feat
                improved = True
        if improved:
            selected.append(best_candidate)
            remaining.remove(best_candidate)
    return selected

def cfs_merit(x_data, y_data):
    k = x_data.shape[1]
    if k == 0:
        return 0
    rcf = np.mean([abs(pearsonr(x_data[:, i], y_data)[0]) for i in range(k)])
    rff = np.mean([abs(pearsonr(x_data[:, i], x_data[:, j])[0]) for i in range(k) for j in range(i + 1, k)]) if k > 1 else 0
    return (k * rcf) / np.sqrt(k + k * (k - 1) * rff) if rff != 0 else rcf

if __name__ == "__main__":
    file = "data/TR_starPep_AB_training.fasta_AAC_class.csv"
    data = pd.read_csv(file)
    column_class_label = 'Class'
    X = data.drop(columns=[column_class_label])
    y = data[column_class_label]

    index_mdi, importance_mdi = mdi_feature_importance(X, y, top_n=10)
    print("MDI Feature Importance:")
    for i in range(len(index_mdi)):
        print(f"Feature {index_mdi[i]}--{X.columns[index_mdi[i]]}: {importance_mdi[i]:.4f}")

    index_IM, importance_IM = mutual_info_selection(X, y, top_n=10)
    print("MDI Feature Importance:")
    for i in range(len(index_mdi)):
        print(f"Feature {index_IM[i]}--{X.columns[index_IM[i]]}: {importance_IM[i]:.4f}")

    index_rfe = rfe_selection(X, y)
    print("RF Selected Features:")
    for i in range(len(index_rfe)):
        print(f"Feature {index_rfe[i]}--{X.columns[index_rfe[i]]}")

    index_sfs= sfs_selection(X, y)
    print("SFS Selected Features:")
    for i in range(len(index_sfs)):
        print(f"Feature {index_sfs[i]}--{X.columns[index_sfs[i]]}")

