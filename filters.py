import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.feature_selection import mutual_info_classif


def remove_highly_correlated_features(df, threshold=0.95, label_column='label'):
    df_features = df.drop(columns=[label_column])
    corr_matrix = df_features.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    corr_with_label = df.corr()[label_column].drop(label_column).abs()

    to_drop = set()

    for col in upper.columns:
        for row in upper.index:
            if upper.loc[row, col] > threshold:
                corr_row = corr_with_label.get(row, 0)
                corr_col = corr_with_label.get(col, 0)

                if corr_row < corr_col:
                    to_drop.add(row)
                else:
                    to_drop.add(col)

    df_filtered = df_features.drop(columns=to_drop)
    df_filtered[label_column] = df[label_column]
    return df_filtered



def remove_lower_correlated_with_class(df, threshold=0.05, label_column='label'):
    correlations = df.corr()[label_column].drop(label_column).abs()
    selected = correlations[correlations > threshold].index
    return df[selected.tolist() + [label_column]]

def compute_entropy(data, n_bins=100):
    hist, _ = np.histogram(data, bins=n_bins, density=True)
    return entropy(hist, base=np.e)


def entropy_filter(df, threshold=0.01, label_column='label'):
    n_bins = df.shape[0]

    X = df.drop(columns=[label_column])
    entropies = X.apply(lambda col: compute_entropy(col, n_bins=n_bins), axis=0)

    # maxima entropia
    uniform_distribution = np.ones(n_bins) / n_bins  # Probabilidades iguales
    max_entropy = entropy(uniform_distribution, base=np.e)

    # max etropia computed
    max_entropy_computed = -np.log( 1.0/ n_bins)
    # Normalizar las entropías
    entropies = entropies / max_entropy

    filtered_columns = entropies[entropies >= threshold].index
    return df[[label_column]+filtered_columns.tolist()]

def drop_columns_with_nan(df):
    return df.dropna(axis=1)

def fill_nan_with_zero(df):
    return df.fillna(0)


def fillna_numeric_mean(df, label_column='label'):
    df_copy = df.copy()

    for col in df_copy.columns:
        if col != label_column and df_copy[col].dtype in ['float64', 'int64']:
            df_copy[col] = df_copy[col].fillna(df_copy[col].mean())

    return df_copy


if __name__ == "__main__":
    path = "data/TR_starPep_AB_training.fasta_AAC_class.csv"
    column_class_lavel = 'Class'
    data = pd.read_csv(path)
    print("Original data shape:", data.shape)

    data[column_class_lavel] = data[column_class_lavel].map({'ABP': 1, 'NoNABP': 0})

    data = remove_lower_correlated_with_class(data, threshold=0.05, label_column='Class')
    print("After removing low correlation features:", data.shape)

    data = entropy_filter(data, threshold=0.3, label_column='Class')
    print("After entropy filtering:", data.shape)

    data = remove_highly_correlated_features(data, threshold=0.90, label_column='Class')
    print("After removing highly correlated features:", data.shape)
