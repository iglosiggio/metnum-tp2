from itertools import chain
from pickle import dump
from time import perf_counter_ns
from multiprocessing import Pool
from pandas import DataFrame
from sklearn.model_selection import KFold
from metnum import KNNClassifier, PCA

import pandas as pd

df_train = pd.read_csv("../data/train.csv")

# Uso values para mandar todo a arrays de numpy
X = df_train[df_train.columns[1:]].values
Y = df_train["label"].values.reshape(-1, 1)

RANDOM_SEED = 0xCAFEC170
NUM_WORKERS = 8

def run_knn(arg):
    k, X_train, Y_train, X_val, Y_val = arg
    print(f'=== START K={k} ===', flush=True)

    time_start = perf_counter_ns()
    clf_knn = KNNClassifier(k)
    clf_knn.fit(X_train, Y_train)
    time_knn_init_and_fit = perf_counter_ns()
    Y_pred = clf_knn.predict(X_val)
    time_knn_predict = perf_counter_ns()

    result = {
        'kind': 'knn',
        'k': k,
        'elapsed_time': time_knn_predict - time_start,
        'knn_init_and_fit_time': time_knn_init_and_fit - time_start,
        'knn_predict_time': time_knn_predict - time_knn_init_and_fit,
        'y_pred': Y_pred,
        'y_val': Y_val,
    }
    print(f'=== END K={k} ===', flush=True)
    return result

def run_pca(arg):
    k, alpha, X_train, Y_train, X_val, Y_val = arg
    print(f'=== START K={k}, Alpha={alpha} ===', flush=True)
    time_start = perf_counter_ns()
    pca = PCA(alpha)
    pca.fit(X_train)
    time_pca_init_and_fit = perf_counter_ns()
    pc_X_train = pca.transform(X_train)
    pc_X_val = pca.transform(X_val)
    time_pca_transform = perf_counter_ns()
    clf_knn = KNNClassifier(k)
    clf_knn.fit(pc_X_train, Y_train)
    time_knn_init_and_fit = perf_counter_ns()
    Y_pred = clf_knn.predict(pc_X_val)
    time_knn_predict = perf_counter_ns()

    result = {
        'kind': 'knnpca',
        'k': k,
        'alpha': alpha,
        'elapsed_time': time_knn_predict - time_start,
        'pca_init_and_fit_time': time_pca_init_and_fit - time_start,
        'pca_transform_time': time_pca_transform - time_pca_init_and_fit,
        'knn_init_and_fit_time': time_knn_init_and_fit - time_pca_transform,
        'knn_predict_time': time_knn_predict - time_knn_init_and_fit,
        'y_pred': Y_pred,
        'y_val': Y_val,
    }

    print(f'=== STOP K={k}, Alpha={alpha} ===', flush=True)
    return result

def main():
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    pca_tasks = (
        (k, alpha, X[train_idx], Y[train_idx], X[test_idx], Y[test_idx])
        for k in range(1, 30)
        for alpha in range(1, 30)
        for train_idx, test_idx in kf.split(X, Y)
    )
    knn_tasks = (
        (k, X[train_idx], Y[train_idx], X[test_idx], Y[test_idx])
        for k in range(1, 30)
        for train_idx, test_idx in kf.split(X, Y)
    )

    with Pool(NUM_WORKERS) as p:
        pca_results = p.imap_unordered(run_pca, pca_tasks)
        knn_results = p.imap_unordered(run_knn, knn_tasks)

        results = DataFrame(chain(pca_results, knn_results))

    with open('data.pickle', 'wb') as file:
        dump(results, file)

    return results

if __name__ == '__main__':
    main()
