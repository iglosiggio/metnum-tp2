from time import perf_counter_ns
from metnum import KNNClassifier, PCA
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

import pandas as pd

df_train = pd.read_csv("../data/train.csv")

# Uso values para mandar todo a arrays de numpy
X = df_train[df_train.columns[1:]].values
y = df_train["label"].values.reshape(-1, 1)

limit = int(0.8 * X.shape[0])

X_train, Y_train = X[:limit], y[:limit]
X_val, Y_val = X[limit:], y[limit:]

assert len(X_train) == len(Y_train)
assert len(X_val) == len(Y_val)
print(f"Corriendo con {len(X_train)} instancias de entrenamiento y {len(X_val)} de validación", flush=True)


def run_knn(k, X_train, Y_train, X_val, Y_val):
    clf_knn = KNNClassifier(k)
    clf_knn.fit(X_train, Y_train)
    Y_pred = clf_knn.predict(X_val)
    accuracy = accuracy_score(Y_val, Y_pred)
    cohen_kappa = cohen_kappa_score(Y_val, Y_pred)
    f1 = f1_score(Y_val, Y_pred, average='micro')
    print(f'Accuracy: {accuracy}')
    print(f'Cohen kappa: {cohen_kappa}')
    print(f'F1: {f1}', flush=True)
    return accuracy, cohen_kappa, f1

def main():
    results = []

    for k in range(1, 30):
        for alpha in range(1, 200):
            print(f'=== K={k}, Alpha={alpha} ===', flush=True)
            start_time = perf_counter_ns()
            pca = PCA(alpha)
            pca.fit(X_train)
            pc_X_train = pca.transform(X_train)
            pc_X_val = pca.transform(X_val)
            accuracy, cohen_kappa, f1 = run_knn(k, pc_X_train, Y_train, pc_X_val, Y_val)
            end_time = perf_counter_ns()
            results.append({
                'kind': 'knnpca',
                'k': k,
                'alpha': alpha,
                'elapsed_time': end_time - start_time,
                'accuracy': accuracy,
                'cohen_kappa': cohen_kappa,
                'f1': f1
            })
        print(f'=== K={k} ===', flush=True)
        start_time = perf_counter_ns()
        accuracy, cohen_kappa, f1 = run_knn(k, X_train, Y_train, X_val, Y_val)
        end_time = perf_counter_ns()
        results.append({
            'kind': 'knn',
            'k': k,
            'elapsed_time': end_time - start_time,
            'accuracy': accuracy,
            'cohen_kappa': cohen_kappa,
            'f1': f1
        })

    print(results, flush=True)
    return results

if __name__ == '__main__':
    main()
