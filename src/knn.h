#pragma once

#include "types.h"


class KNNClassifier {
public:
    KNNClassifier(unsigned int n_neighbors);

    void fit(Matrix X, Matrix y);

    Vector predict(Matrix X);

    uint predecir_fila(Vector fila, unsigned k);

private:
		unsigned int cant_vecinos;
		Matrix X_train;
		Matrix y_train;

};
