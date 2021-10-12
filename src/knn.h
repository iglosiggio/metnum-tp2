#pragma once

#include "types.h"
#include "kdtree.hpp"

class KNNClassifier {
public:
    KNNClassifier(unsigned int n_neighbors);

    void fit(Matrix X, Matrix y);

    Vector predict(Matrix X);
    Vector predict_kdtree(Matrix X);

    uint predecir_fila(const Vector& fila);

private:
		unsigned int cant_vecinos;
		Matrix X_train;
		Matrix y_train;
};
