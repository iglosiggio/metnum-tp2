#include <algorithm>
#include <chrono>
#include <iostream>
#include "knn.h"

using namespace std;

KNNClassifier::KNNClassifier(unsigned int n_neighbors) {
	cant_vecinos = n_neighbors;
}

void KNNClassifier::fit(Matrix X, Matrix y) {
	X_train = X;
	y_train = y;
}

Vector KNNClassifier::predict(Matrix X) {
    // Creamos vector columna a devolver
    auto ret = Vector(X.rows());

    for (unsigned k = 0; k < X.rows(); ++k) {
        ret(k) = predecir_fila(X.row(k));
    }

    return ret;
}

uint KNNClassifier::predecir_fila(const Vector& fila) {
	using Dist = tuple<double, uint>;
	vector<Dist> distancias(X_train.rows());
	vector<uint> cantRepeticiones(10, 0);

	// Calculo todas las distancias
	Vector vec_distancias = (X_train.rowwise() - fila.transpose()).rowwise().squaredNorm().transpose();
	for (uint i = 0; i < vec_distancias.size(); i++)
		distancias[i] = make_tuple(vec_distancias(i), i);

	// Agarro las k más chicas
	std::partial_sort(distancias.begin(), distancias.begin() + cant_vecinos, distancias.end());

	// Cuento los votos de cada vecino
	std::for_each(distancias.cbegin(), distancias.cbegin() + cant_vecinos, [&](Dist d) {
		cantRepeticiones[y_train(get<1>(d), 0)]++;
	});

	// Tomo la etiqueta más votada
	auto label_it = std::max_element(cantRepeticiones.cbegin(), cantRepeticiones.cend());
	uint etiqueta = std::distance(cantRepeticiones.cbegin(), label_it);

	return etiqueta;
}
