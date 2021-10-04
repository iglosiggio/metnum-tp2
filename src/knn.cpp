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
        ret(k) = predecir_fila(X.row(k), k); 
    }

    return ret;
}


double calcular_distancia(Vector v1, Vector v2){
	Vector v = v1 - v2;
	return v.squaredNorm(); 
}

vector<uint> obtenerKMenoresIndices(vector<tuple<double,uint>> distancias, Matrix X_train, Matrix y_train, uint cant_vecinos){
	vector<uint> kMenores(cant_vecinos);
	sort(distancias.begin(),distancias.end());

	//obtengo las etiquetas de las k filas con menor distancia
	for (uint i = 0; i < cant_vecinos; ++i) {
		kMenores[i] = y_train(get<1>(distancias[i]),0);
	}

	return kMenores;
}

uint clasificar(vector<uint> etiquetas, Matrix X_train, uint cant_vecinos) {
	//elegimos la etiqueta que más se repite
	vector<uint> cantRepeticiones(10, 0);
	for (uint i = 0; i < cant_vecinos; ++i) {
		cantRepeticiones[etiquetas[i]]++;
	}

	uint label = 0;
	uint maxCantidad = cantRepeticiones[0];

	for (uint i = 1; i < 10; ++i) {
		if (cantRepeticiones[i] > maxCantidad) {
			maxCantidad = cantRepeticiones[i];
			label = i;
		}	
	}

	return label;
}

uint KNNClassifier::predecir_fila(Vector fila, unsigned nro) {
	vector<uint> kMenores(cant_vecinos);
	vector<tuple<double, uint>> distancias(X_train.rows());

	//calculo la distancia entre cada vector de X_train y el vector a predecir
	for (uint i = 0; i < X_train.rows(); ++i) {
		tuple<double, uint> dist_filas;
		get<0>(dist_filas) = calcular_distancia(X_train.row(i),fila);
		get<1>(dist_filas) = i;
		distancias[i] = dist_filas;
	}

	//eligo las k distancias más cercanas a 0 quedandome con los índices
	kMenores = obtenerKMenoresIndices(distancias, X_train, y_train, cant_vecinos);

	//elijo la etiqueta
	uint etiqueta = clasificar(kMenores, X_train, cant_vecinos);

	return etiqueta;
}