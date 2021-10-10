#include <iostream>
#include "pca.h"
#include "eigen.h"

using namespace std;


PCA::PCA(unsigned int n_components){
  alpha = n_components;

}

// Tenemos n muestras de m variables ($X \in \mathbb{R}^{n \cross m}$).
void PCA::fit(Matrix X){
  // Calculamos el vector $\mu$ que contiene la media de cada una de las
  // variables.
  Vector means = X.transpose() * Vector::Constant(X.rows(), 1.0 / X.rows());

  // Construimos la matriz $X \in \mathbb{R}^{n \cross m} dónde cada muestra
  // corresponde a una fila de $X$ y tienen media cero
  // (i.e., $X_i := (x^{(i) - \mu)$).
  for (int i = 0; i < X.rows(); i++) X.row(i) -= means;

  // Diagonalizamos la matriz de covarianzas $M_x = \frac{X^tX}{n - 1}$. La
  // matriz $V$ (ortogonal) contiene los autovectores de $M_x$.
  Matrix covariance = (X.transpose() * X) / (X.rows() - 1);
  tuple<Vector, Matrix> eigens = get_first_eigenvalues(covariance, alpha);
  this->T = get<1>(eigens);
}

MatrixXd PCA::transform(Matrix X){
  return X*T;
}
