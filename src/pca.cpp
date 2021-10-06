#include <iostream>
#include "pca.h"
#include "eigen.h"

using namespace std;


PCA::PCA(unsigned int n_components){
  alpha = n_components;

}

void PCA::fit(Matrix X){

  Vector v(X.rows());
  v.fill((double)1 / (double)X.rows());

  Vector means = X.transpose() * v;

  Vector aux;
  for (int i = 0; i < X.rows(); i++) {
    aux = X.row(i);
    X.row(i) = (aux - means) / sqrt(X.rows() - 1);
  }

  Matrix covariance = X.transpose() * X;

  tuple<Vector, Matrix> eigens = get_first_eigenvalues(covariance,alpha);
  Vector eigenvalues = get<0>(eigens);
  Matrix eigenvectors = get<1>(eigens);

  this->T = eigenvectors;


}


MatrixXd PCA::transform(Matrix X){
  return X*T;
}
