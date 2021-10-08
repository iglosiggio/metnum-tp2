#include <algorithm>
#include <chrono>
#include <iostream>
#include "eigen.h"

using namespace std;

pair<double, Vector> power_iteration(const Matrix& X, unsigned num_iter, double eps)
{
    double eigenvalue;
    Vector b = Vector::Random(X.cols());
    b /= b.norm();

    for (unsigned i = 0; i < num_iter; ++i) {
        Vector old(b);
        b = X * b;
        // Actualizamos la aproximación del autovalor
        eigenvalue = b.norm();
        // Normalizamos la aproximación al autovector
        b /= eigenvalue;

        double cos_angle = b.transpose() * old;
        // Si nos movimos "poquito" entonces ya convergimos
        if (abs(cos_angle - 1) < eps) break;
    }

    return make_pair(eigenvalue, b);
}

pair<Vector, Matrix> get_first_eigenvalues(const Matrix& X, unsigned num, unsigned num_iter, double epsilon)
{
    Matrix A(X);
    Vector eigvalues(num);
    Matrix eigvectors(A.rows(), num);

    for (unsigned i = 0; i < num; ++i) {
        pair<double, Vector> res = power_iteration(A, num_iter, epsilon);
        eigvalues[i] = res.first;
        eigvectors.col(i) = res.second;

        A -= res.first * res.second * res.second.transpose();
    }

    return make_pair(eigvalues, eigvectors);
}
