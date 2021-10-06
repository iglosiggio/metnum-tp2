#include <algorithm>
#include <chrono>
#include <iostream>
#include "eigen.h"

using namespace std;

Matrix outer(const Vector & u, const Vector & v){
    Matrix out(u.size(), v.size());

    for (int i = 0; i < u.size(); ++i) {
        for (int j = 0; j < v.size(); ++j) {
            out(i , j) = u[i] * v[j];
        }
    }

    return out;
}

pair<double, Vector> power_iteration(const Matrix& X, unsigned num_iter, double eps)
{
    Vector b = Vector::Random(X.cols());
    b = b / b.norm();

    for (unsigned i = 0; i < num_iter; ++i) {

        Vector old(b);
        b = X * b;
        b = b / b.norm();

        double cos_angle = b.transpose()*old;
        if ( abs(cos_angle - 1) < eps) {
            break;
        }
    }

    double eigenvalue = b.transpose() * X * b;

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
        for (unsigned j = 0; j < A.rows(); ++j) {
            eigvectors(j , i) = res.second[j];
        }

        A = A - res.first * outer(res.second, res.second);
    }

    return make_pair(eigvalues, eigvectors);
}
