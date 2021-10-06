#pragma once

#include <Eigen/Sparse>
#include <Eigen/Dense>

using Eigen::MatrixXd;

using uint = unsigned int;

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix;
typedef Eigen::SparseMatrix<double> SparseMatrix;

typedef Eigen::VectorXd Vector;
