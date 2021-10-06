#pragma once
#include "types.h"
using namespace std;

class PCA {
public:
    PCA(unsigned int n_components);

    void fit(Matrix X);

    Eigen::MatrixXd transform(Matrix X);
private:
    unsigned int alpha;
    Matrix T;
};
