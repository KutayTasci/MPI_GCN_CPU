#ifndef LOSSFUNCTIONS_H_INCLUDED
#define LOSSFUNCTIONS_H_INCLUDED

#include "matrix.h"

void crossEntropy(Matrix *y, Matrix *y_hat);
void l2Loss(Matrix *y, Matrix *y_hat);
#endif // LOSSFUNCTIONS_H_INCLUDED
