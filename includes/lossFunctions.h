#ifndef LOSSFUNCTIONS_H_INCLUDED
#define LOSSFUNCTIONS_H_INCLUDED

#include "matrix.h"

void totalCrossEntropy(Matrix *y, Matrix *y_hat);
void totalL2Loss(Matrix *y, Matrix *y_hat);
void calcCrossEntropy(Matrix *y, Matrix *y_hat, Matrix *error);
void calcL2Loss(Matrix *y, Matrix *y_hat, Matrix *error);
#endif // LOSSFUNCTIONS_H_INCLUDED
