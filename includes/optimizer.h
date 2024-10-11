//
// Created by serdar on 10/10/24.
//

#ifndef MPI_GCN_CPU_OPTIMIZER_H
#define MPI_GCN_CPU_OPTIMIZER_H

#include "matrix.h"
#include "typedef.h"

void
adam_step(Matrix *gradients, Matrix *weights, Matrix *bias_grad, double *bias, double lr, double beta1, double beta2,
          double epsilon, int t);


#endif //MPI_GCN_CPU_OPTIMIZER_H
