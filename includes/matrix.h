#ifndef MATRIX_H_INCLUDED
#define MATRIX_H_INCLUDED

#include "typedef.h"
#include "../includes/gcnLayer.h"

void matrix_fill(Matrix *m, double n);

void GEMM(Matrix *A, Matrix *B, double *add_v, Matrix *C);

void GEMM_NT(Matrix *A, Matrix *B, Matrix *C);

void GEMM_TN(Matrix *A, Matrix *B, Matrix *C);

void bias_grad(Matrix *Y_prime, double *gradients);

double uniform_distribution(double low, double high);

void init_weights_random(gcnLayer *layer, int scale);

void matrix_print(Matrix *m);

Matrix *matrix_sum_exp(Matrix *m, int axis);

void matrix_subtract(Matrix *m1, Matrix *m2, Matrix *m);

void matrix_sum(Matrix *m1, Matrix *m2, Matrix *m);

void matrix_de_crossEntropy(Matrix *m1, Matrix *m2, Matrix *m, bool *mask);

void matrix_l2Loss(Matrix *m1, Matrix *m2, Matrix *m);

void matrix_multiply(Matrix *m1, Matrix *m2, Matrix *m);

void matrix_divide(Matrix *m1, Matrix *m2, Matrix *m);

Matrix *matrix_copy(Matrix *m);

Matrix *matrix_full_copy(Matrix *m);

bool matrix_equals(Matrix *m1, Matrix *m2);

void matrix_scale(double n, Matrix *mat);

Matrix *matrix_sqrt(Matrix *mat);

Matrix *matrix_scale_return(double n, Matrix *mat);

Matrix *matrix_softmax(Matrix *m);

void matrix_addScalar(Matrix *mat, double n);

void matrix_MinMaxNorm(Matrix *mat);

void metrics(Matrix *y_hat, Matrix *y, bool *mask);

void matrix_clear_buffers(Matrix *m);

#endif // MATRIX_H_INCLUDED
