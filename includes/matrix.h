#ifndef MATRIX_H_INCLUDED
#define MATRIX_H_INCLUDED

#include "typedef.h"
#include "../includes/gcnLayer.h"

void matrix_fill(Matrix *m, double n);
void GEMM (Matrix *A, Matrix *B, Matrix *C);
void GEMM_NT (Matrix *A, Matrix *B, Matrix *C);
void GEMM_TN (Matrix *A, Matrix *B, Matrix *C);

double uniform_distribution(double low, double high);
void init_weights_random(gcnLayer* layer, int scale);

void matrix_print(Matrix* m);

Matrix* matrix_sum_exp(Matrix* m, int axis);

void matrix_subtract(Matrix* m1, Matrix* m2, Matrix *m);
void matrix_sum(Matrix* m1, Matrix* m2, Matrix *m);

void matrix_de_crossEntropy(Matrix* m1, Matrix* m2, Matrix *m);
void matrix_multiply(Matrix* m1, Matrix* m2, Matrix *m);
void matrix_divide(Matrix* m1, Matrix* m2, Matrix *m);
Matrix* matrix_copy(Matrix* m);

void matrix_scale(double n, Matrix* mat);
Matrix* matrix_sqrt(Matrix* mat);
Matrix* matrix_scale_return(double n, Matrix* mat);
Matrix* matrix_softmax(Matrix* m);
void matrix_addScalar(Matrix* mat, double n);

void matrix_MinMaxNorm(Matrix* mat);
void metrics(Matrix* y_hat, Matrix* y);

#endif // MATRIX_H_INCLUDED
