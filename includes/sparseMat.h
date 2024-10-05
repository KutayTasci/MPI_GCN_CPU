#ifndef SPARSEMAT_H_INCLUDED
#define SPARSEMAT_H_INCLUDED

#include "typedef.h"
#include "gcnLayer.h"


void aggregate_gemm_overlap(gcnLayer* layer, Matrix* X, Matrix* Y, Matrix* B, Matrix *C, int step);
void aggregate_csr(gcnLayer* layer, Matrix* X, Matrix* Y, int step);
void aggregate(gcnLayer* layer, Matrix* X, Matrix* Y, int step);
void aggregate_cco(gcnLayer* layer, Matrix* X, Matrix* Y, int step);
void aggregate_partial_cco(gcnLayer* layer, Matrix* X, Matrix* Y, int step);
void aggregate_csc(gcnLayer* layer, Matrix* X, Matrix* Y, int step);
void aggregate_cco_csc(gcnLayer* layer, Matrix* X, Matrix* Y, int step);
void aggregate_cco_hybrid(gcnLayer* layer, Matrix* X, Matrix* Y, int step);
void aggregate_no_comm(gcnLayer* layer, Matrix* X, Matrix* Y, int step);
void aggregate_no_comp(gcnLayer* layer, Matrix* X, Matrix* Y, int step);

#endif // SPARSEMAT_H_INCLUDED
