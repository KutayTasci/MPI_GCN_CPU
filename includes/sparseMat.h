#ifndef SPARSEMAT_H_INCLUDED
#define SPARSEMAT_H_INCLUDED

#include "typedef.h"
#include "comm.h"
#include "nodeSampling.h"


void aggregate_gemm_overlap(OPComm *opComm, Matrix *X, Matrix *Y, Matrix *B, Matrix *C, int step);

void aggregate_csr(OPComm *opComm, Matrix *X, Matrix *Y, int step, bool *mask);

void aggregate(OPComm *opComm, Matrix *X, Matrix *Y, int step);

void aggregate_cco(OPComm *opComm, Matrix *X, Matrix *Y, int step);

void aggregate_partial_cco(OPComm *opComm, Matrix *X, Matrix *Y, int step);

void aggregate_csc(OPComm *opComm, Matrix *X, Matrix *Y, int step, bool *mask);

void aggregate_cco_csc(OPComm *opComm, Matrix *X, Matrix *Y, int step);

void aggregate_cco_hybrid(OPComm *opComm, Matrix *X, Matrix *Y, int step);

void aggregate_no_comm(OPComm *opComm, Matrix *X, Matrix *Y, int step);

void aggregate_no_comp(OPComm *opComm, Matrix *X, Matrix *Y, int step);

void aggregate_tp(TPW *tpw, Matrix *X, Matrix *Y, int step, bool *mask);

void aggregate_sampled(NodeSamplingComm *samplingComm, Matrix *X, Matrix *Y, int step, bool *mask);

#endif // SPARSEMAT_H_INCLUDED
