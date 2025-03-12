#ifndef GCNLAYER_H_INCLUDED
#define GCNLAYER_H_INCLUDED

#include "typedef.h"
#include "basic.h"
#include "comm.h"
#include "masking.h"


typedef struct {
    int size_m;
    int size_f;
    int size_output;
    ParMatrix *input;
    ParMatrix *output;
    Matrix *weights;
    Matrix *m_weights; // first moment weights
    Matrix *v_weights; // second moment weights
    double *bias;
    double *m_bias; // first moment bias
    double *v_bias; // second moment bias
    Matrix *gradients;
    double *gradients_bias;
    void *comm;
    CommType comm_type;
    bool **masks; // size: 3
} gcnLayer;

gcnLayer *gcn_init(SparseMat *adj, void *comm, CommType comm_type, int size_f, int size_out);

void setMode(int i);

void gcn_forward(gcnLayer *layer, int eval, double *time);

Matrix *gcn_backward(gcnLayer *layer, Matrix *out_error, double *time);

void gcn_step(gcnLayer *layer, double lr, int t);

void gcn_free(gcnLayer *layer);

void initBias(gcnLayer *layer, double d);

ParMatrix *create_gcn_output_matrix(ParMatrix *X, gcnLayer *gcn_layer, bool is_input);

#endif // GCNLAYER_H_INCLUDED
