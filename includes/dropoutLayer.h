//
// Created by serdar on 10/8/24.
//

#ifndef MPI_GCN_CPU_DROPOUTLAYER_H
#define MPI_GCN_CPU_DROPOUTLAYER_H

#include <stdbool.h>
#include "typedef.h"

// dropout layer
typedef struct {
    double dropout_rate;
    ParMatrix *input;
    ParMatrix *output;
    bool *mask;
} dropoutLayer;

dropoutLayer *dropout_init(double dropout_rate);

void dropout_forward(dropoutLayer *layer, int mask_type);

void dropout_backward(dropoutLayer *layer, Matrix *error, double lr);

void dropout_free(dropoutLayer *layer);

#endif //MPI_GCN_CPU_DROPOUTLAYER_H
