#ifndef ACTIVATIONLAYER_H_INCLUDED
#define ACTIVATIONLAYER_H_INCLUDED

#include "../includes/matrix.h"
#include <stdbool.h>

enum activation_type {
    SIGMOID,
    TANH,
    RELU,
    SOFTMAX,
    LEAKY_RELU
};

typedef struct {
    enum activation_type type;
    ParMatrix *input;
    ParMatrix *output;
} activationLayer;

activationLayer *activation_init(enum activation_type type);

void activation_forward(activationLayer *layer);

Matrix *activation_backward(activationLayer *layer, Matrix *error, double lr);

void activation_free(activationLayer *layer);

void set_leaky_relu_alpha(double alpha);

#endif // ACTIVATIONLAYER_H_INCLUDED
