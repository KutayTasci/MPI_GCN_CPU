#include "../includes/activationLayer.h"
#include <math.h>
#include <stdlib.h>
#include "../includes/matrix.h"
//Local function

void apply(double (*func)(double), activationLayer* layer) {
    for (int i = 0; i < layer->input->mat->m; i++) {
        for (int j = 0; j < layer->input->mat->n; j++) {
            layer->output->mat->entries[i][j] = (*func)(layer->input->mat->entries[i][j]);
        }
    }
}

void apply_mat(double (*func)(double), Matrix* mat) {
    for (int i = 0; i < mat->m; i++) {
        for (int j = 0; j < mat->n; j++) {
            mat->entries[i][j] = (*func)(mat->entries[i][j]);
        }
    }
}

double sigmoid(double input) {
    return 1.0 / (1 + exp(-1 * input));
}

Matrix* sigmoidPrime(Matrix* mat) {
    Matrix* ones = matrix_create(mat->m, mat->n);
    matrix_fill(ones, 1);
    Matrix* temp = matrix_create(mat->m, mat->n);
    matrix_subtract(ones, mat, temp);
    matrix_multiply(mat, temp, ones);
    matrix_free(temp);
    return ones;
}

double relu(double input) {
    if (input > 0) {
        return input;
    } else {
        return 0;
    }
}

double reluPrimeAtomic(double input) {
    if (input > 0) {
        return 1;
    } else {
        return 0;
    }
}

Matrix* reluPrime(Matrix* m) {
    Matrix* mat = matrix_copy(m);
    apply_mat(reluPrimeAtomic, mat);
    return mat;
}


double tanh_ops(double input) {
    return tanh(input);
}


Matrix* tanhPrime(Matrix* mat) {
    Matrix* ones = matrix_create(mat->m, mat->n);
    matrix_fill(ones, 1);
    Matrix* temp = matrix_create(mat->m, mat->n);
    Matrix* out = matrix_create(mat->m, mat->n);
    matrix_multiply(mat, mat, temp);
    matrix_subtract(ones, temp, out);
    matrix_free(ones);
    matrix_free(temp);
    return out;
}

activationLayer* activation_init(enum activation_type type) {
    activationLayer* layer = malloc(sizeof(activationLayer));
    layer->type = type;
    layer->init = false;
    return layer;
}

void activation_forward(activationLayer* layer) {
    if (!layer->init) {
        layer->output = (ParMatrix*) malloc(sizeof(ParMatrix));
        layer->output->gm = layer->input->gm;
        layer->output->gn = layer->input->gn;
        layer->output->inPart = layer->input->inPart;
        layer->output->l2gMap = layer->input->l2gMap;
        layer->output->store = layer->input->store;
        layer->output->mat = matrix_create(layer->input->mat->m, layer->input->mat->n);
    }

    if (layer->type == SIGMOID) {
        apply(sigmoid, layer);
    } else if (layer->type == TANH) {
        apply(tanh_ops, layer);
    } else if (layer->type == RELU) {
        apply(relu, layer);
    }
}

Matrix* activation_backward(activationLayer* layer, Matrix* error, double lr) {
    Matrix* input_error;
    Matrix* out_error = matrix_create(layer->input->mat->m, layer->input->mat->n);
    if (layer->type == SIGMOID) {
        input_error = sigmoidPrime(layer->input->mat);
        //matrix_scale(lr, input_error);
    } else if (layer->type == TANH) {
        input_error = tanhPrime(layer->input->mat);
        //matrix_scale(lr, input_error);
    } else if (layer->type == RELU) {
        input_error = reluPrime(layer->input->mat);
        //matrix_scale(lr, input_error);
    } else {
        exit(1);
    }
    matrix_multiply(error, input_error, out_error);
    
    matrix_free(input_error);
    return out_error;
}

void activation_free(activationLayer* layer) {
    matrix_free(layer->input->mat);
    matrix_free(layer->output->mat);

    free(layer);
}
