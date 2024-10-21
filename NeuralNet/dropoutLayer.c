//
// Created by serdar on 10/8/24.
//

#include "../includes/dropoutLayer.h"
#include <stdlib.h>
#include "../includes/matrix.h"

dropoutLayer *dropout_init(double dropout_rate) {
    dropoutLayer *layer = (dropoutLayer *) malloc(sizeof(dropoutLayer));
    layer->dropout_rate = dropout_rate;
    layer->input = NULL;
    layer->output = NULL;
    return layer;
}

void dropout_forward(dropoutLayer *layer) {
    for (int i = 0; i < layer->input->mat->m; i++) {
        for (int j = 0; j < layer->input->mat->n; j++) {
            bool mask = uniform_distribution(0, 1) < layer->dropout_rate;
            layer->mask[i * layer->input->mat->n + j] = mask;
            layer->output->mat->entries[i][j] = mask * layer->input->mat->entries[i][j];
        }
    }
}

void dropout_backward(dropoutLayer *layer, Matrix *error, double lr) {
    for (int i = 0; i < error->m; i++) {
        for (int j = 0; j < error->n; j++) {
            error->entries[i][j] = layer->mask[i * error->n + j] * error->entries[i][j];
        }
    }
}

void dropout_free(dropoutLayer *layer) {
    if (layer->input != NULL) {
        free(layer->input);
    }
    if (layer->output != NULL) {
        free(layer->output);
    }
    free(layer);
}




