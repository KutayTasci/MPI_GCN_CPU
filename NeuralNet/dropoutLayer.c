//
// Created by serdar on 10/8/24.
//

#include "../includes/dropoutLayer.h"
#include <stdlib.h>
#include "../includes/matrix.h"

dropoutLayer* dropout_init(double dropout_rate) {
    dropoutLayer* layer = (dropoutLayer*)malloc(sizeof(dropoutLayer));
    layer->dropout_rate = dropout_rate;
    layer->input = NULL;
    layer->output = NULL;
    layer->init = false;
    return layer;
}

void dropout_forward(dropoutLayer* layer) {
    if (!layer->init) {
        layer->output = create_output_matrix(layer->input);
        layer->mask = (bool*)malloc(sizeof(bool) * layer->input->mat->m * layer->input->mat->n);
        layer->init = true;
    }
    for (int i = 0; i < layer->input->mat->m; i++) {
        for (int j = 0; j < layer->input->mat->n; j++) {
            if (uniform_distribution(0, 1) < layer->dropout_rate) {
                layer->mask[i * layer->input->mat->n + j] = false;
                layer->output->mat->entries[i][j] = 0;
            } else {
                layer->mask[i * layer->input->mat->n + j] = true;
                layer->output->mat->entries[i][j] = layer->input->mat->entries[i][j];
            }
        }
    }
}

void dropout_backward(dropoutLayer* layer, Matrix* error, double lr) {
    for (int i = 0; i < error->m; i++) {
        for (int j = 0; j < error->n; j++) {
            if (layer->mask[i * error->n + j]) {
                error->entries[i][j] = error->entries[i][j];
            } else {
                error->entries[i][j] = 0;
            }
        }
    }
}

void dropout_free(dropoutLayer* layer) {
    if (layer->input != NULL) {
        free(layer->input);
    }
    if (layer->output != NULL) {
        free(layer->output);
    }
    free(layer);
}




