//
// Created by serdar on 10/10/24.
//
#include <math.h>
#include "../includes/optimizer.h"

void
adam_step(Matrix *gradients, Matrix *gradients_bias, gcnLayer *layer, double lr, double beta1, double beta2,
          double epsilon, int t) {
    // compute the learning rate
    double beta1_t = pow(beta1, t);
    double beta2_t = pow(beta2, t);
    double lr_t = lr * sqrt(1 - beta2_t) / (1 - beta1_t);

    for (int i = 0; i < gradients->m; i++) {
        for (int j = 0; j < gradients->n; j++) {
            // first and second moments for weights
            layer->m_weights->entries[i][j] =
                    beta1 * layer->m_weights->entries[i][j] + (1 - beta1) * gradients->entries[i][j];
            layer->v_weights->entries[i][j] = beta2 * layer->v_weights->entries[i][j] +
                                              (1 - beta2) * gradients->entries[i][j] * gradients->entries[i][j];

            layer->weights->entries[i][j] -=
                    lr_t * layer->m_weights->entries[i][j] / (sqrt(layer->v_weights->entries[i][j]) + epsilon);
        }
    }

    for (int j = 0; j < layer->size_output; j++) {
        // first and second moments for bias
        layer->m_bias[j] = beta1 * layer->m_bias[j] + (1 - beta1) * gradients_bias->entries[0][j];
        layer->v_bias[j] = beta2 * layer->v_bias[j] + (1 - beta2) * gradients_bias->entries[0][j] *
                                                      gradients_bias->entries[0][j];

        layer->bias[j] -= lr_t * layer->m_bias[j] / (sqrt(layer->v_bias[j]) + epsilon);
    }
}