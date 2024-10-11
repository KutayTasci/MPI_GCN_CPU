//
// Created by serdar on 10/10/24.
//
#include <math.h>
#include "../includes/optimizer.h"

void
adam_step(Matrix *gradients, Matrix *weights, Matrix *bias_grad, double *bias, double lr, double beta1, double beta2,
          double epsilon, int t) {
    // compute the learning rate
    double beta1_t = pow(beta1, t);
    double beta2_t = pow(beta2, t);
    double lr_t = lr * sqrt(1 - beta2_t) / (1 - beta1_t);

    double m_weight, v_weight, m_bias, v_bias;

    for (int i = 0; i < gradients->m; i++) {
        for (int j = 0; j < gradients->n; j++) {
            // first and second moments for weights
            m_weight = beta1 * m_weight + (1 - beta1) * gradients->entries[i][j];
            v_weight = beta2 * v_weight + (1 - beta2) * gradients->entries[i][j] * gradients->entries[i][j];

            weights->entries[i][j] -= lr_t * m_weight / (sqrt(v_weight) + epsilon);
        }
    }

    for (int j = 0; j < bias_grad->n; j++) {
        // first and second moments for biases
        m_bias = beta1 * m_bias + (1 - beta1) * bias_grad->entries[0][j];
        v_bias = beta2 * v_bias + (1 - beta2) * bias_grad->entries[0][j] * bias_grad->entries[0][j];

        bias[j] -= lr_t * m_bias / (sqrt(v_bias) + epsilon);
    }
}