#include "../includes/gcnLayer.h"
#include "../includes/sparseMat.h"
#include "../includes/matrix.h"
#include "../includes/optimizer.h"
#include "../includes/comm.h"
#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <sys/sysinfo.h>

int MODE = 0;

inline void initBias(gcnLayer *layer, double d) {
    layer->bias = (double *) malloc(sizeof(double) * layer->size_output);
    layer->m_bias = (double *) malloc(sizeof(double) * layer->size_output);
    layer->v_bias = (double *) malloc(sizeof(double) * layer->size_output);
    layer->gradients_bias = (double *) malloc(sizeof(double) * layer->size_output);
    memset(layer->bias, d, sizeof(double) * layer->size_output);
    memset(layer->m_bias, 0, sizeof(double) * layer->size_output);
    memset(layer->v_bias, 0, sizeof(double) * layer->size_output);
}

/*
Function is ready to be integrated into sparse_Mat struct
Then use implement aggregation function
Don't forget memory management
*/

gcnLayer *gcn_init(SparseMat *adj, int size_f, int size_out) {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    gcnLayer *layer = (gcnLayer *) malloc(sizeof(gcnLayer));
    // size_m is declared later
    layer->size_f = size_f;
    layer->size_output = size_out;
    if (world_rank == 0) {
        init_weights_random(layer, 10);
    } else {
        layer->weights = matrix_create(layer->size_f, layer->size_output, 0);
    }
    MPI_Bcast(&(layer->weights->entries[0][0]), layer->weights->m * layer->weights->n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    initBias(layer, 0);

    layer->gradients = matrix_create(layer->size_f, layer->size_output, 0);
    layer->m_weights = matrix_create(layer->size_f, layer->size_output, 0);
    matrix_fill(layer->m_weights, 0);
    layer->v_weights = matrix_create(layer->size_f, layer->size_output, 0);
    matrix_fill(layer->v_weights, 0);

    return layer;
}

void setMode(int i) {
    MODE = i;
}

void gcn_forward(gcnLayer *layer, int mask_type, NodeSamplingComm *samplingComm) {
    Matrix *temp = matrix_create(layer->size_m, layer->size_f, 0);
    aggregate_sampled(samplingComm, layer->input->mat, temp, FORWARD, layer->masks[mask_type]);
    GEMM(temp, layer->weights, layer->bias, layer->output->mat);
    matrix_free(temp);
}

Matrix *gcn_backward(gcnLayer *layer, Matrix *out_error, NodeSamplingComm *samplingComm) {
    Matrix *temp = matrix_create(layer->size_m, layer->size_output, 0);
    Matrix *out = matrix_create(layer->size_m, layer->size_f, out_error->total_m - out_error->m);
    aggregate_sampled(samplingComm, out_error, temp, BACKWARD, layer->masks[TRAIN_IDX]);
    GEMM_NT(temp, layer->weights, out);
    GEMM_TN(layer->input->mat, temp, layer->gradients);
    bias_grad(temp, layer->gradients_bias);
    matrix_free(temp);
    return out;
}


//Later change this as adam and generate normal
void gcn_step(gcnLayer *layer, double lr, int t) {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    double eta = 0.01;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 0.00000001;
    Matrix *temp = matrix_create(layer->gradients->m, layer->gradients->n, 0);
    MPI_Allreduce(&(layer->gradients->entries[0][0]), &(temp->entries[0][0]),
                  layer->gradients->m * layer->gradients->n,
                  MPI_DOUBLE,
                  MPI_SUM,
                  MPI_COMM_WORLD);
    // update weights
    // update bias
    Matrix *sum_bias_grad = matrix_create(1, layer->size_output, 0); // 1xsize_output
    MPI_Allreduce(&(layer->gradients_bias[0]), &(sum_bias_grad->entries[0][0]),
                  layer->size_output, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    adam_step(temp, sum_bias_grad, layer, lr, beta1, beta2, epsilon, t);
    matrix_free(sum_bias_grad);
    matrix_free(temp);
}

void gcn_free(gcnLayer *layer) {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    /*
    sparseMatFree(layer->adjacency);
    sparseMatFree(layer->adjacency_T);
    sendBufferListFree(layer->sendBuffer,world_size, world_rank);
    recvBufferListFree(layer->recvBufferList,world_size, world_rank);
    sendBufferListFree(layer->sendBuffer_backward,world_size, world_rank);
    recvBufferListFree(layer->recvBufferList_backward,world_size, world_rank);
    free(layer);
    */
    layer = NULL;

}

ParMatrix *create_gcn_output_matrix(ParMatrix *X, gcnLayer *gcn_layer, bool is_input) {
    if (!gcn_layer) return create_output_matrix(X);
    int new_n = is_input ? gcn_layer->size_f : gcn_layer->size_output;
    if (gcn_layer->comm_type != TP) return create_output_matrix_size(X, new_n, 0);
    int buffer_size = get_buffer_space(gcn_layer->comm);
    return create_output_matrix_size(X, new_n, buffer_size);
}