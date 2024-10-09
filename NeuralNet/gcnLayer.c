#include "../includes/gcnLayer.h"
#include "../includes/sparseMat.h"
#include "../includes/matrix.h"
#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <sys/sysinfo.h>

int MODE = 0;

/*
Function is ready to be intergrated into sparse_Mat struct
Then use implement aggregation function
Don't forget memory management
*/


gcnLayer *gcn_init(SparseMat *adj, SparseMat *adj_T, int size_f, int size_out) {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int i, temp;
    gcnLayer *layer = (gcnLayer *) malloc(sizeof(gcnLayer));
    layer->size_n = adj->m;
    layer->size_f = size_f;
    layer->size_output = size_out;
    layer->adjacency = adj;
    layer->adjacency_T = adj_T;

    sendTable *sTable = initSendTable(adj);
    layer->sendBuffer = initSendBuffer(sTable, adj->l2gMap, size_f);

    recvTable *rTable = initRecvTable(layer->sendBuffer, adj_T);
    layer->recvBuffer = initRecvBuffer(rTable, size_f);


    sendTableFree(sTable);
    recvTableFree(rTable);

    sendTable *sTable_b = initSendTable(adj_T);
    layer->sendBuffer_backward = initSendBuffer(sTable_b, adj_T->l2gMap, size_out);

    recvTable *rTable_b = initRecvTable(layer->sendBuffer_backward, adj);
    layer->recvBuffer_backward = initRecvBuffer(rTable_b, size_out);


    sendTableFree(sTable_b);
    recvTableFree(rTable_b);

    layer->recvBuffMap = (int *) malloc(sizeof(int) * adj_T->gm);
    memset(layer->recvBuffMap, -1, sizeof(int) * adj_T->gm);

    for (i = 0; i < adj_T->m; i++) {
        temp = adj_T->l2gMap[i];
        layer->recvBuffMap[temp] = i;
    }

    for (i = 0; i < layer->recvBuffer->recv_count; i++) {
        temp = layer->recvBuffer->vertices[i];
        layer->recvBuffMap[temp] = i;
    }

    layer->recvBuffMap_backward = (int *) malloc(sizeof(int) * adj->gm);
    memset(layer->recvBuffMap_backward, -1, sizeof(int) * adj->gm);


    for (i = 0; i < adj->m; i++) {
        temp = adj->l2gMap[i];
        layer->recvBuffMap_backward[temp] = i;
    }

    for (i = 0; i < layer->recvBuffer_backward->recv_count; i++) {
        temp = layer->recvBuffer_backward->vertices[i];
        layer->recvBuffMap_backward[temp] = i;
    }


    if (adj_T->init == 0) {
        for (i = 0; i < adj_T->nnz; i++) {
            temp = adj_T->ja[i];
            adj_T->ja_mapped[i] = layer->recvBuffMap[temp];
        }

        generate_parCSR(adj_T, layer->recvBuffMap, world_size, world_rank);
        adj_T->init = 1;
    }

    if (adj->init == 0) {
        for (i = 0; i < adj->nnz; i++) {
            temp = adj->ja[i];
            adj->ja_mapped[i] = layer->recvBuffMap_backward[temp];
        }
        generate_parCSR(adj, layer->recvBuffMap_backward, world_size, world_rank);
        adj->init = 1;
    }

    layer->msgSendCount = 0;
    layer->msgRecvCount = 0;
    for (i = 0; i < world_size; i++) {
        int range = layer->sendBuffer->pid_map[i + 1] - layer->sendBuffer->pid_map[i];
        int rRange = layer->recvBuffer->pid_map[i + 1] - layer->recvBuffer->pid_map[i];
        if (i != world_rank) {
            if (range != 0) {
                layer->msgSendCount++;
            }
            if (rRange != 0) {
                layer->msgRecvCount++;
            }
        }
    }

    layer->sendBuffer->list = (int *) malloc(sizeof(int) * layer->msgSendCount);
    layer->recvBuffer->list = (int *) malloc(sizeof(int) * layer->msgRecvCount);

    int ctr = 0, ctr_r = 0;
    for (i = 0; i < world_size; i++) {
        int range = layer->sendBuffer->pid_map[i + 1] - layer->sendBuffer->pid_map[i];
        int rRange = layer->recvBuffer->pid_map[i + 1] - layer->recvBuffer->pid_map[i];
        if (i != world_rank) {
            if (range != 0) {
                layer->sendBuffer->list[ctr] = i;
                ctr++;
            }
            if (rRange != 0) {
                layer->recvBuffer->list[ctr_r] = i;
                ctr_r++;
            }
        }
    }


    layer->msgSendCount_b = 0;
    layer->msgRecvCount_b = 0;
    for (i = 0; i < world_size; i++) {
        int range = layer->sendBuffer_backward->pid_map[i + 1] - layer->sendBuffer_backward->pid_map[i];
        int rRange = layer->recvBuffer_backward->pid_map[i + 1] - layer->recvBuffer_backward->pid_map[i];
        if (i != world_rank) {
            if (range != 0) {
                layer->msgSendCount_b++;
            }
            if (rRange != 0) {
                layer->msgRecvCount_b++;
            }
        }
    }

    layer->sendBuffer_backward->list = (int *) malloc(sizeof(int) * layer->msgSendCount_b);
    layer->recvBuffer_backward->list = (int *) malloc(sizeof(int) * layer->msgRecvCount_b);

    ctr = 0, ctr_r = 0;
    for (i = 0; i < world_size; i++) {
        int range = layer->sendBuffer_backward->pid_map[i + 1] - layer->sendBuffer_backward->pid_map[i];
        int rRange = layer->recvBuffer_backward->pid_map[i + 1] - layer->recvBuffer_backward->pid_map[i];
        if (i != world_rank) {
            if (range != 0) {
                layer->sendBuffer_backward->list[ctr] = i;
                ctr++;
            }
            if (rRange != 0) {
                layer->recvBuffer_backward->list[ctr_r] = i;
                ctr_r++;
            }
        }
    }

    if (adj->init == 0) {
        int local_send_volume = layer->sendBuffer->send_count;
        int local_recv_volume = layer->recvBuffer->recv_count;
        int local_send_volume_b = layer->sendBuffer_backward->send_count;
        int local_recv_volume_b = layer->recvBuffer_backward->recv_count;

        int total_send_volume = local_send_volume;
        int total_recv_volume = local_recv_volume;

        int local_msg_count_send = layer->msgSendCount;
        int local_msg_count_recv = layer->msgRecvCount;

        int max_send_volume, max_recv_volume;
        int max_msg_count_send, max_msg_count_recv;
        int total_send_volume_sum, total_recv_volume_sum;
        int total_msg_count_send_sum, total_msg_count_recv_sum;

        // Compute max and average send and receive communication volumes and message counts
        MPI_Reduce(&total_send_volume, &max_send_volume, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&total_recv_volume, &max_recv_volume, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_msg_count_send, &max_msg_count_send, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_msg_count_recv, &max_msg_count_recv, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

        MPI_Reduce(&total_send_volume, &total_send_volume_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&total_recv_volume, &total_recv_volume_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_msg_count_send, &total_msg_count_send_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_msg_count_recv, &total_msg_count_recv_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        int avg_send_volume = total_send_volume_sum / world_size;
        int avg_recv_volume = total_recv_volume_sum / world_size;
        int avg_msg_count_send = total_msg_count_send_sum / world_size;
        int avg_msg_count_recv = total_msg_count_recv_sum / world_size;

        if (world_rank == 0) {
            printf("Initialization Complete:\n");
            printf("Max Send Volume: %d\n", max_send_volume);
            printf("Average Send Volume: %d\n", avg_send_volume);
            printf("Max Receive Volume: %d\n", max_recv_volume);
            printf("Average Receive Volume: %d\n", avg_recv_volume);
            printf("Max Send Message Count: %d\n", max_msg_count_send);
            printf("Average Send Message Count: %d\n", avg_msg_count_send);
            printf("Max Receive Message Count: %d\n", max_msg_count_recv);
            printf("Average Receive Message Count: %d\n", avg_msg_count_recv);
        }
    }
    layer->output = (ParMatrix *) malloc(sizeof(ParMatrix));
    layer->output->gm = adj->gm;
    layer->output->gn = size_out;
    layer->output->store = adj->store;
    layer->output->l2gMap = adj->l2gMap;
    layer->output->inPart = adj->inPart;
    layer->output->mat = matrix_create(adj->m, size_out);

    if (world_rank == 0) {
        init_weights_random(layer, 10);
    } else {
        layer->weights = matrix_create(layer->size_f, layer->size_output);
    }
    MPI_Bcast(&(layer->weights->entries[0][0]), layer->weights->m * layer->weights->n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    layer->gradients = matrix_create(layer->size_f, layer->size_output);
    layer->m_weights = matrix_create(layer->size_f, layer->size_output);
    matrix_fill(layer->m_weights, 0);
    layer->v_weights = matrix_create(layer->size_f, layer->size_output);
    matrix_fill(layer->v_weights, 0);


    free(layer->recvBuffMap);
    free(layer->recvBuffMap_backward);


    return layer;
}

void setMode(int i) {
    MODE = i;
}

void gcn_forward(gcnLayer *layer) {

    Matrix *temp = matrix_create(layer->size_n, layer->size_f);

    switch (MODE) {
        case 0:
            aggregate_csr(layer, layer->input->mat, temp, FORWARD);
            break;
        case 1:
            aggregate(layer, layer->input->mat, temp, FORWARD);
            break;
        case 2:
            aggregate_cco(layer, layer->input->mat, temp, FORWARD);
            break;
        case 3:
            aggregate_partial_cco(layer, layer->input->mat, temp, FORWARD);
            break;
        case 4:
            aggregate_csc(layer, layer->input->mat, temp, FORWARD);
            break;
        case 5:
            aggregate_cco_csc(layer, layer->input->mat, temp, FORWARD);
            break;
        case 6:
            aggregate_cco_hybrid(layer, layer->input->mat, temp, FORWARD);
            break;
        case 7:
            aggregate_gemm_overlap(layer, layer->input->mat, temp, layer->weights, layer->output->mat, FORWARD);
            matrix_free(temp);
            return;
        default:
            printf("No aggregation mode exists.\n");
            printf("Modes exist for 1=>All-to-ALL blocking Cycle\n");
            printf("Modes exist for 2=>All-to-ALL Non-blocking Overlapping\n");
            printf("Modes exist for 3=>All-to-ALL Non-blocking non-Overlapping\n");
            exit(1);
    }
    GEMM(temp, layer->weights, layer->output->mat);
    matrix_free(temp);
    //TO DO
}

Matrix *gcn_backward(gcnLayer *layer, Matrix *out_error) {
    Matrix *temp = matrix_create(layer->size_n, layer->size_output);
    Matrix *out = matrix_create(layer->size_n, layer->size_f);

    switch (MODE) {
        case 0:
            aggregate_csr(layer, out_error, temp, BACKWARD);
            break;
        case 1:
            aggregate(layer, out_error, temp, BACKWARD);
            break;
        case 2:
            aggregate_cco(layer, out_error, temp, BACKWARD);
            break;
        case 3:
            aggregate_partial_cco(layer, out_error, temp, BACKWARD);
            break;
        case 4:
            aggregate_csc(layer, out_error, temp, BACKWARD);
            break;
        case 5:
            aggregate_cco_csc(layer, out_error, temp, BACKWARD);
            break;
        case 6:
            aggregate_cco_hybrid(layer, out_error, temp, BACKWARD);
            break;
        case 7:
            aggregate_partial_cco(layer, out_error, temp, BACKWARD);
            break;
        default:
            printf("No aggregation mode exists.\n");
            printf("Modes exist for 1=>All-to-ALL blocking Cycle\n");
            printf("Modes exist for 2=>All-to-ALL Non-blocking Overlapping\n");
            printf("Modes exist for 3=>All-to-ALL Non-blocking non-Overlapping\n");
            exit(1);
    }

    GEMM_NT(temp, layer->weights, out);
    GEMM_TN(layer->input->mat, temp, layer->gradients);
    matrix_free(temp);
    //printf("flag_4 \n");
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
    Matrix *temp = matrix_create(layer->gradients->m, layer->gradients->n);
    MPI_Allreduce(&(layer->gradients->entries[0][0]), &(temp->entries[0][0]),
                  layer->gradients->m * layer->gradients->n,
                  MPI_DOUBLE,
                  MPI_SUM,
                  MPI_COMM_WORLD);
    //printf("%lf \n", layer->gradients->entries[0][0]);
    matrix_scale(lr, temp);

    matrix_scale(beta1, layer->m_weights);
    Matrix *temp_m = matrix_scale_return((1 - beta1), temp);
    matrix_sum(layer->m_weights, temp_m, layer->m_weights);

    //matrix_free(temp_m);

    matrix_scale(beta2, layer->v_weights);

    matrix_multiply(temp, temp, temp_m);

    matrix_scale((1 - beta2), temp_m);
    matrix_sum(layer->v_weights, temp_m, layer->v_weights);

    matrix_free(temp_m);

    //bias correction can go here
    Matrix *m_dw_corr = matrix_scale_return(1 / (1 - pow(beta1, t + 1)), layer->m_weights);
    Matrix *v_dw_corr = matrix_scale_return(1 / (1 - pow(beta2, t + 1)), layer->v_weights);


    Matrix *tmp_sqrt = matrix_sqrt(v_dw_corr);
    matrix_addScalar(tmp_sqrt, epsilon);
    matrix_divide(m_dw_corr, tmp_sqrt, tmp_sqrt);
    matrix_scale(eta, tmp_sqrt);


    matrix_subtract(layer->weights, tmp_sqrt, layer->weights);


    //printf("%lf \n", layer->weights->entries[0][0]);
    matrix_free(temp);
    matrix_free(tmp_sqrt);
    matrix_free(m_dw_corr);
    matrix_free(v_dw_corr);
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
