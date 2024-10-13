#include "../includes/sparseMat.h"
#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

void aggregate_gemm_overlap(gcnLayer *layer, Matrix *X, Matrix *Y, Matrix *B, Matrix *C, int step) {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int i, j, k;

    int m = Y->m;
    int n = Y->n;
    int f = B->n;
    double val;
    sendBuffer *bufferS;
    recvBuffer *bufferR;
    int msgSendCount, msgRecvCount;
    int *buffMap;
    SparseMat *A;
    if (step == FORWARD) {
        A = layer->adjacency_T;
        bufferS = layer->sendBuffer;
        bufferR = layer->recvBuffer;
        buffMap = layer->adjacency_T->jc_mapped;
        msgSendCount = layer->msgSendCount;
        msgRecvCount = layer->msgRecvCount;
    } else if (step == BACKWARD) {
        A = layer->adjacency;
        bufferS = layer->sendBuffer_backward;
        bufferR = layer->recvBuffer_backward;
        buffMap = layer->adjacency->jc_mapped;
        msgSendCount = layer->msgSendCount_b;
        msgRecvCount = layer->msgRecvCount_b;
    } else {
        printf("Aggregate step can only execute in FORWARD or BACKWARD mode.\n");
        return;
    }


    int range, base, rRange;

    //Fill send table
    int ind, ind_c;

    MPI_Request *request_send = (MPI_Request *) malloc((msgSendCount) * sizeof(MPI_Request));
    MPI_Request *request_recv = (MPI_Request *) malloc((msgRecvCount) * sizeof(MPI_Request));
    //MPI_Status* status_list_r = (MPI_Status*) malloc((msgRecvCount) * sizeof(MPI_Status));
    //MPI_Status* status_list_s = (MPI_Status*) malloc((msgSendCount) * sizeof(MPI_Status));

    ind_c = 0;
    initRecvBufferSpace(bufferR);
    for (i = 0; i < msgRecvCount; i++) {
        k = bufferR->list[i];
        range = bufferR->pid_map[k + 1] - bufferR->pid_map[k];
        base = bufferR->pid_map[k];
        MPI_Irecv(&(bufferR->data[base][0]),
                  range * bufferR->feature_size,
                  MPI_DOUBLE,
                  k,
                  AGG_COMM + k,
                  MPI_COMM_WORLD,
                  &(request_recv[ind_c]));
        ind_c++;


    }
    ind_c = 0;
    initSendBufferSpace(bufferS);
    for (i = 0; i < msgSendCount; i++) {
        k = bufferS->list[i];
        range = bufferS->pid_map[k + 1] - bufferS->pid_map[k];
        base = bufferS->pid_map[k];
        for (j = 0; j < range; j++) {
            ind = bufferS->vertices_local[base + j];
            memcpy(bufferS->data[base + j], X->entries[ind], sizeof(double) * bufferR->feature_size);
        }
        MPI_Isend(&(bufferS->data[base][0]),
                  range * bufferS->feature_size,
                  MPI_DOUBLE,
                  k,
                  AGG_COMM + world_rank,
                  MPI_COMM_WORLD,
                  &(request_send[ind_c]));
        ind_c++;


    }
    memset(Y->entries[0], 0,
           Y->m * Y->n * sizeof(double));


    //Local comp will be here
//    int vertice;
    for (i = A->proc_map[world_rank]; i < A->proc_map[world_rank + 1]; i++) {
//        vertice = A->l2gMap[i];
        int target_node = A->ic[i].v_id;
        for (j = A->ic[i].indptr; j < A->ic[i + 1].indptr; j++) {
            for (k = 0; k < Y->n; k++) {
                Y->entries[target_node][k] += A->val_c[j] * X->entries[buffMap[j]][k];
            }
        }
    }

    for (i = 0; i < m; i++) {
        for (j = 0; j < f; j++) {
            C->entries[i][j] = 0;
        }
        for (k = 0; k < n; k++) {
            val = Y->entries[i][k];
            for (j = 0; j < f; j++) {
                C->entries[i][j] += val * B->entries[k][j];
            }
        }
    }

    MPI_Waitall(msgRecvCount, request_recv, MPI_STATUS_IGNORE);
    MPI_Waitall(msgSendCount, request_send, MPI_STATUS_IGNORE);
    //Computation and communication
    //MPI_Status status;

    memset(Y->entries[0], 0,
           Y->m * Y->n * sizeof(double));

    int part;
    for (int t = 0; t < msgRecvCount; t++) {
        part = bufferR->list[t];
        for (i = A->proc_map[part]; i < A->proc_map[part + 1]; i++) {
            int target_node = A->ic[i].v_id;
            for (j = A->ic[i].indptr; j < A->ic[i + 1].indptr; j++) {
                int tmp = A->jc_mapped[j];
                for (k = 0; k < Y->n; k++) {
                    Y->entries[target_node][k] += A->val_c[j] * bufferR->data[tmp][k];
                }
            }
        }
    }

    for (i = 0; i < m; i++) {
        for (k = 0; k < n; k++) {
            val = Y->entries[i][k];
            for (j = 0; j < f; j++) {
                C->entries[i][j] += val * B->entries[k][j];
            }
        }
    }

    free(request_send);
    free(request_recv);

    recvBufferSpaceFree(bufferR);
    sendBufferSpaceFree(bufferS);

}

void aggregate(gcnLayer *layer, Matrix *X, Matrix *Y, int step) {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int i, j, k;
    int msgSendCount;
    int msgRecvCount;
    sendBuffer *bufferS;
    recvBuffer *bufferR;
    SparseMat *A;
    //int* buffMap;
    if (step == FORWARD) {
        A = layer->adjacency_T;
        bufferS = layer->sendBuffer;
        bufferR = layer->recvBuffer;
        //buffMap = layer->adjacency_T->jc_mapped;
        msgSendCount = layer->msgSendCount;
        msgRecvCount = layer->msgRecvCount;
    } else if (step == BACKWARD) {
        A = layer->adjacency;
        bufferS = layer->sendBuffer_backward;
        bufferR = layer->recvBuffer_backward;
        //buffMap = layer->adjacency->jc_mapped;
        msgSendCount = layer->msgSendCount_b;
        msgRecvCount = layer->msgRecvCount_b;
    } else {
        printf("Aggregate step can only execute in FORWARD or BACKWARD mode.\n");
        return;
    }

    MPI_Request *request_recv = (MPI_Request *) malloc((msgRecvCount) * sizeof(MPI_Request));
    //MPI_Status* status_list_r = (MPI_Status*) malloc((msgRecvCount) * sizeof(MPI_Status));



    //Fill send table
    int ind, ind_c;
    MPI_Request request;
    int range;
    int base;


    initSendBufferSpace(bufferS);
    for (i = 0; i < msgSendCount; i++) {
        k = bufferS->list[i];
        range = bufferS->pid_map[k + 1] - bufferS->pid_map[k];
        base = bufferS->pid_map[k];
        for (j = 0; j < range; j++) {
            ind = bufferS->vertices_local[base + j];
            memcpy(bufferS->data[base + j], X->entries[ind], sizeof(double) * bufferR->feature_size);
        }

    }


    initRecvBufferSpace(bufferR);
    for (i = 0; i < msgRecvCount; i++) {
        k = bufferR->list[i];
        range = bufferR->pid_map[k + 1] - bufferR->pid_map[k];
        base = bufferR->pid_map[k];

        MPI_Irecv(&(bufferR->data[base][0]),
                  range * bufferR->feature_size,
                  MPI_DOUBLE,
                  k,
                  AGG_COMM + k,
                  MPI_COMM_WORLD,
                  &(request_recv[i]));

    }


    for (i = 0; i < msgSendCount; i++) {
        k = bufferS->list[i];
        range = bufferS->pid_map[k + 1] - bufferS->pid_map[k];
        base = bufferS->pid_map[k];
        MPI_Send(&(bufferS->data[base][0]),
                 range * bufferS->feature_size,
                 MPI_DOUBLE,
                 k,
                 AGG_COMM + world_rank,
                 MPI_COMM_WORLD);

    }

    //MPI_Barrier(MPI_COMM_WORLD);
    MPI_Waitall(msgRecvCount, request_recv, MPI_STATUS_IGNORE);
    memset(Y->entries[0], 0,
           Y->m * Y->n * sizeof(double));


    // local computations
    int p = world_rank;
    for (i = A->proc_map[p]; i < A->proc_map[p + 1]; i++) {
        int target_node = A->ic[i].v_id;
        for (j = A->ic[i].indptr; j < A->ic[i + 1].indptr; j++) {
            int tmp = A->jc_mapped[j];
            for (k = 0; k < Y->n; k++) {
                Y->entries[target_node][k] += A->val_c[j] * X->entries[tmp][k];
            }
        }
    }

    // global computations
    for (int p_t = 0; p_t < msgRecvCount; p_t++) {
        p = bufferR->list[p_t];
        for (i = A->proc_map[p]; i < A->proc_map[p + 1]; i++) {
            int target_node = A->ic[i].v_id;
            for (j = A->ic[i].indptr; j < A->ic[i + 1].indptr; j++) {
                int tmp = A->jc_mapped[j];
                for (k = 0; k < Y->n; k++) {
                    Y->entries[target_node][k] += A->val_c[j] * bufferR->data[tmp][k];
                }
            }
        }
    }

    recvBufferSpaceFree(bufferR);
    sendBufferSpaceFree(bufferS);


    MPI_Request_free(&request);

}

void aggregate_csr(gcnLayer *layer, Matrix *X, Matrix *Y, int step) {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int i, j, k;
    sendBuffer *bufferS;
    recvBuffer *bufferR;
    SparseMat *A;
    int msgSendCount;
    int msgRecvCount;
    if (step == FORWARD) {
        A = layer->adjacency_T;
        bufferS = layer->sendBuffer;
        bufferR = layer->recvBuffer;
        msgSendCount = layer->msgSendCount;
        msgRecvCount = layer->msgRecvCount;
    } else if (step == BACKWARD) {
        A = layer->adjacency;
        bufferS = layer->sendBuffer_backward;
        bufferR = layer->recvBuffer_backward;
        msgSendCount = layer->msgSendCount_b;
        msgRecvCount = layer->msgRecvCount_b;
    } else {
        printf("Aggregate step can only execute in FORWARD or BACKWARD mode.\n");
        return;
    }

    int ind, ind_c;
    MPI_Request request;
    int range;
    int base;

    initSendBufferSpace(bufferS);
    for (i = 0; i < msgSendCount; i++) {
        k = bufferS->list[i];
        range = bufferS->pid_map[k + 1] - bufferS->pid_map[k];
        base = bufferS->pid_map[k];
        for (j = 0; j < range; j++) {
            ind = bufferS->vertices_local[base + j];
            memcpy(bufferS->data[base + j], X->entries[ind], sizeof(double) * bufferR->feature_size);
        }

    }

    MPI_Request *request_send = (MPI_Request *) malloc((msgSendCount) * sizeof(MPI_Request));
    MPI_Request *request_recv = (MPI_Request *) malloc((msgRecvCount) * sizeof(MPI_Request));
    //MPI_Status* status_list_r = (MPI_Status*) malloc((msgRecvCount) * sizeof(MPI_Status));
    //MPI_Status* status_list_s = (MPI_Status*) malloc((msgSendCount) * sizeof(MPI_Status));



    initRecvBufferSpace(bufferR);
    for (i = 0; i < msgRecvCount; i++) {
        k = bufferR->list[i];
        range = bufferR->pid_map[k + 1] - bufferR->pid_map[k];
        base = bufferR->pid_map[k];

        MPI_Irecv(&(bufferR->data[base][0]),
                  range * bufferR->feature_size,
                  MPI_DOUBLE,
                  k,
                  AGG_COMM + k,
                  MPI_COMM_WORLD,
                  &(request_recv[i]));

    }

    for (i = 0; i < msgSendCount; i++) {
        k = bufferS->list[i];
        range = bufferS->pid_map[k + 1] - bufferS->pid_map[k];
        base = bufferS->pid_map[k];
        MPI_Send(&(bufferS->data[base][0]),
                 range * bufferS->feature_size,
                 MPI_DOUBLE,
                 k,
                 AGG_COMM + world_rank,
                 MPI_COMM_WORLD);
        //&(request_send[i]));

    }
    memset(Y->entries[0], 0,
           Y->m * Y->n * sizeof(double));

    MPI_Waitall(msgRecvCount, request_recv, MPI_STATUS_IGNORE);
    MPI_Waitall(msgSendCount, request_send, MPI_STATUS_IGNORE);

    int vertice;
    for (i = 0; i < A->m; i++) {
        for (j = A->ia[i]; j < A->ia[i + 1]; j++) {
            int target_node = A->ja[j];
            int tmp = A->ja_mapped[j];
            if (A->inPart[target_node] == world_rank) {
                for (k = 0; k < Y->n; k++) {
                    Y->entries[i][k] += A->val[j] * X->entries[tmp][k];
                }
            } else {
                for (k = 0; k < Y->n; k++) {
                    Y->entries[i][k] += A->val[j] * bufferR->data[tmp][k];
                }
            }

        }

    }

    recvBufferSpaceFree(bufferR);
    sendBufferSpaceFree(bufferS);
    //MPI_Request_free(&request);

}

void aggregate_partial_cco(gcnLayer *layer, Matrix *X, Matrix *Y, int step) {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int i, j, k;
    int msgSendCount;
    int msgRecvCount;
    sendBuffer *bufferS;
    recvBuffer *bufferR;
    SparseMat *A;
    int *buffMap;
    if (step == FORWARD) {
        A = layer->adjacency_T;
        bufferS = layer->sendBuffer;
        bufferR = layer->recvBuffer;
        buffMap = layer->adjacency_T->jc_mapped;
        msgSendCount = layer->msgSendCount;
        msgRecvCount = layer->msgRecvCount;
    } else if (step == BACKWARD) {
        A = layer->adjacency;
        bufferS = layer->sendBuffer_backward;
        bufferR = layer->recvBuffer_backward;
        buffMap = layer->adjacency->jc_mapped;
        msgSendCount = layer->msgSendCount_b;
        msgRecvCount = layer->msgRecvCount_b;
    } else {
        printf("Aggregate step can only execute in FORWARD or BACKWARD mode.\n");
        return;
    }
    int range, base, rRange;;
    //Fill send table
    int ind, ind_c;

    MPI_Request *request_send = (MPI_Request *) malloc((msgSendCount) * sizeof(MPI_Request));
    MPI_Request *request_recv = (MPI_Request *) malloc((msgRecvCount) * sizeof(MPI_Request));
    //MPI_Status* status_list_r = (MPI_Status*) malloc((msgRecvCount) * sizeof(MPI_Status));
    //MPI_Status* status_list_s = (MPI_Status*) malloc((msgSendCount) * sizeof(MPI_Status));



    initRecvBufferSpace(bufferR);
    for (i = 0; i < msgRecvCount; i++) {
        k = bufferR->list[i];
        range = bufferR->pid_map[k + 1] - bufferR->pid_map[k];
        base = bufferR->pid_map[k];

        MPI_Irecv(&(bufferR->data[base][0]),
                  range * bufferR->feature_size,
                  MPI_DOUBLE,
                  k,
                  AGG_COMM + k,
                  MPI_COMM_WORLD,
                  &(request_recv[i]));

    }

    initSendBufferSpace(bufferS);
    for (i = 0; i < msgSendCount; i++) {
        k = bufferS->list[i];
        range = bufferS->pid_map[k + 1] - bufferS->pid_map[k];
        base = bufferS->pid_map[k];
        for (j = 0; j < range; j++) {
            ind = bufferS->vertices_local[base + j];
            memcpy(bufferS->data[base + j], X->entries[ind], sizeof(double) * bufferR->feature_size);
        }
        MPI_Isend(&(bufferS->data[base][0]),
                  range * bufferS->feature_size,
                  MPI_DOUBLE,
                  k,
                  AGG_COMM + world_rank,
                  MPI_COMM_WORLD,
                  &(request_send[i]));
    }
    memset(Y->entries[0], 0,
           Y->m * Y->n * sizeof(double));


    //Local comp will be here
    int vertice;
    for (i = A->proc_map[world_rank]; i < A->proc_map[world_rank + 1]; i++) {
//        vertice = A->l2gMap[i];
        int target_node = A->ic[i].v_id;
        for (j = A->ic[i].indptr; j < A->ic[i + 1].indptr; j++) {
            for (k = 0; k < Y->n; k++) {
                Y->entries[target_node][k] += A->val_c[j] * X->entries[buffMap[j]][k];
            }
        }
    }
    MPI_Waitall(msgRecvCount, request_recv, MPI_STATUS_IGNORE);
    MPI_Waitall(msgSendCount, request_send, MPI_STATUS_IGNORE);
    //Computation and communication
    //MPI_Status status;

    int part;
    for (int t = 0; t < msgRecvCount; t++) {
        part = bufferR->list[t];
        for (i = A->proc_map[part]; i < A->proc_map[part + 1]; i++) {
            int target_node = A->ic[i].v_id;
            for (j = A->ic[i].indptr; j < A->ic[i + 1].indptr; j++) {
                int tmp = A->jc_mapped[j];
                for (k = 0; k < Y->n; k++) {
                    Y->entries[target_node][k] += A->val_c[j] * bufferR->data[tmp][k];
                }
            }
        }
    }

    //free(request_send);
    //free(request_recv);

    recvBufferSpaceFree(bufferR);
    sendBufferSpaceFree(bufferS);

}

void aggregate_cco(gcnLayer *layer, Matrix *X, Matrix *Y, int step) {

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int i, j, k;
    int msgSendCount;
    int msgRecvCount;
    sendBuffer *bufferS;
    recvBuffer *bufferR;
    SparseMat *A;
    int *buffMap;
    if (step == FORWARD) {
        A = layer->adjacency_T;
        bufferS = layer->sendBuffer;
        bufferR = layer->recvBuffer;
        buffMap = layer->adjacency_T->jc_mapped;
        msgSendCount = layer->msgSendCount;
        msgRecvCount = layer->msgRecvCount;
    } else if (step == BACKWARD) {
        A = layer->adjacency;
        bufferS = layer->sendBuffer_backward;
        bufferR = layer->recvBuffer_backward;
        buffMap = layer->adjacency->jc_mapped;
        msgSendCount = layer->msgSendCount_b;
        msgRecvCount = layer->msgRecvCount_b;
    } else {
        printf("Aggregate step can only execute in FORWARD or BACKWARD mode.\n");
        return;
    }
    int ind;
    int ind_c;
    int range, base, rRange;


    MPI_Request *request_send = (MPI_Request *) malloc((msgSendCount) * sizeof(MPI_Request));
    MPI_Request *request_recv = (MPI_Request *) malloc((msgRecvCount) * sizeof(MPI_Request));


    initRecvBufferSpace(bufferR);
    for (i = 0; i < msgRecvCount; i++) {
        k = bufferR->list[i];
        range = bufferR->pid_map[k + 1] - bufferR->pid_map[k];
        base = bufferR->pid_map[k];

        MPI_Irecv(&(bufferR->data[base][0]),
                  range * bufferR->feature_size,
                  MPI_DOUBLE,
                  k,
                  AGG_COMM + k,
                  MPI_COMM_WORLD,
                  &(request_recv[i]));

    }

    initSendBufferSpace(bufferS);
    for (i = 0; i < msgSendCount; i++) {
        k = bufferS->list[i];
        range = bufferS->pid_map[k + 1] - bufferS->pid_map[k];
        base = bufferS->pid_map[k];
        for (j = 0; j < range; j++) {
            ind = bufferS->vertices_local[base + j];
            memcpy(bufferS->data[base + j], X->entries[ind], sizeof(double) * bufferR->feature_size);
        }
        MPI_Isend(&(bufferS->data[base][0]),
                  range * bufferS->feature_size,
                  MPI_DOUBLE,
                  k,
                  AGG_COMM + world_rank,
                  MPI_COMM_WORLD,
                  &(request_send[i]));
    }

    memset(Y->entries[0], 0,
           Y->m * Y->n * sizeof(double));

    MPI_Status status;
    int completed[msgRecvCount];
    int ready;
    //memset(completed, 0, sizeof(completed));

    //Local comp will be here
    for (i = A->proc_map[world_rank]; i < A->proc_map[world_rank + 1]; i++) {
        int vertice = A->l2gMap[i];
        int target_node = A->ic[i].v_id;
        for (j = A->ic[i].indptr; j < A->ic[i + 1].indptr; j++) {
            for (k = 0; k < Y->n; k++) {
                Y->entries[target_node][k] += A->val_c[j] * X->entries[buffMap[j]][k];
            }
        }
    }
    //MPI_Testall(msgSendCount, request_send, &ready, MPI_STATUSES_IGNORE);


    //Computation and communication
    int part;
    //MPI_Waitall(msgSendCount, request_send, MPI_STATUS_IGNORE);
    for (int t = 0; t < msgRecvCount; t++) {
        MPI_Waitany(msgRecvCount, request_recv, completed, &status);
        part = status.MPI_SOURCE;
        for (i = A->proc_map[part]; i < A->proc_map[part + 1]; i++) {
            int target_node = A->ic[i].v_id;
            for (j = A->ic[i].indptr; j < A->ic[i + 1].indptr; j++) {
                int tmp = A->jc_mapped[j];
                for (k = 0; k < Y->n; k++) {
                    Y->entries[target_node][k] += A->val_c[j] * bufferR->data[tmp][k];
                }
            }
        }
    }
    //printf(" ---- Proc %d is finished ----\n", world_rank);
    MPI_Waitall(msgSendCount, request_send, MPI_STATUS_IGNORE);

    //free(request_send);
    //free(request_recv);


    recvBufferSpaceFree(bufferR);
    sendBufferSpaceFree(bufferS);

}

void aggregate_csc(gcnLayer *layer, Matrix *X, Matrix *Y, int step, bool eval) {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int i, j, k;
    int msgSendCount;
    int msgRecvCount;
    sendBuffer *bufferS;
    recvBuffer *bufferR;
    SparseMat *A;
    int *buffMap;
    if (step == FORWARD) {
        A = layer->adjacency_T;
        bufferS = layer->sendBuffer;
        bufferR = layer->recvBuffer;
        buffMap = layer->adjacency_T->jc_mapped;
        msgSendCount = layer->msgSendCount;
        msgRecvCount = layer->msgRecvCount;
    } else if (step == BACKWARD) {
        A = layer->adjacency;
        bufferS = layer->sendBuffer_backward;
        bufferR = layer->recvBuffer_backward;
        buffMap = layer->adjacency->jc_mapped;
        msgSendCount = layer->msgSendCount_b;
        msgRecvCount = layer->msgRecvCount_b;
    } else {
        printf("Aggregate step can only execute in FORWARD or BACKWARD mode.\n");
        return;
    }
    int range, base, rRange;;
    //Fill send table
    int ind, ind_c;

    //MPI_Request* request_send = (MPI_Request*) malloc((msgSendCount) * sizeof(MPI_Request));
    MPI_Request *request_recv = (MPI_Request *) malloc((msgRecvCount) * sizeof(MPI_Request));
    //MPI_Status* status_list_r = (MPI_Status*) malloc((msgRecvCount) * sizeof(MPI_Status));
    //MPI_Status* status_list_s = (MPI_Status*) malloc((msgSendCount) * sizeof(MPI_Status));



    initRecvBufferSpace(bufferR);
    for (i = 0; i < msgRecvCount; i++) {
        k = bufferR->list[i];
        range = bufferR->pid_map[k + 1] - bufferR->pid_map[k];
        base = bufferR->pid_map[k];

        MPI_Irecv(&(bufferR->data[base][0]),
                  range * bufferR->feature_size,
                  MPI_DOUBLE,
                  k,
                  AGG_COMM + k,
                  MPI_COMM_WORLD,
                  &(request_recv[i]));

    }

    initSendBufferSpace(bufferS);
    for (i = 0; i < msgSendCount; i++) {
        k = bufferS->list[i];
        range = bufferS->pid_map[k + 1] - bufferS->pid_map[k];
        base = bufferS->pid_map[k];
        for (j = 0; j < range; j++) {
            ind = bufferS->vertices_local[base + j];
            bool mask_factor = layer->mask[ind] ^ eval;
//            memcpy(bufferS->data[base + j], X->entries[ind], sizeof(double) * bufferR->feature_size);
            if (mask_factor) {
                memcpy(bufferS->data[base + j], X->entries[ind], sizeof(double) * bufferR->feature_size);
            } else {
                memset(bufferS->data[base + j], 0, sizeof(double) * bufferR->feature_size);
            }
        }
        MPI_Send(&(bufferS->data[base][0]),
                 range * bufferS->feature_size,
                 MPI_DOUBLE,
                 k,
                 AGG_COMM + world_rank,
                 MPI_COMM_WORLD);
        //&(request_send[i]));
    }
    memset(Y->entries[0], 0,
           Y->m * Y->n * sizeof(double));


    //Local comp will be here
    int vertice;
    for (i = A->proc_map[world_rank]; i < A->proc_map[world_rank + 1]; i++) {
//        vertice = A->l2gMap[i];
        int target_node = A->ic[i].v_id;
        for (j = A->ic[i].indptr; j < A->ic[i + 1].indptr; j++) {
            for (k = 0; k < Y->n; k++) {
                Y->entries[target_node][k] += A->val_c[j] * X->entries[buffMap[j]][k];
            }
        }
    }
    MPI_Waitall(msgRecvCount, request_recv, MPI_STATUS_IGNORE);
    //MPI_Waitall(msgSendCount, request_send, MPI_STATUS_IGNORE);
    //Computation and communication
    //MPI_Status status;

    int part;
    for (int t = 0; t < msgRecvCount; t++) {
        part = bufferR->list[t];
        for (i = A->proc_map[part]; i < A->proc_map[part + 1]; i++) {
            int target_node = A->ic[i].v_id;
            for (j = A->ic[i].indptr; j < A->ic[i + 1].indptr; j++) {
                int tmp = A->jc_mapped[j];
                for (k = 0; k < Y->n; k++) {
                    Y->entries[target_node][k] += A->val_c[j] * bufferR->data[tmp][k];
                }
            }
        }
    }

    //free(request_send);
    //free(request_recv);

    recvBufferSpaceFree(bufferR);
    sendBufferSpaceFree(bufferS);
}

void aggregate_cco_csc(gcnLayer *layer, Matrix *X, Matrix *Y, int step) {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int i, j, k;
    int msgSendCount;
    int msgRecvCount;
    sendBuffer *bufferS;
    recvBuffer *bufferR;
    SparseMat *A;
    int *buffMap;
    if (step == FORWARD) {
        A = layer->adjacency_T;
        bufferS = layer->sendBuffer;
        bufferR = layer->recvBuffer;
        buffMap = layer->adjacency_T->jc_mapped;
        msgSendCount = layer->msgSendCount;
        msgRecvCount = layer->msgRecvCount;
    } else if (step == BACKWARD) {
        A = layer->adjacency;
        bufferS = layer->sendBuffer_backward;
        bufferR = layer->recvBuffer_backward;
        buffMap = layer->adjacency->jc_mapped;
        msgSendCount = layer->msgSendCount_b;
        msgRecvCount = layer->msgRecvCount_b;
    } else {
        printf("Aggregate step can only execute in FORWARD or BACKWARD mode.\n");
        return;
    }
    int ind;
    int ind_c;
    int range, base, rRange;


    //MPI_Request* request_send = (MPI_Request*) malloc((msgSendCount) * sizeof(MPI_Request));
    MPI_Request *request_recv = (MPI_Request *) malloc((msgRecvCount) * sizeof(MPI_Request));


    initRecvBufferSpace(bufferR);
    for (i = 0; i < msgRecvCount; i++) {
        k = bufferR->list[i];
        range = bufferR->pid_map[k + 1] - bufferR->pid_map[k];
        base = bufferR->pid_map[k];

        MPI_Irecv(&(bufferR->data[base][0]),
                  range * bufferR->feature_size,
                  MPI_DOUBLE,
                  k,
                  AGG_COMM + k,
                  MPI_COMM_WORLD,
                  &(request_recv[i]));

    }

    initSendBufferSpace(bufferS);
    for (i = 0; i < msgSendCount; i++) {
        k = bufferS->list[i];
        range = bufferS->pid_map[k + 1] - bufferS->pid_map[k];
        base = bufferS->pid_map[k];
        for (j = 0; j < range; j++) {
            ind = bufferS->vertices_local[base + j];
            memcpy(bufferS->data[base + j], X->entries[ind], sizeof(double) * bufferR->feature_size);
        }
        MPI_Send(&(bufferS->data[base][0]),
                 range * bufferS->feature_size,
                 MPI_DOUBLE,
                 k,
                 AGG_COMM + world_rank,
                 MPI_COMM_WORLD);
        //&(request_send[i]));
    }

    memset(Y->entries[0], 0,
           Y->m * Y->n * sizeof(double));

    MPI_Status status;
    int completed[msgRecvCount];
    int ready;
    //memset(completed, 0, sizeof(completed));

    //Local comp will be here
    for (i = A->proc_map[world_rank]; i < A->proc_map[world_rank + 1]; i++) {
//        int vertice = A->l2gMap[i];
        int target_node = A->ic[i].v_id;
        for (j = A->ic[i].indptr; j < A->ic[i + 1].indptr; j++) {
            for (k = 0; k < Y->n; k++) {
                Y->entries[target_node][k] += A->val_c[j] * X->entries[buffMap[j]][k];
            }
        }
    }
    //MPI_Testall(msgSendCount, request_send, &ready, MPI_STATUSES_IGNORE);


    //Computation and communication
    int part;
    //MPI_Waitall(msgSendCount, request_send, MPI_STATUS_IGNORE);
    for (int t = 0; t < msgRecvCount; t++) {
        MPI_Waitany(msgRecvCount, request_recv, completed, &status);
        part = status.MPI_SOURCE;
        for (i = A->proc_map[part]; i < A->proc_map[part + 1]; i++) {
            int target_node = A->ic[i].v_id;
            for (j = A->ic[i].indptr; j < A->ic[i + 1].indptr; j++) {
                int tmp = A->jc_mapped[j];
                for (k = 0; k < Y->n; k++) {
                    Y->entries[target_node][k] += A->val_c[j] * bufferR->data[tmp][k];
                }
            }
        }
    }
    //printf(" ---- Proc %d is finished ----\n", world_rank);
    //MPI_Waitall(msgSendCount, request_send, MPI_STATUS_IGNORE);

    //free(request_send);
    //free(request_recv);


    recvBufferSpaceFree(bufferR);
    sendBufferSpaceFree(bufferS);
}

void aggregate_cco_hybrid(gcnLayer *layer, Matrix *X, Matrix *Y, int step) {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int i, j, k;
    int msgSendCount;
    int msgRecvCount;
    sendBuffer *bufferS;
    recvBuffer *bufferR;
    SparseMat *A;
    int *buffMap;
    if (step == FORWARD) {
        A = layer->adjacency_T;
        bufferS = layer->sendBuffer;
        bufferR = layer->recvBuffer;
        buffMap = layer->adjacency_T->jc_mapped;
        msgSendCount = layer->msgSendCount;
        msgRecvCount = layer->msgRecvCount;
    } else if (step == BACKWARD) {
        A = layer->adjacency;
        bufferS = layer->sendBuffer_backward;
        bufferR = layer->recvBuffer_backward;
        buffMap = layer->adjacency->jc_mapped;
        msgSendCount = layer->msgSendCount_b;
        msgRecvCount = layer->msgRecvCount_b;
    } else {
        printf("Aggregate step can only execute in FORWARD or BACKWARD mode.\n");
        return;
    }
    int ind;
    int ind_c;
    int range, base, rRange;


    MPI_Request *request_send = (MPI_Request *) malloc((msgSendCount) * sizeof(MPI_Request));
    MPI_Request *request_recv = (MPI_Request *) malloc((msgRecvCount) * sizeof(MPI_Request));

    ind_c = 0;
    initRecvBufferSpace(bufferR);
    for (i = 0; i < world_size; i++) {
        if (i != world_rank) {
            range = bufferR->pid_map[i + 1] - bufferR->pid_map[i];
            base = bufferR->pid_map[i];
            if (range != 0) {
                MPI_Irecv(&(bufferR->data[base][0]),
                          range * bufferR->feature_size,
                          MPI_DOUBLE,
                          i,
                          AGG_COMM + i,
                          MPI_COMM_WORLD,
                          &request_recv[ind_c]);
                ind_c++;
            }
        }
    }


    initSendBufferSpace(bufferS);
    ind_c = 0;
    for (i = 0; i < world_size; i++) {
        range = bufferS->pid_map[i + 1] - bufferS->pid_map[i];
        base = bufferS->pid_map[i];
        if (i != world_rank) {
            if (range != 0) {
                for (j = 0; j < range; j++) {
                    ind = bufferS->vertices_local[base + j];
                    memcpy(bufferS->data[base + j], X->entries[ind], sizeof(double) * bufferR->feature_size);
                }

                MPI_Isend(&(bufferS->data[base][0]),
                          range * bufferS->feature_size,
                          MPI_DOUBLE,
                          i,
                          AGG_COMM + world_rank,
                          MPI_COMM_WORLD,
                          &(request_send[ind_c]));
                ind_c++;
            }
        }
    }

    memset(Y->entries[0], 0,
           Y->m * Y->n * sizeof(double));

    MPI_Status status;
    int completed[msgRecvCount];
    memset(completed, 0, sizeof(completed));

    //Local comp will be here
    for (i = A->proc_map[world_rank]; i < A->proc_map[world_rank + 1]; i++) {
//        int vertice = A->l2gMap[i];
        int target_node = A->ic[i].v_id;
        for (j = A->ic[i].indptr; j < A->ic[i + 1].indptr; j++) {
            for (k = 0; k < Y->n; k++) {
                Y->entries[target_node][k] += A->val_c[j] * X->entries[buffMap[j]][k];
            }
        }
    }


    //Computation and communication
    int part, vertice;
    for (int t = 0; t < msgRecvCount; t++) {
        MPI_Waitany(msgRecvCount, request_recv, completed, &status);
        part = status.MPI_SOURCE;
        range = bufferR->pid_map[part + 1] - bufferR->pid_map[part];
        base = bufferR->pid_map[part];
        for (i = 0; i < range; i++) {
            vertice = bufferR->vertices[base + i];
            for (j = A->jb[vertice]; j < A->jb[vertice + 1]; j++) {
                int target_node = A->ib[j];
                for (k = 0; k < Y->n; k++) {
                    Y->entries[target_node][k] += A->val_b[j] * bufferR->data[base + i][k];
                }
            }
        }
    }
    //printf(" ---- Proc %d is finished ----\n", world_rank);
    MPI_Waitall(msgSendCount, request_send, MPI_STATUS_IGNORE);
    //printf(" ---- Proc %d is finished ----\n", world_rank);
    MPI_Waitall(msgSendCount, request_send, MPI_STATUS_IGNORE);

    free(request_send);
    free(request_recv);


    recvBufferSpaceFree(bufferR);
    sendBufferSpaceFree(bufferS);

}

void aggregate_no_comp(gcnLayer *layer, Matrix *X, Matrix *Y, int step) {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int i, j, k;
    int msgSendCount;
    int msgRecvCount;
    sendBuffer *bufferS;
    recvBuffer *bufferR;
    //SparseMat* A;
    //int* buffMap;
    if (step == FORWARD) {
        //A = layer->adjacency_T;
        bufferS = layer->sendBuffer;
        bufferR = layer->recvBuffer;
        //buffMap = layer->adjacency_T->jc_mapped;
        msgSendCount = layer->msgSendCount;
        msgRecvCount = layer->msgRecvCount;
    } else if (step == BACKWARD) {
        //A = layer->adjacency;
        bufferS = layer->sendBuffer_backward;
        bufferR = layer->recvBuffer_backward;
        //buffMap = layer->adjacency->jc_mapped;
        msgSendCount = layer->msgSendCount_b;
        msgRecvCount = layer->msgRecvCount_b;
    } else {
        printf("Aggregate step can only execute in FORWARD or BACKWARD mode.\n");
        return;
    }

    MPI_Request *request_recv = (MPI_Request *) malloc((msgRecvCount) * sizeof(MPI_Request));
    //MPI_Status* status_list_r = (MPI_Status*) malloc((msgRecvCount) * sizeof(MPI_Status));



    //Fill send table

    int ind, ind_c;
    MPI_Request request;
    int range;
    int base;


    initSendBufferSpace(bufferS);


    initRecvBufferSpace(bufferR);
    for (i = 0; i < msgRecvCount; i++) {
        k = bufferR->list[i];
        range = bufferR->pid_map[k + 1] - bufferR->pid_map[k];
        base = bufferR->pid_map[k];

        MPI_Irecv(&(bufferR->data[base][0]),
                  range * bufferR->feature_size,
                  MPI_DOUBLE,
                  k,
                  AGG_COMM + k,
                  MPI_COMM_WORLD,
                  &(request_recv[i]));

    }


    for (i = 0; i < msgSendCount; i++) {
        k = bufferS->list[i];
        range = bufferS->pid_map[k + 1] - bufferS->pid_map[k];
        base = bufferS->pid_map[k];
        MPI_Send(&(bufferS->data[base][0]),
                 range * bufferS->feature_size,
                 MPI_DOUBLE,
                 k,
                 AGG_COMM + world_rank,
                 MPI_COMM_WORLD);

    }

    //MPI_Barrier(MPI_COMM_WORLD);
    MPI_Waitall(msgRecvCount, request_recv, MPI_STATUS_IGNORE);
    memset(Y->entries[0], 0,
           Y->m * Y->n * sizeof(double));


    recvBufferSpaceFree(bufferR);
    sendBufferSpaceFree(bufferS);


    //MPI_Request_free(&request);

}

void aggregate_no_comm(gcnLayer *layer, Matrix *X, Matrix *Y, int step) {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int i, j, k;
    int msgSendCount;
    int msgRecvCount;
    sendBuffer *bufferS;
    recvBuffer *bufferR;
    SparseMat *A;
    //int* buffMap;
    if (step == FORWARD) {
        A = layer->adjacency_T;
        bufferS = layer->sendBuffer;
        bufferR = layer->recvBuffer;
        //buffMap = layer->adjacency_T->jc_mapped;
        msgSendCount = layer->msgSendCount;
        msgRecvCount = layer->msgRecvCount;
    } else if (step == BACKWARD) {
        A = layer->adjacency;
        bufferS = layer->sendBuffer_backward;
        bufferR = layer->recvBuffer_backward;
        //buffMap = layer->adjacency->jc_mapped;
        msgSendCount = layer->msgSendCount_b;
        msgRecvCount = layer->msgRecvCount_b;
    } else {
        printf("Aggregate step can only execute in FORWARD or BACKWARD mode.\n");
        return;
    }

    MPI_Request *request_recv = (MPI_Request *) malloc((msgRecvCount) * sizeof(MPI_Request));
    //MPI_Status* status_list_r = (MPI_Status*) malloc((msgRecvCount) * sizeof(MPI_Status));



    //Fill send table

    int ind, ind_c;
    MPI_Request request;
    int range;
    int base;


    initSendBufferSpace(bufferS);
    for (i = 0; i < msgSendCount; i++) {
        k = bufferS->list[i];
        range = bufferS->pid_map[k + 1] - bufferS->pid_map[k];
        base = bufferS->pid_map[k];
        for (j = 0; j < range; j++) {
            ind = bufferS->vertices_local[base + j];
            memcpy(bufferS->data[base + j], X->entries[ind], sizeof(double) * bufferR->feature_size);
        }

    }
    initRecvBufferSpace(bufferR);
    memset(Y->entries[0], 0,
           Y->m * Y->n * sizeof(double));


    int p = world_rank;
    for (i = A->proc_map[p]; i < A->proc_map[p + 1]; i++) {
        int target_node = A->ic[i].v_id;
        for (j = A->ic[i].indptr; j < A->ic[i + 1].indptr; j++) {
            int tmp = A->jc_mapped[j];
            for (k = 0; k < Y->n; k++) {
                Y->entries[target_node][k] += A->val_c[j] * X->entries[tmp][k];
            }
        }
    }


    for (int p_t = 0; p_t < msgRecvCount; p_t++) {
        p = bufferR->list[p_t];

        for (i = A->proc_map[p]; i < A->proc_map[p + 1]; i++) {
            int target_node = A->ic[i].v_id;
            for (j = A->ic[i].indptr; j < A->ic[i + 1].indptr; j++) {
                int tmp = A->jc_mapped[j];
                for (k = 0; k < Y->n; k++) {
                    Y->entries[target_node][k] += A->val_c[j] * bufferR->data[tmp][k];
                }
            }
        }
    }


    sendBufferSpaceFree(bufferS);
    recvBufferSpaceFree(bufferR);

    //MPI_Request_free(&request);

}


