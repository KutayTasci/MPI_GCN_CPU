//
// Created by serdar on 12/21/24.
//
#ifndef MPI_GCN_CPU_NODESAMPLING_H
#define MPI_GCN_CPU_NODESAMPLING_H

#include "typedef.h"
#include <stdbool.h>


typedef struct {
    sendBuffer *sendBuffer;
    recvBuffer *recvBuffer;
    int msgSendCount;
    int msgRecvCount;
    int n;
    int total_m; // buffer size
    double p;
    SparseMat *A;
    SparseMat *A_T;
} NodeSamplingComm;


void reallocNodeSamplingComm(NodeSamplingComm *comm, int new_m);

NodeSamplingComm *nodeSamplingCommInit(SparseMat *A, SparseMat *A_T, double p, int feature_size);


#endif //MPI_GCN_CPU_NODESAMPLING_H
