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
    int *recvIdxs; // local indexes
    int *recvBuffMap; // global indexes to local indexes
    int *cscR; // csc format only for the edges starting from the sampled nodes (row idxs)
    int *cscC; // column idxs (local)
    double *cscV; // values
    int *sendIdxs; // this is set to max of sendBuffer->vertices
} NodeSamplingComm;


void reallocNodeSamplingComm(NodeSamplingComm *comm, int new_m);

NodeSamplingComm *nodeSamplingCommInit(SparseMat *A, SparseMat *A_T, double p, int feature_size);

void sampleNodes(NodeSamplingComm *comm, int step, ParMatrix *X);

int bns(int base_idx, int size, int *recvIdxs, double p);

#endif //MPI_GCN_CPU_NODESAMPLING_H
