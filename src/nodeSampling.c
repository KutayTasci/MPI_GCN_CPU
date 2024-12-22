//
// Created by serdar on 12/21/24.
//
#include "../includes/nodeSampling.h"
#include "../includes/basic.h"
#include <stdlib.h>
#include <mpi.h>

NodeSamplingComm *nodeSamplingCommInit(SparseMat *A, SparseMat *A_T, double p, int feature_size) {
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    NodeSamplingComm *comm = (NodeSamplingComm *) malloc(sizeof(NodeSamplingComm));
    // calculate average send buffer size
    sendTable *sTable = initSendTable(A);
    comm->sendBuffer->feature_size = A->n;

    comm->sendBuffer->send_count = 0;
    comm->msgSendCount = 0;
    for (int i = 0; i < world_size; i++) {
        comm->sendBuffer->send_count += sTable->send_count[i];
        if (sTable->send_count[i] > 0) {
            comm->msgSendCount++;
        }
    }
    comm->sendBuffer->data = (double **) malloc(sizeof(double *) * comm->msgSendCount);
    for (int i = 0; i < sTable->p_count; i++) {
        if (sTable->send_count[i] == 0) {
            continue;
        }
        int max_buffer_size = sTable->send_count[i] * p * 2;
        if (max_buffer_size > sTable->send_count[i]) {
            max_buffer_size = sTable->send_count[i];
        }
        comm->sendBuffer->data[i] = (double *) malloc(
                sizeof(double) * max_buffer_size * comm->sendBuffer->feature_size);
    }

    // calculate average recv buffer size
    recvTable *rTable = initRecvTable(comm->sendBuffer, A_T); // todo remove vertex mappings
    comm->recvBuffer = initRecvBuffer(rTable, feature_size);
    // calculate what messages to send by using csc and csr
    
}

