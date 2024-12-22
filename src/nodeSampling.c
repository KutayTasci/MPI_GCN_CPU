//
// Created by serdar on 12/21/24.
//
#include "../includes/nodeSampling.h"
#include "../includes/basic.h"
#include <stdlib.h>
#include <mpi.h>
#include <string.h>

NodeSamplingComm *nodeSamplingCommInit(SparseMat *A, SparseMat *A_T, double p, int feature_size) {
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    NodeSamplingComm *comm = (NodeSamplingComm *) malloc(sizeof(NodeSamplingComm));
    // calculate average send buffer size
    sendTable *sTable = initSendTable(A);
    initSendBuffer(sTable, A->l2gMap, feature_size);

    // calculate average recv buffer size
    recvTable *rTable = initRecvTable(comm->sendBuffer, A_T); // todo remove vertex mappings
    comm->recvBuffer = initRecvBuffer(rTable, feature_size);
    comm->recvBuffer->count = (int *) malloc(sizeof(int) * world_size);
    memset(comm->recvBuffer->count, 0, sizeof(int) * world_size);
    // calculate what messages to send by using csc and csr

    return comm;
}

void sampleNodes(NodeSamplingComm *comm, int step) {
    // send the global indices of the nodes to be sampled
    // then send the features
    // update comm->recvBuffer->count everytime do not update pid_map
    for (int i = 0; i < comm->msgSendCount; i++) {
        int proc_id = comm->sendBuffer->list[i];
        // select random vertices to be sampled
    }
}
