//
// Created by serdar on 12/21/24.
//
#include "../includes/nodeSampling.h"
#include "../includes/basic.h"
#include <stdlib.h>
#include <mpi.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

void setSamplingProbability(NodeSamplingComm *comm){
    int tot_boundary = 0;
    int max_boundary = 0;
    int noOfPartition = 8;
    int max_k = 1;
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    for(int i = 0; i < size; i++) {
        int count = comm->boundaryCounts[i];
        max_boundary = (count > max_boundary) ? count : max_boundary;
        tot_boundary += count;
    }

    if( tot_boundary > 0) {
        max_k  = tot_boundary / noOfPartition
    }
    if( max_k > (noOfPartition / 2 - noOfPartition / 16)) {
        max_k = (noOfPartition / 2 - noOfPartition / 16);
    }

    double tot_rec = 0.0;
    double max_rec = 0.0;
    double imbalance_rat = 20000.0; // big enough
    int best_k = 1;
    for(int k = 1; k <= max_k; k++) {
        for(int i = 0; i < size; i++) {
            int count = comm->boundaryCounts[i];
            double probability = max(0.1, 1 - count * k / tot_boundary);
            double rec_vol = math.ceil(count * probability);

            max_rec = (rec_vol > max_rec) ? rec_vol : max_rec;
            tot_rec += rec_vol;
        }
        double avg_rec = tot_rec / noOfPartition;

        if(avg_rec > 0) {
            imbalance_rat = max_rec / avg_rec;
            best_k = k;
        }
    }

    double max_prob = 0;
    for(int i = 0; i < size; i++) {
        int count = comm->boundaryCounts[i];
        double probability = max(0.1, 1 - count * best_k / tot_boundary);
        max_prob = max(max_prob, probability);
        comm->samplingProb[i] = probability;
    }
    double maxMultp =  1 / max_prob;
}

NodeSamplingComm *nodeSamplingCommInit(SparseMat *A, SparseMat *A_T, double p, int feature_size) {
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    NodeSamplingComm *comm = (NodeSamplingComm *) malloc(sizeof(NodeSamplingComm));
    // calculate average send buffer size
    sendTable *sTable = initSendTable(A);
    comm->sendBuffer = initSendBuffer(sTable, A->l2gMap, feature_size);

    // calculate average recv buffer size
    recvTable *rTable = initRecvTable(comm->sendBuffer, A_T); // todo remove vertex mappings
    comm->recvBuffer = initRecvBuffer(rTable, feature_size);
    comm->recvBuffMap = (int *) malloc(sizeof(int) * A_T->gm);
    memset(comm->recvBuffMap, -1, sizeof(int) * A_T->gm);
    for (int i = 0; i < A_T->m; i++) {
        int temp = A_T->l2gMap[i];
        comm->recvBuffMap[temp] = i; // global to local recv buffer mapping
    }
    // calculate what messages to send by using csc and csr
    for (int i = 0; i < comm->recvBuffer->recv_count; i++) {
        int temp = comm->recvBuffer->vertices[i];
        comm->recvBuffMap[temp] = i;
    }
    comm->msgSendCount = 0;
    comm->msgRecvCount = 0;
    for (int i = 0; i < world_size; i++) {
        int range = comm->sendBuffer->pid_map[i + 1] - comm->sendBuffer->pid_map[i];
        int rRange = comm->recvBuffer->pid_map[i + 1] - comm->recvBuffer->pid_map[i];
        if (i != world_rank) {
            if (range != 0) {
                comm->msgSendCount++;
            }
            if (rRange != 0) {
                comm->msgRecvCount++;
            }
        }
    }
    comm->sendBuffer->list = (int *) malloc(sizeof(int) * comm->msgSendCount);
    comm->recvBuffer->list = (int *) malloc(sizeof(int) * comm->msgRecvCount);

    int ctr = 0, ctr_r = 0;
    for (int i = 0; i < world_size; i++) {
        int range = comm->sendBuffer->pid_map[i + 1] - comm->sendBuffer->pid_map[i];
        int rRange = comm->recvBuffer->pid_map[i + 1] - comm->recvBuffer->pid_map[i];
        if (i != world_rank) {
            if (range != 0) {
                comm->sendBuffer->list[ctr] = i;
                ctr++;
            }
            if (rRange != 0) {
                comm->recvBuffer->list[ctr_r] = i;
                ctr_r++;
            }
        }
    }
    initSendBufferSpace(comm->sendBuffer);
    initRecvBufferSpace(comm->recvBuffer);
    comm->p = p;
    comm->cscR = (int *) malloc(sizeof(int) * (comm->recvBuffer->recv_count + 1)); // first act as a counter
    memset(comm->cscR, 0, sizeof(int) * (comm->recvBuffer->recv_count + 1));
    int v_j, part;
    for (int i = 0; i < A->m; i++) {
        for (int j = A->ia[i]; j < A->ia[i + 1]; j++) {
            v_j = A->ja[j];
            part = A->inPart[v_j];
            if (part != world_rank) {
                int local_idx = comm->recvBuffMap[v_j];
                comm->cscR[local_idx + 1]++;
            }
        }
    }
    for (int i = 1; i <= comm->recvBuffer->recv_count; i++) {
        comm->cscR[i] += comm->cscR[i - 1];
    }
    comm->cscC = (int *) malloc(sizeof(int) * comm->cscR[comm->recvBuffer->recv_count]);
    int *counter = (int *) malloc(sizeof(int) * comm->recvBuffer->recv_count);
    memset(counter, 0, sizeof(int) * comm->recvBuffer->recv_count);
    for (int i = 0; i < A->m; i++) {
        for (int j = A->ia[i]; j < A->ia[i + 1]; j++) {
            v_j = A->ja[j];
            part = A->inPart[v_j];
            if (part != world_rank) {
                int local_idx = comm->recvBuffMap[v_j];
                int idx = comm->cscR[local_idx] + counter[local_idx];
                comm->cscC[idx] = i; // save local index
                counter[local_idx]++;
            }
        }
    }
    free(counter);
    // init recvIdxs
    comm->recvIdxs = (int *) malloc(sizeof(int) * comm->recvBuffer->recv_count); // use pid map for proc idxs
    // init sendIdxs
    int max_vtx_cnt = 0;
    for (int i = 0; i < world_size; i++) {
        if (sTable->send_count[i] > max_vtx_cnt) {
            max_vtx_cnt = sTable->send_count[i];
        }
    }
    comm->sendIdxs = (int *) malloc(sizeof(int) * max_vtx_cnt);
    sendTableFree(sTable);
    recvTableFree(rTable);
    comm->A = A;
    comm->A_T = A_T;

    comm->boundaryCounts = (int *) malloc(sizeof(int) * world_size);
    memset(comm->boundaryCounts, 0, sizeof(int) * world_size);
    comm->boundaryCounts[world_rank] = comm->recvBuffer->recv_count;
    MPI_Allreduce(MPI_IN_PLACE, comm->boundaryCounts, world_size, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    setSamplingProbability(comm)
    return comm;
}

int bns(int base_idx, int size, int *recvIdxs, double p) {
    int recv_count = 0;
    for (int i = 0; i < size; i++) {
        bool add = rand() < p * RAND_MAX;
        if (add) {
            recvIdxs[recv_count++] = base_idx + i;
        }
    }
    if (recv_count < size) {
        recvIdxs[recv_count] = -1;
    } // otherwise it is full
    return recv_count;
}

int gbr(int base_idx, int size)

#define set_seed(rank) srand(rank + step * world_size)

void sampleNodes(NodeSamplingComm *comm, int step, ParMatrix *X) {
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Request *requests = (MPI_Request *) malloc((comm->msgRecvCount) * sizeof(MPI_Request));
    // first learn what you'll recieve
    for (int i = 0; i < comm->msgRecvCount; i++) {
        int proc_id = comm->recvBuffer->list[i];
        int base = comm->recvBuffer->pid_map[proc_id];
        int *recvIdxs = &comm->recvIdxs[base];
        set_seed(proc_id); // make sure the seed is the same for the sender and receiver
        int bufferSize = bns(base, comm->recvBuffer->recv_count, recvIdxs, comm->p);
        MPI_Irecv(comm->recvBuffer->data[base], bufferSize * comm->recvBuffer->feature_size, MPI_DOUBLE, proc_id, 0,
                  MPI_COMM_WORLD, &requests[i]);
    }

    // send the global indices of the nodes to be sampled
    // then send the features
    // update comm->recvBuffer->count everytime do not update pid_map
    for (int i = 0; i < comm->msgSendCount; i++) {
        int proc_id = comm->sendBuffer->list[i];
        int base = comm->sendBuffer->pid_map[proc_id];
        int next_base = comm->sendBuffer->pid_map[proc_id + 1];
        int count = next_base - base;
        // select random vertices to be sampled
        set_seed(world_rank);
        int bufferSize = bns(base, count, comm->sendIdxs, comm->p);
        for (int j = 0; j < bufferSize; j++) {
            int data_idx = comm->sendIdxs[j];
            int local_idx = comm->sendBuffer->vertices_local[data_idx];
            memcpy(comm->sendBuffer->data[data_idx], X->mat->entries[local_idx],
                   comm->sendBuffer->feature_size * sizeof(double));
        }
        MPI_Send(comm->sendBuffer->data[base], bufferSize * comm->sendBuffer->feature_size, MPI_DOUBLE, proc_id, 0,
                 MPI_COMM_WORLD);
    }
    MPI_Waitall(comm->msgRecvCount, requests, MPI_STATUSES_IGNORE);
    free(requests);
}
