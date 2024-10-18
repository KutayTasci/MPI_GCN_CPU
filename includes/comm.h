//
// Created by serdar on 10/16/24.
//

#ifndef MPI_GCN_CPU_COMM_H
#define MPI_GCN_CPU_COMM_H

#include "mpi.h"
#include <stdbool.h>
#include "typedef.h"

typedef struct {
    sendBuffer *sendBuffer;
    recvBuffer *recvBuffer;
    sendBuffer *sendBuffer_backward;
    recvBuffer *recvBuffer_backward;
    int msgSendCount;
    int msgRecvCount;
    int msgSendCount_b;
    int msgRecvCount_b;
    int *recvBuffMap;
    int *recvBuffMap_backward;
    SparseMat *adjacency;
    SparseMat *adjacency_T;
} OPComm;

typedef struct {
    int *proc_map; //world_size+1
    int *row_map; //count
    int *row_map_lcl; //count
    double **buffer; //count * f
    int count;
    int f;
} CommBuffer;


typedef struct {
    CommBuffer sendBuffer;
    int msgSendCount;
    MPI_Request *send_ls;
    int *send_proc_list;

    CommBuffer recvBuffer;
    int msgRecvCount;
    MPI_Request *recv_ls;
    int *recv_proc_list;
} OP_Comm;

typedef struct {
    bool init;

    int reduce_count;
    int *reduce_list;
    int *reduce_list_mapped;
    int **reduce_source_mapped; // first element is the length of the list
    int *reduce_local;
    int lcl_count;
    int *reduce_nonlocal;
    int nlcl_count;
} Reducer;

typedef struct {
    Reducer reducer;

    CommBuffer sendBuffer_p1;
    int msgSendCount_p1;
    MPI_Request *send_ls_p1;
    int *send_proc_list_p1;

    CommBuffer recvBuffer_p1;
    int msgRecvCount_p1;
    MPI_Request *recv_ls_p1;
    int *recv_proc_list_p1;

    CommBuffer sendBuffer_p2;
    int msgSendCount_p2;
    MPI_Request *send_ls_p2;
    int *send_proc_list_p2;

    CommBuffer recvBuffer_p2;
    int msgRecvCount_p2;
    MPI_Request *recv_ls_p2;
    int *recv_proc_list_p2;
    SparseMat *A;
} TP_Comm;

typedef struct {
    TP_Comm tpComm;
    TP_Comm tpComm_backward;
} TPW;

typedef enum {
    NO_OVER_CSR = 0,
    NO_OVER_CSR_DATA_STRUCTURE = 1,
    OVER_CSR_DATA_STRUCTURE = 2,
    PARTIAL_OVER_CSR_DATA_STRUCTURE = 3,
    NO_OVER_CSC = 4,
    OVER_CSC = 5,
    HYBRID_CSC_CSR = 6,
    FULL_FULL = 7,
    TP = 8
} CommType;

OPComm *initOPComm(SparseMat *adj, SparseMat *adj_T, int size_f, int size_out);

TPW *initTPComm(SparseMat *adjacency, SparseMat *adjacency_T, int size_f, int size_out, bool preduce, char *comm_file,
                char *comm_file_T);

void readTPComm(char *fname, int f, bool preduce, TP_Comm *comm);

void map_csr(SparseMat *A, TP_Comm *comm);

void CommBufferInit(CommBuffer *buff);

void CommBufferFree(CommBuffer *buff);

#endif //MPI_GCN_CPU_COMM_H
