#ifndef TYPEDEF_H_INCLUDED
#define TYPEDEF_H_INCLUDED

#pragma once

#include <stdbool.h>

#define STORE_BY_COLUMNS 0
#define STORE_BY_ROWS    1

#define AGG_COMM 0
#define FORWARD 0
#define BACKWARD 1

typedef struct {
    int indptr;
    int v_id;
} csrPtr;

typedef struct {
    int *ia;    // rows of A in csr format
    int *ja;    // cols of A in csr format
    int *ja_mapped; // cols of A in csr format locally mapped
    double *val; // values of A in csr format

    int *ib; // rows of A in csc format
    int *jb; // cols of A in csc format
    double *val_b; // values of A in csc format

    int *proc_map;
    csrPtr *ic;
    int *jc;
    int *jc_mapped;
    double *val_c;

    int n;
    int m;
    int nnz;
    int gn, gm;
    int store;
    int *l2gMap;
    int *inPart;
    int init;
} SparseMat;


typedef struct {
    double **entries;
    int m;
    int n;
    int phase_1, phase_2; // used in tp communication
} Matrix;

typedef struct {
    Matrix *mat;
    int gm; // global rows
    int gn; // global cols
    int store; // 0 for column major, 1 for row major
    int *l2gMap; // local to global map
    int *inPart; // partitioning
} ParMatrix;

typedef struct node {
    int val;
    struct node *next;
} node_t;

typedef struct {
    int p_count;
    int myId;
    int n;
    int *send_count;
    int **table;

    node_t **table_t;
} sendTable;

typedef struct {
    int p_count;
    int myId;
    int n;
    int *recv_count;
    int **table;

} recvTable;

typedef struct {
    int send_count;
    int *pid_map;
    int *list;
    int feature_size;
    int *vertices;
    int *vertices_local;
    double **data;
} sendBuffer;

typedef struct {
    int recv_count;
    int *pid_map;
    int *list;
    int feature_size;
    int *vertices;
    double **data;
} recvBuffer;

void sparseMatInit(SparseMat *A);

void sparseMatFree(SparseMat *A);

void csrToCsc(SparseMat *A);

void generate_parCSR(SparseMat *A, int *recv_map, int world_size, int world_rank);

Matrix *matrix_create(int row, int col);

void matrix_free(Matrix *m);

ParMatrix *init_ParMatrix(SparseMat *A, int n);

void parMatrixFree(ParMatrix *X);

ParMatrix *create_output_matrix(ParMatrix *input);

sendTable *sendTableCreate(int p_count, int myId, int n); //n equals A->m
void sendTableFree(sendTable *table);

recvTable *recvTableCreate(int p_count, int myId, int n); //n equals A->gn
void recvTableFree(recvTable *table);

//sendBuffer* sendBufferCreate(int send_count, int feature_size, int p_id);
void sendBufferFree(sendBuffer *buffer);

void initSendBufferSpace(sendBuffer *buffer);

void sendBufferSpaceFree(sendBuffer *buffer);
//void sendBufferListFree(sendBuffer** bufferList, int world_size, int world_rank);

//recvBuffer* recvBufferCreate(int send_count, int feature_size, int p_id);
void initRecvBufferSpace(recvBuffer *buffer);

void recvBufferSpaceFree(recvBuffer *buffer);

void recvBufferFree(recvBuffer *buffer);
//void recvBufferListFree(recvBuffer** bufferList, int world_size, int world_rank);
#endif // TYPEDEF_H_INCLUDED
