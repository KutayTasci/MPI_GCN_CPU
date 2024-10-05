#ifndef GCNLAYER_H_INCLUDED
#define GCNLAYER_H_INCLUDED
#include "typedef.h"
#include "basic.h"


#define FORWARD 101
#define BACKWARD 102

typedef struct {
    int size_n;
    int size_f;
    int size_output;
    int* recvBuffMap;
    int* recvBuffMap_backward;
    SparseMat* adjacency;
    SparseMat* adjacency_T;
    ParMatrix* input;
    ParMatrix* output;
    Matrix* weights;
    Matrix* m_weights;
    Matrix* v_weights;
    Matrix* gradients;
    sendBuffer* sendBuffer;
    recvBuffer* recvBuffer;
    sendBuffer* sendBuffer_backward;
    recvBuffer* recvBuffer_backward;
    int msgSendCount;
    int msgRecvCount;
    int msgSendCount_b;
    int msgRecvCount_b;
} gcnLayer;

gcnLayer* gcn_init(SparseMat* adj, SparseMat* adj_T, int size_f,int size_out);
void setMode(int i);
void gcn_forward(gcnLayer* layer);
Matrix* gcn_backward(gcnLayer* layer, Matrix* out_error);
void gcn_step(gcnLayer* layer, double lr, int t);
void gcn_free(gcnLayer* layer);
#endif // GCNLAYER_H_INCLUDED
