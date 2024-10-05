#ifndef BASIC_H_INCLUDED
#define BASIC_H_INCLUDED

#pragma once
#include "typedef.h"

int mapG2L(int i, int offset);
int mapProcessor(int i);
sendTable* initSendTable(SparseMat* A);
recvTable* initRecvTable(sendBuffer* send_table, SparseMat* A_T);
sendBuffer* initSendBuffer(sendTable* table,int *l2gMap, int feature_size);
recvBuffer* initRecvBuffer(recvTable* table, int feature_size);

#endif // BASIC_H_INCLUDED
