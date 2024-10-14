//
// Created by serdar on 10/13/24.
//

#ifndef MPI_GCN_CPU_MASKING_H
#define MPI_GCN_CPU_MASKING_H

#include <stdbool.h>
#include "typedef.h"


bool *masking_init(int numLocalVertices, double ratio, unsigned int seed);

void masking_forward(bool eval);

#endif //MPI_GCN_CPU_MASKING_H
