//
// Created by serdar on 10/13/24.
//

#ifndef MPI_GCN_CPU_MASKING_H
#define MPI_GCN_CPU_MASKING_H

#include <stdbool.h>
#include "argParse.h"


#define TRAIN_IDX 0
#define TEST_IDX 1
#define EVAL_IDX 2

bool **random_masking_init(int numLocalVertices, unsigned int seed, double train_ratio, double test_ratio);

bool **
load_masking(char *train_mask_file, char *test_mask_file, char *eval_mask_file, int numLocalVertices, int *partition);

void free_masks(bool **masks);

bool **mask_init(int m, args arg, int *partition);

#endif //MPI_GCN_CPU_MASKING_H
