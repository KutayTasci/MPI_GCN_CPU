//
// Created by serdar on 10/13/24.
//
#include "../includes/masking.h"
#include <stdlib.h>
#include <mpi.h>

bool *masking_init(int numLocalVertices, double ratio, unsigned int seed) {
    bool *mask_arr = (bool *) malloc(sizeof(bool) * numLocalVertices);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    srand(seed + world_rank);

    for (int i = 0; i < numLocalVertices; i++) {
        bool mask = (double) rand() / RAND_MAX < ratio;
        mask_arr[i] = mask;
    } // mask should be the same for all processes
    return mask_arr;
}