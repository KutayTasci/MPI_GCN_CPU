//
// Created by serdar on 10/13/24.
//
#include "../includes/masking.h"
#include <stdlib.h>
#include <mpi.h>
#include <stdio.h>

bool **random_masking_init(int numLocalVertices, unsigned int seed, double train_ratio, double test_ratio) {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    srand(seed + world_rank);
    if (train_ratio + test_ratio > 1) {
        if (world_rank == 0)
            printf("Train and test ratio should be less than 1.0\n");
        exit(0);
    }
    bool **masks = malloc(3 * sizeof(bool *));
    for (int i = 0; i < 3; i++) {
        masks[i] = malloc(numLocalVertices * sizeof(bool));
    }
    for (int i = 0; i < numLocalVertices; i++) {
        double r = (double) rand() / RAND_MAX;
        if (r < train_ratio) {
            masks[0][i] = true;
            masks[1][i] = false;
            masks[2][i] = false;
        } else if (r < train_ratio + test_ratio) {
            masks[0][i] = false;
            masks[1][i] = true;
            masks[2][i] = false;
        } else {
            masks[0][i] = false;
            masks[1][i] = false;
            masks[2][i] = true;
        }
    }
    return masks;
}

bool **
load_masking(char *train_mask_file, char *test_mask_file, char *eval_mask_file, int numLocalVertices, int *partition) {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    bool **masks = malloc(3 * sizeof(bool *));
    char *mask_files[3] = {train_mask_file, test_mask_file, eval_mask_file};
    for (int i = 0; i < 3; i++) {
        masks[i] = malloc(numLocalVertices * sizeof(bool));
        FILE *file = fopen(mask_files[i], "rb");
        if (file == NULL) {
            fclose(file);
            if (i == 2) continue; // eval masks is optional
            printf("Mask file %s not found\n", mask_files[i]);
            exit(0);
        }
        int mask_v, l_idx = 0, g_idx = 0;
        while (fread(&mask_v, sizeof(bool), 1, file) == 1) {
            if (world_rank == partition[g_idx]) {
                masks[i][l_idx++] = mask_v;
            }
            g_idx++;
        }
        fclose(file);
        if (l_idx != numLocalVertices) {
            printf("Mask file %s does not contain enough entries l,numVertices: %d, %d\n", mask_files[i], l_idx,
                   numLocalVertices);
            exit(0);
        }
    }
    return masks;
}

void free_masks(bool **masks) {
    for (int i = 0; i < 3; i++) {
        free(masks[i]);
    }
    free(masks);
}

bool **mask_init(int m, args arg, int *partition) {
    if (arg.random_masking) {
        return random_masking_init(m, 123, 0.7, 0.2);
    } else {
        return load_masking(arg.train_mask_file, arg.test_mask_file, arg.eval_mask_file, m, partition);
    }
}