//
// Created by serdar on 10/16/24.
//

#ifndef MPI_GCN_CPU_ARGPARSE_H
#define MPI_GCN_CPU_ARGPARSE_H

#include <stdio.h>
#include <stdbool.h>
#include "comm.h"

typedef struct {
    double lr;
    double dropout_rate;
    int n_epochs;
    int n_threads;
    int hidden_size;
    unsigned int seed;
    bool symmetric;
    char adj_file[256];
    char inpart[256];
    char tp_comm_file[256];
    char inpart_T[256];
    char adj_T_file[256];
    char tp_comm_file_T[256];
    char features_file[256];
    char labels_file[256];
    char train_mask_file[256];
    char test_mask_file[256];
    char eval_mask_file[256];
    char output_file[256];
    CommType comm_type;
    bool random_masking;
    double p;
} args;

args parseArgs(int argc, char **argv);

#endif //MPI_GCN_CPU_ARGPARSE_H
