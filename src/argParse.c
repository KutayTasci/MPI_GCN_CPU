//
// Created by serdar on 10/16/24.
//

#include "../includes/argParse.h"

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>
#include <dirent.h>


void copyPath(const char *dir, char *file_name, char *dest) {
    strcpy(dest, dir);
    strcat(dest, "/");
    strcat(dest, file_name);
}

void process_directory(const char *dir_path, char *adj_file, char *inpart, char *tp_comm_file) {
    DIR *dir = opendir(dir_path);
    if (dir) {
        struct dirent *ent;
        while ((ent = readdir(dir)) != NULL) {
            if (strstr(ent->d_name, "inpart") != NULL) {
                // if ends with .bin
                if (strstr(ent->d_name, ".bin") != NULL) {
                    copyPath(dir_path, ent->d_name, adj_file);
                } else {
                    copyPath(dir_path, ent->d_name, inpart);
                }
            } else if (strstr(ent->d_name, "tp_comm") != NULL) {
                copyPath(dir_path, ent->d_name, tp_comm_file);
            }
        }
        closedir(dir);
    }
}

args parseArgs(int argc, char **argv) {
    args ret;
    char *usage = "Usage: MPI_GCN_CPU <dataset_folder> <inpart_path> <n_threads> <n_epochs> <agg_mode> <-l lr> <-d dropout_rate> <-t inpart_transpose_path> <-s hidden_size>\n";
    if (argc < 6 || strcmp(argv[1], "-h") == 0) {
        printf("%s", usage);
        printf("inpart_path, inpart_transpose_path: must contain inpart and inpart.bin files\n");
        printf("dataset_folder: must contain features.csv and labels.csv\n");
        exit(1);
    }
    strcpy(ret.features_file, argv[1]);
    strcat(ret.features_file, "/features.csv");
    strcpy(ret.labels_file, argv[1]);
    strcat(ret.labels_file, "/labels.csv");

    process_directory(argv[2], ret.adj_file, ret.inpart, ret.tp_comm_file);

    ret.n_threads = atoi(argv[3]);
    ret.n_epochs = atoi(argv[4]);
    ret.comm_type = atoi(argv[5]);
    if (ret.comm_type < 0 || ret.comm_type > 8) {
        printf("Invalid aggregation mode. Must be between 0 and 8\n");
        exit(1);
    }
    // set defaults
    ret.dropout_rate = 0.5;
    ret.symmetric = true;
    ret.hidden_size = 64;
    ret.tp_comm_file[0] = '\0';
    ret.tp_comm_file_T[0] = '\0';

    for (int i = 6; i < argc; i++) {
        if (strcmp(argv[i], "-d") == 0) {
            ret.dropout_rate = atof(argv[i + 1]);
            if (ret.dropout_rate < 0 || ret.dropout_rate > 1) {
                printf("Invalid dropout rate. Must be between 0 and 1\n");
                exit(1);
            }
        } else if (strcmp(argv[i], "-t") == 0) {
            process_directory(argv[i + 1], ret.adj_T_file, ret.inpart_T, ret.tp_comm_file_T);
            ret.symmetric = false;
        } else if (strcmp(argv[i], "-l") == 0) {
            ret.lr = atof(argv[i + 1]);
            if (ret.lr <= 0) {
                printf("Invalid learning rate. Must be greater than 0\n");
                exit(1);
            }
        } else if (strcmp(argv[i], "-s") == 0) {
            ret.hidden_size = atoi(argv[i + 1]);
            if (ret.hidden_size <= 0) {
                printf("Invalid hidden size. Must be greater than 0\n");
                exit(1);
            }
        }
    }
    int world_rank, world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    char *agg_info[] = {"Default Non-Overlapping CSR",
                        "Non-Overlapping CSR with with our datas structure",
                        "Overlapping CSR with with our datas structure",
                        "Partial-Overlapping CSR with with our datas structure",
                        "Default Non-Overlapping CSC",
                        "Overlapping CSC",
                        "Overlapping CSC and CSR Hybrid",
                        "Full+Full",
                        "TP"};
    if (world_rank == 0) {
        printf("Experimental settings\n");
        printf("Dataset: %s\n", argv[1]);
        printf("Processor Count:%d - Hidden_Parameter:%d\n", world_size, ret.hidden_size);
        printf("Aggregation Mode: %s\n", agg_info[ret.comm_type]);
        printf("--------------\n");
    }
    if (ret.comm_type == 8) {
        if (ret.tp_comm_file[0] == '\0') {
            printf("TP aggregation mode selected but no tp_comm file found\n");
            exit(1);
        }
        if (!ret.symmetric && ret.tp_comm_file_T[0] == '\0') {
            printf("TP aggregation mode selected but no tp_comm file found for transpose\n");
            exit(1);
        }
    }
    return ret;
}