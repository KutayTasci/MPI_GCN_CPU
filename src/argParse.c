//
// Created by serdar on 10/16/24.
//

#include "../includes/argParse.h"

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>
#include <dirent.h>

#define printf_r0(...) if (world_rank == 0) printf(__VA_ARGS__)

void copyPath(const char *dir, char *file_name, char *dest) {
    strcpy(dest, dir);
    if (dest[strlen(dest) - 1] != '/')
        strcat(dest, "/");
    strcat(dest, file_name);
}

bool checkReduced(char *file_name) {
    return strstr(file_name, ".reduced") != NULL;
}

void process_directory(const char *dir_path, char *adj_file, char *inpart, char *tp_comm_file, CommType comm_type) {
    bool reduced = comm_type > 7;
    DIR *dir = opendir(dir_path);
    if (dir) {
        struct dirent *ent;
        while ((ent = readdir(dir)) != NULL) {
            bool wrong_file = reduced ^ checkReduced(ent->d_name);
            if (wrong_file) continue;
            if (strstr(ent->d_name, ".inpart") != NULL) {
                // if ends with .bin
                if (strstr(ent->d_name, ".bin") != NULL) {
                    copyPath(dir_path, ent->d_name, adj_file);
                } else {
                    copyPath(dir_path, ent->d_name, inpart);
                }
            } else if (strstr(ent->d_name, ".phases") != NULL) {
                if (strstr(ent->d_name, ".one") == NULL) // ignore one phase file
                    copyPath(dir_path, ent->d_name, tp_comm_file);
            }
        }
        closedir(dir);
    }
}


void exit_safe() {
    MPI_Finalize();
    exit(0);
}

bool checkPathExists(const char *path) {
    FILE *file = fopen(path, "r");
    if (file) {
        fclose(file);
        return true;
    }
    return false;
}

args parseArgs(int argc, char **argv) {
    int world_rank, world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    args ret;
    char *usage = "Usage: MPI_GCN_CPU <dataset_folder> <inpart_folder> <n_threads> <n_epochs> <agg_mode> <-l lr> <-d dropout_rate> <-t inpart_T_folder> <-s hidden_size>\n";
    if (argc < 6 || strcmp(argv[1], "-h") == 0) {
        printf_r0("%s", usage);
        printf_r0("inpart_path, inpart_transpose_path: must contain inpart and inpart.bin files\n");
        printf_r0("dataset_folder: must contain features.csv and labels.csv\n");
        exit_safe();
    }
    // set inpart and adj file
    ret.inpart[0] = '\0';
    ret.adj_file[0] = '\0';
    ret.inpart_T[0] = '\0';
    ret.adj_T_file[0] = '\0';
    ret.tp_comm_file[0] = '\0';
    ret.tp_comm_file_T[0] = '\0';

    copyPath(argv[1], "features.csv", ret.features_file);
    copyPath(argv[1], "labels.csv", ret.labels_file);
    if (!checkPathExists(ret.features_file) || !checkPathExists(ret.labels_file)) {
        copyPath(argv[1], "features.bin", ret.features_file);
        copyPath(argv[1], "labels.bin", ret.labels_file);
        if (!checkPathExists(ret.features_file) || !checkPathExists(ret.labels_file)) {
            printf_r0("features.csv/.bin or labels.csv/.bin not found\n");
            exit_safe();
        }
    }
    copyPath(argv[1], "train_mask.bin", ret.train_mask_file);
    copyPath(argv[1], "test_mask.bin", ret.test_mask_file);
    copyPath(argv[1], "eval_mask.bin", ret.eval_mask_file);

    ret.n_threads = atoi(argv[3]);
    ret.n_epochs = atoi(argv[4]);
    ret.comm_type = atoi(argv[5]);
    if (ret.comm_type < 0 || ret.comm_type > 8) {
        printf_r0("Invalid aggregation mode. Must be between 0 and 8\n");
        exit_safe();
    }
    // set defaults
    ret.dropout_rate = 0.5;
    ret.symmetric = true;
    ret.hidden_size = 64;
    ret.lr = 0.01;

    process_directory(argv[2], ret.adj_file, ret.inpart, ret.tp_comm_file, ret.comm_type);
    for (int i = 6; i < argc; i++) {
        if (strcmp(argv[i], "-d") == 0) {
            ret.dropout_rate = atof(argv[i + 1]);
            if (ret.dropout_rate < 0 || ret.dropout_rate > 1) {
                printf_r0("Invalid dropout rate. Must be between 0 and 1\n");
                exit_safe();
            }
        } else if (strcmp(argv[i], "-t") == 0) {
            process_directory(argv[i + 1], ret.adj_T_file, ret.inpart_T, ret.tp_comm_file_T, ret.comm_type);
            ret.symmetric = false;
        } else if (strcmp(argv[i], "-l") == 0) {
            ret.lr = atof(argv[i + 1]);
            if (ret.lr <= 0) {
                printf_r0("Invalid learning rate. Must be greater than 0\n");
                exit_safe();
            }
        } else if (strcmp(argv[i], "-s") == 0) {
            ret.hidden_size = atoi(argv[i + 1]);
            if (ret.hidden_size <= 0) {
                printf_r0("Invalid hidden size. Must be greater than 0\n");
                exit_safe();
            }
        }
    }
    // print args
    if (world_rank == 0) {
        printf("MODE: %d\n", ret.comm_type);
        printf("Epochs: %d\n", ret.n_epochs);
        printf("Hidden size: %d\n", ret.hidden_size);
        printf("Learning rate: %f\n", ret.lr);
        printf("Dropout rate: %f\n", ret.dropout_rate);
        printf("Inpart: %s\n", ret.inpart);
        printf("Inpart_T: %s\n", ret.inpart_T);
        printf("Adj file: %s\n", ret.adj_file);
        printf("Adj_T file: %s\n", ret.adj_T_file);
        printf("Features file: %s\n", ret.features_file);
        printf("Labels file: %s\n", ret.labels_file);
        printf("TP Comm file: %s\n", ret.tp_comm_file);
    }

    if (ret.comm_type == 8) {
        if (ret.tp_comm_file[0] == '\0') {
            printf_r0("TP aggregation mode selected but no tp_comm file found\n");
            exit_safe();
        }
        if (!ret.symmetric && ret.tp_comm_file_T[0] == '\0') {
            printf_r0("TP aggregation mode selected but no tp_comm file found for transpose\n");
            exit_safe();
        }
    }
    if (ret.inpart[0] == '\0' || ret.adj_file[0] == '\0') {
        printf_r0("inpart or adj file not found\n");
        exit_safe();
    }
    if (!ret.symmetric && (ret.inpart_T[0] == '\0' || ret.adj_T_file[0] == '\0')) {
        printf_r0("inpart_T or adj_T file not found\n");
        exit_safe();
    }
    if (ret.symmetric) {
        strcpy(ret.tp_comm_file_T, ret.tp_comm_file);
    }

    char *agg_info[] = {"Default Non-Overlapping CSR",
                        "Non-Overlapping CSR with with our datas structure",
                        "Overlapping CSR with with our datas structure",
                        "Partial-Overlapping CSR with with our datas structure",
                        "Default Non-Overlapping CSC",
                        "Overlapping CSC",
                        "Overlapping CSC and CSR Hybrid",
                        "Full+Full",
                        "TP"};
    printf_r0("Experimental settings\n");
    printf_r0("Dataset: %s\n", argv[1]);
    printf_r0("Processor Count:%d - Hidden_Parameter:%d\n", world_size, ret.hidden_size);
    printf_r0("Aggregation Mode: %s\n", agg_info[ret.comm_type]);
    printf_r0("--------------\n");
    return ret;
}