#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "includes/fileio.h"
#include "includes/typedef.h"
#include "includes/basic.h"
#include "includes/gcnLayer.h"
#include "includes/activationLayer.h"
#include "includes/neuralNet.h"
#include "includes/matrix.h"
#include "includes/lossFunctions.h"
#include "includes/sparseMat.h"
#include <sys/sysinfo.h>


//CMD args dataset  MODE epoch

int main(int argc, char **argv) {
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    //printf("running on %d proc\n", world_size);
    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    neural_net *net = net_init(3);

    setMode(atoi(argv[2])); //Agregation iterator in gcnLayer.c
    int epoch = atoi(argv[3]);
    int hidden_p = atoi(argv[4]);
    int tst_n = atoi(argv[5]);

    char fname[100];//input file of Adj matrix
    char fname_T[100]; //input file of Transpose Adj
    char inpart[100]; //Partitioning of the inputs
    char densefname[100]; //Feature matrix
    char labels[100];
    char trainMaskf[100];
    char valMaskf[100];

    strcpy(fname, argv[1]);//input file of Adj matrix
    strcpy(fname_T, argv[1]); //input file of Transpose Adj
    strcpy(inpart, argv[1]); //Partitioning of the inputs
    strcpy(densefname, argv[1]); //Feature matrix
    strcpy(labels, argv[1]);
    strcpy(trainMaskf, argv[1]);
    strcpy(valMaskf, argv[1]);

    if (world_rank == 0) {
        printf("Experimental settings\n");
        printf("Dataset: %s\n", fname);
        printf("Processor Count:%d - Hidden_Parameter:%d - Partitioning: %d\n", world_size, hidden_p, tst_n);
        switch (atoi(argv[2])) {
            case 0:
                printf("Aggregate Mode: Default Non-Overlapping CSR\n");
                break;
            case 1:
                printf("Aggregate Mode: Non-Overlapping CSR with with our datas structure\n");
                break;
            case 2:
                printf("Aggregate Mode: Overlapping CSR with with our datas structure\n");
                break;
            case 3:
                printf("Aggregate Mode: Partial-Overlapping CSR with with our datas structure\n");
                break;
            case 4:
                printf("Aggregate Mode: Default Non-Overlapping CSC\n");
                break;
            case 5:
                printf("Aggregate Mode: Overlapping CSC\n");
                break;
            case 6:
                printf("Aggregate Mode: Overlapping CSC and CSR Hybrid\n");
                break;
            case 7:
                printf("Aggregate Mode: Full+Full\n");
                break;
        }
        printf("--------------\n");
    }

    char snum[5];
    char partf[100];
    char partf_T[100];
    sprintf(snum, "%d", world_size);
    sprintf(partf, "/part_files_K%d_M%d/adj.mtx.inpart.", hidden_p, tst_n);

    strcat(fname, partf);//input file of Adj matrix
    strcat(fname, snum);
    strcat(fname, ".bin");
    //directed graph
    //sprintf(partf_T, "/part_files_K%d_M%d_T/adj.mtx.inpart.",hidden_p, tst_n);
    //strcat(fname_T, partf_T);
    //strcat(fname_T,snum);
    //strcat(fname_T,".bin"); //input file of Transpose Adj
    //directed graph
    strcat(inpart, partf); //Partitioning of the inputs
    strcat(inpart, snum);
    strcat(densefname, "/features.csv"); //Feature matrix
    strcat(labels, "/labels.csv");

    SparseMat *A = readSparseMat(fname, STORE_BY_ROWS, inpart);
    //SparseMat* A_T = readSparseMat(fname_T, STORE_BY_ROWS, inpart); 
    SparseMat *A_T = A; // Change for directed graph
    ParMatrix *X = readDenseMat(densefname, A);
    ParMatrix *Y = readDenseMat(labels, A);

    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0) {
        printf("reading finished\n");
        struct sysinfo sys_info;
        if (sysinfo(&sys_info) != 0) {
            perror("sysinfo");
            return 1;
        }
        unsigned long free_memory = sys_info.freeram * sys_info.mem_unit;

        printf("Free memory: %lu bytes\n", free_memory);

    }


    layer_super *gcn_1 = layer_init_gcn(A, A_T, X->gn, Y->gn);
    layer_super *act_1 = layer_init_activation(RELU);
    layer_super *gcn_2 = layer_init_gcn(A, A_T, hidden_p, Y->gn);

    net_addLayer(net, gcn_1);
    net_addLayer(net, act_1);
    net_addLayer(net, gcn_2);

    //for memory opt
    if (atoi(argv[2]) != 0) {
        free(A_T->val);
        free(A_T->ia);
        free(A_T->ja);
        free(A_T->ja_mapped);
    }


    if (world_rank == 0) {
        printf("model generated\n");
    }


    if (world_rank == 0) {
        struct sysinfo sys_info;
        if (sysinfo(&sys_info) != 0) {
            perror("sysinfo");
            return 1;
        }
        unsigned long free_memory = sys_info.freeram * sys_info.mem_unit;

        printf("Free memory: %lu bytes\n", free_memory);
        printf("Starting\n");
    }
    ParMatrix *output;
    double t1, t2, t3;
    output = net_forward(net, X);
    double tot = 0;
    double min = 99999;

    for (int i = 0; i < epoch; i++) {
        Matrix *tempErr = matrix_create(Y->mat->m, Y->gn);

        MPI_Barrier(MPI_COMM_WORLD);
        t1 = MPI_Wtime();

        output = net_forward(net, X);
        Matrix *soft = matrix_softmax(output->mat);
        matrix_de_crossEntropy(soft, Y->mat, tempErr);
        //totalCrossEntropy(Y->mat, soft);
        //metrics(soft, Y->mat);

        net_backward(net, tempErr, 0.001, i);


        MPI_Barrier(MPI_COMM_WORLD);
        t2 = MPI_Wtime();
        matrix_free(soft);


        tot += t2 - t1;
        if (min > t2 - t1) {
            min = t2 - t1;
        }
    }
    if (world_rank == 0) {
        printf("Average runtime for current experiment=> %lf\n", tot / epoch);
        printf("Min runtime for current experiment=> %lf\n", min);
        printf("---------------------------------------------\n");
    }

    if (atoi(argv[2]) == 1) {
        gcnLayer *gcn_1l = (gcnLayer *) gcn_1->layer;
        Matrix *temp = matrix_create(gcn_1l->size_n, gcn_1l->size_f);
        MPI_Barrier(MPI_COMM_WORLD);
        t1 = MPI_Wtime();
        aggregate_no_comm(gcn_1l, X->mat, temp, 101);
        MPI_Barrier(MPI_COMM_WORLD);
        t2 = MPI_Wtime();
        if (world_rank == 0) {
            printf("Aggregate with no communication=> %lf\n", t2 - t1);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        t1 = MPI_Wtime();
        aggregate_no_comp(gcn_1l, X->mat, temp, 101);
        MPI_Barrier(MPI_COMM_WORLD);
        t2 = MPI_Wtime();
        if (world_rank == 0) {
            printf("Aggregate with no computation=> %lf\n", t2 - t1);
            printf("---------------------------------------------\n");
        }
        matrix_free(temp);
    }


    MPI_Barrier(MPI_COMM_WORLD);
    //parMatrixFree(X);
    //net_free(net);

    MPI_Finalize();
    return 0;
}
