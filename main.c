#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "includes/fileio.h"
#include "includes/typedef.h"
#include "includes/basic.h"
#include "includes/activationLayer.h"
#include "includes/neuralNet.h"
#include "includes/matrix.h"
#include "includes/sparseMat.h"
#include "includes/masking.h"
#include "includes/argParse.h"


//CMD args dataset  MODE epoch

int main(int argc, char **argv) {
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    args arg = parseArgs(argc, argv);

//    sleep(10); // to attach gdb
    SparseMat *A = readSparseMat(arg.adj_file, STORE_BY_ROWS, arg.inpart);
    SparseMat *A_T;
    if (arg.symmetric) {
        A_T = A;
    } else {
        A_T = readSparseMat(arg.adj_T_file, STORE_BY_ROWS, arg.inpart_T);
    }
    ParMatrix *X = readDenseMat(arg.features_file, A);
    ParMatrix *Y = readDenseMat(arg.labels_file, A);
    neural_net *net = net_init(10);

    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0) {
        if (!printFreeMemory()) exit(1);
        printf("Total rows, local rows: %d %d\n", X->mat->total_m, X->mat->m);
    }
    void *comm1, *comm2;
    if (arg.comm_type == TP) {
        comm1 = initTPComm(A, A_T, X->gn, arg.hidden_size, true, arg.tp_comm_file, arg.tp_comm_file_T);
        comm2 = initTPComm(A, A_T, arg.hidden_size, Y->gn, true, arg.tp_comm_file, arg.tp_comm_file_T);
        bind_recv_buffers(X->mat, comm1);
        bind_recv_buffers(X->mat, comm2);
    } else {
        comm1 = initOPComm(A, A_T, X->gn, arg.hidden_size);
        comm2 = initOPComm(A, A_T, arg.hidden_size, Y->gn);
    }
    double train_ratio = 0.8;
    bool *train_mask = masking_init(X->mat->m, train_ratio, 123);
    layer_super *gcn_1 = layer_init_gcn(A, comm1, arg.comm_type, X->gn, arg.hidden_size, train_mask);
    layer_super *dropout_1 = layer_init_dropout(0.3);
    layer_super *act_1 = layer_init_activation(RELU);
    layer_super *gcn_2 = layer_init_gcn(A, comm2, arg.comm_type, arg.hidden_size, Y->gn, train_mask);

    net_addLayer(net, gcn_1);
    net_addLayer(net, dropout_1);
    net_addLayer(net, act_1);
    net_addLayer(net, gcn_2);

    //for memory opt
//    if (atoi(argv[2]) != 0) {
//        free(A_T->val);
//        free(A_T->ia);
//        free(A_T->ja);
//        free(A_T->ja_mapped);
//    }

    if (world_rank == 0) {
        printf("model generated\n");
        if (!printFreeMemory()) exit(1);
        printf("Starting training\n");
    }
    ParMatrix *output;
    double t1, t2, t3;
    output = net_forward(net, X, false);
    double tot = 0;
    double min = 99999;
    for (int i = 0; i < arg.n_epochs; i++) {
        Matrix *tempErr = matrix_create(Y->mat->m, Y->gn);
        MPI_Barrier(MPI_COMM_WORLD);
        t1 = MPI_Wtime();

        output = net_forward(net, X, false);
        Matrix *soft = matrix_softmax(output->mat);
        matrix_de_crossEntropy(soft, Y->mat, tempErr, train_mask);
//        totalCrossEntropy(Y->mat, soft);

        net_backward(net, tempErr, 0.001, i);
        MPI_Barrier(MPI_COMM_WORLD);
        t2 = MPI_Wtime();
        matrix_free(soft);
        tot += t2 - t1;
        if (min > t2 - t1) {
            min = t2 - t1;
        }
        // test
        output = net_forward(net, X, true);
        soft = matrix_softmax(output->mat);
        metrics(soft, Y->mat, train_mask);
        matrix_free(soft);
    }
    if (world_rank == 0) {
        printf("Average runtime for current experiment=> %lf\n", tot / arg.n_epochs);
        printf("Min runtime for current experiment=> %lf\n", min);
        printf("---------------------------------------------\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    // free memory
//    net_free(net);
    MPI_Finalize();
    return 0;
}
