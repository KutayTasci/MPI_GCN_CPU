#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include "includes/fileio.h"
#include "includes/typedef.h"
#include "includes/basic.h"
#include "includes/activationLayer.h"
#include "includes/neuralNet.h"
#include "includes/matrix.h"
#include "includes/sparseMat.h"
#include "includes/masking.h"
#include "includes/argParse.h"
#include "includes/lossFunctions.h"
#include "includes/nodeSampling.h"


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
    int feature_size = readFeatureSize(arg.features_file);
    int output_size = readFeatureSize(arg.labels_file);
    neural_net *net = net_init(10);
    ParMatrix *X, *Y = readDenseMat(arg.labels_file, A, 0);
    MPI_Barrier(MPI_COMM_WORLD);
    NodeSamplingComm *comm = nodeSamplingCommInit(A, A_T, 0.1, feature_size);
    srand(arg.seed);
    bool **masks = mask_init(X->mat->m, arg, A->inPart);
    layer_super *gcn_1 = layer_init_gcn(A, comm, arg.comm_type, X->gn, arg.hidden_size, masks);
    layer_super *dropout_1 = layer_init_dropout(0.3);
    layer_super *act_1 = layer_init_activation(RELU);
    layer_super *gcn_2 = layer_init_gcn(A, comm, arg.comm_type, arg.hidden_size, Y->gn, masks);

    net_addLayer(net, gcn_1);
    net_addLayer(net, dropout_1);
    net_addLayer(net, act_1);
    net_addLayer(net, gcn_2);

    net_prepare(net, X);

    MPI_Barrier(MPI_COMM_WORLD);

    if (world_rank == 0) {
        printf("model generated\n");
        if (!printFreeMemory()) exit(1);
        printf("Starting training\n");
    }
    ParMatrix *output;
    double t1, t2, t3;
    output = net_forward(net, X, TRAIN_IDX);

    double times[arg.n_epochs];
    for (int i = 0; i < arg.n_epochs; i++) {
        Matrix *tempErr = matrix_create(Y->mat->m, Y->gn, X->mat->total_m - X->mat->m);
        MPI_Barrier(MPI_COMM_WORLD);
        t1 = MPI_Wtime();

        output = net_forward(net, X, TRAIN_IDX);
        Matrix *soft = matrix_softmax(output->mat);
        matrix_de_crossEntropy(soft, Y->mat, tempErr, masks[TRAIN_IDX]);
//        totalCrossEntropy(Y->mat, soft);

        net_backward(net, tempErr, 0.001, i);
        MPI_Barrier(MPI_COMM_WORLD);
        t2 = MPI_Wtime();
        matrix_free(soft);
        times[i] = t2 - t1;
        // test
        output = net_forward(net, X, TEST_IDX);
        soft = matrix_softmax(output->mat);
        metrics(soft, Y->mat, masks[TEST_IDX]);
        matrix_free(soft);
    }
    if (world_rank == 0) {
        double tot = 0, min = 9999999;
        for (int i = 0; i < arg.n_epochs; i++) {
            tot += times[i];
            if (times[i] < min) min = times[i];
        }
        printf("Average runtime for current experiment=> %lf\n", tot / arg.n_epochs);
        printf("Min runtime for current experiment=> %lf\n", min);
        printf("---------------------------------------------\n");
        for (int i = 0; i < arg.n_epochs - 1; i++) {
            printf("%lf,", times[i]);
        }
        printf("%lf\n", times[arg.n_epochs - 1]);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    // free memory
//    net_free(net);
//    free_masks(masks);
    MPI_Finalize();
    return 0;
}
