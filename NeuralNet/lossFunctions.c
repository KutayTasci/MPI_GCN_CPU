#include "../includes/lossFunctions.h"
#include <math.h>
#include <stdlib.h>
#include <mpi.h>
#include <stdio.h>

double min(double num1, double num2)
{
    if (num1 < num2) {
        return num1;
    } else {
        return  num2;
    }
}


double max(double num1, double num2)
{
    if (num1 > num2) {
        return num1;
    } else {
        return  num2;
    }
}

void totalCrossEntropy(Matrix *y, Matrix *y_hat) {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if ((y->m == y_hat->m) && (y->n == y_hat->n)) {
        double total= 0;
        for (int i = 0; i < y->m; i++) {
           for (int j = 0; j < y_hat->n; j++) {
                if (y->entries[i][j] == 1) {
                    total += (-1 * log(max(y_hat->entries[i][j], 0.001)));
                }
            }
        }
        double global_total;
        MPI_Reduce(&total, &global_total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (world_rank == 0) {
            printf("Cross-Entropy Loss %lf\n", global_total);
        }
    } else {
        printf("Dimension mistmatch cross entropy: %dx%d %dx%d\n", y->m, y->n, y_hat->m, y_hat->n);
        exit(1);
    }
}

void totalL2Loss(Matrix *y, Matrix *y_hat) {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if ((y->m == y_hat->m) && (y->n == y_hat->n)) {
        double total= 0;
        for (int i = 0; i < y->m; i++) {
           for (int j = 0; j < y_hat->n; j++) {
                total += pow(y->entries[i][j] - y_hat->entries[i][j], 2);
            }
        }
        double global_total;
        MPI_Reduce(&total, &global_total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (world_rank == 0) {
            printf("L2 Loss %lf\n", global_total);
        }
    } else {
        printf("Dimension mistmatch l2 loss: %dx%d %dx%d\n", y->m, y->n, y_hat->m, y_hat->n);
        exit(1);
    }
}

void calcCrossEntropy(Matrix *y, Matrix *y_hat, Matrix *error) {
    if ((y->m == y_hat->m) && (y->n == y_hat->n)) {
        for (int i = 0; i < y->m; i++) {
           for (int j = 0; j < y_hat->n; j++) {
                error->entries[i][j] = -1 * (y->entries[i][j] * log(max(y_hat->entries[i][j], 0.001)));
           }
        }
    } else {
        printf("Dimension mistmatch entropy: %dx%d %dx%d\n", y->m, y->n, y_hat->m, y_hat->n);
        exit(1);
    }
}

void calcL2Loss(Matrix *y, Matrix *y_hat, Matrix *error) {
    if ((y->m == y_hat->m) && (y->n == y_hat->n)) {
        for (int i = 0; i < y->m; i++) {
           for (int j = 0; j < y_hat->n; j++) {
                error->entries[i][j] = pow(y->entries[i][j] - y_hat->entries[i][j], 2);
           }
        }
    } else {
        printf("Dimension mistmatch l2 loss: %dx%d %dx%d\n", y->m, y->n, y_hat->m, y_hat->n);
        exit(1);
    }
}