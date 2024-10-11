#include "../includes/matrix.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <mpi.h>
#include <time.h>

void matrix_fill(Matrix *m, double n) {
    for (int i = 0; i < m->m; i++) {
        for (int j = 0; j < m->n; j++) {
            m->entries[i][j] = n;
        }
    }
}

void GEMM(Matrix *A, Matrix *B, double *bias, Matrix *C) {
    int m = A->m;
    int n = A->n;
    int f = B->n;
    int i, j, k;

    if (A->n != B->m) {
        printf("Matrix sizes are incompitable\n");
        return;
    }
    double val;
    for (i = 0; i < m; i++) {
        for (j = 0; j < f; j++) {
            C->entries[i][j] = 0;
        }
        for (k = 0; k < n; k++) {
            val = A->entries[i][k];
            for (j = 0; j < f; j++) {
                C->entries[i][j] += val * B->entries[k][j];
            }
        }
        for (j = 0; j < f; j++) {
            C->entries[i][j] += bias[j];
        }
    }
}

void GEMM_NT(Matrix *A, Matrix *B, Matrix *C) {
    int m = A->m;
    int n = A->n;
    int f = B->m;
    int i, j, k;

    //printf("Hello from inside\n");
    //printf("Acessing A => %lf\n", A->entries[0][0]);
    //printf("Acessing B => %lf\n", B->entries[0][0]);
    //printf("Acessing C => %lf\n", C->entries[0][0]);

    if (A->n != B->n) {
        printf("Matrix dimensions are not compatible for multiplication!\n");
        return;
    }

    for (i = 0; i < m; i++) {
        for (j = 0; j < f; j++) {
            C->entries[i][j] = 0;
            for (k = 0; k < n; k++) {
                C->entries[i][j] += A->entries[i][k] * B->entries[j][k];
            }
        }
    }
}

void GEMM_TN(Matrix *A, Matrix *B, Matrix *C) {
    int m = A->m;
    int n = A->n;
    int f = B->n;
    int i, j, k;

    memset(C->entries[0], 0, C->m * C->n * sizeof(double));
    //printf("Test Start\n");
    //printf("Acessing A => %d %d\n", A->m, A->n);
    //printf("Acessing B => %d %d\n", B->m, B->n);
    //printf("Acessing C => %d %d\n", C->m, C->nmake);


    if (A->m != B->m) {
        printf("Matrix dimensions are not compatible for multiplication!\n");
        return;
    }

    //printf("flag_2 \n");
    //MPI_Barrier(MPI_COMM_WORLD);

    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            for (k = 0; k < f; k++) {
                C->entries[j][k] += A->entries[i][j] * B->entries[i][k];
            }
        }
    }

    //printf("flag_3 \n");
    //MPI_Barrier(MPI_COMM_WORLD);
}

// Bias gradient
void bias_grad(Matrix *y_prime, double *gradients) {
    for (int i = 0; i < y_prime->m; i++) {
        gradients[i] = 0;
        for (int j = 0; j < y_prime->n; j++) {
            gradients[i] += y_prime->entries[i][j];
        }
    }
}

double uniform_distribution(double low, double high) {
    double difference = high - low; // The difference between the two
    double div = RAND_MAX / difference;

    return low + (rand() / div);
}

void init_weights_random(gcnLayer *layer, int scale) {

    int m = layer->size_f;
    int n = layer->size_output;
    double min = -sqrt(6.0) / sqrt((double) m + n);
    double max = sqrt(6.0) / sqrt((double) m + n);
    srand(time(NULL));
    layer->weights = matrix_create(m, n);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            layer->weights->entries[i][j] = uniform_distribution(min, max) / 10;

        }
    }
}

void matrix_print(Matrix *m) {
    printf("Rows: %d Columns: %d\n", m->m, m->n);
    for (int i = 0; i < m->m; i++) {
        for (int j = 0; j < m->n; j++) {
            printf("%1.7f ", m->entries[i][j]);
        }
        printf("\n");
    }
}

Matrix *matrix_sum_exp(Matrix *m, int axis) {
    if (axis == 0) {
        /*    
        int cols = m->n;
        int rows = m->m;
        int total;
        Matrix* sum = matrix_create(1, cols);
        for(int i = 0;i<cols;i++) {
            for (int j = 0;j<rows;j++) {
                total += exp(m->entries[j][i]);
            }
            sum->entries[0][i] = total;
        }
        return sum;
        */
        exit(1);
    } else if (axis == 1) {
        int cols = m->n;
        int rows = m->m;
        double total;
        Matrix *sum = matrix_create(1, rows);
        for (int i = 0; i < rows; i++) {
            total = 0;
            for (int j = 0; j < cols; j++) {
                total += exp(m->entries[i][j]);
            }
            sum->entries[0][i] = total;
        }
        return sum;
    } else {
        exit(1);
    }
}

void matrix_subtract(Matrix *m1, Matrix *m2, Matrix *m) {
    if ((m1->m == m2->m) && (m1->n == m2->n)) {
        for (int i = 0; i < m1->m; i++) {
            for (int j = 0; j < m2->n; j++) {
                m->entries[i][j] = m1->entries[i][j] - m2->entries[i][j];
            }
        }
    } else {
        printf("Dimension mistmatch subtract: %dx%d %dx%d\n", m1->m, m1->n, m2->m, m2->n);
        exit(1);
    }
}

void matrix_sum(Matrix *m1, Matrix *m2, Matrix *m) {
    if ((m1->m == m2->m) && (m1->n == m2->n)) {
        for (int i = 0; i < m1->m; i++) {
            for (int j = 0; j < m2->n; j++) {
                m->entries[i][j] = m1->entries[i][j] + m2->entries[i][j];
            }
        }
    } else {
        printf("Dimension mistmatch subtract: %dx%d %dx%d\n", m1->m, m1->n, m2->m, m2->n);
        exit(1);
    }
}

void matrix_de_crossEntropy(Matrix *m1, Matrix *m2, Matrix *m) {
    if ((m1->m == m2->m) && (m1->n == m2->n)) {
        double total = 0;
        for (int i = 0; i < m1->m; i++) {
            for (int j = 0; j < m1->n; j++) {
                m->entries[i][j] = m1->entries[i][j] - m2->entries[i][j];

            }
            total++;
        }
    } else {
        printf("Dimension mistmatch subtract: %dx%d %dx%d\n", m1->m, m1->n, m2->m, m2->n);
        exit(1);
    }
}

void matrix_multiply(Matrix *m1, Matrix *m2, Matrix *m) {
    if ((m1->m == m2->m) && (m1->n == m2->n)) {
        for (int i = 0; i < m1->m; i++) {
            for (int j = 0; j < m2->n; j++) {
                m->entries[i][j] = m1->entries[i][j] * m2->entries[i][j];
            }
        }
    } else {
        printf("Dimension mistmatch multiply: %dx%d %dx%d\n", m1->m, m1->n, m2->m, m2->n);
        exit(1);
    }
}

void matrix_divide(Matrix *m1, Matrix *m2, Matrix *m) {
    if ((m1->m == m2->m) && (m1->n == m2->n)) {
        for (int i = 0; i < m1->m; i++) {
            for (int j = 0; j < m2->n; j++) {
                m->entries[i][j] = m1->entries[i][j] / m2->entries[i][j];
            }
        }
    } else {
        printf("Dimension mistmatch multiply: %dx%d %dx%d\n", m1->m, m1->n, m2->m, m2->n);
        exit(1);
    }

}

Matrix *matrix_copy(Matrix *m) {
    Matrix *mat = matrix_create(m->m, m->n);
    for (int i = 0; i < m->m; i++) {
        for (int j = 0; j < m->n; j++) {
            mat->entries[i][j] = m->entries[i][j];
        }
    }
    return mat;
}

void matrix_scale(double n, Matrix *mat) {
    for (int i = 0; i < mat->m; i++) {
        for (int j = 0; j < mat->n; j++) {
            mat->entries[i][j] *= n;
        }
    }
}

Matrix *matrix_sqrt(Matrix *mat) {
    Matrix *mat_new = matrix_create(mat->m, mat->n);
    for (int i = 0; i < mat->m; i++) {
        for (int j = 0; j < mat->n; j++) {
            mat_new->entries[i][j] = sqrt(mat->entries[i][j]);
        }
    }
    return mat_new;
}

Matrix *matrix_scale_return(double n, Matrix *mat) {
    Matrix *mat_new = matrix_create(mat->m, mat->n);
    for (int i = 0; i < mat->m; i++) {
        for (int j = 0; j < mat->n; j++) {
            mat_new->entries[i][j] = mat->entries[i][j] * n;
        }
    }
    return mat_new;
}

void matrix_addScalar(Matrix *mat, double n) {
    for (int i = 0; i < mat->m; i++) {
        for (int j = 0; j < mat->n; j++) {
            mat->entries[i][j] += n;
        }
    }
}

void matrix_MinMaxNorm(Matrix *mat) {
    for (int i = 0; i < mat->m; i++) {
        double min = mat->entries[i][0];
        double max = mat->entries[i][0];
        for (int j = 1; j < mat->n; j++) {
            if (mat->entries[i][j] < min) {
                min = mat->entries[i][j];
            }
            if (mat->entries[i][j] > max) {
                max = mat->entries[i][j];
            }
        }

        for (int j = 0; j < mat->n; j++) {
            mat->entries[i][j] = (mat->entries[i][j] - min) / (max - min);
        }
    }
}

Matrix *matrix_softmax(Matrix *m) {
    //Matrix* m = matrix_copy(mat_inp);
    //matrix_MinMaxNorm(m);
    Matrix *sum = matrix_sum_exp(m, 1);
    Matrix *mat = matrix_create(m->m, m->n);
    for (int i = 0; i < mat->m; i++) {
        for (int j = 0; j < mat->n; j++) {
            mat->entries[i][j] = exp(m->entries[i][j]) / sum->entries[0][i];
        }
    }
    matrix_free(sum);
    //matrix_free(m);
    return mat;
}

void metrics(Matrix *y_hat, Matrix *y) {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    double tp = 0;
    double fp = 0;
    int total = 0;
    for (int i = 0; i < y->m; i++) {
        double max = y_hat->entries[i][0];
        int max_ind = 0;
        for (int j = 1; j < y->n; j++) {
            if (y_hat->entries[i][j] > max) {
                max = y_hat->entries[i][j];
                max_ind = j;
            }
        }
        if (y->entries[i][max_ind] >= 0.5) {
            tp += 1;
        } else {
            fp += 1;
        }
        total += 1;

    }
    double global_tp;
    int global_total;
    MPI_Reduce(&tp, &global_tp, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total, &global_total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (world_rank == 0) {
        printf("Accuracy = %lf\n", global_tp / global_total);
    }
}