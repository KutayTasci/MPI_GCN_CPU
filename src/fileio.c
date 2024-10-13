#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <stdint.h>
#include "../includes/fileio.h"
#include "../includes/typedef.h"

//Local function for generating csc format
SparseMat *readSparseMat(char *fName, int partScheme, char *inPartFile) {
    if (partScheme == STORE_BY_COLUMNS) {
        printf("STORE_BY_COLUMNS not implemented.");
        exit(EXIT_FAILURE);
    } else {
        int world_size;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        // Get the rank of the process
        int world_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

        int64_t sloc;

        SparseMat *A = (SparseMat *) malloc(sizeof(SparseMat));

        FILE *fpmat = fopen(fName, "rb");
        fread(&(A->gm), sizeof(int), 1, fpmat);
        fread(&(A->gn), sizeof(int), 1, fpmat);
        fseek(fpmat, 2 * sizeof(int) + (world_rank * sizeof(int64_t)), SEEK_SET);
        fread(&sloc, sizeof(int64_t), 1, fpmat);

        fseek(fpmat, sloc, SEEK_SET);
        fread(&(A->m), sizeof(int), 1, fpmat);
        fread(&(A->nnz), sizeof(int), 1, fpmat);

        sparseMatInit(A); // To initialize A we need gm gn m and nnz

        fread(A->ia, sizeof(int), A->m + 1, fpmat);
        fread(A->ja, sizeof(int), A->nnz, fpmat);
        fread(A->val, sizeof(double), A->nnz, fpmat);

        A->store = STORE_BY_ROWS;

        A->inPart = malloc(sizeof(*(A->inPart)) * A->gn);
        A->l2gMap = malloc(sizeof(int) * A->m);

        FILE *pf = fopen(inPartFile, "r");
        for (int i = 0; i < A->gn; ++i)
            fscanf(pf, "%d", &(A->inPart[i]));
        fclose(pf);

        int ctr = 0;
        for (int i = 0; i < A->gn; ++i) {
            if (A->inPart[i] == world_rank) {
                A->l2gMap[ctr++] = i;
            }
        }

        int *tmp = malloc(sizeof(*tmp) * A->gn);
        memset(tmp, 0, sizeof(*tmp) * A->gn);
        A->n = 0;
        for (int i = 0; i < A->m; ++i) {
            for (int j = A->ia[i]; j < A->ia[i + 1]; ++j)
                ++(tmp[A->ja[j]]);
        }

        for (int j = 0; j < A->gn; ++j) {
            if (world_rank == A->inPart[j])
                ++(tmp[j]);
        }

        for (int j = 0; j < A->gn; ++j) {
            if (tmp[j])
                ++(A->n);
        }

        //csrToCsc(A);//Fill CSC
        free(tmp);

        fclose(fpmat);

        return A;
    }
}

ParMatrix *readDenseMat(char *fName, SparseMat *A) {

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    char line[MAXCHAR];
    char *ptr;
    ParMatrix *X = (ParMatrix *) malloc(sizeof(ParMatrix));

    FILE *file = fopen(fName, "r");
    fgets(line, MAXCHAR, file);
    int gm = atoi(line);
    fgets(line, MAXCHAR, file);
    int feat_size = atoi(line);
    X->gm = gm;
    X->gn = feat_size;

    X->mat = matrix_create(A->m, feat_size);

    X->store = STORE_BY_ROWS;
    X->inPart = A->inPart;
    X->l2gMap = A->l2gMap;
    int r_ctr = 0;
    for (int i = 0; i < X->gm; i++) {

        if (X->inPart[i] == world_rank) {
            int c_ctr = 0;
            char *tok;
            fgets(line, MAXCHAR, file);
            for (tok = strtok(line, ","); tok && *tok; tok = strtok(NULL, ",")) {
                X->mat->entries[r_ctr][c_ctr++] = strtod(tok, &ptr);
            }
            //printf("%d\n", c_ctr);
            r_ctr++;
        } else {
            fgets(line, MAXCHAR, file);
        }

    }
    fclose(file);
    return X;
}
