//
// Created by serdar on 10/16/24.
//

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include "../includes/comm.h"
#include "../includes/basic.h"

// used to send messages in random order
void shuffle(int array[], int n) {
//    srand(time(NULL)); // Seed the random number generator
    for (int i = n - 1; i > 0; i--) {
        // Generate a random index j such that 0 <= j <= i
        int j = rand() % (i + 1);

        // Swap array[i] and array[j]
        int temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

OPComm *initOPComm(SparseMat *adj, SparseMat *adj_T, int size_f, int size_out) {
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    OPComm *comm = (OPComm *) malloc(sizeof(OPComm));
    int i, temp;
    comm->adjacency = adj;
    comm->adjacency_T = adj_T;

    sendTable *sTable = initSendTable(adj);
    comm->sendBuffer = initSendBuffer(sTable, adj->l2gMap, size_f);

    recvTable *rTable = initRecvTable(comm->sendBuffer, adj_T);
    comm->recvBuffer = initRecvBuffer(rTable, size_f);


    sendTableFree(sTable);
    recvTableFree(rTable);

    sendTable *sTable_b = initSendTable(adj_T);
    comm->sendBuffer_backward = initSendBuffer(sTable_b, adj_T->l2gMap, size_out);

    recvTable *rTable_b = initRecvTable(comm->sendBuffer_backward, adj);
    comm->recvBuffer_backward = initRecvBuffer(rTable_b, size_out);


    sendTableFree(sTable_b);
    recvTableFree(rTable_b);

    comm->recvBuffMap = (int *) malloc(sizeof(int) * adj_T->gm);
    memset(comm->recvBuffMap, -1, sizeof(int) * adj_T->gm);

    for (i = 0; i < adj_T->m; i++) {
        temp = adj_T->l2gMap[i];
        comm->recvBuffMap[temp] = i;
    }

    for (i = 0; i < comm->recvBuffer->recv_count; i++) {
        temp = comm->recvBuffer->vertices[i];
        comm->recvBuffMap[temp] = i;
    }

    comm->recvBuffMap_backward = (int *) malloc(sizeof(int) * adj->gm);
    memset(comm->recvBuffMap_backward, -1, sizeof(int) * adj->gm);


    for (i = 0; i < adj->m; i++) {
        temp = adj->l2gMap[i];
        comm->recvBuffMap_backward[temp] = i;
    }

    for (i = 0; i < comm->recvBuffer_backward->recv_count; i++) {
        temp = comm->recvBuffer_backward->vertices[i];
        comm->recvBuffMap_backward[temp] = i;
    }


    if (adj_T->init == 0) {
        for (i = 0; i < adj_T->nnz; i++) {
            temp = adj_T->ja[i];
            adj_T->ja_mapped[i] = comm->recvBuffMap[temp];
        }

        generate_parCSR(adj_T, comm->recvBuffMap, world_size, world_rank);
        adj_T->init = 1;
    }

    if (adj->init == 0) {
        for (i = 0; i < adj->nnz; i++) {
            temp = adj->ja[i];
            adj->ja_mapped[i] = comm->recvBuffMap_backward[temp];
        }
        generate_parCSR(adj, comm->recvBuffMap_backward, world_size, world_rank);
        adj->init = 1;
    }

    comm->msgSendCount = 0;
    comm->msgRecvCount = 0;
    for (i = 0; i < world_size; i++) {
        int range = comm->sendBuffer->pid_map[i + 1] - comm->sendBuffer->pid_map[i];
        int rRange = comm->recvBuffer->pid_map[i + 1] - comm->recvBuffer->pid_map[i];
        if (i != world_rank) {
            if (range != 0) {
                comm->msgSendCount++;
            }
            if (rRange != 0) {
                comm->msgRecvCount++;
            }
        }
    }

    comm->sendBuffer->list = (int *) malloc(sizeof(int) * comm->msgSendCount);
    comm->recvBuffer->list = (int *) malloc(sizeof(int) * comm->msgRecvCount);

    int ctr = 0, ctr_r = 0;
    for (i = 0; i < world_size; i++) {
        int range = comm->sendBuffer->pid_map[i + 1] - comm->sendBuffer->pid_map[i];
        int rRange = comm->recvBuffer->pid_map[i + 1] - comm->recvBuffer->pid_map[i];
        if (i != world_rank) {
            if (range != 0) {
                comm->sendBuffer->list[ctr] = i;
                ctr++;
            }
            if (rRange != 0) {
                comm->recvBuffer->list[ctr_r] = i;
                ctr_r++;
            }
        }
    }


    comm->msgSendCount_b = 0;
    comm->msgRecvCount_b = 0;
    for (i = 0; i < world_size; i++) {
        int range = comm->sendBuffer_backward->pid_map[i + 1] - comm->sendBuffer_backward->pid_map[i];
        int rRange = comm->recvBuffer_backward->pid_map[i + 1] - comm->recvBuffer_backward->pid_map[i];
        if (i != world_rank) {
            if (range != 0) {
                comm->msgSendCount_b++;
            }
            if (rRange != 0) {
                comm->msgRecvCount_b++;
            }
        }
    }

    comm->sendBuffer_backward->list = (int *) malloc(sizeof(int) * comm->msgSendCount_b);
    comm->recvBuffer_backward->list = (int *) malloc(sizeof(int) * comm->msgRecvCount_b);

    ctr = 0, ctr_r = 0;
    for (i = 0; i < world_size; i++) {
        int range = comm->sendBuffer_backward->pid_map[i + 1] - comm->sendBuffer_backward->pid_map[i];
        int rRange = comm->recvBuffer_backward->pid_map[i + 1] - comm->recvBuffer_backward->pid_map[i];
        if (i != world_rank) {
            if (range != 0) {
                comm->sendBuffer_backward->list[ctr] = i;
                ctr++;
            }
            if (rRange != 0) {
                comm->recvBuffer_backward->list[ctr_r] = i;
                ctr_r++;
            }
        }
    }
    initSendBufferSpace(comm->sendBuffer);
    initRecvBufferSpace(comm->recvBuffer);
    initSendBufferSpace(comm->sendBuffer_backward);
    initRecvBufferSpace(comm->recvBuffer_backward);

    if (adj->init == 0) {
        int local_send_volume = comm->sendBuffer->send_count;
        int local_recv_volume = comm->recvBuffer->recv_count;
        int local_send_volume_b = comm->sendBuffer_backward->send_count;
        int local_recv_volume_b = comm->recvBuffer_backward->recv_count;

        int total_send_volume = local_send_volume;
        int total_recv_volume = local_recv_volume;

        int local_msg_count_send = comm->msgSendCount;
        int local_msg_count_recv = comm->msgRecvCount;

        int max_send_volume, max_recv_volume;
        int max_msg_count_send, max_msg_count_recv;
        int total_send_volume_sum, total_recv_volume_sum;
        int total_msg_count_send_sum, total_msg_count_recv_sum;

        // Compute max and average send and receive communication volumes and message counts
        MPI_Reduce(&total_send_volume, &max_send_volume, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&total_recv_volume, &max_recv_volume, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_msg_count_send, &max_msg_count_send, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_msg_count_recv, &max_msg_count_recv, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

        MPI_Reduce(&total_send_volume, &total_send_volume_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&total_recv_volume, &total_recv_volume_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_msg_count_send, &total_msg_count_send_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_msg_count_recv, &total_msg_count_recv_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        int avg_send_volume = total_send_volume_sum / world_size;
        int avg_recv_volume = total_recv_volume_sum / world_size;
        int avg_msg_count_send = total_msg_count_send_sum / world_size;
        int avg_msg_count_recv = total_msg_count_recv_sum / world_size;

        if (world_rank == 0) {
            printf("Initialization Complete:\n");
            printf("Max Send Volume: %d\n", max_send_volume);
            printf("Average Send Volume: %d\n", avg_send_volume);
            printf("Max Receive Volume: %d\n", max_recv_volume);
            printf("Average Receive Volume: %d\n", avg_recv_volume);
            printf("Max Send Message Count: %d\n", max_msg_count_send);
            printf("Average Send Message Count: %d\n", avg_msg_count_send);
            printf("Max Receive Message Count: %d\n", max_msg_count_recv);
            printf("Average Receive Message Count: %d\n", avg_msg_count_recv);
        }
    }
    free(comm->recvBuffMap);
    free(comm->recvBuffMap_backward);
    return comm;
}

TPW *initTPComm(SparseMat *adjacency, SparseMat *adjacency_T, int size_f, int size_out, bool preduce,
                char *comm_file, char *comm_file_T) {
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    TPW *comm = (TPW *) malloc(sizeof(TPW));
    readTPComm(comm_file, size_f, preduce, &(comm->tpComm));
    readTPComm(comm_file_T, size_out, preduce, &(comm->tpComm_backward));
    map_csr(adjacency, &(comm->tpComm));
    map_csr(adjacency_T, &(comm->tpComm_backward));
    prep_comm_tp(&(comm->tpComm));
    prep_comm_tp(&(comm->tpComm_backward));
    return comm;
}


void readTPComm(char *fName, int f, bool partial_reduce, TP_Comm *Comm) {
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int64_t sloc;
    FILE *fpmat = fopen(fName, "rb");
    if (fpmat == NULL) {
        if (world_rank == 0)
            printf("Phase comm file named '%s' not found\n", fName);
        exit(1);
    }
    fseek(fpmat, (world_rank * sizeof(int64_t)), SEEK_SET);
    fread(&sloc, sizeof(int64_t), 1, fpmat);


    fseek(fpmat, sloc, SEEK_SET);
    fread(&(Comm->sendBuffer_p1.count), sizeof(int), 1, fpmat);
    fread(&(Comm->recvBuffer_p1.count), sizeof(int), 1, fpmat);
    fread(&(Comm->sendBuffer_p2.count), sizeof(int), 1, fpmat);
    fread(&(Comm->recvBuffer_p2.count), sizeof(int), 1, fpmat);


    Comm->sendBuffer_p1.proc_map = (int *) malloc((world_size + 1) * sizeof(int));
    Comm->sendBuffer_p1.row_map = (int *) malloc(Comm->sendBuffer_p1.count * sizeof(int));

    Comm->recvBuffer_p1.proc_map = (int *) malloc((world_size + 1) * sizeof(int));
    Comm->recvBuffer_p1.row_map = (int *) malloc(Comm->recvBuffer_p1.count * sizeof(int));

    Comm->sendBuffer_p2.proc_map = (int *) malloc((world_size + 1) * sizeof(int));
    Comm->sendBuffer_p2.row_map = (int *) malloc(Comm->sendBuffer_p2.count * sizeof(int));

    Comm->recvBuffer_p2.proc_map = (int *) malloc((world_size + 1) * sizeof(int));
    Comm->recvBuffer_p2.row_map = (int *) malloc(Comm->recvBuffer_p2.count * sizeof(int));

    fread(Comm->sendBuffer_p1.proc_map, sizeof(int), world_size + 1, fpmat);
    fread(Comm->recvBuffer_p1.proc_map, sizeof(int), world_size + 1, fpmat);
    fread(Comm->sendBuffer_p2.proc_map, sizeof(int), world_size + 1, fpmat);
    fread(Comm->recvBuffer_p2.proc_map, sizeof(int), world_size + 1, fpmat);
    Comm->msgRecvCount_p1 = 0;
    Comm->msgRecvCount_p2 = 0;
    Comm->msgSendCount_p1 = 0;
    Comm->msgSendCount_p2 = 0;

    for (int i = 1; i <= world_size; ++i) {
        if (Comm->sendBuffer_p1.proc_map[i] - Comm->sendBuffer_p1.proc_map[i - 1] != 0) {
            Comm->msgSendCount_p1++;
        }
        if (Comm->sendBuffer_p2.proc_map[i] - Comm->sendBuffer_p2.proc_map[i - 1] != 0) {
            Comm->msgSendCount_p2++;
        }
        if (Comm->recvBuffer_p1.proc_map[i] - Comm->recvBuffer_p1.proc_map[i - 1] != 0) {
            Comm->msgRecvCount_p1++;
        }
        if (Comm->recvBuffer_p2.proc_map[i] - Comm->recvBuffer_p2.proc_map[i - 1] != 0) {
            Comm->msgRecvCount_p2++;
        }
    }

    fread(Comm->sendBuffer_p1.row_map, sizeof(int), Comm->sendBuffer_p1.count, fpmat);
    fread(Comm->recvBuffer_p1.row_map, sizeof(int), Comm->recvBuffer_p1.count, fpmat);
    fread(Comm->sendBuffer_p2.row_map, sizeof(int), Comm->sendBuffer_p2.count, fpmat);
    fread(Comm->recvBuffer_p2.row_map, sizeof(int), Comm->recvBuffer_p2.count, fpmat);

    Comm->sendBuffer_p1.f = f;
    Comm->recvBuffer_p1.f = f;
    Comm->sendBuffer_p2.f = f;
    Comm->recvBuffer_p2.f = f;

    CommBufferInit(&(Comm->sendBuffer_p1));
    CommBufferInit(&(Comm->sendBuffer_p2));

    if (partial_reduce != 0) {
        Comm->reducer.init = true;
        fread(&(Comm->reducer.reduce_count), sizeof(int), 1, fpmat);
        Comm->reducer.reduce_list = (int *) malloc(Comm->reducer.reduce_count * sizeof(int));
        Comm->reducer.reduce_list_mapped = (int *) malloc(Comm->reducer.reduce_count * sizeof(int));
        Comm->reducer.reduce_source_mapped = (int **) malloc(Comm->reducer.reduce_count * sizeof(int *));
        Comm->reducer.reduce_source_factors = (double **) malloc(Comm->reducer.reduce_count * sizeof(double *));
        for (int i = 0; i < Comm->reducer.reduce_count; i++) {
            fread(&(Comm->reducer.reduce_list[i]), sizeof(unsigned int), 1, fpmat);
            int tmp;
            fread(&(tmp), sizeof(int), 1, fpmat);
            Comm->reducer.reduce_source_mapped[i] = (int *) malloc((tmp + 1) * sizeof(int));
            Comm->reducer.reduce_source_mapped[i][0] = tmp;
            fread(&(Comm->reducer.reduce_source_mapped[i][1]), sizeof(int), tmp, fpmat);
            Comm->reducer.reduce_source_factors[i] = (double *) malloc(tmp * sizeof(double));
            fread(Comm->reducer.reduce_source_factors[i], sizeof(double), tmp, fpmat);
        }
    } else {
        Comm->reducer.init = false;
    }
    fclose(fpmat);
}

void CommBufferInit(CommBuffer *buff) {
    double *data = (double *) malloc(buff->count * buff->f * sizeof(double));
    buff->buffer = (double **) malloc(buff->count * sizeof(double *));

    for (int i = 0; i < buff->count; i++) {
        buff->buffer[i] = &(data[buff->f * i]);
    }

}

void CommBufferFree(CommBuffer *buff) {
    free(buff->proc_map);
    free(buff->row_map);
    free(buff->buffer[0]);
    free(buff->buffer);
    free(buff);
    buff = NULL;
}

void map_csr(SparseMat *A, TP_Comm *comm) {
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int *global_map = (int *) malloc(A->gn * sizeof(int));

    for (int i = 0; i < A->gn; i++) {
        global_map[i] = -1;
    }

    for (int i = 0; i < A->m; i++) {
        global_map[A->l2gMap[i]] = i;
    }
    int base = A->m;
    for (int i = 0; i < comm->recvBuffer_p1.count; i++) {
        global_map[comm->recvBuffer_p1.row_map[i]] = base + i;
    }


    base += comm->recvBuffer_p1.count;
    for (int i = 0; i < comm->recvBuffer_p2.count; i++) {
        global_map[comm->recvBuffer_p2.row_map[i]] = base + i;
    }

    for (int i = 0; i < A->nnz; i++) {
        A->ja_mapped[i] = global_map[A->ja[i]];
        if (A->ja_mapped[i] == -1) {
            printf("Incoming seg fault 1 -> %d\n", A->ja[i]);
        }
    }

    comm->sendBuffer_p1.row_map_lcl = (int *) malloc(comm->sendBuffer_p1.count * sizeof(int));
    for (int i = 0; i < comm->sendBuffer_p1.count; ++i) {
        comm->sendBuffer_p1.row_map_lcl[i] = global_map[comm->sendBuffer_p1.row_map[i]];
        if (comm->sendBuffer_p1.row_map_lcl[i] == -1) {
            printf("Incoming seg fault 2\n");
        }
    }

    comm->sendBuffer_p2.row_map_lcl = (int *) malloc(comm->sendBuffer_p2.count * sizeof(int));
    for (int i = 0; i < comm->sendBuffer_p2.count; ++i) {
        comm->sendBuffer_p2.row_map_lcl[i] = global_map[comm->sendBuffer_p2.row_map[i]];
        if (comm->sendBuffer_p2.row_map_lcl[i] == -1) {
            printf("Incoming seg fault 3\n");
        }
    }

    comm->recvBuffer_p1.row_map_lcl = (int *) malloc(comm->recvBuffer_p1.count * sizeof(int));
    for (int i = 0; i < comm->recvBuffer_p1.count; ++i) {
        comm->recvBuffer_p1.row_map_lcl[i] = global_map[comm->recvBuffer_p1.row_map[i]];
        if (comm->recvBuffer_p1.row_map_lcl[i] == -1) {
            printf("Incoming seg fault 4\n");
        }
    }
    comm->recvBuffer_p2.row_map_lcl = (int *) malloc(comm->recvBuffer_p2.count * sizeof(int));
    for (int i = 0; i < comm->recvBuffer_p2.count; ++i) {
        comm->recvBuffer_p2.row_map_lcl[i] = global_map[comm->recvBuffer_p2.row_map[i]];
        if (comm->recvBuffer_p2.row_map_lcl[i] == -1) {
            printf("Incoming seg fault 5\n");
        }
    }

    if (comm->reducer.init) {
        int *tmp = (int *) malloc(comm->reducer.reduce_count * sizeof(int));
        int ctr = 0;
        int flag;
        for (int i = 0; i < comm->reducer.reduce_count; i++) {
            comm->reducer.reduce_list_mapped[i] = global_map[comm->reducer.reduce_list[i]];
            flag = 0;
            for (int j = 1; j <= comm->reducer.reduce_source_mapped[i][0]; j++) {
                comm->reducer.reduce_source_mapped[i][j] = global_map[comm->reducer.reduce_source_mapped[i][j]];
                if (comm->reducer.reduce_source_mapped[i][j] == -1) {
                    printf("Incoming seg fault 6\n");
                }
                if (A->inPart[comm->reducer.reduce_source_mapped[i][j]] != world_rank) {
                    flag = 1;
                }
            }
            if (flag == 1) {
                ctr++;
                tmp[i] = 1;
            } else {
                tmp[i] = 0;
            }
        }

        comm->reducer.reduce_local = (int *) malloc((comm->reducer.reduce_count - ctr) * sizeof(int));
        comm->reducer.reduce_nonlocal = (int *) malloc(ctr * sizeof(int));
        comm->reducer.lcl_count = comm->reducer.reduce_count - ctr;
        comm->reducer.nlcl_count = ctr;
        int ctr_0 = 0, ctr_1 = 0;
        for (int i = 0; i < comm->reducer.reduce_count; i++) {
            if (tmp[i] == 1) {
                comm->reducer.reduce_nonlocal[ctr_0++] = i;
            } else {
                comm->reducer.reduce_local[ctr_1++] = i;
            }
        }
    }
    free(global_map);
    comm->A = A;
}

void prep_comm_tp(TP_Comm *Comm) {
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int i;
    int base, range, part;
    int *send_ls1 = (int *) malloc(Comm->msgSendCount_p1 * sizeof(int));
    int *recv_ls1 = (int *) malloc(Comm->msgRecvCount_p1 * sizeof(int));
    int *send_ls2 = (int *) malloc(Comm->msgSendCount_p2 * sizeof(int));
    int *recv_ls2 = (int *) malloc(Comm->msgRecvCount_p2 * sizeof(int));

    int ctr = 0, ctrp = 0;
    for (int i = 0; i < world_size; ++i) {
        if (Comm->sendBuffer_p1.proc_map[i + 1] - Comm->sendBuffer_p1.proc_map[i] != 0) {
            send_ls1[ctr] = i;
            ctr++;
        }


        if (Comm->recvBuffer_p1.proc_map[i + 1] - Comm->recvBuffer_p1.proc_map[i] != 0) {
            recv_ls1[ctrp] = i;
            ctrp++;
        }
    }
    ctr = 0, ctrp = 0;
    for (int i = 0; i < world_size; ++i) {
        if (Comm->sendBuffer_p2.proc_map[i + 1] - Comm->sendBuffer_p2.proc_map[i] != 0) {
            send_ls2[ctr] = i;
            ctr++;
        }


        if (Comm->recvBuffer_p2.proc_map[i + 1] - Comm->recvBuffer_p2.proc_map[i] != 0) {
            recv_ls2[ctrp] = i;
            ctrp++;
        }
    }

    shuffle(send_ls1, Comm->msgSendCount_p1);
    Comm->send_proc_list_p1 = send_ls1;
    Comm->recv_proc_list_p1 = recv_ls1;
    shuffle(send_ls2, Comm->msgSendCount_p2);
    Comm->send_proc_list_p2 = send_ls2;
    Comm->recv_proc_list_p2 = recv_ls2;

    Comm->send_ls_p1 = (MPI_Request *) malloc((Comm->msgSendCount_p1) * sizeof(MPI_Request));
    Comm->recv_ls_p1 = (MPI_Request *) malloc((Comm->msgRecvCount_p1) * sizeof(MPI_Request));
    Comm->send_ls_p2 = (MPI_Request *) malloc((Comm->msgSendCount_p2) * sizeof(MPI_Request));
    Comm->recv_ls_p2 = (MPI_Request *) malloc((Comm->msgRecvCount_p2) * sizeof(MPI_Request));
}

void map_comm_tp(TP_Comm *Comm, Matrix *B) {
    int i;
    int base, range, part;
    for (i = 0; i < Comm->msgRecvCount_p1; i++) {
        part = Comm->recv_proc_list_p1[i];
        range = Comm->recvBuffer_p1.proc_map[part + 1] - Comm->recvBuffer_p1.proc_map[part];
        base = B->m + Comm->recvBuffer_p1.proc_map[part] + Comm->reducer.reduce_count;
        if (base + range > B->total_m) {
            printf("yo error!!\n");
        }
        MPI_Recv_init(&(B->entries[base][0]),
                      range * B->n,
                      MPI_DOUBLE,
                      part,
                      0,
                      MPI_COMM_WORLD,
                      &(Comm->recv_ls_p1[i]));
    }


    for (i = 0; i < Comm->msgRecvCount_p2; i++) {
        part = Comm->recv_proc_list_p2[i];
        range = Comm->recvBuffer_p2.proc_map[part + 1] - Comm->recvBuffer_p2.proc_map[part];
        base = B->m + Comm->recvBuffer_p1.count + Comm->recvBuffer_p2.proc_map[part] + Comm->reducer.reduce_count;
        if (base + range > B->total_m) {
            printf("yo error!!\n");
        }
        MPI_Recv_init(&(B->entries[base][0]),
                      range * B->n,
                      MPI_DOUBLE,
                      part,
                      1,
                      MPI_COMM_WORLD,
                      &(Comm->recv_ls_p2[i]));
    }

}

int get_buffer_space(TPW *comm) {
    return comm->tpComm.recvBuffer_p1.count + comm->tpComm.recvBuffer_p2.count + comm->tpComm.reducer.reduce_count;
}

int get_comm_buffer_space(TPW *comm) {
    return comm->tpComm.recvBuffer_p1.count + comm->tpComm.recvBuffer_p2.count;
}
