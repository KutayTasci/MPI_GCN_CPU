#include "../includes/basic.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <sys/sysinfo.h>

sendTable *initSendTable(SparseMat *A) {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    sendTable *table = sendTableCreate(world_size, world_rank, A->m);
    node_t *tmp_node;
    int ctr = 0;
    for (int i = 0; i < A->m; i++) {
        for (int j = A->ia[i]; j < A->ia[i + 1]; j++) {
            int u_id = A->inPart[A->ja[j]];
            if (u_id != world_rank) {
                tmp_node = table->table_t[u_id];


                while (tmp_node->next != NULL && tmp_node->val != i) {
                    tmp_node = tmp_node->next;
                }

                if (tmp_node->val == -1 && tmp_node->val != i) {
                    tmp_node->val = i;

                    table->send_count[u_id]++;
                    node_t *tmp = (node_t *) malloc(sizeof(node_t));
                    tmp->next = NULL;
                    tmp->val = -1;
                    tmp_node->next = tmp;
                }

            }
        }
    }
    //printf("Send count in processor %d => %d\n", world_rank, table->send_count[1]);

    return table;
}

recvTable *initRecvTable(sendBuffer *send_table, SparseMat *A_T) {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    recvTable *table = recvTableCreate(world_size, world_rank, A_T->gn);
    /*
    for (int i = 0; i < A_T->m; i++) {
        for (int j = A_T->ia[i]; j < A_T->ia[i+1]; j++) {
            int u_id = A_T->inPart[A_T->ja[j]];
            if (u_id != world_rank) {
                if (table->table[u_id][A_T->ja[j]] == 0) {
                    table->table[u_id][A_T->ja[j]] = 1;
                    table->recv_count[u_id]++;
                }
            }
        }
    }
    */
    MPI_Request *requests = (MPI_Request *) malloc((world_size - 1) * sizeof(MPI_Request));
    MPI_Request *requests2 = (MPI_Request *) malloc((world_size - 1) * sizeof(MPI_Request));
    int ind = 0;
    for (int i = 0; i < world_size; i++) {
        if (i != world_rank) {
            MPI_Irecv(&(table->recv_count[i]), 1, MPI_INT, i, 0, MPI_COMM_WORLD, &(requests[ind++]));
        }
    }
    for (int i = 0; i < world_size; i++) {
        if (i != world_rank) {
            int tmp = send_table->pid_map[i + 1] - send_table->pid_map[i];
            MPI_Send(&(tmp), 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Waitall(world_size - 1, requests, MPI_STATUSES_IGNORE);
    table->table = (int **) malloc(world_size * sizeof(int *));
    for (int i = 0; i < world_size; i++) {
        if (i != world_rank && table->recv_count[i] != 0) {
            table->table[i] = (int *) malloc(table->recv_count[i] * sizeof(int));
        }
    }


    ind = 0;
    for (int i = 0; i < world_size; i++) {
        if (i != world_rank && table->recv_count[i] != 0) {
            MPI_Irecv(table->table[i], table->recv_count[i], MPI_INT, i, 1, MPI_COMM_WORLD, &(requests2[ind++]));
        }
    }


    for (int i = 0; i < world_size; i++) {
        int range = send_table->pid_map[i + 1] - send_table->pid_map[i];
        int base = send_table->pid_map[i];
        if (i != world_rank && range != 0) {
            MPI_Send(&(send_table->vertices[base]), range, MPI_INT, i, 1, MPI_COMM_WORLD);
        }
    }

    MPI_Waitall(ind, requests2, MPI_STATUSES_IGNORE);


    free(requests);
    free(requests2);


    //printf("Send count in processor %d => %d\n", world_rank, table->send_count[1]);



    return table;
}

sendBuffer *initSendBuffer(sendTable *table, int *l2gMap, int feature_size) {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    sendBuffer *buffer = (sendBuffer *) malloc(sizeof(sendBuffer));
    buffer->feature_size = feature_size;

    buffer->send_count = 0;
    for (int i = 0; i < world_size; i++) {
        buffer->send_count += table->send_count[i];
    }
    buffer->vertices_local = (int *) malloc(sizeof(int) * buffer->send_count);
    buffer->vertices = (int *) malloc(sizeof(int) * buffer->send_count);
    int ind = 0;
    buffer->pid_map = (int *) malloc(sizeof(int) * (world_size + 1));
    for (int i = 0; i < world_size; i++) {
        buffer->pid_map[i] = ind;
        ind += table->send_count[i];
    }
    buffer->pid_map[world_size] = ind;
    for (int i = 0; i < world_size; i++) {
        if (table->send_count[i] != 0) {
            ind = 0;
            node_t *curr = table->table_t[i];
            /*
            for (int j = 0;j < table->n; j++) {
                if (table->table[i][j] != 0) {
                    buffer->vertices_local[buffer->pid_map[i] + ind] = j;
                    buffer->vertices[buffer->pid_map[i] + ind++] = l2gMap[j];
                }
            }
            */
            while (curr->next != NULL) {
                buffer->vertices_local[buffer->pid_map[i] + ind] = curr->val;
                buffer->vertices[buffer->pid_map[i] + ind++] = l2gMap[curr->val];
                curr = curr->next;
            }
        }
    }

    return buffer;
}

recvBuffer *initRecvBuffer(recvTable *table, int feature_size) {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    recvBuffer *buffer = (recvBuffer *) malloc(sizeof(recvBuffer));
    buffer->feature_size = feature_size;

    buffer->recv_count = 0;
    for (int i = 0; i < world_size; i++) {
        buffer->recv_count += table->recv_count[i];
    }
    buffer->vertices = (int *) malloc(sizeof(int) * buffer->recv_count);
    int ind = 0;

    buffer->pid_map = (int *) malloc(sizeof(int) * (world_size + 1));
    for (int i = 0; i < world_size; i++) {
        buffer->pid_map[i] = ind;
        ind += table->recv_count[i];

    }
    buffer->pid_map[world_size] = ind;

    for (int i = 0; i < world_size; i++) {

        if (i != world_rank && table->recv_count[i] != 0) {
            for (int j = 0; j < table->recv_count[i]; j++) {
                buffer->vertices[buffer->pid_map[i] + j] = table->table[i][j];
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    return buffer;
}

bool printFreeMemory() {
    struct sysinfo sys_info;
    if (sysinfo(&sys_info) != 0) {
        perror("sysinfo");
        return 0;
    }
    unsigned long free_memory = sys_info.freeram * sys_info.mem_unit;
    printf("Free memory: %lu bytes\n", free_memory);
    return 1;
}