#include "../includes/neuralNet.h"

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

neural_net *net_init(int capacity) {
    neural_net *net = malloc(sizeof(neural_net));
    net->layers = malloc(capacity * sizeof(layer_super));
    net->layer_capacity = capacity;
    net->n_layers = 0;
    return net;
}

layer_super *layer_init(enum layer_type type) {
    layer_super *layer = malloc(sizeof(layer_super));
    layer->type = type;

    return layer;
}


layer_super *layer_init_activation(enum activation_type type) {
    layer_super *layer = layer_init(ACTIVATION);
    layer->layer = activation_init(type);
    return layer;
}

layer_super *layer_init_dropout(double dropout_rate) {
    layer_super *layer = layer_init(DROPOUT);
    layer->layer = dropout_init(dropout_rate);
    return layer;
}

layer_super *layer_init_gcn(SparseMat *adj, int size_f, int size_out, bool **masks) {
    layer_super *layer = layer_init(GCN);
    layer->layer = gcn_init(adj, size_f, size_out);
    gcnLayer *gcn_layer = (gcnLayer *) layer->layer;
    gcn_layer->masks = masks;
    return layer;
}

void net_addLayer(neural_net *net, layer_super *layer) {
    if (net->layer_capacity > net->n_layers) {
        net->layers[net->n_layers] = layer;
        net->n_layers += 1;
    } else {
        printf("Layer capacity of neural net is full. Capacity:%d \n", net->layer_capacity);
    }
}

ParMatrix *net_forward(neural_net *net, ParMatrix *input, int mask_type) {

    for (int i = 0; i < net->n_layers; i++) {
        if (net->layers[i]->type == ACTIVATION) {
            activationLayer *activation_layer = (activationLayer *) net->layers[i]->layer;
            activation_forward(activation_layer);
        } else if (net->layers[i]->type == GCN) {
            gcnLayer *gcn_layer = (gcnLayer *) net->layers[i]->layer;
            gcn_forward(gcn_layer, mask_type, net->samplingComm);
        } else if (net->layers[i]->type == DROPOUT) {
            dropoutLayer *dropout_layer = (dropoutLayer *) net->layers[i]->layer;
            dropout_forward(dropout_layer, mask_type);

        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    int last = net->n_layers - 1;
    if (net->layers[last]->type == GCN) {
        gcnLayer *gcn_layer = (gcnLayer *) net->layers[last]->layer;
        return gcn_layer->output;
    } else if (net->layers[last]->type == ACTIVATION) {
        activationLayer *activation_layer = (activationLayer *) net->layers[last]->layer;
        return activation_layer->output;
    } else if (net->layers[last]->type == DROPOUT) {
        dropoutLayer *dropout_layer = (dropoutLayer *) net->layers[last]->layer;
        return dropout_layer->output;
    } else {
        printf("Invalid layer type at last layer\n");
        exit(1);
    }
}

void net_backward(neural_net *net, Matrix *error, double lr, int t) {
    Matrix *tmp;
    for (int i = net->n_layers - 1; i >= 0; i--) {
        if (net->layers[i]->type == GCN) {
            gcnLayer *gcn_layer = (gcnLayer *) net->layers[i]->layer;
            tmp = error;
            error = gcn_backward(gcn_layer, error);
            matrix_free(tmp);
            gcn_step(gcn_layer, lr, t);
        } else if (net->layers[i]->type == ACTIVATION) {
            activationLayer *activation_layer = (activationLayer *) net->layers[i]->layer;
            tmp = error;
            error = activation_backward(activation_layer, error, lr);
            matrix_free(tmp);
        } else if (net->layers[i]->type == DROPOUT) {
            dropoutLayer *dropout_layer = (dropoutLayer *) net->layers[i]->layer;
            dropout_backward(dropout_layer, error, lr); // updates 'error' in place
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    matrix_free(error);
}


void net_free(neural_net *net) {
    for (int i = 0; i < net->n_layers; i++) {
        layer_free(net->layers[i]);
    }
    free(net);
}

void layer_free(layer_super *layer) {
    if (layer->type == ACTIVATION) {
        activationLayer *activation_layer = (activationLayer *) layer->layer;
        activation_free(activation_layer);
    } else if (layer->type == GCN) {
        gcnLayer *gcn_layer = (gcnLayer *) layer->layer;
        gcn_free(gcn_layer);
    } else if (layer->type == DROPOUT) {
        dropoutLayer *dropout_layer = (dropoutLayer *) layer->layer;
        dropout_free(dropout_layer);
    }
    free(layer);
}

gcnLayer *isNextGCN(neural_net *net, int i) {
    if (i + 1 < net->n_layers && net->layers[i + 1]->type == GCN) {
        return (gcnLayer *) net->layers[i + 1]->layer;
    } else {
        return NULL;
    }
}

void net_prepare(neural_net *net, ParMatrix *X) {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    ParMatrix *input = X;
    for (int i = 0; i < net->n_layers; i++) {
        if (net->layers[i]->type == GCN) {
            gcnLayer *gcn_layer = (gcnLayer *) net->layers[i]->layer;
            gcn_layer->output = create_output_matrix_size(input, gcn_layer->size_output, 0);
            gcn_layer->input = input;
            gcn_layer->size_m = input->mat->m;
            input = gcn_layer->output;
            if (gcn_layer->comm_type == TP) {
                TPW *tpw = (TPW *) gcn_layer->comm;
                map_comm_tp(&tpw->tpComm, gcn_layer->input->mat);
            }
        } else if (net->layers[i]->type == ACTIVATION) {
            activationLayer *activation_layer = (activationLayer *) net->layers[i]->layer;
            activation_layer->output = create_gcn_output_matrix(input, isNextGCN(net, i), true);
            activation_layer->input = input;
            input = activation_layer->output;
        } else if (net->layers[i]->type == DROPOUT) {
            dropoutLayer *dropout_layer = (dropoutLayer *) net->layers[i]->layer;
            dropout_layer->output = create_gcn_output_matrix(input, isNextGCN(net, i), true);
            dropout_layer->input = input;
            input = dropout_layer->output;
            dropout_layer->mask = (bool *) malloc(
                    sizeof(bool) * dropout_layer->input->mat->m * dropout_layer->input->mat->n);
        }
    }
}
