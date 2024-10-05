#include "../includes/neuralNet.h"

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

neural_net* net_init(int capacity) {
    neural_net* net = malloc(sizeof(neural_net));
    net->layers = malloc(capacity * sizeof(layer_super));
    net->layer_capacity = capacity;
    net->n_layers = 0;
    return net;
}

layer_super* layer_init(enum layer_type type) {
    layer_super* layer = malloc(sizeof(layer_super));
    layer->type = type;

    return layer;
}


layer_super* layer_init_activation(enum activation_type type) {
    layer_super* layer = layer_init(ACTIVATION);
    layer->activation_layer = activation_init(type);
    layer->gcn_layer = NULL;
    return layer;
}

layer_super* layer_init_gcn(SparseMat* adj, SparseMat* adj_T,int size_f,int size_out) {
    layer_super* layer = layer_init(GCN);
    layer->gcn_layer = gcn_init(adj, adj_T, size_f, size_out);
    layer->activation_layer = NULL;

    return layer;
}

void net_addLayer(neural_net* net,layer_super* layer) {
    if (net->layer_capacity > net->n_layers) {
        net->layers[net->n_layers] = layer;
        net->n_layers += 1;
    } else {
        printf("Layer capacity of neural net is full. Capacity:%d \n", net->layer_capacity);
    }
}

ParMatrix* net_forward(neural_net* net, ParMatrix* input) {
    for (int i = 0;i<net->n_layers;i++) {
        if(net->layers[i]->type == ACTIVATION) {
            //printf("activation %d\n",i);
            net->layers[i]->activation_layer->input = input;
            activation_forward(net->layers[i]->activation_layer);
            input = net->layers[i]->activation_layer->output;
            //printf("end %d\n",i);
        }else if(net->layers[i]->type == GCN) {
            //printf("gcn %d\n",i);
            net->layers[i]->gcn_layer->input = input;
            gcn_forward(net->layers[i]->gcn_layer);
            input = net->layers[i]->gcn_layer->output;
            //printf("end %d\n",i);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    int last = net->n_layers - 1;
    if(net->layers[last]->type == GCN){
        return net->layers[last]->gcn_layer->output;
    } else if(net->layers[last]->type == ACTIVATION){
        return net->layers[last]->activation_layer->output;
    }
    printf("Something went wrong in forwarding.\n");
    exit(0);
}

void net_backward(neural_net* net, Matrix* error, double lr, int t) {
    Matrix* tmp;
    for (int i= net->n_layers-1;i >= 0; i--) {
        if(net->layers[i]->type == GCN) {
            tmp = error;     
            error = gcn_backward(net->layers[i]->gcn_layer, error);
            matrix_free(tmp);
            gcn_step(net->layers[i]->gcn_layer, lr, t);
            //printf("end %d\n",i);
        } else if(net->layers[i]->type == ACTIVATION) {
            //printf("activation gcn %d\n",i);
            tmp = error;
            error = activation_backward(net->layers[i]->activation_layer, error, lr);
            matrix_free(tmp);
            //printf("end %d\n",i);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    matrix_free(error);
}


void net_free(neural_net* net) {
    for (int i = 0;i<net->n_layers;i++) {
        layer_free(net->layers[i]);
    }
    free(net);
}

void layer_free(layer_super* layer) {
    if (layer->type == ACTIVATION) {
        activation_free(layer->activation_layer);
    } else if (layer->type == GCN) {
        gcn_free(layer->gcn_layer);
    }
    free(layer);
}
