# MPI_GCN_CPU
A basic distributed implementation of a GCN model with using MPI
usage:
mpirun -np <n> bin/gcn <input_folder> <agg_mode> <epochs> <hidden_parameters> <partioning> 