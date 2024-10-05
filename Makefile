CC=mpicc

CFLAGS=-Wall -std=c99 -O2 -Wno-unused-variable -Wno-unused-function -g -Werror
LIBS=
SRC_FILE= src/typedef.c src/basic.c src/fileio.c
MAT_FILE= Matrix/Matrix.c Matrix/sparseMat.c
NN_FILE = NeuralNet/activationLayer.c NeuralNet/gcnLayer.c NeuralNet/neuralNet.c NeuralNet/lossFunctions.c
OBJ_FILE= typedef.o basic.o fileio.o Matrix.o sparseMat.o activationLayer.o gcnLayer.o neuralNet.o lossFunctions.o

all: typedef matrix_cmp neuralNet_cmp
	$(CC) $(CFLAGS) -o bin/gcn main.c $(OBJ_FILE) $(LIBS)
	make clean

typedef: build $(SRC_FILE)
	$(CC) $(CFLAGS) -c $(SRC_FILE) $(LIBS)

matrix_cmp: $(MAT_FILE)
	$(CC) $(CFLAGS) -c $(MAT_FILE) $(LIBS)

neuralNet_cmp: $(NN_FILE)
	$(CC) $(CFLAGS) -c $(NN_FILE) $(LIBS)

build:
	test -d bin || mkdir bin

.PHONY: clean
clean:
	rm -f *.o
