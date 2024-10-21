#ifndef FILEIO_H_INCLUDED
#define FILEIO_H_INCLUDED

#pragma once

#define MAXCHAR 200000

#include "typedef.h"

SparseMat *readSparseMat(char *fName, int partScheme, char *inPartFile);

ParMatrix *readDenseMat(char *fName, SparseMat *A, int buffer_size);

int readFeatureSize(char *fName);

#endif // FILEIO_H_INCLUDED
