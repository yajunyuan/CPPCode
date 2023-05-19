#pragma once
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>   //º∆À„ ±º‰
using namespace std::chrono;
#include "cuda_runtime.h"
#include <cuda.h>
#include "device_launch_parameters.h"
#include "cublas_v2.h"
#define RESULTSIZE 500
class GPUCAL
{
public:
	cublasHandle_t cuHandle;
	int imgArrayRow, imgArrayCol;
	int* result, *result1;
	int* dimgArray;
	int* dresult, * dresult1;
	GPUCAL(int imgArrayrow, int imgArraycol);
	~GPUCAL();
	void MatCal(int* imgArray, int* result, double upper, double lower);
};