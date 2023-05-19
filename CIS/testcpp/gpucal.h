#include <iostream>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>   //º∆À„ ±º‰
using namespace std::chrono;
#include "cuda_runtime.h"
#include <cuda.h>
#include "device_launch_parameters.h"
#include "cublas_v2.h"
const int Numsgongwei = 4;
const int ROIRow = 1000;
const int ROICol = 60;
const int xhigh = 40;
const int xlength = 60;
const int precnum = 100;
const int xdlength = xlength * precnum;
const int roi_array_col = ROICol;
const int xlist_row = xlength;
const int xlist_col = xhigh;
const int xdlist_row = xhigh;
const int xdlist_col = xdlength;
const int result_col = 4;
class GPUCAL
{
public:
    cublasHandle_t cuHandle;
    double* droi, * dxlist, * dxlistre, * dxdlist, * dxdlistre, * dgrad1d, * dgrad2d, * dmaxmin,*ddiff, * dresult;
    double* dstart_position, *droiparam, *droiinput, *droisum;
    double* xlistre, * xdlistre, * grad1d, * grad2d, * maxmin, *diff, *result;
    int* dindex;
    int roinumber, sin_roi_row, sin_roi_col, roi_array_row;
    GPUCAL(int roi_number, int roi_row, int roi_col);
    ~GPUCAL();
    void initxlist(double* xlist, double* xdlist);
    void operator()() {
        
    }
    void rands(double* a, int row, int col);
    int matcal(double* roi, double* start_position, double* result);
};
