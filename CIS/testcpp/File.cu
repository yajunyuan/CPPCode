#include "cuda_runtime.h"  
#include "device_launch_parameters.h"
#include <iostream>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
__constant__ int constant_roi_row;
__constant__ int constant_xdlist_col;
dim3 dg(16, 16);
dim3 dbs((48000 + dg.x - 1) / dg.x, (600 + dg.y - 1) / dg.y);

__global__ void sumroi(double* da, double* db, int col_a, int roirow, int roinumber)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int i = col * 48000 + row;
    if (i < col_a * roinumber) {
        for (int j = 0; j < roirow; j++) {
            db[i] += da[int(i/ col_a)* roirow* col_a+j * col_a + i% col_a];
        }
    }
}

__global__ void sumimg(double* da, double* db, double* roiparam, int row_a, int col_a, int roinumber)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int i = col * 48000 + row;
    if (i < int(roiparam[2]) * roinumber) {
        for (int j = 0; j < row_a; j++) {
            db[i] += da[col_a * j + int(roiparam[i / int(roiparam[2])])+i% int(roiparam[2])];
        }
    }
}

__global__ void getindex(double* da, double* db, double* dc, double* dstartposition, int* dindex, int col_a,int row_b, int col_b, int roirow, int col_c)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int i = col * 48000 + row;
    if (i < row_b) {
        double tmp = db[0];
        //int index = 0;
        for (int j = 1; j < col_b; j++) {
            if (db[i * col_b + j] < tmp) {
                tmp = db[i * col_b + j];
                dindex[i] =  j;
            }
        }
        //for (int l = 0; l < roirow; l++) {
        //    for (int k = index - 30; k < index + 30; k++) {
        //        dc[i * col_c * roirow+l * col_c + k - (index - 30)] = da[i* col_a *roirow+l * col_a + k];
        //    }
        //}
        dstartposition[i] = dstartposition[i] + dindex[i] - 35;
        //printf("%d 是  %f  %d\n", i, tmp, dindex[i]);
    }
}

__global__ void realroi(double* da, double* db, int* index, int row_b,int col_a, int col_b, int roirow)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    //if (row < row_b && col < col_b) {
    //    db[row * col_b + col] = da[(row % roirow) * col_a + index[int(row / roirow)] - 30];
    //}
    if (row < row_b && col < col_b) {
        db[row * col_b + col] = da[row * col_a + col + index[int(row/roirow)] - 35];
    }
}

__global__ void mul(double* da, double* db, double* dc, int row_a, int col_a, int col_b)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < row_a && col < col_b) {
        dc[row * col_b + col] = 0;
        for (int i = 0; i < col_a; i++) {
            dc[row * col_b + col] += da[row * col_a + i] * db[i * col_b + col];
        }
    }
}

__global__ void gradient(double *da, double *dc, int row_a, int col_a)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int i = col* 48000+ row;
    if (i < row_a) {
        for (int j = 1; j < col_a - 1; j++) {
            dc[i * (col_a) + j] = (da[i * col_a + j + 1] - da[i * col_a + j - 1]) / 2.0;
        }
    }
}

__global__ void grad1com(double* da, double* db, double* dc, int row_a, int col_a, int row_b, int col_b, int pre, double* dstart_position, int roirow)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int i = col * 48000 + row;
    if (i < row_a) {
        dc[i * 4 + 1] = 255.0;
        for (int j = col_b / pre; j < col_b - col_b / pre; j++) {
            if (da[i * col_a + j + 1] < dc[i * 4 + 1] && (db[i * col_b + j] < 0 && db[i * col_b + j + 1] > 0)) {
                dc[i * 4] = (double)(j) / pre + dstart_position[int(i / roirow)];
                dc[i * 4 + 1] = da[i * col_a + j+1];
            }
        }
    }
}
__global__ void com(double* da, double* db, double* dc, int row_a, int col_a, int row_b, int col_b, int pre, double* dstart_position, int roirow)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int i = col * 48000 + row;
    if (i < row_a) {
        dc[i * 4 + 2] = 255.0;
        dc[i * 4 + 3] = -255.0;
        for (int j = col_b / pre; j < col_b - col_b / pre; j++) {
            if (da[i * col_a + j + 1] < dc[i * 4 + 2] && (db[i * col_b + j] < 0 && db[i * col_b + j + 1] > 0)) {
                dc[i * 4] = (double)(j) / pre + dstart_position[int(i / roirow)];
                dc[i * 4 + 2] = da[i * col_a + j];
            }
            if (da[i * col_a + j + 1] > dc[i * 4 + 3] && (db[i * col_b + j] > 0 && db[i * col_b + j + 1] < 0)) {
                dc[i * 4 + 1] = (double)(j) / pre + dstart_position[int(i / roirow)];
                dc[i * 4 + 3] = da[i * col_a + j];
            }
        }
    }
}

__global__ void diff(double* da, double* dc, int row_c, int roirow)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int i = col * 48000 + row;
    if (i < row_c) {
        dc[i] = da[(i + roirow) * 4] - da[i * 4];
        //dc[i] = da[(i + roirow) * 4 ] - da[i * 4 + 1];
    }
}

__global__ void med(double* da, double* dc, int row_c, int roirow)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int i = col * 48000 + row;
    if (i < row_c) {
        dc[i] = (da[i * roirow + int(roirow/2)] + da[i * roirow + int(roirow / 2)+1]) / 2;
    }
}

extern "C" void mextix1(double* da, double* db, double* dc, int row_a, int col_a, int col_b)
{
    mul <<<dbs, dg >>> (da, db, dc, row_a, col_a, col_b);
}

extern "C" void getgradient(double* da, double* dc, int row_a, int col_a)
{
    gradient << <dbs, dg>> > (da,  dc, row_a, col_a);
}

extern "C" void compare(double* da, double* db, double* dc, int row_a, int col_a, int row_b, int col_b, int pre, double* dstart_position, int roirow)
{

    grad1com << <dbs, dg >> > (da, db, dc, row_a, col_a, row_b, col_b, pre, dstart_position, roirow);
}

extern "C" void diffcal(double* da, double* dc, int row_c, int roirow)
{

    diff << <dbs, dg >> > (da, dc, row_c, roirow);
}

extern "C" void medcal(double* da, double* dc, int row_c, int roirow)
{
    for (int i = 0; i < row_c; i++) {
        thrust::sort(thrust::device_pointer_cast(da + i * roirow), thrust::device_pointer_cast(da + (i + 1) * roirow));
    }
    med << <dbs, dg >> > (da, dc, row_c, roirow);
}


extern "C" void getroisum(double* da, double* db, int col_a, int roirow, int roinumber)
{
    sumroi << <dbs, dg >> > (da, db, col_a, roirow, roinumber);
}

extern "C" void getrealindex(double* da, double* db, double* dc, double* dstartposition, int* dindex, int col_a, int roinumber, int col_b, int roirow, int col_c)
{
    getindex << <dbs, dg >> > (da, db, dc, dstartposition, dindex, col_a, roinumber, col_b, roirow, col_c);
}

extern "C" void getrealroi(double* da, double* db, double* dc, double* dstartposition, int* dindex, int col_a, int roinumber, int col_b, int roirow, int col_c)
{
    getindex << <dbs, dg >> > (da, db, dc, dstartposition, dindex, col_a, roinumber, col_b, roirow, col_c);
    realroi << <dbs, dg >> > (da, dc, dindex, roinumber * roirow, col_a, col_c, roirow);
}