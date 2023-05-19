#include "cuda_runtime.h"  
#include "device_launch_parameters.h"
#include "math.h"
#include <iostream>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include "device_functions.h"
typedef struct resultarray
{
    int resultRow, resultCol;

}ResultArray;

dim3 dbs(16, 16);
dim3 dgs((14100 + dbs.x - 1) / dbs.x, (2000 + dbs.y - 1) / dbs.y);
struct Lock {
    int* mutex;
    Lock(void) {
        int state = 0;
        cudaMalloc((void**)&mutex, sizeof(int));
        cudaMemcpy(mutex, &state, sizeof(int), cudaMemcpyHostToDevice);
    }
    ~Lock(void) {
        cudaFree(mutex);
    }
    __device__ void lock(void) {
        while (atomicCAS(mutex, 0, 1) != 0);
    }
    __device__ void unlock(void) {
        atomicExch(mutex, 0);
    }
};

__device__ volatile int resultIndex = 0;

__global__ void SearchDefect(double* da, int row_a, int col_a, double upper ,double lower, int* result)
{

    __shared__ int bsresultIndex;
    int bsresult[100] = {0};
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (row % 100 == 0 && (row + 100)<= row_a && col %100 == 0 && (col + 100) <= col_a) {
            double Pweight[100];
            double grayValues_Cropprd[100];
            double AfterCalman[100];
            int SingleImage_count = 0;
            for (int i = row; i < row + 100; i++) {
                for (int j = col; j < col + 100; j++) {
                    int index = j - col;
                    double grayValue;
                    if (i == row) {
                        Pweight[index]=0.1;
                        AfterCalman[index] = da[i * row_a + j];
                        grayValue = AfterCalman[index];
                    }
                    else {
                        double Xkk_1_middle =  AfterCalman[index];
                        double Pkk_1_middle =  Pweight[index] + 0.001;
                        double Kg_middle = Pkk_1_middle / (Pkk_1_middle + 0.05);
                        double Width_middle = da[i * row_a + j];
                        double afterCalmanValue = Xkk_1_middle + Kg_middle * (Width_middle - Xkk_1_middle); //预测的灰度
                        //printf("%f   %f   %f   %f\n", Xkk_1_middle, Kg_middle, Width_middle, afterCalmanValue);
                        AfterCalman[index] = afterCalmanValue;
                        Pweight[index] = (1 - Kg_middle ) * Pkk_1_middle;
                        grayValue = afterCalmanValue;
                        if (grayValue > upper | grayValue < lower) {
                            //printf("%d row, %d col, %f  %f value\n", i, j, da[i * row_a + j], grayValue);
                            SingleImage_count++;
                        }
                    }
                    
                }
            }
            if (SingleImage_count >= 2 && SingleImage_count != 100 * 100) //需要在其中增加对针孔的判断，像素数量1个就剪切出来//将背景滤除//数量和blockSize * blockSize一致大概率为边缘
            {
                //printf("%d, %d ---\n", row, col);
                bsresult[bsresultIndex +1] = row;
                bsresult[bsresultIndex + 2] = col;
                bsresultIndex += 2;
                bsresult[0] = bsresultIndex;
            }
        }
        __syncthreads();
        if (bsresult[0] != 0) {
            for (int i = resultIndex; i < resultIndex+bsresult[0]; i++) {
                result[i] = bsresult[i - resultIndex+1];
            }
            resultIndex += bsresult[0];
            //printf("%d \n", resultIndex);
        }
        //result[bid] = shared[0];
        
}

__global__ void SearchDefect(int* da, int row_a, int col_a, int* result)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row  <row_a && col  <col_a-3){
        int dis = pow((da[row * col_a + col] - da[row * col_a + col + 2]), 2);
        if (dis >= 256) {
            //if (row <= 100 && col <= 100) {
            //    printf("row is %d  %d ", row, col);
            //}
            result[row * (col_a-2) + col] = 1; 
        }
        else {
            result[row * (col_a - 2) + col] = 0;
        }
    }
}

__global__ void SearchDefect1(int* da, int row_a, int col_a, int* result)
{
    //__shared__ int bsresultIndex;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_in_block = threadIdx.x * blockDim.y + threadIdx.y;
    if (row % 100 == 0 && (row + 100) <= row_a && col % 100 == 0 && (col + 100) <= col_a)
    //if (row % 100 == 0 && (row ) <= row_a && col % 100 == 0 && (col) <= col_a)
    {
        int SingleImage_count = 0;
        for (int i = row; i < row + 100; i++) {
            for (int j = col; j < col + 100; j++) {
                if (da[i * col_a + j] == 1) {
                    SingleImage_count++;
                }
            }
        }
        
        
        //printf("row is %d  %d SingleImage_count is %d ", row, col, SingleImage_count);
        if (SingleImage_count >= 6) //需要在其中增加对针孔的判断，像素数量1个就剪切出来//将背景滤除//数量和blockSize * blockSize一致大概率为边缘
        {
            atomicAdd((int*)&resultIndex, 2);
            result[resultIndex - 1] = row;
            result[resultIndex] = col;
            result[0] = resultIndex;
            //printf("index is %d    %d, %d ---\n", result[0], result[resultIndex - 1], result[resultIndex]);
        }
        __syncthreads();
        //__threadfence();
        //if (tid_in_block == 0)
        //{
        //    atomicAdd((int*)&g_mutex, 1);
        //    // only when all blocks add 1 go g_mutex
        //    // will g_mutex equal to goalVal
        //    while (g_mutex != goalVal)
        //    {
        //        // Do nothing here
        //    }
        //}
        
        //printf("index is %d", resultIndex);
        
    }
}

extern "C" void SearchDefectFunc(int* da, int row_a, int col_a, double upper, double lower, int* result)
{
    //SearchDefect<< <dgs, dbs >> > (da,  row_a,  col_a,  upper,  lower, result);
    SearchDefect << <dgs, dbs >> > (da, row_a, col_a,result);
}

extern "C" void SearchDefectFunc1(int* da, int row_a, int col_a, int* result)
{
    SearchDefect1 << <dgs, dbs >> > (da, row_a, col_a, result);
}

