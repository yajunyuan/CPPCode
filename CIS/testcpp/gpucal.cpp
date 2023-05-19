#include<iostream>
#include"gpucal.h"

using namespace std;
extern "C" void getroisum(double* da, double* db, int col_a, int roirow, int roinumber);
extern "C" void getrealindex(double* da, double* db, double* dc, double* dstartposition, int* dindex, int col_a, int roinumber, int col_b, int roirow, int col_c);
extern "C" void getrealroi(double* da, double* db, double* dc, double* dstartposition, int* dindex, int col_a, int roinumber, int col_b, int roirow, int col_c);
extern "C" void mextix1(double* da, double* db, double* dc, int row_a, int col_a, int col_b);
extern "C" void getgradient(double* da, double* dc, int row_a, int col_a);
extern "C" void compare(double* da, double* db, double* dc, int row_a, int col_a, int row_b, int col_b, int pre, double* dstart_position, int roirow);
extern "C" void diffcal(double* da, double* dc, int row_c, int roirow);
extern "C" void medcal(double* da, double* dc, int row_c, int roirow);

GPUCAL::GPUCAL(int roi_number,int roi_row, int roi_col)
{
    roinumber = roi_number;
    roi_array_row = roi_number * roi_row;
    sin_roi_row = roi_row;
    sin_roi_col = roi_col;
    clock_t t1 = clock();
    cublasStatus_t status = cublasCreate(&cuHandle);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        if (status == CUBLAS_STATUS_NOT_INITIALIZED) {
            cout << "CUBLAS 对象实例化出错" << endl;
        }
        return;
    }
 
    xlistre = (double*)malloc(roi_array_row * xlist_col * sizeof(double));
    xdlistre = (double*)malloc(roi_array_row * xdlist_col * sizeof(double));
    grad1d = (double*)malloc(roi_array_row * xdlist_col * sizeof(double));

    grad2d = (double*)malloc(roi_array_row * (xdlist_col) * sizeof(double));
    maxmin = (double*)malloc(roi_array_row * result_col * sizeof(double));
    diff = (double*)malloc((roi_array_row - roi_row) * sizeof(double));
    result = (double*)malloc((roinumber - 1) * sizeof(double));
    ////生成随机数组
    //rands(a, row_a, col_a);
    //rands(b, row_b, col_b);

    //分配内存 GPU申请空间所需时间
    //cpy(roi_array_row, xdlist_col);
    cudaMalloc((void**)&droiinput, roinumber * sin_roi_row * sin_roi_col * sizeof(double));
    cudaMalloc((void**)&droisum, roinumber * sin_roi_col * sizeof(double));
    cudaMalloc((void**)&dindex, roinumber * sizeof(int));
    cudaMalloc((void**)&droi, roi_array_row * roi_array_col * sizeof(double));
    cudaMalloc((void**)&dstart_position, roinumber * sizeof(double));
    cudaMalloc((void**)&dxlist, xlist_row * xlist_col * sizeof(double));
    cudaMalloc((void**)&dxlistre, roi_array_row * xlist_col * sizeof(double));
    cudaMalloc((void**)&dxdlist, xdlist_row * xdlist_col * sizeof(double));
    cudaMalloc((void**)&dxdlistre, roi_array_row * xdlist_col * sizeof(double));
    cudaMalloc((void**)&dgrad1d, roi_array_row * xdlist_col * sizeof(double));
    cudaMalloc((void**)&dgrad2d, roi_array_row * xdlist_col * sizeof(double));
    cudaMalloc((void**)&dmaxmin, roi_array_row * result_col * sizeof(double));
    cudaMalloc((void**)&ddiff, (roi_array_row - roi_row) * sizeof(double));
    cudaMalloc((void**)&dresult, (roinumber - 1) * sizeof(double));

    //cudaMalloc((void**)&db, row_b * col_b * sizeof(double));
    //cudaMalloc((void**)&dc, row_a * col_b * sizeof(double));
    //cudaMalloc((void**)&dc1, row_a * col_b * sizeof(double));
    //cudaMalloc((void**)&time,blocks_num*sizeof(clock_t)*2);
    clock_t t2 = clock();
    double ts = (double)(t2 - t1);
    //CLOCKS_PER_SEC表示一秒钟内CPU运行的时钟周期数
    printf("GPU 分配空间 costtime : %lf ms\n ROI number : %d 个\n", ts / CLOCKS_PER_SEC * 1000, roinumber);

}
GPUCAL::~GPUCAL(void)
{

    //释放内存
    free(xlistre);
    xlistre = NULL;
    free(xdlistre);
    xdlistre = NULL;
    free(grad1d);
    grad1d = NULL;
    free(grad2d);
    grad2d = NULL;
    free(maxmin);
    maxmin = NULL;
    free(diff);
    diff = NULL;
    cudaFree(droiinput);
    cudaFree(droisum);
    cudaFree(dindex);
    cudaFree(droi);
    cudaFree(dstart_position);
    cudaFree(dxlist);
    cudaFree(dxlistre);
    cudaFree(dxdlist);
    cudaFree(dxdlistre);
    cudaFree(dgrad1d);
    cudaFree(dgrad2d);
    cudaFree(dmaxmin);
    cudaFree(ddiff);
    cudaFree(dresult);
    cout << "free GPUCAL memory" << endl;
}

void GPUCAL::initxlist(double* xlist, double* xdlist)
{
    //for (int i = 0; i < 10; i++) {
    //    for (int j = 0; j < 4; j++) {
    //        cout << xlist[i * xlist_col + j] << "  ";
    //    }
    //    cout << "xlist" << endl;
    //}
    //for (int i = 0; i < 10; i++) {
    //    for (int j = 0; j < 4; j++) {
    //        cout << xdlist[i * xdlist_col + j] << "  ";
    //    }
    //    cout << "xdlist" << endl;
    //}
    auto starttime = system_clock::now();
    cudaMemcpy(dxlist, xlist, xlist_row * xlist_col * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dxdlist, xdlist, xdlist_row * xdlist_col * sizeof(double), cudaMemcpyHostToDevice);
    duration<double> time1 = system_clock::now() - starttime;
    cout << "list传送所耗时间为：" << time1.count() << "s" << endl;
}

//随机生成矩阵
void GPUCAL::rands(double* a, int row, int col)
{
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            a[i * col + j] = double(rand() % 10 + 1);
        }
    }
}

int GPUCAL::matcal(double* roi, double* start_position, double* result)
{
    auto starttime = system_clock::now();
    
    duration<double> dif = system_clock::now() - starttime;
    cout << "GPU运算开辟空间所耗时间为：" << dif.count() << "s" << endl;
    cudaMemcpy(droiinput, roi, roi_array_row * sin_roi_col * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dstart_position, start_position, roinumber * sizeof(double), cudaMemcpyHostToDevice);

    getroisum(droiinput, droisum, sin_roi_col, sin_roi_row, roinumber);
    //getrealindex(droiinput, droisum, droi, dstart_position, dindex, sin_roi_col, roinumber, sin_roi_col, sin_roi_row, roi_array_col);
    getrealroi(droiinput, droisum, droi, dstart_position, dindex, sin_roi_col, roinumber, sin_roi_col, sin_roi_row, roi_array_col);

    dif = system_clock::now() - starttime;
    cout << "GPU运算1所耗时间为：" << dif.count() << "s" << endl;
    //存到GPU
    //cudaMemcpy(droi, roi, roi_array_row * roi_array_col * sizeof(double), cudaMemcpyHostToDevice);
    //cudaMemcpy(dstart_position, start_position, roinumber * sizeof(double), cudaMemcpyHostToDevice);
    
    //cublasSetVector(row_a * col_b, sizeof(double), c1, 1, dc1, 1);
    /*
        GPU运算  并行运算时间
        计算代码运行时间
    */
    double a1 = 1; double b1 = 0;
    cublasDgemm(cuHandle, CUBLAS_OP_N, CUBLAS_OP_N, xlist_col, roi_array_row, roi_array_col, &a1, dxlist, xlist_col, droi, roi_array_col, &b1, dxlistre, xlist_col);
    //cublasDgemm(cuHandle, CUBLAS_OP_N, CUBLAS_OP_N, col_b, row_a, col_a, &a1, db, col_b, da, col_a, &b1, dc1, col_b);
    cublasDgemm(cuHandle, CUBLAS_OP_N, CUBLAS_OP_N, xdlist_col, roi_array_row, xlist_col, &a1, dxdlist, xdlist_col, dxlistre, xlist_col, &b1, dxdlistre, xdlist_col);
    //mextix1(droi, dxlist, dxlistre, roi_array_row, roi_array_col, xlist_col);
    //mextix1(dxlistre, dxdlist, dxdlistre, roi_array_row, xlist_col, xdlist_col);
    //compare(da, db, dc, row_a, col_a, row_b, col_b);
    getgradient(dxdlistre, dgrad1d, roi_array_row, xdlist_col);
    //getgradient(dgrad1d, dgrad2d, roi_array_row, xdlist_col);
    compare(dxdlistre, dgrad1d, dmaxmin, roi_array_row, xdlist_col, roi_array_row, xdlist_col, precnum, dstart_position, sin_roi_row);
    diffcal(dmaxmin, ddiff, roi_array_row - sin_roi_row, sin_roi_row);
    //sortcal(ddiff, roinumber - 1);
    cudaMemcpy(diff, ddiff, (roi_array_row - sin_roi_row) * sizeof(double), cudaMemcpyDeviceToHost);
    dif = system_clock::now() - starttime;
    cout << "GPU运算2所耗时间为：" << dif.count() << "s" << endl;
    medcal(ddiff, dresult, roinumber - 1, sin_roi_row);

    dif = system_clock::now() - starttime;
    cout << "GPU运算所耗时间为：" << dif.count() << "s" << endl;
    cudaDeviceSynchronize();
    dif = system_clock::now() - starttime;
    cout << "同步所耗时间为：" << dif.count() << "s" << endl;
    //cudaMemcpy(xlistre, dxlistre, roi_array_row * (xlist_col) * sizeof(double), cudaMemcpyDeviceToHost);
    //cudaMemcpy(xdlistre, dxdlistre, roi_array_row * (xdlist_col) * sizeof(double), cudaMemcpyDeviceToHost);
    //cudaMemcpy(diff, ddiff, (roi_array_row - sin_roi_row) * sizeof(double), cudaMemcpyDeviceToHost);
    //从GPU取回
    /*cudaMemcpy(xlistre, dxlistre, roi_array_row * (xlist_col) * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(xdlistre, dxdlistre, roi_array_row * (xdlist_col) * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(grad1d, dgrad1d, roi_array_row * (xdlist_col) * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(grad2d, dgrad2d, roi_array_row * (xdlist_col) * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(maxmin, dmaxmin, roi_array_row * result_col * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(diff, ddiff, (roinumber - 1) * sizeof(double), cudaMemcpyDeviceToHost);*/
    
    cudaMemcpy(result, dresult, (roinumber - 1) * sizeof(double), cudaMemcpyDeviceToHost);
    //cudaMemcpy(c, dc, row_a * col_b * sizeof(double), cudaMemcpyDeviceToHost);
    //cudaMemcpy(c1, dgrad1d, row_a * col_b * sizeof(double), cudaMemcpyDeviceToHost);
    //GPU运算时间
    dif = system_clock::now() - starttime;
    cout << "GPU运算+取回CPU所耗时间为：" << dif.count() << "s" << endl;
    /*for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 4; j++) {
            cout << xlistre[i * (xlist_col)+j] << "  ";
        }
        cout << "xlistre" << endl;
    }
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 4; j++) {
            cout << xdlistre[i * (xdlist_col) + j] << "  ";
        }
        cout << "xdlistre" << endl;
    }
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 4; j++) {
            cout << grad1d[i * (xdlist_col) + j] << "  ";
        }
        cout << "grad1d" << endl;
    }
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 4; j++) {
            cout << grad2d[i * xdlist_col + j] << "  ";
        }
        cout << "grad2d" << endl;
    }*/
    for (int i = 0; i < roinumber - 1; i++) {
        for (int j = 0; j < 100; j++) {
            cout << diff[i * ROIRow + j] << "  ";
        }
        cout << "diff" << endl;
    }
    for (int i = 0; i < roinumber - 1; i++) {
        cout << result[i] << "  ";
        cout << "dresult" << endl;
    }
    //auto starttime1 = system_clock::now();
    //double a1 = 1; double b1 = 0;
    //cublasDgemm(cuHandle, CUBLAS_OP_N, CUBLAS_OP_N, col_b, row_a, col_a, &a1, db, col_b, da, col_a, &b1, dc1, col_b);
    
    //cublasGetVector(row_a * col_b, sizeof(double), dc1, 1, c1, 1);
    //cudaMemcpy(c1, dc1, size, cudaMemcpyDeviceToHost);
    //duration<double> diff1 = system_clock::now() - starttime1;
    //cout << "cublas库运算所耗时间为：" << diff1.count() << "s" << endl;

    //cublasDgemm(cuHandle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &a1, da, N, dc1, N, &b1, dc, N);
    //cublasGetVector(row_a * col_b, sizeof(double), dc, 1, c, 1);

    return 2;
}