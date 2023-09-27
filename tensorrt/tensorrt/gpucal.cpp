#include"gpucal.h"

extern "C" void SearchDefectFunc(int* da, int row_a, int col_a, double upper, double lower, int* result, cudaStream_t streams[]);
extern "C" void SearchDefectFunc1(int* da, int row_a, int col_a, int* result, cudaStream_t streams[]);
extern "C" void Reset(int* result, cudaStream_t streams[]);
GPUCAL::GPUCAL(int imgArrayrow, int imgArraycol)
{
    cublasStatus_t status = cublasCreate(&cuHandle);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        if (status == CUBLAS_STATUS_NOT_INITIALIZED) {
            std::cout << "CUBLAS 对象实例化出错" << std::endl;
        }
        return;
    }
    imgArrayRow = imgArrayrow;
    imgArrayCol = imgArraycol;
    result = (int*)malloc(RESULTSIZE * sizeof(int));
    result1 = (int*)malloc(imgArrayRow * (imgArrayCol) * sizeof(int));
    cudaMalloc((void**)&dimgArray, imgArrayRow * imgArrayCol * sizeof(int));
    cudaMalloc((void**)&dresult, imgArrayRow * (imgArrayCol-2) * sizeof(int));
    cudaMalloc((void**)&dresult1, RESULTSIZE * sizeof(int));
}

GPUCAL::~GPUCAL(void)
{
    free(result);
    result = NULL;
    cudaFree(dresult);
    std::cout << "free GPU memory" << std::endl;

}

void GPUCAL::MatCal(cudaStream_t streams[],int* imgArray, int* result, double upper, double lower)
{
    cudaMemcpyAsync(dimgArray, imgArray, imgArrayRow * imgArrayCol * sizeof(int), cudaMemcpyHostToDevice, streams[1]);
    auto starttime = system_clock::now();
    SearchDefectFunc(dimgArray, imgArrayRow, imgArrayCol, upper, lower, dresult, streams);
    //std::cout << imgArrayRow << "imgArrayRow" << imgArrayCol << std::endl;
    SearchDefectFunc1(dresult, imgArrayRow, imgArrayCol - 2, dresult1, streams);
    duration<double> dif = system_clock::now() - starttime;
    //std::cout << "GPU运算并取回所耗时间为：" << dif.count() << "s" << std::endl;
    Reset(dresult1, streams);
    cudaMemcpyAsync(result, dresult1, RESULTSIZE * sizeof(int), cudaMemcpyDeviceToHost, streams[1]);

    dif = system_clock::now() - starttime;
    //std::cout << "GPU运算并取回所耗时间为：" << dif.count() << "s" << std::endl;
    //for (int i = 40; i < 46; i++) {
    //    for (int j = 0; j < imgArrayCol; j++) {
    //        std::cout << result1[i* imgArrayRow+j] << "  ";
    //    }
    //    std::cout << std::endl;
    //}
}