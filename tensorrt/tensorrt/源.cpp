#include<stdio.h>
#include<iostream>
#include<opencv2/opencv.hpp>
#include <time.h>
#include<vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>
#include <chrono>
#include <vector>
#include <opencv2/opencv.hpp>
#include "dirent.h"
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#define CUDA_CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

std::string label_map = "D:/python/yolov5-6.1/classes.txt";
int main(int argc, char** argv) {
    //step1:����runtime
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    //step2:�����л�����engine
    ICudaEngine* engine = runtime->deserializeCudaEngine(modelData, modelSize, nullptr);
    assert(engine != nullptr);
    // ��ӡ���������
    printf("Bindings after deserializing:\n");
    for (int bi = 0; bi < engine->getNbBindings(); bi++)
    {
        if (engine->bindingIsInput(bi) == true)
        {
            printf("Binding %d (%s): Input.\n", bi, engine->getBindingName(bi));
        }
        else
        {
            printf("Binding %d (%s): Output.\n", bi, engine->getBindingName(bi));
        }
    }



    //step3:����context,����һЩ�ռ����洢�м伤��ֵ
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);

    //step4:�����������blob���ֻ�ȡ�����������
    int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    //step5:ʹ����Щ����������buffersָ�� GPU ����������������
    void* buffers[2];
    //buffers[inputIndex] = inputBuffer;
    //buffers[outputIndex] = outputBuffer;
    //step6��Ϊ�����������GPU�Դ�
    CUDA_CHECK(cudaMalloc(&buffers[inputIndex], batchSize * inputDim.c() * inputDim.h() * inputDim.w() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[outputIndex], batchSize * outputDim.c() * outputDim.h() * outputDim.w() * sizeof(float)));
    //step6������cuda��
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    //step7����CPU��GPU----����input����
    CUDA_CHECK(cudaMemcpyAsync(buffers[inputIndex],//�Դ��ϵĴ洢�������ڴ����������
        input, //�����ڴ��е�����
        batchSize * inputDim.c() * inputDim.h() * inputDim.w() * sizeof(float),
        cudaMemcpyHostToDevice,
        stream));
    //step8:�첽����
    context->enqueueV2(buffers, stream, nullptr);

    //step9����GPU��CPU----����output����
    CUDA_CHECK(cudaMemcpyAsync(output,//���ڴ��е�����
        buffers[outputIndex],//���Դ��еĴ洢��,���ģ�����
        batchSize * outputDim.c() * outputDim.h() * outputDim.w() * sizeof(float),
        cudaMemcpyDeviceToHost,
        stream));
    //step10��ͬ��cuda��
    CUDA_CHECK(cudaStreamSynchronize(stream));
    //step11���ͷ���Դ
    cudaStreamDestroy(stream);
    context->destroy();
    engine->destroy();
    runtime->destroy();
    CUDA_CHECK(cudaFree(buffers[inputIndex]));
    CUDA_CHECK(cudaFree(buffers[outputIndex]));

}