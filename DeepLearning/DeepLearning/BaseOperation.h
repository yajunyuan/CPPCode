#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include <numeric>
#pragma warning(disable:4996)
using namespace nvinfer1;
#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cout << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

#define DEVICE 0  // GPU id
#define NMS_THRESH 0.45
#define BBOX_CONF_THRESH 0.25
#define MASK_THRESHOLD 0.5;

static double myfunction(double num) {
    return exp(num);
}

template <typename T>
void softmax(const typename::std::vector<T>&v, typename::std::vector<T>&s) {
    double sum = 0.0;
    transform(v.begin(), v.end(), s.begin(), myfunction);
    sum = accumulate(s.begin(), s.end(), sum);
    for (size_t i = 0; i < s.size(); ++i)
        s.at(i) /= sum;
}

template <typename T, typename A>
int arg_max(std::vector<T, A> const& vec) {
    return static_cast<int>(std::distance(vec.begin(), max_element(vec.begin(), vec.end())));
}

class BaseOperation
{
public:
     int MASK_NUM=32;
      int INPUT_W;
      int INPUT_H;
      int NUM_CLASSES;
        typedef struct
    {
        std::string engine_mode;
        float* data;
        float* prob;
        float* prob1;
        int yolomode; //0:yolov5  1:yolov8
        int inputindex;
        int output_size;
        int outputindex;
        int output1_size;
        int output1index;
        IRuntime* runtime;
        ICudaEngine* engine;
        IExecutionContext* context;
        void** buffers;
        cudaStream_t stream;
        int _segWidth, _segHeight, _segChannels, num_box;

    }Yolov5TRTContext;

    struct RecResult {
        char imgname[100];
        int reallabel;
        int id;             //结果类别id
        double confidence;   //结果置信度
        int box[4];       //矩形框
        double radian;
        uchar* boxMask;
    };

    struct Object
    {
        //cv::Rect_<float> rect;
        float bbox[4];
        float prob;
        float label;
        float radian;
        std::vector<float> picked_proposals;
        uchar* boxMask;
    };


    void GetConfigValue(const char* keyName, char* keyValue);

    void doInference(IExecutionContext& context, cudaStream_t& stream, ICudaEngine& engine, std::string engine_mode,
        void** buffers, float* input, float* output, const int output_size, float* output1, const int output1_size,
        Yolov5TRTContext* trt);

    cv::Mat static_resize(cv::Mat img, std::vector<int>& padsize, std::string engine_mode);

    float* blobFromImage(cv::Mat img, std::string engine_mode, int channels);

    bool cmp(const Object& a, const Object& b);

    float iou(float lbox[4], float rbox[4]);

    void Radian(cv::Mat img, const Object& input_box);

    void ObjPostprocess(std::string engine_mode, std::vector<Object>& res, float* output, int num_box, float conf_thresh, float nms_thresh, int yolomode);

    void ObjUniqueprocess(std::vector<Object>& res, float nms_thresh);

    float SigmoidFunction(float a);

    void SegPostprocess(std::vector<Object>& res, float* prob, float* prob1, cv::Mat img, const std::vector<int>& padsize,
        const std::vector<int>& segMaskParam, int yolomode);
};

