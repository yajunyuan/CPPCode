#include <fstream> 
#include <iostream> 

#include <NvInfer.h> 
#include <NvOnnxParser.h> 
#include "logging.h"

using namespace nvinfer1;
using namespace nvonnxparser;
//using namespace sample;

//int main1(int argc, char** argv)
//{
//    // Create builder 
//    Logger m_logger;
//    IBuilder* builder = createInferBuilder(m_logger);
//    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
//    IBuilderConfig* config = builder->createBuilderConfig();
//
//    // Create model to populate the network 
//    INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
//
//    // Parse ONNX file 
//    IParser* parser = nvonnxparser::createParser(*network, m_logger);
//    bool parser_status = parser->parseFromFile("model.onnx", static_cast<int>(ILogger::Severity::kWARNING));
//
//    // Get the name of network input 
//    Dims dim = network->getInput(0)->getDimensions();
//    if (dim.d[0] == -1)  // -1 means it is a dynamic model 
//    {
//        const char* name = network->getInput(0)->getName();
//        IOptimizationProfile* profile = builder->createOptimizationProfile();
//        profile->setDimensions(name, OptProfileSelector::kMIN, Dims4(1, dim.d[1], dim.d[2], dim.d[3]));
//        profile->setDimensions(name, OptProfileSelector::kOPT, Dims4(1, dim.d[1], dim.d[2], dim.d[3]));
//        profile->setDimensions(name, OptProfileSelector::kMAX, Dims4(1, dim.d[1], dim.d[2], dim.d[3]));
//        config->addOptimizationProfile(profile);
//    }
//
//
//    // Build engine 
//    config->setMaxWorkspaceSize(1 << 20);
//    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
//
//    // Serialize the model to engine file 
//    IHostMemory* modelStream{ nullptr };
//    assert(engine != nullptr);
//    modelStream = engine->serialize();
//
//    std::ofstream p("model.engine", std::ios::binary);
//    if (!p) {
//        std::cerr << "could not open output file to save model" << std::endl;
//        return -1;
//    }
//    p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
//    std::cout << "generate file success!" << std::endl;
//
//    // Release resources 
//    modelStream->destroy();
//    network->destroy();
//    engine->destroy();
//    builder->destroy();
//    config->destroy();
//    return 0;
//}


#include <iostream>
#include <fstream>
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvOnnxParser.h"
#include "logging.h"
using namespace nvinfer1;


#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <sys/stat.h>
#include <time.h>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "NvOnnxParser.h"
#define DebugP(x) std::cout << "Line" << __LINE__ << "  " << #x << "=" << x << std::endl


using namespace nvinfer1;

Logger gLogger;
// LogStreamConsumer gLogError;

static const int INPUT_H = 640;
static const int INPUT_W = 640;
static const int INPUT_C = 3;
static const int OUTPUT_SIZE = 8400;

const char* INPUT_BLOB_NAME = "input";
const char* OUTPUT_BLOB_NAME = "output";

const std::string gSampleName = "TensorRT.sample_onnx_image";
const std::string onnxFile = R"(D:\yolov8\ultralytics\runs\segment\train9\weights\best.onnx)";
const std::string engineFile = R"(D:\yolov8\ultralytics\runs\segment\train9\weights\int8.engine)";
const std::string calibFile = R"(D:\yolov8\ultralytics\data\segdata/calibfile.txt)";

std::vector<float> prepareImage(cv::Mat& img) {
    int c = 3;
    int h = INPUT_H;
    int w = INPUT_W;

    // 1 Resize the source Image to a specific size(这里保持原图长宽比进行resize)
    float scale = std::min(float(w) / img.cols, float(h) / img.rows);
    auto scaleSize = cv::Size(img.cols * scale, img.rows * scale);

    // Convert BGR to RGB
    cv::Mat rgb;
    cv::cvtColor(img, rgb, CV_BGR2RGB);

    cv::Mat resized;
    cv::resize(rgb, resized, scaleSize, 0, 0, cv::INTER_CUBIC);

    // 2 Crop Image(将resize后的图像放在(H, W, C)的中心, 周围用127做padding)
    cv::Mat cropped(h, w, CV_8UC3, 127);
        // Rect(left_top_x, left_top_y, width, height)
    cv::Rect rect((w - scaleSize.width) / 2, (h - scaleSize.height) / 2, scaleSize.width, scaleSize.height);
    resized.copyTo(cropped(rect));

    // 3 Type conversion, convert unsigned int 8 to float 32
    cv::Mat img_float;
    cropped.convertTo(img_float, CV_32FC3, 1.f / 255.0);

    // HWC to CHW, and convert Mat to std::vector<float>
    std::vector<cv::Mat> input_channels(c);
    cv::split(cropped, input_channels);

    std::vector<float> result(h * w * c);
    auto data = result.data();
    int channelLength = h * w;
    for (int i = 0; i < c; ++i) {
        memcpy(data, input_channels[i].data, channelLength * sizeof(float));
        data += channelLength;
    }
    return result;
}

// 实现自己的calibrator类
namespace nvinfer1 {
    class int8EntropyCalibrator : public nvinfer1::IInt8EntropyCalibrator {
    public:
        int8EntropyCalibrator(const int& batchSize,
            const std::string& imgPath,
            const std::string& calibTablePath);

        virtual ~int8EntropyCalibrator();

        int getBatchSize() const noexcept override { return batchSize; }

        bool getBatch(void* bindings[], const char* names[], int32_t nbBindings) noexcept override;

        const void* readCalibrationCache(std::size_t& length) noexcept override;

        void writeCalibrationCache(const void* ptr, std::size_t length) noexcept override;

    private:
        int batchSize;
        size_t inputCount;
        size_t imageIndex;
        std::string calibTablePath;
        std::vector<std::string> imgPaths;

        float* batchData{ nullptr };
        void* deviceInput{ nullptr };

        bool readCache;
        std::vector<char> calibrationCache;
    };

    int8EntropyCalibrator::int8EntropyCalibrator(const int& batchSize, const std::string& imgPath,
        const std::string& calibTablePath) : batchSize(batchSize), calibTablePath(calibTablePath), imageIndex(0) {
        int inputChannel = 3;
        int inputH = 256;
        int inputW = 256;
        inputCount = batchSize * inputChannel * inputH * inputW;

        std::fstream f(imgPath);
        if (f.is_open()) {
            std::string temp;
            while (std::getline(f, temp)) imgPaths.push_back(temp);
        }
        int len = imgPaths.size();
        for (int i = 0; i < len; i++) {
            std::cout << imgPaths[i] << std::endl;
        }

        // allocate memory for a batch of data, batchData is for CPU, deviceInput is for GPU
        batchData = new float[inputCount];
        cudaMalloc(&deviceInput, inputCount * sizeof(float));
    }

    int8EntropyCalibrator::~int8EntropyCalibrator() {
        cudaFree(deviceInput);
        if (batchData) {
            delete[] batchData;
        }
    }

    bool int8EntropyCalibrator::getBatch(void* bindings[], const char* names[], int32_t nbBindings) noexcept {
        std::cout << imageIndex << " " << batchSize << std::endl;
        std::cout << imgPaths.size() << std::endl;
        if (imageIndex + batchSize > int(imgPaths.size()))
            return false;
        // load batch
        float* ptr = batchData;
        for (size_t j = imageIndex; j < imageIndex + batchSize; ++j) {
            cv::Mat img = cv::imread(imgPaths[j]);
            std::vector<float> inputData = prepareImage(img);
            if (inputData.size() != inputCount) {
                std::cout << "InputSize Error" << std::endl;
                return false;
            }
            assert(inputData.size() == inputCount);
            memcpy(ptr, inputData.data(), (int)(inputData.size()) * sizeof(float));
            ptr += inputData.size();
            std::cout << "load image " << imgPaths[j] << " " << (j + 1) * 100. / imgPaths.size() << "%" << std::endl;
        }
        imageIndex += batchSize;
        // copy bytes from Host to Device
        cudaMemcpy(deviceInput, batchData, inputCount * sizeof(float), cudaMemcpyHostToDevice);
        bindings[0] = deviceInput;
        return true;
    }

    const void* int8EntropyCalibrator::readCalibrationCache(std::size_t& length) noexcept {
        calibrationCache.clear();
        std::ifstream input(calibTablePath, std::ios::binary);
        input >> std::noskipws;
        if (readCache && input.good()) {
            std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(),
                std::back_inserter(calibrationCache));
        }
        length = calibrationCache.size();
        return length ? &calibrationCache[0] : nullptr;
    }

    void int8EntropyCalibrator::writeCalibrationCache(const void* cache, std::size_t length) noexcept {
        std::ofstream output(calibTablePath, std::ios::binary);
        output.write(reinterpret_cast<const char*>(cache), length);
    }
}

bool onnxToTRTModel(const std::string& modelFile, // name of the onnx model
    unsigned int maxBatchSize,    // batch size - NB must be at least as large as the batch we want to run with
    IHostMemory*& trtModelStream, // output buffer for the TensorRT model
    const std::string& engineFile)
{
    
    // create the builder
    IBuilder* builder = createInferBuilder(gLogger.getTRTLogger());
    assert(builder != nullptr);

    // create the config
    auto config = builder->createBuilderConfig();
    assert(config != nullptr);

    if (!builder->platformHasFastInt8()) {
        std::cout << "builder platform do not support Int8" << std::endl;
        return false;
    }

    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    std::cout << "explicitBatch is: " << explicitBatch << std::endl;
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);

    auto parser = nvonnxparser::createParser(*network, gLogger.getTRTLogger());

    //Optional - uncomment below lines to view network layer information
    //config->setPrintLayerInfo(true);
    //parser->reportParsingInfo();

    //if (!parser->parseFromFile(locateFile(modelFile, gArgs.dataDirs).c_str(), static_cast<int>(gLogger.getReportableSeverity())))
    //{
    //    gLogger.gLogError << "Failure while parsing ONNX file" << std::endl;
    //    return false;
    //}

    // config
    config->setAvgTimingIterations(1);
    config->setMinTimingIterations(1);
    config->setMaxWorkspaceSize(1 << 20);

    // Build the engine
    builder->setMaxBatchSize(maxBatchSize);
    //builder->setMaxWorkspaceSize(1 << 20);
    //builder->setMaxWorkspaceSize(10 << 20);

    nvinfer1::int8EntropyCalibrator* calibrator = nullptr;
    if (calibFile.size() > 0) calibrator = new nvinfer1::int8EntropyCalibrator(maxBatchSize, calibFile, "");

    // builder->setFp16Mode(gArgs.runInFp16);
    // builder->setInt8Mode(gArgs.runInInt8);

    // 对builder进行设置, 告诉它使用Int8模式, 并利用编写好的calibrator类进行calibration
    //builder->setInt8Mode(true);
    //builder->setInt8Calibrator(calibrator);


    // if (gArgs.runInInt8)
    // {
    //     samplesCommon::setAllTensorScales(network, 127.0f, 127.0f);
    // }
    config->setFlag(BuilderFlag::kINT8);
    config->setInt8Calibrator(calibrator);

    // 如果使用了calibrator, 应该参考https://github.com/enazoe/yolo-tensorrt/blob/dd4cb522625947bfe6bfbdfbb6890c3f7558864a/modules/yolo.cpp, 把下面这行注释掉，使用数据集校准得到dynamic range；否则使用下面这行手动设置dynamic range。
    // setAllTensorScales函数在官方TensorRT开源代码里有
    //samplesCommon::setAllTensorScales(network, 127.0f, 127.0f);
    // samplesCommon::enableDLA(builder, gArgs.useDLACore);

    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    assert(engine);

    if (calibrator) {
        delete calibrator;
        calibrator = nullptr;
    }

    // we can destroy the parser
    parser->destroy();

    // serialize the engine, then close everything down
    trtModelStream = engine->serialize();
    std::ofstream file;
    file.open(engineFile, std::ios::binary | std::ios::out);
    file.write((const char*)trtModelStream->data(), trtModelStream->size());
    file.close();

    engine->destroy();
    config->destroy();
    network->destroy();
    builder->destroy();

    return true;
}

int main1(int argc, char** argv)
{

    auto sampleTest = gLogger.defineTest(gSampleName, argc, const_cast<const char**>(argv));

    gLogger.reportTestStart(sampleTest);

    // create a TensorRT model from the onnx model and serialize it to a stream
    nvinfer1::IHostMemory* trtModelStream{ nullptr };

    if (!onnxToTRTModel(onnxFile, 1, trtModelStream, engineFile))
        gLogger.reportFail(sampleTest);

    assert(trtModelStream != nullptr);
}


//int main1()
//{
//    // 加载双精度的Engine模型
//    std::ifstream engineFile("your_fp64_engine.engine", std::ios::binary);
//    if (!engineFile)
//    {
//        std::cerr << "Error: Failed to open FP64 engine file!" << std::endl;
//        return 1;
//    }
//
//    // 创建执行引擎
//    Logger gLogger;
//    IRuntime* runtime = createInferRuntime(gLogger.getTRTLogger());
//    if (!runtime)
//    {
//        std::cerr << "Error: Unable to create TensorRT IRuntime!" << std::endl;
//        return 1;
//    }
//
//    // 反序列化FP64 Engine模型
//    int runtimeVersion = -1;
//    std::unique_ptr<nvinfer1::ICudaEngine> engine(runtime->deserializeCudaEngine(engineFile, nullptr, &runtimeVersion));
//    if (!engine)
//    {
//        std::cerr << "Error: Failed to deserialize FP64 engine model!" << std::endl;
//        return 1;
//    }
//
//    // 创建INT8量化器
//    IInt8Calibrator* calibrator = new MyInt8Calibrator();
//    builder->setInt8Calibrator(calibrator);
//    builder->setInt8Mode(true);
//
//    // INT8量化
//    builder->setInt8Mode(true);
//    builder->setInt8Calibrator(calibrator);
//
//    // 保存INT8量化后的Engine模型
//    std::ofstream int8EngineFile("your_int8_engine.engine", std::ios::binary);
//    if (!int8EngineFile)
//    {
//        std::cerr << "Error: Failed to create INT8 engine file!" << std::endl;
//        return 1;
//    }
//    nvinfer1::IHostMemory* serializedInt8Engine = engine->serialize();
//    if (!serializedInt8Engine)
//    {
//        std::cerr << "Error: Failed to serialize INT8 engine model!" << std::endl;
//        return 1;
//    }
//    int8EngineFile.write(static_cast<const char*>(serializedInt8Engine->data()), serializedInt8Engine->size());
//
//    // 释放资源
//    serializedInt8Engine->destroy();
//    int8EngineFile.close();
//    engineFile.close();
//    runtime->destroy();
//
//    return 0;
//}