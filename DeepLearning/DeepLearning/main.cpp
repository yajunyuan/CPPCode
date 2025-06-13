#include "logging.h"
#include <chrono>
#include "BaseOperation.h"
#include <thread>
#include <atomic>

//#include<windows.h>
using namespace nvinfer1;
static Logger gLogger;
BaseOperation baseOperation;
std::atomic<bool> isRunning{ true };
Loger loger("log/log.txt");
extern "C"
{
    __declspec(dllexport) void* AiGPUInit(const char* modelpath, const char* modelmode, bool gpuflag = false);
    __declspec(dllexport) void AIGPUDetectImg(void* h, uchar* data, int width, int height, int stride, BaseOperation::RecResult*& result, int& outlen);
    __declspec(dllexport) void AiGPUDetectPath(void* h, const char* imgpath, BaseOperation::RecResult*& output, int& outlen);
    __declspec(dllexport) void ReleaseStruct(BaseOperation::RecResult* structArray);
}

void ResizeBox(int imgcols, int imgrows,const std::vector<int>& padsize, float bbox[]) {
    int newh = padsize[0], neww = padsize[1], padh = padsize[2], padw = padsize[3];
    float ratio_h = (float)imgrows / newh;
    float ratio_w = (float)imgcols / neww;
    float x = (bbox[0] - padw) * ratio_w;  //x
    float y = (bbox[1] - padh) * ratio_h;  //y
    float w = bbox[2] * ratio_w;  //w
    float h = bbox[3] * ratio_h;  //h
    float left = MAX((x - 0.5 * w), 0);
    float top = MAX((y - 0.5 * h), 0);
    float colstmp = MIN((w + left), imgcols);
    float rowstmp = MIN((h + top), imgrows);
    bbox[0] = left;
    bbox[1] = top;
    bbox[2] = colstmp-left;
    bbox[3] = rowstmp-top;
}

void threadFunction(BaseOperation::Yolov5TRTContext* trt) {
    //std::string modelpathstr = R"(C:\Users\rs\Desktop\best.engine)";
    //const char* modelpath = modelpathstr.c_str();
    //std::string modelmodestr = "obj";
    //const char* modelmode = modelmodestr.c_str();
    //BaseOperation::Yolov5TRTContext* trt = (BaseOperation::Yolov5TRTContext*)AiGPUInit(modelpath, modelmode);
    //std::string imgpath = R"(E:\dataset\pengda\all\test1)";
    //const char* p = imgpath.c_str();
    //BaseOperation::RecResult* result;
    //int outlen = 0;
    float* blob = new float[baseOperation.INPUT_W * baseOperation.INPUT_H * 3];
    while (isRunning) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        //Sleep(5);
        //AiGPUDetectPath(trt, p, result, outlen);
        //cv::Mat img= cv::Mat::zeros(2500, 2500, CV_8UC3);
        //std::vector<BaseOperation::RecResult> output;
        //AIGPUDetect(trt, img, "test", output);
        baseOperation.doInference(*trt->context, trt->stream, *(trt->engine), trt->engine_mode, trt->buffers, blob, trt->prob, trt->output_size, trt->prob1, trt->output1_size, trt);
    }
    delete[] blob;
    loger.log("Running Thread is exiting...", Loger::ERROR);
    //std::cout << "Thread is exiting..." << std::endl;
}

void* AiGPUInit(const char* engineFilePath, const char* engineMode, bool gpuflag)
{
    //detect init
    int dev_num = 0;
    cudaError_t error_id = cudaGetDeviceCount(&dev_num);
    if (error_id != cudaSuccess)
    {
        loger.log("cudaGetDeviceCount returned "+std::to_string(error_id)+"-> "+cudaGetErrorString(error_id), Loger::ERROR);
        //printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
        return nullptr;
    }
    int cudaRuntimeVersion;
    cudaRuntimeGetVersion(&cudaRuntimeVersion);
    loger.log("CUDA Runtime Version: " + std::to_string(cudaRuntimeVersion));
    //std::cout << "CUDA Runtime Version: " << cudaRuntimeVersion << std::endl;
    //char engine_filepath[1000] = { 0 };
    //char enginemode[100] = { 0 };
    //char input_w[100] = { 0 };
    //char input_h[100] = { 0 };
    //char num_class[100] = { 0 };
    //baseOperation.GetConfigValue("engine_file_path", engine_filepath);
    //baseOperation.GetConfigValue("engine_mode", enginemode);
    //GetConfigValue("INPUT_W", input_w);
    //GetConfigValue("INPUT_H", input_h);
    //GetConfigValue("NUM_CLASSES", num_class);
    //engine_filepath[strlen(engine_filepath) - 1] = 0;
    //enginemode[strlen(enginemode) - 1] = 0;
    //INPUT_W = atoi(input_w);
    //INPUT_H = atoi(input_h);
    //NUM_CLASSES = atoi(num_class);
    std::string engine_mode = engineMode;
    /*std::string engine_mode = (char*)engineMode;*/
    cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    char* trtModelStream{ nullptr };
    size_t size{ 0 };
    //if (argc == 3 && std::string(argv[2]) == "-i") {
    std::string engine_file = engineFilePath;
    /*const std::string engine_file = (char*)engineFilePath;*/
    //std::cout << engine_file.length() << " -------"<< engine_file << std::endl;
    std::ifstream file(engine_file, std::ios::binary);
    BaseOperation::Yolov5TRTContext* trt = new BaseOperation::Yolov5TRTContext();
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        //for (size_t i = 0; i < size; ++i) {
        //    trtModelStream[i] ^= 0x88; // 解密数据（异或）
        //}
        file.close();
        loger.log(engine_file+" read engine ok");
        //std::cout << "read engine ok" << std::endl;
    }
    else {
        loger.log(engine_file + " read engine failed");
        //std::cout << "read engine failed" << std::endl;
    }

    trt->runtime = createInferRuntime(gLogger);
    assert(trt->runtime != nullptr);
    trt->engine = trt->runtime->deserializeCudaEngine(trtModelStream, size);
    assert(trt->engine != nullptr);

    // 多线程处理
    //for (int i = 0; i < 6; i++) {
    //    trt->context[i] = trt->engine->createExecutionContext();
    //}
    trt->context = trt->engine->createExecutionContext();
    //assert(trt->context != nullptr);
    delete[] trtModelStream;
    void** buffers;
    const int bindingnum = trt->engine->getNbBindings();
    buffers = new void* [bindingnum];
    //assert(trt->engine->getNbBindings() == 2);
    //void* buffers[2];
    trt->context->setBindingDimensions(0, Dims4(1, 3, 640, 640));
    std::vector<Dims> inputDims;
    std::vector<Dims> outputDims;
    for (int i = 0; i < bindingnum; i++) {
        if (trt->engine->bindingIsInput(i)) {
            inputDims.push_back(trt->context->getBindingDimensions(i));
        }
        else {
            outputDims.push_back(trt->context->getBindingDimensions(i));
        }
    }
    Dims outDims;

    int outputIndex = 0;
    if (outputDims.size() > 1) {
        for (int i = 0; i < outputDims.size(); ++i)
        {
            if (outputDims[i].nbDims == 3) {
                outDims = outputDims[i];
                outputIndex = i + 1;
                break;
            }
        }
    }
    else {
        outDims = outputDims[0];
        outputIndex = 1;
    }


    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    //const int inputIndex = trt->engine->getBindingIndex(INPUT_BLOB_NAME);
    const int inputIndex = 0;
    assert(trt->engine->getBindingDataType(inputIndex) == nvinfer1::DataType::kFLOAT);
    //const int outputIndex = trt->engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(trt->engine->getBindingDataType(outputIndex) == nvinfer1::DataType::kFLOAT);
    int mBatchSize = trt->engine->getMaxBatchSize();

    auto out_dims = trt->context->getBindingDimensions(outputIndex);
    trt->output_size = 1;
    trt->engine_mode = engine_mode;
    cudaStream_t streams;
    cudaStreamCreate(&streams);
    trt->stream = streams;
    baseOperation.INPUT_H = trt->context->getBindingDimensions(inputIndex).d[2];
    baseOperation.INPUT_W = trt->context->getBindingDimensions(inputIndex).d[3];
    if (out_dims.nbDims>2) {
        if (out_dims.d[1] < out_dims.d[2]) {
            trt->yolomode = 1;
            trt->num_box = out_dims.d[2];
            loger.log(engine_file + " the engine is yolov8; engine mode is "+ engine_mode + " engine out dims is "+
                std::to_string(out_dims.d[0]) +"*" + std::to_string(out_dims.d[1]) + "*" + std::to_string(out_dims.d[2]));
        }
        else {
            trt->yolomode = 0;
            trt->num_box = out_dims.d[1];
            loger.log(engine_file + " the engine is yolov5; engine mode is " + engine_mode + " engine out dims is " +
                std::to_string(out_dims.d[0]) + "*" + std::to_string(out_dims.d[1]) + "*" + std::to_string(out_dims.d[2]));
        }
    }
    if (engine_mode == "cls") {
        baseOperation.NUM_CLASSES = out_dims.d[1];
        
    }
    else if (engine_mode == "obj" || engine_mode == "obb") {
        if (trt->yolomode == 1) {
            if (engine_mode == "obj") {
                baseOperation.NUM_CLASSES = out_dims.d[1] - 4;
            }
            else {
                baseOperation.NUM_CLASSES = out_dims.d[1] - 5;
            }
        }
        else {
            baseOperation.NUM_CLASSES = out_dims.d[2] - 5;
        }
        
    }
    else {
        if (trt->yolomode == 1) {
            baseOperation.NUM_CLASSES = out_dims.d[1] - baseOperation.MASK_NUM - 4;
        }
        else {
            baseOperation.NUM_CLASSES = out_dims.d[2] - baseOperation.MASK_NUM - 5;
        }
        
    }
    for (int j = 0; j < out_dims.nbDims; j++) {
        trt->output_size *= out_dims.d[j];
    }
    trt->prob = new float[trt->output_size];
    trt->inputindex = inputIndex;
    trt->outputindex = outputIndex;
    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], 3 * baseOperation.INPUT_H * baseOperation.INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], trt->output_size * sizeof(float)));
    if (engine_mode == "seg") {
        Dims out1Dims;
        int outputIndex1 = 0;
        for (int i = 0; i < outputDims.size(); ++i)
        {
            if (outputDims[i].nbDims == 4) {
                out1Dims = outputDims[i];
                outputIndex1 = i + 1;
                break;
            }
        }
        //const int outputIndex1 = trt->engine->getBindingIndex(OUTPUT_BLOB_NAME1);
        auto out_dims = trt->engine->getBindingDimensions(outputIndex1);
        trt->output1index = outputIndex1;
        trt->output1_size = 1;
        for (int j = 0; j < out_dims.nbDims; j++) {
            trt->output1_size *= out_dims.d[j];
        }
        trt->_segWidth = out_dims.d[3];
        trt->_segHeight = out_dims.d[2];
        trt->_segChannels = out_dims.d[1];

        trt->prob1 = new float[trt->output1_size];
        if (bindingnum > 3) {
            CHECK(cudaMalloc(&buffers[2], 3 * 80 * 80 * 52 * sizeof(float)));
            CHECK(cudaMalloc(&buffers[3], 3 * 40 * 40 * 52 * sizeof(float)));
            CHECK(cudaMalloc(&buffers[4], 3 * 20 * 20 * 52 * sizeof(float)));
        }
        CHECK(cudaMalloc(&buffers[outputIndex1], trt->output1_size * sizeof(float)));
    }
    trt->buffers = buffers;
    //// 处理第一次推理时间长
    //float* blob = new float[baseOperation.INPUT_W * baseOperation.INPUT_H * 3];
    //baseOperation.doInference(*trt->context, trt->stream, *(trt->engine), trt->engine_mode, trt->buffers, blob, trt->prob, trt->output_size, trt->prob1, trt->output1_size, trt);
    //delete[] blob;
    //for (int i = 0; i < 6; i++) {
    //    float* blob = new float[INPUT_W * INPUT_H * 3];
    //    doInference(*trt->context[i], trt->stream, *(trt->engine), trt->engine_mode, trt->buffers, blob, trt->prob, trt->output_size, trt->prob1, trt->output1_size, trt);
    //    delete[] blob;
    //}
    if (gpuflag) {
        isRunning = gpuflag;
        BaseOperation::Yolov5TRTContext* copytrt = (BaseOperation::Yolov5TRTContext*)AiGPUInit(engineFilePath, engineMode, false);
        std::thread myThread(threadFunction, copytrt);
        if (myThread.joinable()) {
            myThread.detach(); // 分离线程，释放相关资源 
        }
    }
    loger.log("the engine classes is " + std::to_string(baseOperation.NUM_CLASSES));
    return (void*)trt;
}

double CalcAreaRatio(const cv::Mat img, int box[]) {
    cv::Mat imgtmp;
    cv::cvtColor(img, imgtmp, cv::COLOR_BGR2GRAY);
    //imgtmp = imgtmp(cv::Rect(0, 0, 1000, 1528));
    cv::Rect rect1(box[0], box[1]+105, box[2], box[3]-345);
    cv::Mat binaryImg, binaryImg2;
    cv::Mat equalizedImg, equalizedImg2;
    cv::equalizeHist(imgtmp, equalizedImg);
    cv::Mat imagetmp1 = imgtmp(rect1);

    //double minVal, maxVal;
    //cv::minMaxLoc(imagetmp1, &minVal, &maxVal);
    //cv::Mat normalizedImg;
    //imagetmp1.convertTo(normalizedImg, CV_32F); // 转换为浮点型，方便后续计算  
    //normalizedImg = (normalizedImg - minVal) * (255.0 / (maxVal - minVal));
    //normalizedImg.convertTo(normalizedImg, CV_8U);

    cv::threshold(imagetmp1, binaryImg, 100, 255, cv::THRESH_BINARY); // 阈值可以根据需求调整 
    double total = box[2] * (box[3] - 345);
    double countratio1 = cv::countNonZero(binaryImg)/ total;
    //cv::Rect rect2(600, 150, 320, 1100);
    //cv::Mat imagetmp2 = imgtmp(rect2);
    //cv::equalizeHist(imagetmp2, equalizedImg2);
    //cv::threshold(imagetmp2, binaryImg2, 100, 255, cv::THRESH_BINARY);
    //double countratio2 = cv::countNonZero(binaryImg2) / 320.0 / 1100;
    return countratio1;
}

bool CalcHeight(const cv::Mat img, int box[]) {
    cv::Mat imgtmp;
    cv::cvtColor(img, imgtmp, cv::COLOR_BGR2GRAY);
    cv::Rect rect(box[0], box[1], box[2], box[3]);
    cv::Mat iamgeori1 = imgtmp(rect);
    cv::Mat imgtmp1;
    cv::medianBlur(iamgeori1, iamgeori1, 7);
    threshold(iamgeori1, imgtmp1, 70, 255, cv::THRESH_BINARY_INV);
    cv::Mat labels, stats, centroids;
    int num = connectedComponentsWithStats(imgtmp1, labels, stats, centroids);
    for (int i = 1; i < num; i++) {
        int left = stats.at<int>(i, cv::CC_STAT_LEFT);
        int top = stats.at<int>(i, cv::CC_STAT_TOP);
        int width = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int height = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (left == 0 || top == 0 || width > 100 || height > 400 || area < 5000) {
            continue;
        }
        //std::cout << height << std::endl;
        if (height > 340) {
            loger.log("是6406 NG;");
            std::cout << "是6406ng， " << rect.tl() << std::endl;
            return true;
        }
    }
    return false;
}

void AIGPUDetect(void* h, cv::Mat img, std::string imgname, std::vector<BaseOperation::RecResult>& output) {
    loger.log("AI Img Detect Start; img height is " + std::to_string(img.rows) + " width is " + std::to_string(img.cols));
    BaseOperation::Yolov5TRTContext* trt = (BaseOperation::Yolov5TRTContext*)h;
    //int img_w = img.cols;
    //int img_h = img.rows;
    std::vector<int> padsize;
    //auto start1 = std::chrono::system_clock::now();
    cv::Mat pr_img = baseOperation.static_resize(img, padsize, trt->engine_mode);
    float* blob;
    blob = baseOperation.blobFromImage(pr_img, trt->engine_mode, 3);
    //std::cout << "preprocess: " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start1).count() / 1000.0 << "ms" << std::endl;
    
    float scale = (std::min)(baseOperation.INPUT_W / (img.cols * 1.0), baseOperation.INPUT_H / (img.rows * 1.0));
    //auto start = std::chrono::system_clock::now();
    baseOperation.doInference(*trt->context, trt->stream, *(trt->engine), trt->engine_mode, trt->buffers, blob, trt->prob, trt->output_size, trt->prob1, trt->output1_size, trt);
    //auto end = std::chrono::system_clock::now();
    //std::cout << "doInference: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0 << "ms" << std::endl;
    
    BaseOperation::RecResult resulttmp;
    //auto start2 = std::chrono::system_clock::now();
    strcpy(resulttmp.imgname, imgname.c_str());
    if (trt->engine_mode == "cls") {
        std::vector<float> vecprob(trt->prob, trt->prob + trt->output_size), vecprob1(vecprob);;
        if (find_if(vecprob.begin(), vecprob.end(), [](float i) { return i > 1; }) != vecprob.end()) {
            softmax(vecprob, vecprob1);//yolov8不需要softmax
        }
        resulttmp.id = arg_max(vecprob1);
        resulttmp.confidence = vecprob1[resulttmp.id];
        output.push_back(resulttmp);
        loger.log("predict class: " + std::to_string(resulttmp.id) + " score: " + std::to_string(resulttmp.confidence));
        //std::cout << resulttmp.id << " " << resulttmp.confidence << std::endl;
    }
    if (trt->engine_mode == "obj" || trt->engine_mode == "obb") {
        std::vector<BaseOperation::Object> objects;
        baseOperation.ObjPostprocess(trt->engine_mode, objects, trt->prob, trt->num_box, BBOX_CONF_THRESH, NMS_THRESH, trt->yolomode);
        baseOperation.ObjUniqueprocess(objects, NMS_THRESH);
        //for (auto objects : batchobjects) {
        for (int i = 0; i < objects.size(); i++)
        {
            ResizeBox(img.cols, img.rows, padsize, objects[i].bbox);
            //if (objects[i].bbox[3] < 400) {
            //    continue;
            //}
            //if (objects[i].bbox[3] < 500 || objects[i].bbox[3] > 1100) {
            //    continue;
            //}
            if (objects[i].bbox[2] < 200){ // || objects[i].bbox[3] < 650) {
                continue;
            }
            for (auto j = 0; j < 4; j++) {
                resulttmp.box[j] = objects[i].bbox[j];
            }
            resulttmp.id = objects[i].label;
            resulttmp.confidence = objects[i].prob;
            if (trt->engine_mode == "obj") {
                if (resulttmp.id == 1 && CalcHeight(img, resulttmp.box)) {
                    resulttmp.id = 0;
                }
                //if (resulttmp.id == 0  && CalcAreaRatio(img, resulttmp.box) > 0.5) {
                //    resulttmp.confidence = 0.9;
                //}
                //cv::rectangle(img, cv::Point(resulttmp.box[0], resulttmp.box[1]), cv::Point(resulttmp.box[0] + resulttmp.box[2], resulttmp.box[1] + resulttmp.box[3]), cv::Scalar(0, 0, 255), 2);
            }
            else {
                resulttmp.radian = objects[i].radian;
                baseOperation.Radian(img, objects[i]);
            }
            output.push_back(resulttmp);
            loger.log("predict class: "+std::to_string(resulttmp.id)+" score: " +
                std::to_string(resulttmp.confidence) + " box: " + std::to_string(resulttmp.box[0])+"  "
                + std::to_string(resulttmp.box[1]) + "  " + std::to_string(resulttmp.box[2]) + "  "
                + std::to_string(resulttmp.box[3]));
            //std::cout << resulttmp.id << " " << resulttmp.confidence << " " << resulttmp.box[3] << std::endl;
        }
    }
    if (trt->engine_mode == "seg") {
        std::vector<int> imgSize = { img.cols, img.rows };
        std::vector<int> segMaskParams = { trt->num_box, trt->_segChannels, trt->_segWidth, trt->_segHeight };
        std::vector<BaseOperation::Object> objects;
        baseOperation.SegPostprocess(objects, trt->prob, trt->prob1, img, padsize, segMaskParams, trt->yolomode);
        //outputResultLen = objects.size();
        //outputResult = new BaseOperation::RecResult[outputResultLen];
        for (int i = 0; i < objects.size(); i++)
        {
            for (auto j = 0; j < 4; j++) {
                resulttmp.box[j] = objects[i].bbox[j];
            }
            resulttmp.id = objects[i].label;
            resulttmp.confidence = objects[i].prob;
            resulttmp.boxMask = objects[i].boxMask;
            output.push_back(resulttmp);
            loger.log("predict class: " + std::to_string(resulttmp.id) + " score: " +
                std::to_string(resulttmp.confidence) + " box: " + std::to_string(resulttmp.box[0]) + "  "
                + std::to_string(resulttmp.box[1]) + "  " + std::to_string(resulttmp.box[2]) + "  "
                + std::to_string(resulttmp.box[3]));
            //temp_mask_proposals.push_back(picked_proposals[idx]);
        }
    }
    //std::cout << "postprocess: " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start2).count() / 1000.0 << "ms" << std::endl;
    delete blob;
}

void AIGPUDetectImg(void* h, uchar* data, int width, int height, int stride, BaseOperation::RecResult*& result, int& outlen) {
    cv::Mat img = cv::Mat(cv::Size(width, height), CV_8UC3, data, stride).clone();
    std::vector<BaseOperation::RecResult> output;
    AIGPUDetect(h, img, "img", output);
    outlen = output.size();
    result = new BaseOperation::RecResult[outlen];
    memcpy(result, &output[0], outlen * sizeof(BaseOperation::RecResult));
}

void AiGPUDetectPath(void* h, const char* imgpath, BaseOperation::RecResult*& result, int& outlen) {
    std::vector<cv::String> imgLists;
    std::string path;
    path = imgpath;
    cv::glob(path, imgLists, true);
    std::vector<BaseOperation::RecResult> output;
    for (auto img : imgLists) {
        std::cout << std::string(img) << std::endl;
        cv::Mat srcimg = cv::imread(img);
        std::string imgname = img.substr(path.size() + 1);
        AIGPUDetect(h, srcimg, imgname, output);
    }
    //cv::Mat srcimg, srcimg1;
    //for (auto i = 0; i < imgLists.size(); i++) {
    //    srcimg = cv::imread(imgLists[0]);
    //    std::string imgname = imgLists[0].substr(path.size() + 1);
    //    srcimg1 = cv::imread(imgLists[1]);
    //    AIGPUDetect(h, srcimg, srcimg1,imgname, output);
    //}
    outlen = output.size();
    result = new BaseOperation::RecResult[outlen];
    memcpy(result, &output[0], outlen * sizeof(BaseOperation::RecResult));
}

void ReleaseStruct(BaseOperation::RecResult* structArray) {
    if (structArray) {
        delete structArray;
    }
    if (isRunning) {
        isRunning = false;
    }

}
#include <thread>
void main() {
    int outputResultLen;
    std::vector<BaseOperation::RecResult> outputResult;
    std::string modelpathstr = R"(D:\yolov8\ultralytics\runs\obb\train5\weights\best.engine)";

    modelpathstr = R"(E:\yolov8\ultralytics\weight\datasetnew2\train6\weights\best.engine)";
    //modelpathstr = R"(D:\data\yolov10\weight\all\train\weights\best.engine)";
    //modelpathstr = R"(D:\yolov8\ultralytics\weight\all\train23\weights\jsenginemodel1.jsmodel)";
    const char* modelpath = modelpathstr.c_str();
    std::string modelmodestr = "obj";
    const char* modelmode = modelmodestr.c_str();
    BaseOperation::Yolov5TRTContext* trt = (BaseOperation::Yolov5TRTContext*)AiGPUInit(modelpath, modelmode,true);
    std::string imgpath = R"(D:\yolov8\ultralytics\coco128-seg\images)";
    //cv::Mat imgdata = cv::imread(imgpath);
    //std::cout << nvinfer1::kNV_TENSORRT_VERSION_IMPL << std::endl;
    //AIGPUDetect(trt, imgdata, imgpath, outputResult);
    imgpath = R"(D:\yolov8\ultralytics\dota8\images\val)";
    imgpath = R"(G:\pengda2)";//1023tmp 1113
    BaseOperation::RecResult* result; 
    int outlen = 0;
    const char* p = imgpath.c_str();
    AiGPUDetectPath(trt, p, result, outlen);
    //for (int i = 0; i < 20; i++) {
    //    
    //    Sleep(1000);
    //}
    ReleaseStruct(result);
    std::this_thread::sleep_for(std::chrono::milliseconds(5000));
}