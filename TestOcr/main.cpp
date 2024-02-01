// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "include/logging.h"
#include "NvInfer.h"
#include "cuda_runtime_api.h"

#include <include/args.h>
#include <include/paddleocr.h>
#include <include/paddlestructure.h>
#include <gflags/gflags.h>

//#include "yolo.h"
using namespace PaddleOCR;
using namespace nvinfer1;
void check_params() {
  if (FLAGS_det) {
    if (FLAGS_det_model_dir.empty()) {
      std::cout << "Usage[det]: ./ppocr "
                   "--det_model_dir=/PATH/TO/DET_INFERENCE_MODEL/ "
                << "--image_dir=/PATH/TO/INPUT/IMAGE/" << std::endl;
      exit(1);
    }
  }
  if (FLAGS_rec) {
    std::cout
        << "In PP-OCRv3, rec_image_shape parameter defaults to '3, 48, 320',"
           "if you are using recognition model with PP-OCRv2 or an older "
           "version, "
           "please set --rec_image_shape='3,32,320"
        << std::endl;
    if (FLAGS_rec_model_dir.empty() ) {
      std::cout << "Usage[rec]: ./ppocr "
                   "--rec_model_dir=/PATH/TO/REC_INFERENCE_MODEL/ "
                << "--image_dir=/PATH/TO/INPUT/IMAGE/" << std::endl;
      exit(1);
    }
  }
  if (FLAGS_cls && FLAGS_use_angle_cls) {
    if (FLAGS_cls_model_dir.empty() ) {
      std::cout << "Usage[cls]: ./ppocr "
                << "--cls_model_dir=/PATH/TO/REC_INFERENCE_MODEL/ "
                << "--image_dir=/PATH/TO/INPUT/IMAGE/" << std::endl;
      exit(1);
    }
  }
  if (FLAGS_table) {
    if (FLAGS_table_model_dir.empty() || FLAGS_det_model_dir.empty() ||
        FLAGS_rec_model_dir.empty() ) {
      std::cout << "Usage[table]: ./ppocr "
                << "--det_model_dir=/PATH/TO/DET_INFERENCE_MODEL/ "
                << "--rec_model_dir=/PATH/TO/REC_INFERENCE_MODEL/ "
                << "--table_model_dir=/PATH/TO/TABLE_INFERENCE_MODEL/ "
                << "--image_dir=/PATH/TO/INPUT/IMAGE/" << std::endl;
      exit(1);
    }
  }
  if (FLAGS_layout) {
    if (FLAGS_layout_model_dir.empty() ) {
      std::cout << "Usage[layout]: ./ppocr "
                << "--layout_model_dir=/PATH/TO/LAYOUT_INFERENCE_MODEL/ "
                << "--image_dir=/PATH/TO/INPUT/IMAGE/" << std::endl;
      exit(1);
    }
  }
  if (FLAGS_precision != "fp32" && FLAGS_precision != "fp16" &&
      FLAGS_precision != "int8") {
    std::cout << "precison should be 'fp32'(default), 'fp16' or 'int8'. "
              << std::endl;
    exit(1);
  }
}

struct RecResult {
    int box[4];
    int label=-1;
    int textlen=0;
    char text[100];
    double score = -1.0;
    bool flag=false;
};
//void ocr(std::vector<cv::String> &cv_all_img_names) {
void ocr(std::vector<cv::Mat> & img_list, int& outputResultLen, RecResult*& outputResult) {
  PPOCR ocr = PPOCR();

  if (FLAGS_benchmark) {
    ocr.reset_timer();
  }

  //std::vector<cv::Mat> img_list;
  //std::vector<cv::String> img_names;
  //for (int i = 0; i < img_list.size(); ++i) {
  //  cv::Mat img = cv::imread(cv_all_img_names[i], cv::IMREAD_COLOR);
  //  if (!img.data) {
  //    std::cerr << "[ERROR] image read failed! image path: "
  //              << cv_all_img_names[i] << std::endl;
  //    continue;
  //  }
  //  img_list.push_back(img);
  //  img_names.push_back(cv_all_img_names[i]);
  //}
  auto pre_start = std::chrono::system_clock::now();
  std::vector<std::vector<OCRPredictResult>> ocr_results =
      ocr.ocr(img_list, FLAGS_det, FLAGS_rec, FLAGS_cls);
  std::cout << "doInference: " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - pre_start).count() / 1000.0 << "ms" << std::endl;
  std::vector<RecResult> ocrresults;
  for (int i = 0; i < img_list.size(); ++i) {
      for (int j = 0; j < ocr_results[i].size(); ++j) {
          RecResult ocrresulttmp;
          std::vector<int> boxtmp;
          for (const auto& box : ocr_results[i][j].box) {
              boxtmp.insert(boxtmp.end(), box.begin(), box.end());
          }
          memcpy(ocrresulttmp.box, &boxtmp[0], 2 * sizeof(int));
          ocrresulttmp.box[2] = boxtmp[4] - boxtmp[0];
          ocrresulttmp.box[3] = boxtmp[5] - boxtmp[1];
          strcpy(ocrresulttmp.text, ocr_results[i][j].text.c_str());
          //ocrresulttmp.text = &ocr_results[i][j].text[0];
          //ocrresulttmp.text = ocr_results[i][j].text.c_str();
          ocrresulttmp.textlen = strlen(ocr_results[i][j].text.c_str());
          ocrresulttmp.score = ocr_results[i][j].score;
          ocrresulttmp.flag = true;
          ocrresults.push_back(ocrresulttmp);
      }
  }
  outputResultLen = ocrresults.size();
  outputResult = new RecResult[outputResultLen];
  memcpy(outputResult, &ocrresults[0], outputResultLen * sizeof(RecResult));
  for (int i = 0; i < img_list.size(); i++) {
      Utility::print_result(ocr_results[i]);
  }
  //for (int i = 0; i < img_names.size(); ++i) {
  //  std::cout << "predict img: " << cv_all_img_names[i] << std::endl;
  //  Utility::print_result(ocr_results[i]);
  //  if (FLAGS_visualize && FLAGS_det) {
  //    std::string file_name = Utility::basename(img_names[i]);
  //    cv::Mat srcimg = img_list[i];
  //    Utility::VisualizeBboxes(srcimg, ocr_results[i],
  //                             FLAGS_output + "/" + file_name);
  //  }
  //}
  //if (FLAGS_benchmark) {
  //  ocr.benchmark_log(cv_all_img_names.size());
  //}
}

void structure(std::vector<cv::String> &cv_all_img_names) {
  PaddleOCR::PaddleStructure engine = PaddleOCR::PaddleStructure();

  if (FLAGS_benchmark) {
    engine.reset_timer();
  }

  for (int i = 0; i < cv_all_img_names.size(); i++) {
    std::cout << "predict img: " << cv_all_img_names[i] << std::endl;
    cv::Mat img = cv::imread(cv_all_img_names[i], cv::IMREAD_COLOR);
    if (!img.data) {
      std::cerr << "[ERROR] image read failed! image path: "
                << cv_all_img_names[i] << std::endl;
      continue;
    }

    std::vector<StructurePredictResult> structure_results = engine.structure(
        img, FLAGS_layout, FLAGS_table, FLAGS_det && FLAGS_rec);

    for (int j = 0; j < structure_results.size(); j++) {
      std::cout << j << "\ttype: " << structure_results[j].type
                << ", region: [";
      std::cout << structure_results[j].box[0] << ","
                << structure_results[j].box[1] << ","
                << structure_results[j].box[2] << ","
                << structure_results[j].box[3] << "], score: ";
      std::cout << structure_results[j].confidence << ", res: ";

      if (structure_results[j].type == "table") {
        std::cout << structure_results[j].html << std::endl;
        if (structure_results[j].cell_box.size() > 0 && FLAGS_visualize) {
          std::string file_name = Utility::basename(cv_all_img_names[i]);

          Utility::VisualizeBboxes(img, structure_results[j],
                                   FLAGS_output + "/" + std::to_string(j) +
                                       "_" + file_name);
        }
      } else {
        std::cout << "count of ocr result is : "
                  << structure_results[j].text_res.size() << std::endl;
        if (structure_results[j].text_res.size() > 0) {
          std::cout << "********** print ocr result "
                    << "**********" << std::endl;
          Utility::print_result(structure_results[j].text_res);
          std::cout << "********** end print ocr result "
                    << "**********" << std::endl;
        }
      }
    }
  }
  if (FLAGS_benchmark) {
    engine.benchmark_log(cv_all_img_names.size());
  }
}

extern "C"
{
    __declspec(dllexport) void* AIInit();
    __declspec(dllexport) void DetectInter(void* h, cv::Mat img, int& outputResultLen, RecResult*& outputResult, int detectmode);
}

typedef struct
{
    std::string engine_mode;
    float* data;
    float* prob;
    float* prob1;
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
    int _segWidth, _segHeight, _segChannels, Num_box;

}Yolov5TRTContext;
#define DEVICE 0  // GPU id
#define NMS_THRESH 0.45
#define BBOX_CONF_THRESH 0.25
#define MASK_THRESHOLD 0.5;
static int MASK_NUM = 32;
static  int INPUT_W = 64;
static  int INPUT_H = 64;
static  int NUM_CLASSES = 7;
//static const int _segWidth = 56;
//static const int _segHeight = 56;
//static const int _segChannels = 32;
//static const int Num_box = 3087;
//static const int OUTPUT_SIZE1 = _segChannels * _segWidth * _segHeight;//output1
const char* INPUT_BLOB_NAME = "images";
const char* OUTPUT_BLOB_NAME = "output0";
//const char* OUTPUT_BLOB_NAME1 = "output1";//mask
//const char* OUTPUT_BLOB_NAME1 = "1326";
const char* OUTPUT_BLOB_NAME1 = "645";
static Logger gLogger;

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

void GetConfigValue(const char* keyName, char* keyValue)
{
    std::string config_file = "./config/recconfig.conf";
    char buff[300] = { 0 };
    FILE* file = fopen(config_file.c_str(), "r");
    while (fgets(buff, 300, file))
    {
        char* tempKeyName = strtok(buff, "=");
        if (!tempKeyName) continue;
        char* tempKeyValue = strtok(NULL, "=");

        if (!strcmp(tempKeyName, keyName))
            strcpy(keyValue, tempKeyValue);
    }
    fclose(file);
}
void doInference(IExecutionContext& context, cudaStream_t& stream, ICudaEngine& engine, std::string engine_mode, void** buffers, float* input, float* output, const int output_size, float* output1, const int output1_size, Yolov5TRTContext* trt) {
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host

    //CHECK(cudaMemcpyAsync(buffers[engine.getBindingIndex(INPUT_BLOB_NAME)], input, 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(buffers[trt->inputindex], input, 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    //context.executeV2(buffers);
    context.enqueueV2(buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[trt->outputindex], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    if (engine_mode == "seg") {
        CHECK(cudaMemcpyAsync(output1, buffers[trt->output1index], output1_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    }
    cudaStreamSynchronize(stream);
}

void* AIInit()//(char* engineFilePath, char* engineMode)
{
    //ocr param init
    char* argv[1] = { };
    google::ReadFromFlagsFile(FLAGS_config_file.c_str(), argv[0], true);
    check_params();

    //detect init
    int dev_num = 0;
    cudaError_t error_id = cudaGetDeviceCount(&dev_num);
    if (error_id != cudaSuccess)
    {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        return nullptr;
    }
    char engine_filepath[1000] = { 0 };
    char enginemode[100] = { 0 };
    char input_w[100] = { 0 };
    char input_h[100] = { 0 };
    char num_class[100] = { 0 };
    GetConfigValue("engine_file_path", engine_filepath);
    GetConfigValue("engine_mode", enginemode);
    //GetConfigValue("INPUT_W", input_w);
    //GetConfigValue("INPUT_H", input_h);
    //GetConfigValue("NUM_CLASSES", num_class);
    engine_filepath[strlen(engine_filepath) - 1] = 0;
    //enginemode[strlen(enginemode) - 1] = 0;
    //INPUT_W = atoi(input_w);
    //INPUT_H = atoi(input_h);
    //NUM_CLASSES = atoi(num_class);
    std::string engine_mode = enginemode;
    /*std::string engine_mode = (char*)engineMode;*/
    cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    char* trtModelStream{ nullptr };
    size_t size{ 0 };
    //if (argc == 3 && std::string(argv[2]) == "-i") {
    std::string engine_file = engine_filepath;
    /*const std::string engine_file = (char*)engineFilePath;*/
    //std::cout << engine_file.length() << " -------"<< engine_file << std::endl;
    std::ifstream file(engine_file, std::ios::binary);
    Yolov5TRTContext* trt = new Yolov5TRTContext();
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
        std::cout << "read engine ok" << std::endl;
    }
    else {
        std::cout << "read engine failed" << std::endl;
    }

    trt->runtime = createInferRuntime(gLogger);
    assert(trt->runtime != nullptr);
    std::cout << "createInferRuntime ok" << std::endl;
    trt->engine = trt->runtime->deserializeCudaEngine(trtModelStream, size);
    assert(trt->engine != nullptr);
    std::cout << "deserializeCudaEngine ok" << std::endl;

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
    std::vector<Dims> inputDims;
    std::vector<Dims> outputDims;
    for (int i = 0; i < bindingnum; i++) {
        if (trt->engine->bindingIsInput(i)) {
            inputDims.push_back(trt->engine->getBindingDimensions(i));
        }
        else {
            outputDims.push_back(trt->engine->getBindingDimensions(i));
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

    auto out_dims = trt->engine->getBindingDimensions(outputIndex);
    trt->output_size = 1;
    trt->engine_mode = engine_mode;
    cudaStream_t streams;
    cudaStreamCreate(&streams);
    trt->stream = streams;
    INPUT_H = trt->engine->getBindingDimensions(inputIndex).d[2];
    INPUT_W = trt->engine->getBindingDimensions(inputIndex).d[3];
    if (engine_mode == "cls") {
        NUM_CLASSES = out_dims.d[1];
    }
    else if (engine_mode == "obj") {
        //NUM_CLASSES = out_dims.d[2] - 5;
        NUM_CLASSES = out_dims.d[1] - 4;
    }
    else {
        //NUM_CLASSES = out_dims.d[2] - MASK_NUM - 5;
        NUM_CLASSES = out_dims.d[1] - MASK_NUM - 4;
    }
    for (int j = 0; j < out_dims.nbDims; j++) {
        trt->output_size *= out_dims.d[j];
    }

    trt->prob = new float[trt->output_size];
    //trt->Num_box = out_dims.d[1];
    trt->Num_box = out_dims.d[2];
    trt->inputindex = inputIndex;
    trt->outputindex = outputIndex;
    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], trt->output_size * sizeof(float)));
    if (engine_mode == "seg") {
        Dims out1Dims;
        int outputIndex1 = 0;
        for (int i = 0; i < outputDims.size(); ++i)
        {
            if (outputDims[i].nbDims == 4) {
                out1Dims = outputDims[i];
                outputIndex1 = i+1;
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
    // 处理第一次推理时间长
     float* blob = new float[INPUT_W * INPUT_H * 3];
     doInference(*trt->context, trt->stream, *(trt->engine), trt->engine_mode, trt->buffers, blob, trt->prob, trt->output_size, trt->prob1, trt->output1_size, trt);
     delete[] blob;
    //for (int i = 0; i < 6; i++) {
    //    float* blob = new float[INPUT_W * INPUT_H * 3];
    //    doInference(*trt->context[i], trt->stream, *(trt->engine), trt->engine_mode, trt->buffers, blob, trt->prob, trt->output_size, trt->prob1, trt->output1_size, trt);
    //    delete[] blob;
    //}
    return (void*)trt;
}
cv::Mat static_resize(cv::Mat img, std::vector<int>& padsize, std::string engine_mode) {
    if (engine_mode == "cls") {
        int crop_size = std::min(img.cols, img.rows);
        int  left = (img.cols - crop_size) / 2, top = (img.rows - crop_size) / 2;
        cv::Mat crop_image = img(cv::Rect(left, top, crop_size, crop_size));
        cv::Mat out;
        cv::resize(crop_image, out, cv::Size(INPUT_W, INPUT_H));
        return out;
    }
    else {
        int w, h, x, y;
        float r_w = INPUT_W / (img.cols * 1.0);
        float r_h = INPUT_H / (img.rows * 1.0);
        if (r_h > r_w) {
            w = INPUT_W;
            h = r_w * img.rows;
            x = 0;
            y = (INPUT_H - h) / 2;
        }
        else {
            w = r_h * img.cols;
            h = INPUT_H;
            x = (INPUT_W - w) / 2;
            y = 0;
        }
        cv::Mat re(h, w, CV_8UC3);
        cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
        cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(114, 114, 114));
        re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
        padsize.push_back(h);
        padsize.push_back(w);
        padsize.push_back(y);
        padsize.push_back(x);// int newh = padsize[0], neww = padsize[1], padh = padsize[2], padw = padsize[3];

        return out;
        //float r = (std::min)(INPUT_W / (img.cols*1.0), INPUT_H / (img.rows*1.0));
        //// r = std::min(r, 1.0f);
        //int unpad_w = r * img.cols;
        //int unpad_h = r * img.rows;
        //cv::Mat re(unpad_h, unpad_w, CV_8UC3);
        //cv::resize(img, re, re.size());
        //cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(114, 114, 114));
        //re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
        //return out;
    }  
}
float* blobFromImage(cv::Mat img, std::string engine_mode, int channels) {
    std::cout << img.total() << std::endl;
    float* blob = new float[img.total() * channels];
    //int channels = 3;
    int img_h = img.rows;
    int img_w = img.cols;
    for (size_t c = 0; c < channels; c++) {
        for (int h = 0; h < img.rows; ++h)
        {
            //获取第i行首像素指针
            cv::Vec3b* p1 = img.ptr<cv::Vec3b>(h);
            for (int w = 0; w < img.cols; ++w)
            {
                //将img的bgr转为image的rgb 
                blob[c * img_w * img_h + h * img_w + w] = (float)(p1[w][2-c] / 255.0);
                if (engine_mode == "cls") {
                    // RGB mean(0.485, 0.456, 0.406) std(0.229, 0.224, 0.225)
                    if (c == 0) {
                        blob[c * img_w * img_h + h * img_w + w] = (blob[c * img_w * img_h + h * img_w + w] -0.485) / 0.229;
                    }
                    else if (c == 1) {
                        blob[c * img_w * img_h + h * img_w + w] = (blob[c * img_w * img_h + h * img_w + w] - 0.456) / 0.224;
                    }
                    else {
                        blob[(c)*img_w * img_h + h * img_w + w] = (blob[c * img_w * img_h + h * img_w + w] - 0.406) / 0.225;
                    }
                }
            }
        }
    }
    
    //for (size_t c = 0; c < channels; c++)
    //{
    //    for (size_t h = 0; h < img_h; h++)
    //    {
    //        for (size_t w = 0; w < img_w; w++)
    //        {
    //            blob[c * img_w * img_h + h * img_w + w] =
    //                (float)(img.at<cv::Vec3b>(h, w)[c] / 255.0);
    //            if (engine_mode == "cls") {
    //                if (c == 0) {
    //                    blob[c * img_w * img_h + h * img_w + w] = (blob[c * img_w * img_h + h * img_w + w] - 0.406) / 0.225;
    //                }
    //                else if (c == 1) {
    //                    blob[c * img_w * img_h + h * img_w + w] = (blob[c * img_w * img_h + h * img_w + w] - 0.456) / 0.224;
    //                }
    //                else {
    //                    blob[(c) * img_w * img_h + h * img_w + w] = (blob[c * img_w * img_h + h * img_w + w] - 0.485) / 0.229;
    //                }
    //            }
    //        }
    //    }
    //}
    return blob;
}

double myfunction(double num) {
    return exp(num);
}
template <typename T>
void softmax(const typename::std::vector<T>& v, typename::std::vector<T>& s) {
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
struct Object
{
    //cv::Rect_<float> rect;
    float bbox[4];
    float prob;
    float label;
};
bool cmp(const Object& a, const Object& b) {
    return a.prob > b.prob;
}
float iou(float lbox[4], float rbox[4]) {
    float interBox[] = {
        (std::max)(lbox[0] - lbox[2] / 2.f , rbox[0] - rbox[2] / 2.f), //left
        (std::min)(lbox[0] + lbox[2] / 2.f , rbox[0] + rbox[2] / 2.f), //right
        (std::max)(lbox[1] - lbox[3] / 2.f , rbox[1] - rbox[3] / 2.f), //top
        (std::min)(lbox[1] + lbox[3] / 2.f , rbox[1] + rbox[3] / 2.f), //bottom
    };

    if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);
    return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
}

void nms(std::vector<Object>& res, float* output, float conf_thresh, float nms_thresh = 0.5) {
    int mi = NUM_CLASSES + 4;
    int det_size = sizeof(Object) / sizeof(float);
    std::map<float, std::vector<Object>> m;
    for (int i = 0; i < 8400; i++) {
        //if (output[mi * i + 4] <= conf_thresh) continue; //yolov5
        float tmp = 0.0;
        float labeltmp = -1;
        for (int j = 4; j < mi; j++) {
            //output[mi * i + j] *= output[mi * i + 4]; //yolov5
            if (output[mi * i + j] > tmp) {
                tmp = output[mi * i + j];
                labeltmp = j - 4; //yolov8
                //output[mi * i + 5] = j - 5;  //yolov5
            }
        }
        output[mi * i + 4] = tmp;
        output[mi * i + 5] = labeltmp;

        if (tmp < conf_thresh) continue;
        Object det;
        memcpy(&det, &output[mi * i], det_size * sizeof(float));
        if (m.count(det.label) == 0) m.emplace(det.label, std::vector<Object>());
        m[det.label].push_back(det);
    }
    for (auto it = m.begin(); it != m.end(); it++) {
        //std::cout << it->second[0].class_id << " --- " << std::endl;
        auto& dets = it->second;
        std::sort(dets.begin(), dets.end(), cmp);
        for (size_t m = 0; m < dets.size(); ++m) {
            auto& item = dets[m];
            res.push_back(item);
            for (size_t n = m + 1; n < dets.size(); ++n) {
                if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
                    dets.erase(dets.begin() + n);
                    --n;
                }
            }
        }
    }
}

void AIDetect(void* h, cv::Mat img, int& outputResultLen, RecResult*& outputResult) {
    Yolov5TRTContext* trt = (Yolov5TRTContext*)h;
    int img_w = img.cols;
    int img_h = img.rows;
    std::vector<int> padsize;
    cv::Mat pr_img = static_resize(img, padsize, trt->engine_mode);
    std::cout << "blob image" << std::endl;

    float* blob;
    blob = blobFromImage(pr_img, trt->engine_mode, 3);
    float scale = (std::min)(INPUT_W / (img.cols * 1.0), INPUT_H / (img.rows * 1.0));
    auto start = std::chrono::system_clock::now();
    doInference(*trt->context, trt->stream, *(trt->engine), trt->engine_mode, trt->buffers, blob, trt->prob, trt->output_size, trt->prob1, trt->output1_size, trt);

    auto end = std::chrono::system_clock::now();
    std::cout << "doInference: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0 << "ms" << std::endl;

    if (trt->engine_mode == "cls") {
        std::vector<float> vecprob(trt->prob, trt->prob + trt->output_size), vecprob1(vecprob);;
        softmax(vecprob, vecprob1);
        outputResultLen = 1;
        outputResult = new RecResult[outputResultLen];
        outputResult[0].label = arg_max(vecprob1);
        outputResult[0].score = vecprob1[outputResult[0].label];
        std::cout << outputResult[0].label<< " " <<outputResult[0].score << std::endl;
    }
    if (trt->engine_mode == "obj") {
        //yolov8 
        cv::Mat output = cv::Mat(NUM_CLASSES+4, trt->Num_box,  CV_32F, trt->prob).t();
        float* array =  output.ptr<float>();
        std::vector<Object> objects;
        nms(objects, array, BBOX_CONF_THRESH, NMS_THRESH);
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        outputResultLen = objects.size();
        outputResult = new RecResult[outputResultLen];
        for (int i = 0; i < objects.size(); i++)
        {
            for (auto j = 0; j < 4; j++) {
                outputResult[i].box[j] = objects[i].bbox[j];
                std::cout << outputResult[i].box[j] << std::endl;
            }
            outputResult[i].label = objects[i].label;
            outputResult[i].score = objects[i].prob;
            std::cout << outputResult[i].label << " " << outputResult[i].score << std::endl;
        }
    }
    if (trt->engine_mode == "seg") {
        std::vector<int> classIds;//结果id数组
        std::vector<float> confidences;//结果每个id对应置信度数组
        std::vector<cv::Rect> boxes;//每个id矩形框
        std::vector<std::vector<float>> picked_proposals;  //存储output0[:,:, 5 + _className.size():net_width]用以后续计算mask

        int newh = padsize[0], neww = padsize[1], padh = padsize[2], padw = padsize[3];
        //printf("newh:%d,neww:%d,padh:%d,padw:%d", newh, neww, padh, padw);
        float ratio_h = (float)img_h / newh;
        float ratio_w = (float)img_w / neww;

        // 处理box
        int net_width = NUM_CLASSES + 5 + trt->_segChannels;
        float* pdata = trt->prob;
        for (int j = 0; j < trt->Num_box; ++j) {
            float box_score = pdata[4]; ;//获取每一行的box框中含有某个物体的概率
            if (box_score >= BBOX_CONF_THRESH) {
                cv::Mat scores(1, NUM_CLASSES, CV_32FC1, pdata + 5);
                cv::Point classIdPoint;
                double max_class_socre;
                minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
                max_class_socre = (float)max_class_socre;
                if (max_class_socre >= BBOX_CONF_THRESH) {

                    std::vector<float> temp_proto(pdata + 5 + NUM_CLASSES, pdata + net_width);
                    picked_proposals.push_back(temp_proto);

                    float x = (pdata[0] - padw) * ratio_w;  //x
                    float y = (pdata[1] - padh) * ratio_h;  //y
                    float w = pdata[2] * ratio_w;  //w
                    float h = pdata[3] * ratio_h;  //h

                    int left = MAX((x - 0.5 * w), 0);
                    int top = MAX((y - 0.5 * h), 0);
                    classIds.push_back(classIdPoint.x);
                    confidences.push_back(max_class_socre * box_score);
                    boxes.push_back(cv::Rect(left, top, int(w), int(h)));
                }
            }
            pdata += net_width;//下一行
        }
        //std::vector<OutputSeg> output;
        if (!classIds.empty()) {
            //执行非最大抑制以消除具有较低置信度的冗余重叠框（NMS）
            std::vector<int> nms_result;
            cv::dnn::NMSBoxes(boxes, confidences, BBOX_CONF_THRESH, NMS_THRESH, nms_result);
            std::vector<std::vector<float>> temp_mask_proposals;
            cv::Rect holeImgRect(0, 0, img_w, img_h);
            outputResultLen = nms_result.size();
            outputResult = new RecResult[outputResultLen];
            for (int i = 0; i < nms_result.size(); ++i) {
                int idx = nms_result[i];
                RecResult result;
                outputResult[i].label = classIds[idx];
                outputResult[i].score = confidences[idx];
                cv::Rect rect = boxes[idx] & holeImgRect;
                outputResult[i].box[0] = rect.x;
                outputResult[i].box[1] = rect.y;
                outputResult[i].box[2] = rect.width;
                outputResult[i].box[3] = rect.height;
                std::cout << outputResult[i].box[0] << "  " << outputResult[i].box[1] << "  " << outputResult[i].box[2]
                    << "  " << outputResult[i].box[3] << "  " << outputResult[i].score << std::endl;
                //temp_mask_proposals.push_back(picked_proposals[idx]);
            }
        }
        delete blob;
    }
}

void OcrDetect(cv::Mat img, int& outputResultLen, RecResult*& outputResult) {
    std::vector<cv::Mat> img_lists;
    img_lists.push_back(img);
    ocr(img_lists, outputResultLen, outputResult);
}

void HoleDetect(cv::Mat img, int& outputResultLen, RecResult*& outputResult) {
    outputResultLen = 1;
    outputResult = new RecResult[outputResultLen];
    int flaghei = 0;
    cv::Mat gray;
    if (img.channels() == 3) {
        cvtColor(img, gray, cv::COLOR_RGB2GRAY);
    }
    for (auto i = 0; i < gray.size().height; i = i + 5) {
        uchar* sdata = gray.data + gray.step * i;
        int flagwid = 0;
        for (auto j = 0; j < gray.size().width; j++) {
            if (sdata[j] < 100) {
                flagwid++;
                if (flagwid > 20) {
                    flaghei++;
                    break;
                }
            }
        }
        if (flaghei > 3) {
            outputResult[0].flag = true;
            break;
        }
    }
}

void DetectInter(void* h, cv::Mat img, int& outputResultLen, RecResult*& outputResult, int detectmode) {
    switch (detectmode) {
    // OCR
    case 0:
        OcrDetect(img, outputResultLen, outputResult);
        break;
    // classification、detection、 segmentation
    case 1:
        AIDetect(h, img, outputResultLen, outputResult);
        break;
    //Hole detection
    case 2:
        HoleDetect(img, outputResultLen, outputResult);
        break;
    default:
        std::cout << "param detectmode is invalid" << std::endl;
        break;
    }
}

void testocrori(int& outputResultLen, RecResult*& outputResult) {
    char* argv[1] = { };
    // Parsing command-line
    //google::ParseCommandLineFlags(&argc, &argv, true);
    //FLAGS_flagfile.empty();
    google::ReadFromFlagsFile(FLAGS_config_file.c_str(), argv[0], true);
    google::SetCommandLineOption("image_dir", "./imgs/");
    check_params();

    if (!Utility::PathExists(FLAGS_image_dir)) {
        std::cerr << "[ERROR] image path not exist! image_dir: " << FLAGS_image_dir
            << std::endl;
        exit(1);
    }
    //cv::Mat img = cv::Mat(cv::Size(width, height), CV_8UC3, data, stride);

    std::vector<cv::String> cv_all_img_names;
    cv::glob(FLAGS_image_dir, cv_all_img_names);
    std::cout << "total images num: " << cv_all_img_names.size() << std::endl;
    cv::Mat img = cv::imread(cv_all_img_names[0], cv::IMREAD_COLOR);
    std::vector<cv::Mat> img_lists;
    img_lists.push_back(img);
    if (!Utility::PathExists(FLAGS_output)) {
        Utility::CreateDir(FLAGS_output);
    }
    if (FLAGS_type == "ocr") {
        ocr(img_lists, outputResultLen, outputResult);
    }
    else if (FLAGS_type == "structure") {
        structure(cv_all_img_names);
    }
    else {
        std::cout << "only value in ['ocr','structure'] is supported" << std::endl;
    }
}

//void DnnDetect()
//{
//    std::string model_path = R"(D:\yolov5\yolov5\best.onnx)";
//    std::string img_path = R"(D:\yolov5\yolov5\data\C0\detect\20200404051008-C0910-3.jpg)";
//    Yolov5 test;
//    cv::dnn::Net net;
//    if (test.readModel(net, model_path, false)) {
//        std::cout << "read net ok!" << std::endl;
//    }
//    else {
//        std::cout << "read net error!" << std::endl;;
//    }
//    cv::Mat img = cv::imread(img_path);
//    int outputResultLen;
//    RecResult* outputResult;
//    test.Detect(img, net, outputResultLen, outputResult);
//}

void main() {
    int outputResultLen;
    RecResult* outputResult = new RecResult[4];
    Yolov5TRTContext* trt = (Yolov5TRTContext*)AIInit();
    cv::Mat img = cv::imread("C:\\Users\\rs\\Desktop\\ocr.png", cv::IMREAD_COLOR);
    OcrDetect(img, outputResultLen, outputResult);
    std::string imgpath = R"(D:\yolov8\ultralytics\data\C0\JPEGImages\20200404051008-C0910-5.jpg)";
    cv::Mat imgdata = cv::imread(imgpath);
    AIDetect(trt, imgdata, outputResultLen, outputResult); 
    //DnnDetect();
}
