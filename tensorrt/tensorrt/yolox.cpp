#ifdef _MSC_VER
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT
#endif
#pragma warning(disable:4996)
#undef NDEBUG
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
#include "logging.h"
#include <thread>
#include "safequeue.h"

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

using namespace nvinfer1;
using namespace std;
using namespace cv;


// stuff we know about the network and the input/output blobs
static  int INPUT_W = 640;
static  int INPUT_H = 640;
static  int NUM_CLASSES = 7;
//static const int _segWidth = 56;
//static const int _segHeight = 56;
//static const int _segChannels = 32;
//static const int Num_box = 3087;
//static const int OUTPUT_SIZE1 = _segChannels * _segWidth * _segHeight;//output1
const char* INPUT_BLOB_NAME = "images";
const char* OUTPUT_BLOB_NAME = "output0";
//const char* OUTPUT_BLOB_NAME1 = "output1";//mask
const char* OUTPUT_BLOB_NAME1 = "1326";
static Logger gLogger;

cv::Mat static_resize(cv::Mat& img, std::vector<int>&padsize) {
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
    cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(128, 128, 128));
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

struct Object
{
    //cv::Rect_<float> rect;
    float bbox[4];
    float prob;
    float label;
};

//struct OutputSeg {
//    int id;             //结果类别id
//    float confidence;   //结果置信度
//    cv::Rect box;       //矩形框
//    cv::Mat boxMask;       //矩形框内mask，节省内存空间和加快速度
//};

struct OutputSeg {
    int id;             //结果类别id
    double confidence;   //结果置信度
    int box[4];       //矩形框
    int bytesize;
    BYTE* boxMask;
};

struct GridAndStride
{
    int grid0;
    int grid1;
    int stride;
};

static void generate_grids_and_stride(std::vector<int>& strides, std::vector<GridAndStride>& grid_strides)
{
    for (auto stride : strides)
    {
        int num_grid_y = INPUT_H / stride;
        int num_grid_x = INPUT_W / stride;
        for (int g1 = 0; g1 < num_grid_y; g1++)
        {
            for (int g0 = 0; g0 < num_grid_x; g0++)
            {
                GridAndStride grid_stride = { g0, g1, stride };
                grid_strides.push_back(grid_stride);
            }
        }
    }
}

//static inline float intersection_area(const Object& a, const Object& b)
//{
//    cv::Rect_<float> inter = a.rect & b.rect;
//    return inter.area();
//}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

//static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
//{
//    picked.clear();
//
//    const int n = faceobjects.size();
//
//    std::vector<float> areas(n);
//    for (int i = 0; i < n; i++)
//    {
//        areas[i] = faceobjects[i].rect.area();
//    }
//
//    for (int i = 0; i < n; i++)
//    {
//        const Object& a = faceobjects[i];
//
//        int keep = 1;
//        for (int j = 0; j < (int)picked.size(); j++)
//        {
//            const Object& b = faceobjects[picked[j]];
//
//            // intersection over union
//            float inter_area = intersection_area(a, b);
//            float union_area = areas[i] + areas[picked[j]] - inter_area;
//            // float IoU = inter_area / union_area
//            if (inter_area / union_area > nms_threshold)
//                keep = 0;
//        }
//
//        if (keep)
//            picked.push_back(i);
//    }
//}


//static void generate_yolox_proposals(std::vector<GridAndStride> grid_strides, float* feat_blob, float prob_threshold, std::vector<Object>& objects)
//{
//
//    const int num_anchors = grid_strides.size();
//
//    for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++)
//    {
//        const int grid0 = grid_strides[anchor_idx].grid0;
//        const int grid1 = grid_strides[anchor_idx].grid1;
//        const int stride = grid_strides[anchor_idx].stride;
//
//        const int basic_pos = anchor_idx * (NUM_CLASSES + 5);
//
//        // yolox/models/yolo_head.py decode logic
//        float x_center = (feat_blob[basic_pos+0] + grid0) * stride;
//        float y_center = (feat_blob[basic_pos+1] + grid1) * stride;
//        float w = exp(feat_blob[basic_pos+2]) * stride;
//        float h = exp(feat_blob[basic_pos+3]) * stride;
//        float x0 = x_center - w * 0.5f;
//        float y0 = y_center - h * 0.5f;
//
//        float box_objectness = feat_blob[basic_pos+4];
//        for (int class_idx = 0; class_idx < NUM_CLASSES; class_idx++)
//        {
//            float box_cls_score = feat_blob[basic_pos + 5 + class_idx];
//            float box_prob = box_objectness * box_cls_score;
//            if (box_prob > prob_threshold)
//            {
//                Object obj;
//                obj.rect.x = x0;
//                obj.rect.y = y0;
//                obj.rect.width = w;
//                obj.rect.height = h;
//                obj.label = class_idx;
//                obj.prob = box_prob;
//
//                objects.push_back(obj);
//            }
//
//        } // class loop
//
//    } // point anchor loop
//}

float* blobFromImage(cv::Mat& img, std::string engine_mode, int channels){
    float* blob = new float[img.total()* channels];
    //int channels = 3;
    int img_h = img.rows;
    int img_w = img.cols;
    for (size_t c = 0; c < channels; c++) 
    {
        for (size_t  h = 0; h < img_h; h++) 
        {
            for (size_t w = 0; w < img_w; w++) 
            {
                blob[c * img_w * img_h + h * img_w + w] =
                    (float)(img.at<cv::Vec3b>(h, w)[c]/255.0);
                if (engine_mode == "cls") {
                    blob[c * img_w * img_h + h * img_w + w] = (blob[c * img_w * img_h + h * img_w + w] - 0.5) / 0.5;
                }
            }
        }
    }
    return blob;
}


//static void decode_outputs(float* prob, std::vector<Object>& objects, float scale, const int img_w, const int img_h) {
//        std::vector<Object> proposals;
//        std::vector<int> strides = {8, 16, 32};
//        std::vector<GridAndStride> grid_strides;
//        generate_grids_and_stride(strides, grid_strides);
//        generate_yolox_proposals(grid_strides, prob,  BBOX_CONF_THRESH, proposals);
//        std::cout << "num of boxes before nms: " << proposals.size() << std::endl;
//
//        qsort_descent_inplace(proposals);
//
//        std::vector<int> picked;
//        nms_sorted_bboxes(proposals, picked, NMS_THRESH);
//
//
//        int count = picked.size();
//
//        std::cout << "num of boxes: " << count << std::endl;
//
//        objects.resize(count);
//        for (int i = 0; i < count; i++)
//        {
//            objects[i] = proposals[picked[i]];
//
//            // adjust offset to original unpadded
//            float x0 = (objects[i].rect.x) / scale;
//            float y0 = (objects[i].rect.y) / scale;
//            float x1 = (objects[i].rect.x + objects[i].rect.width) / scale;
//            float y1 = (objects[i].rect.y + objects[i].rect.height) / scale;
//
//            // clip
//            x0 = (std::max)((std::min)(x0, (float)(img_w - 1)), 0.f);
//            y0 = (std::max)((std::min)(y0, (float)(img_h - 1)), 0.f);
//            x1 = (std::max)((std::min)(x1, (float)(img_w - 1)), 0.f);
//            y1 = (std::max)((std::min)(y1, (float)(img_h - 1)), 0.f);
//
//            objects[i].rect.x = x0;
//            objects[i].rect.y = y0;
//            objects[i].rect.width = x1 - x0;
//            objects[i].rect.height = y1 - y0;
//        }
//}

const float color_list[80][3] =
{
    {0.000, 0.447, 0.741},
    {0.850, 0.325, 0.098},
    {0.929, 0.694, 0.125},
    {0.494, 0.184, 0.556},
    {0.466, 0.674, 0.188},
    {0.301, 0.745, 0.933},
    {0.635, 0.078, 0.184},
    {0.300, 0.300, 0.300},
    {0.600, 0.600, 0.600},
    {1.000, 0.000, 0.000},
    {1.000, 0.500, 0.000},
    {0.749, 0.749, 0.000},
    {0.000, 1.000, 0.000},
    {0.000, 0.000, 1.000},
    {0.667, 0.000, 1.000},
    {0.333, 0.333, 0.000},
    {0.333, 0.667, 0.000},
    {0.333, 1.000, 0.000},
    {0.667, 0.333, 0.000},
    {0.667, 0.667, 0.000},
    {0.667, 1.000, 0.000},
    {1.000, 0.333, 0.000},
    {1.000, 0.667, 0.000},
    {1.000, 1.000, 0.000},
    {0.000, 0.333, 0.500},
    {0.000, 0.667, 0.500},
    {0.000, 1.000, 0.500},
    {0.333, 0.000, 0.500},
    {0.333, 0.333, 0.500},
    {0.333, 0.667, 0.500},
    {0.333, 1.000, 0.500},
    {0.667, 0.000, 0.500},
    {0.667, 0.333, 0.500},
    {0.667, 0.667, 0.500},
    {0.667, 1.000, 0.500},
    {1.000, 0.000, 0.500},
    {1.000, 0.333, 0.500},
    {1.000, 0.667, 0.500},
    {1.000, 1.000, 0.500},
    {0.000, 0.333, 1.000},
    {0.000, 0.667, 1.000},
    {0.000, 1.000, 1.000},
    {0.333, 0.000, 1.000},
    {0.333, 0.333, 1.000},
    {0.333, 0.667, 1.000},
    {0.333, 1.000, 1.000},
    {0.667, 0.000, 1.000},
    {0.667, 0.333, 1.000},
    {0.667, 0.667, 1.000},
    {0.667, 1.000, 1.000},
    {1.000, 0.000, 1.000},
    {1.000, 0.333, 1.000},
    {1.000, 0.667, 1.000},
    {0.333, 0.000, 0.000},
    {0.500, 0.000, 0.000},
    {0.667, 0.000, 0.000},
    {0.833, 0.000, 0.000},
    {1.000, 0.000, 0.000},
    {0.000, 0.167, 0.000},
    {0.000, 0.333, 0.000},
    {0.000, 0.500, 0.000},
    {0.000, 0.667, 0.000},
    {0.000, 0.833, 0.000},
    {0.000, 1.000, 0.000},
    {0.000, 0.000, 0.167},
    {0.000, 0.000, 0.333},
    {0.000, 0.000, 0.500},
    {0.000, 0.000, 0.667},
    {0.000, 0.000, 0.833},
    {0.000, 0.000, 1.000},
    {0.000, 0.000, 0.000},
    {0.143, 0.143, 0.143},
    {0.286, 0.286, 0.286},
    {0.429, 0.429, 0.429},
    {0.571, 0.571, 0.571},
    {0.714, 0.714, 0.714},
    {0.857, 0.857, 0.857},
    {0.000, 0.447, 0.741},
    {0.314, 0.717, 0.741},
    {0.50, 0.5, 0}
};

//static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects, std::string f)
//{
//    //static const char* class_names[] = {
//    //    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
//    //    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
//    //    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
//    //    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
//    //    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
//    //    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
//    //    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
//    //    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
//    //    "hair drier", "toothbrush"
//    //};
//    static const char* class_names[] = { "C_MC","C_OTH","C_CM","C_TS","C_PASS" };
//
//    cv::Mat image = bgr.clone();
//
//    for (size_t i = 0; i < objects.size(); i++)
//    {
//        const Object& obj = objects[i];
//
//        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
//                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);
//
//        cv::Scalar color = cv::Scalar(color_list[obj.label][0], color_list[obj.label][1], color_list[obj.label][2]);
//        float c_mean = cv::mean(color)[0];
//        cv::Scalar txt_color;
//        if (c_mean > 0.5){
//            txt_color = cv::Scalar(0, 0, 0);
//        }else{
//            txt_color = cv::Scalar(255, 255, 255);
//        }
//
//        cv::rectangle(image, obj.rect, color * 255, 2);
//
//        char text[256];
//        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);
//
//        int baseLine = 0;
//        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);
//
//        cv::Scalar txt_bk_color = color * 0.7 * 255;
//
//        int x = obj.rect.x;
//        int y = obj.rect.y + 1;
//        //int y = obj.rect.y - label_size.height - baseLine;
//        if (y > image.rows)
//            y = image.rows;
//        //if (x + label_size.width > image.cols)
//            //x = image.cols - label_size.width;
//
//        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
//                      txt_bk_color, -1);
//
//        cv::putText(image, text, cv::Point(x, y + label_size.height),
//                    cv::FONT_HERSHEY_SIMPLEX, 0.4, txt_color, 1);
//    }
//
//    cv::imwrite("det_res.jpg", image);
//    fprintf(stderr, "save vis file\n");
//    /* cv::imshow("image", image); */
//    /* cv::waitKey(0); */
//}


void doInference(IExecutionContext& context, cudaStream_t& stream, ICudaEngine& engine, std::string engine_mode, void** buffers, float* input, float* output, const int output_size, float* output1, const int output1_size) {
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    
    //CHECK(cudaMemcpyAsync(buffers[engine.getBindingIndex(INPUT_BLOB_NAME)], input, 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(buffers[engine.getBindingIndex(INPUT_BLOB_NAME)], input, 1 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    //context.executeV2(buffers);
    context.enqueueV2(buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[engine.getBindingIndex(OUTPUT_BLOB_NAME)], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    if (engine_mode == "seg") {
        CHECK(cudaMemcpyAsync(output1, buffers[engine.getBindingIndex(OUTPUT_BLOB_NAME1)], output1_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    }
    cudaStreamSynchronize(stream);
}


cv::Rect get_rect(cv::Mat& img, float bbox[4]) {
    int l, r, t, b;
    float r_w = INPUT_W / (img.cols * 1.0);
    float r_h = INPUT_H / (img.rows * 1.0);
    if (r_h > r_w) {
        l = bbox[0] - bbox[2] / 2.f;
        r = bbox[0] + bbox[2] / 2.f;
        t = bbox[1] - bbox[3] / 2.f - (INPUT_H - r_w * img.rows) / 2;
        b = bbox[1] + bbox[3] / 2.f - (INPUT_H - r_w * img.rows) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    }
    else {
        l = bbox[0] - bbox[2] / 2.f - (INPUT_W - r_h * img.cols) / 2;
        r = bbox[0] + bbox[2] / 2.f - (INPUT_W - r_h * img.cols) / 2;
        t = bbox[1] - bbox[3] / 2.f;
        b = bbox[1] + bbox[3] / 2.f;
        l = l / r_h;
        r = r / r_h;
        t = t / r_h;
        b = b / r_h;
    }
    return cv::Rect(l, t, r - l, b - t);
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

bool cmp(const Object& a, const Object& b) {
    return a.prob > b.prob;
}

void nms(std::vector<Object>& res, float* output, float conf_thresh, float nms_thresh = 0.5) {
    int mi = NUM_CLASSES + 5;
    int det_size = sizeof(Object) / sizeof(float);
    std::map<float, std::vector<Object>> m;
    for (int i = 0; i < 1000; i++) {
        if (output[mi * i + 4] <= conf_thresh) continue;
        float tmp = 0.0;
        for (int j = 5; j < mi; j++) {
            output[mi * i + j] *= output[mi * i + 4];
            if (output[mi * i + j] > tmp) {
                tmp = output[mi * i + j];
                output[mi * i + 5] = j - 5;
            }
        }
        Object det;
        memcpy(&det, &output[mi * i], det_size * sizeof(float));
        if (m.count(det.label) == 0) m.emplace(det.label , std::vector<Object>());
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

std::vector<std::string> read_images_in_folder(cv::String pattern)
{
    std::vector<cv::String> fn;
    glob(pattern, fn, false);

    std::vector<std::string> images_list;
    size_t count = fn.size(); //number of png files in images folder

    for (auto x : fn)
    {
        std::string suffix_str = x.substr(x.find_last_of('.') + 1);
        if ((suffix_str == "jpg") || (suffix_str == "png") || suffix_str == "bmp" || suffix_str == "jpeg")
        {
            images_list.push_back(std::string(x));
        }

        //cout << "1";
    }
    /*for (size_t i = 0; i < count; i++)
    {

        images_list.push_back(fn[i]);
    }*/

    return images_list;
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

//void DrawPred(cv::Mat& img, std::vector<OutputSeg> result) {
//    //生成随机颜色
//    std::vector < cv:: Scalar > color;
//    srand(time(0));
//    for (int i = 0; i < NUM_CLASSES; i++) {
//        int b = rand() % 256;
//        int g = rand() % 256;
//        int r = rand() % 256;
//        color.push_back(cv::Scalar(b, g, r));
//    }
//    cv::Mat mask = img.clone();
//    for (int i = 0; i < result.size(); i++) {
//        int left, top;
//        //left = result[i].box.x;
//        //top = result[i].box.y;
//        //int color_num = i;
//        //rectangle(img, result[i].box, color[result[i].id], 2, 8);
//        //mask(result[i].box).setTo(color[result[i].id], result[i].boxMask);
//        left = result[i].box[0];
//        top = result[i].box[1];
//        int color_num = i;
//        rectangle(img, cv::Rect(result[i].box[0], result[i].box[1], result[i].box[2], result[i].box[3]), color[result[i].id], 2, 8);
//        int nByte = result[i].box[2] * result[i].box[3];//字节计算
//        Mat outImg = Mat::zeros(result[i].box[3], result[i].box[2], CV_8UC1);
//        memcpy(outImg.data, result[i].boxMask, nByte);
//        mask(Rect(result[i].box[0], result[i].box[1], result[i].box[2], result[i].box[3])).setTo(color[result[i].id], outImg);
//
//        std::string label = std::to_string(result[i].id) + ":" + std::to_string(result[i].confidence);
//        int baseLine;
//        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
//        top = max(top, labelSize.height);
//        putText(img, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 1, color[result[i].id], 2);
//    }
//    addWeighted(img, 0.5, mask, 0.5, 0, img); //将mask加在原图上面
//}

void GetConfigValue(const char* keyName, char* keyValue)
{
    std::string config_file = "./config.cfg";
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


//Queue<float*> blob_queue;
std::queue<float*> blob_queue;
//bool finished = false;
std::atomic_bool finished = ATOMIC_VAR_INIT(false);
void preprocess() {
    auto pre_start = std::chrono::system_clock::now();
    
    for (int i = 0; i < 100000000; i++) {
        float* blob = new float[2];
        blob[0] = i+10;
        blob[1] = i+11;
        blob_queue.push(blob);
        //printf( "插入第%d组数据",i);
    }
    finished = true;
    //blob_queue.finished();
    std::cout << "预处理线程：" << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - pre_start).count() << "ms" << std::endl;
}

void inference() {
    auto prestart = std::chrono::system_clock::now();
    while(!blob_queue.empty() or !finished) {
        if (!blob_queue.empty()) {
            float* blob = new float[2];
            blob = blob_queue.front();
            blob_queue.pop();
            //std::cout << blob[0] << " " << blob[1] << std::endl;
        }
    }
    //blob_queue.quit();
    std::cout << "推理线程：" << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - prestart).count() << "ms" << std::endl;
}

int main1(int argc, char** argv) {
    cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};
    //if (argc == 3 && std::string(argv[2]) == "-i") {
    char engine_filepath[100] = { 0 };
    char enginemode[100] = { 0 };
    char input_w[100] = { 0 };
    char input_h[100] = { 0 };
    char num_class[100] = { 0 };
    GetConfigValue("engine_file_path", engine_filepath);
    GetConfigValue("engine_mode", enginemode);
    GetConfigValue("INPUT_W", input_w);
    GetConfigValue("INPUT_H", input_h);
    GetConfigValue("NUM_CLASSES", num_class);
    engine_filepath[strlen(engine_filepath) - 1] = 0;
    enginemode[strlen(enginemode) - 1] = 0;
    INPUT_W = atoi(input_w);
    INPUT_H = atoi(input_h);
    NUM_CLASSES = atoi(num_class);
    const std::string engine_file_path = engine_filepath;
    const std::string engine_mode = enginemode;
    //const std::string engine_file_path = "./model/VARTA_seg.engine";
    //const std::string engine_mode = "seg";
    std::ifstream file(engine_file_path, std::ios::binary);
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

    //const std::string input_image_path = "20200404051008-C0910-3.jpg";

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr); 
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    //// Pointers to input and output device buffers to pass to engine.
    //// Engine requires exactly IEngine::getNbBindings() number of buffers.
    void** buffers;
    const int bindingnum = engine->getNbBindings();
    //if (engine_mode != "seg") {
    //    assert(bindingnum == 2);
    //}
    //else {
    //    assert(bindingnum == 3);
    //}

    buffers = new void* [bindingnum];

    //void* buffers[3];
    
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);

    assert(engine->getBindingDataType(inputIndex) == nvinfer1::DataType::kFLOAT);
    const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(engine->getBindingDataType(outputIndex) == nvinfer1::DataType::kFLOAT);
    int mBatchSize = engine->getMaxBatchSize();
    

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    auto out_dims = engine->getBindingDimensions(outputIndex);
    auto output_size = 1;
    auto output1_size = 1;
    for(int j=0;j<out_dims.nbDims;j++) {
        output_size *= out_dims.d[j];
    }
    static const int Num_box = out_dims.d[1];
    static float* prob = new float[output_size];
    static float* prob1;
    static int _segWidth, _segHeight, _segChannels;
    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], output_size * sizeof(float)));
    if (engine_mode == "seg") {
        const int outputIndex1 = engine->getBindingIndex(OUTPUT_BLOB_NAME1);
        out_dims = engine->getBindingDimensions(outputIndex1);
        for (int j = 0; j < out_dims.nbDims; j++) {
            output1_size *= out_dims.d[j];
        }
        _segWidth = out_dims.d[3];
        _segHeight = out_dims.d[2];
        _segChannels = out_dims.d[1];

        prob1 = new float[output1_size];
        CHECK(cudaMalloc(&buffers[2], 3*80*80*44 * sizeof(float)));
        CHECK(cudaMalloc(&buffers[3], 3 * 40 * 40 * 44 * sizeof(float)));
        CHECK(cudaMalloc(&buffers[4], 3 * 20 * 20 * 44 * sizeof(float)));
        CHECK(cudaMalloc(&buffers[outputIndex1], output1_size * sizeof(float)));
    }
    float* blobinit = new float[INPUT_W * INPUT_H * 3];
    doInference(*context, stream, *engine, engine_mode, buffers, blobinit, prob, output_size, prob1, output1_size);
    delete[] blobinit;

    auto start = std::chrono::system_clock::now();
    BYTE aaaa[5000] = { "./test_x" };
    std::string aa_find_dir_list = (char*)aaaa;
    std::vector<std::string> aaa_img_list = read_images_in_folder(aa_find_dir_list);
    double inference_time = 0;
    double preprocess_time = 0;
    double nms_time = 0;
    for (int i = 0; i < aaa_img_list.size(); i++)
    {
        std::cout << aaa_img_list[i] << std::endl;
        std::string input_image_path = aaa_img_list[i];
        cv::Mat img = cv::imread(input_image_path);
        int img_w = img.cols;
        int img_h = img.rows;
        std::vector<int> padsize;
        auto pre_start = std::chrono::system_clock::now();
        cv::Mat pr_img = static_resize(img, padsize);

        float* blob;
        blob = blobFromImage(pr_img, engine_mode, 1);
        auto pre_end = std::chrono::system_clock::now();
        if (i != 0) {
            preprocess_time += std::chrono::duration_cast<std::chrono::microseconds>(pre_end - pre_start).count() / 1000.0;
        }
        float scale = (std::min)(INPUT_W / (img.cols * 1.0), INPUT_H / (img.rows * 1.0));

        // run inference
        auto doin_start = std::chrono::system_clock::now();

        doInference(*context, stream, *engine, engine_mode, buffers, blob, prob, output_size, prob1, output1_size);
        auto doin_end = std::chrono::system_clock::now();
        std::cout << "doInference: " << std::chrono::duration_cast<std::chrono::microseconds>(doin_end - pre_start).count() / 1000.0 << "ms" << std::endl;
        if (i != 0) {
            inference_time += std::chrono::duration_cast<std::chrono::microseconds>(doin_end - doin_start).count() / 1000.0;
        }
        //std::cout << std::chrono::duration_cast<std::chrono::microseconds>(doin_end - doin_start).count() << "us" << std::endl;
        
        if (engine_mode == "cls") {
            auto nms_start = std::chrono::system_clock::now();
            std::vector<float> vecprob(prob, prob + output_size), vecprob1(vecprob);;
            softmax(vecprob, vecprob1);
            std::cout << arg_max(vecprob) << std::endl;
            std::cout << arg_max(vecprob1) << std::endl;
            auto nms_end = std::chrono::system_clock::now();
            if (i != 0) {
                nms_time += std::chrono::duration_cast<std::chrono::microseconds>(nms_end - nms_start).count() / 1000.0;
            }
            //for (std::vector<float>::const_iterator it = vecprob1.begin(); it != vecprob1.end(); ++it) {
            //    std::cout << *it << " ";
            //}
            //std::cout << std::endl;
        }
        if (engine_mode == "obj") {
            std::vector<Object> objects;
            auto nms_start = std::chrono::system_clock::now();
            nms(objects, prob, BBOX_CONF_THRESH, NMS_THRESH);
            auto nms_end = std::chrono::system_clock::now();
            if (i != 0) {
                nms_time += std::chrono::duration_cast<std::chrono::microseconds>(nms_end - nms_start).count() / 1000.0;
            }
            //std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(nms_start - nms_end).count() << "ms" << std::endl;
            for (int i = 0; i < objects.size(); i++)
            {
                std::cout << get_rect(img, objects[i].bbox) << " " << objects[i].label << " " << objects[i].prob << std::endl;
            }
            for (size_t j = 0; j < objects.size(); j++) {
                cv::Rect r = get_rect(img, objects[j].bbox);
                cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
                cv::putText(img, std::to_string((int)objects[j].label), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
            }
            cv::imwrite("./result/" + input_image_path.substr(input_image_path.find_last_of('/ \\') + 1), img);
        }
        if (engine_mode == "seg") {
            std::vector<int> classIds;//结果id数组
            std::vector<float> confidences;//结果每个id对应置信度数组
            std::vector<cv::Rect> boxes;//每个id矩形框
            std::vector<std::vector<float>> picked_proposals;  //存储output0[:,:, 5 + _className.size():net_width]用以后续计算mask

            int newh = padsize[0], neww = padsize[1], padh = padsize[2], padw = padsize[3];
            //printf("newh:%d,neww:%d,padh:%d,padw:%d", newh, neww, padh, padw);
            float ratio_h = (float)img_h / newh;
            float ratio_w = (float)img_w / neww;

            // 处理box
            int net_width = NUM_CLASSES + 5 + _segChannels;
            auto nms_start = std::chrono::system_clock::now();
            float* pdata = prob;
            for (int j = 0; j < Num_box; ++j) {
                float box_score = pdata[4]; ;//获取每一行的box框中含有某个物体的概率
                if (box_score >= BBOX_CONF_THRESH) {
                    cv::Mat scores(1, NUM_CLASSES, CV_32FC1, pdata + 5);
                    cv::Point classIdPoint;
                    double max_class_socre;
                    minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint); //classIdPoint 最大值位置（x为列，y为行）
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
            std::vector<OutputSeg> output;
            //if(!classIds.empty()) {
            //    //执行非最大抑制以消除具有较低置信度的冗余重叠框（NMS）
            //    std::vector<int> nms_result;
            //    cv::dnn::NMSBoxes(boxes, confidences, BBOX_CONF_THRESH, NMS_THRESH, nms_result);
            //    std::vector<std::vector<float>> temp_mask_proposals;
            //    cv::Rect holeImgRect(0, 0, img_w, img_h);
            //    for (int i = 0; i < nms_result.size(); ++i) {
            //        int idx = nms_result[i];
            //        OutputSeg result;
            //        result.id = classIds[idx];
            //        result.confidence = confidences[idx];
            //        result.box = boxes[idx] & holeImgRect;
            //        output.push_back(result);
            //        temp_mask_proposals.push_back(picked_proposals[idx]);
            //    }
            //    // 处理mask
            //    cv::Mat maskProposals;
            //    for (int i = 0; i < temp_mask_proposals.size(); ++i)
            //        //std::cout<< Mat(temp_mask_proposals[i]).t().size();
            //        maskProposals.push_back(cv::Mat(temp_mask_proposals[i]).t());
            //    pdata = prob1;
            //    std::vector<float> mask(pdata, pdata + _segChannels * _segWidth * _segHeight);
            //    cv::Mat mask_protos = cv::Mat(mask);
            //    cv::Mat protos = mask_protos.reshape(0, { _segChannels,_segWidth * _segHeight });//将prob1的值 赋给mask_protos
            //    cv::Mat matmulRes = (maskProposals * protos).t();//n*32 32*25600 A*B是以数学运算中矩阵相乘的方式实现的，要求A的列数等于B的行数时
            //    cv::Mat masks = matmulRes.reshape(output.size(), { _segWidth,_segHeight });
            //    //std::cout << protos.size();
            //    std::vector<cv::Mat> maskChannels;
            //    split(masks, maskChannels);
            //    //std::cout << maskChannels.size();
            //    for (int i = 0; i < output.size(); ++i) {
            //        cv::Mat dest, masktmp;
            //        //sigmoid
            //        cv::exp(-maskChannels[i], dest);
            //        dest = 1.0 / (1.0 + dest);//160*160
            //        cv::Rect roi(int((float)padw / INPUT_W * _segWidth), int((float)padh / INPUT_H * _segHeight), int(_segWidth - padw / 2), int(_segHeight - padh / 2));
            //        //std::cout << roi;
            //        dest = dest(roi);
            //        resize(dest, masktmp, img.size(), cv::INTER_NEAREST);
            //        //crop----截取box中的mask作为该box对应的mask
            //        cv::Rect temp_rect = output[i].box;
            //        masktmp = masktmp(temp_rect) > MASK_THRESHOLD;
            //        //Point classIdPoint;
            //        //double max_class_socre;
            //        //minMaxLoc(mask, 0, &max_class_socre, 0, &classIdPoint);
            //        //max_class_socre = (float)max_class_socre;
            //        //printf("最大值:%.2f", max_class_socre);
            //        output[i].boxMask = masktmp;
            //    }
            //    auto nms_end = std::chrono::system_clock::now();
            //    if (i != 0) {
            //        nms_time += std::chrono::duration_cast<std::chrono::microseconds>(nms_end - nms_start).count() / 1000.0;
            //    }
            //}
            //DrawPred(img, output);
            //cv::imwrite("./result/"+ input_image_path.substr(input_image_path.find_last_of('/ \\') + 1), img);

        }


        // delete the pointer to the float
        delete blob;

    }
    std::cout<< std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count() / 1000.0<<" ms" << std::endl;
    std::cout << (aaa_img_list.size() - 1) <<preprocess_time << "  " << inference_time << "  " << nms_time << std::endl;
    inference_time /= (aaa_img_list.size() - 1);
    preprocess_time /= (aaa_img_list.size() - 1);
    nms_time /= (aaa_img_list.size() - 1);
   
    std::cout << "pre process: " << preprocess_time << "ms" << "inference_time: " << inference_time << "ms" << "nms_time: " << nms_time << "ms";
    /*std::vector<Object> objects;
    std::cout << scale << std::endl;
    decode_outputs(prob, objects, scale, img_w, img_h);
    for (int i = 0; i < objects.size(); i++)
    {
        std::cout << objects[i].rect << " "<< objects[i].label<< " "<<objects[i].prob<<std::endl;
    }
    draw_objects(img, objects, input_image_path);*/

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[0]));
    CHECK(cudaFree(buffers[1]));
    // destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
    //return 0;
    
}

extern "C"
{
    DLL_EXPORT void* AIInit(char* engineFilePath, char* engineMode);
    DLL_EXPORT void AIDetectInter(void* h, uchar* data, int width, int height, int stride,  
        int& outputResultLen, OutputSeg*& outputResult);
    DLL_EXPORT uchar* DetectArray(int& rows, int& cols);
    DLL_EXPORT void Freecuda(void* h);

}
extern "C"  DLL_EXPORT int CreateOutputSegs(OutputSeg * *segs);
int CreateOutputSegs(OutputSeg** segs) {
    // 创建一个动态数组，注意需要释放内存
    int n = 10; //数组长度为10，可以根据实际情况修改
    *segs = new OutputSeg[n];
    // 为了简单起见，我们用随机数填充数组
    srand(time(NULL));
    int a[4] = {409,415,54,90};
    for (int i = 0; i < n; i++) {
        (*segs)[i].id = i + 1;
        (*segs)[i].confidence = 0.4;
        for (int j = 0; j < 4; j++) {
            (*segs)[i].box[j] = a[j];
        }
    }
    return n; //返回数组长度
}

struct abc
{
    float x;
    int box[4];
};

extern "C" __declspec(dllexport)
void __stdcall Test(abc * pt, int size)
{
    for (int i = 0; i < size; ++i)
    {
        pt[i].x = (float)i;
        for(auto j=0;j<4;j++)
            pt[i].box[j] = 2 * i*j;
    }
}


typedef struct
{
    std::string engine_mode;
    float* data;
    float* prob;
    float* prob1;
    int output_size;
    int output1_size;
    IRuntime* runtime;
    ICudaEngine* engine;
    IExecutionContext* context[6];
    void** buffers;
    cudaStream_t stream;
    int _segWidth, _segHeight, _segChannels, Num_box;
    int inputIndex;
    int outputIndex;

}Yolov5TRTContext;


void* AIInit(char* engineFilePath, char* engineMode)
{
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
    for (int i = 0; i < 6; i++) {
        trt->context[i] = trt->engine->createExecutionContext();
    }
    //assert(trt->context != nullptr);
    delete[] trtModelStream;
    void** buffers;
    //if (engine_mode != "seg") {
    //    assert(trt->engine->getNbBindings() == 2);
    //}
    //else {
    //    assert(trt->engine->getNbBindings() == 3);
    //}
    const int bindingnum = trt->engine->getNbBindings();
    buffers = new void* [bindingnum];
    //assert(trt->engine->getNbBindings() == 2);
    //void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = trt->engine->getBindingIndex(INPUT_BLOB_NAME);

    assert(trt->engine->getBindingDataType(inputIndex) == nvinfer1::DataType::kFLOAT);
    const int outputIndex = trt->engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(trt->engine->getBindingDataType(outputIndex) == nvinfer1::DataType::kFLOAT);
    int mBatchSize = trt->engine->getMaxBatchSize();

    auto out_dims = trt->engine->getBindingDimensions(outputIndex);
    trt->output_size = 1;
    trt->engine_mode = engine_mode;
    cudaStream_t streams;
    cudaStreamCreate(&streams);
    trt->stream = streams;
    for (int j = 0; j < out_dims.nbDims; j++) {
        trt->output_size *= out_dims.d[j];
    }

    trt->prob = new float[trt->output_size];
    trt->Num_box = out_dims.d[1];
    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], trt->output_size * sizeof(float)));
    if (engine_mode == "seg") {
        const int outputIndex1 = trt->engine->getBindingIndex(OUTPUT_BLOB_NAME1);
        auto out_dims = trt->engine->getBindingDimensions(outputIndex1);
        trt->output1_size = 1;
        for (int j = 0; j < out_dims.nbDims; j++) {
            trt->output1_size *= out_dims.d[j];
        }
        trt->_segWidth = out_dims.d[3];
        trt->_segHeight = out_dims.d[2];
        trt->_segChannels = out_dims.d[1];
        
        trt->prob1 = new float[trt->output1_size];
        CHECK(cudaMalloc(&buffers[2], 3 * 80 * 80 * 44 * sizeof(float)));
        CHECK(cudaMalloc(&buffers[3], 3 * 40 * 40 * 44 * sizeof(float)));
        CHECK(cudaMalloc(&buffers[4], 3 * 20 * 20 * 44 * sizeof(float)));
        CHECK(cudaMalloc(&buffers[outputIndex1], trt->output1_size * sizeof(float)));
    }
    trt->buffers = buffers;
    // 处理第一次推理时间长
    for (int i = 0; i < 6; i++) {
        float* blob = new float[INPUT_W* INPUT_H * 3];
        doInference(*trt->context[i], trt->stream, *(trt->engine), trt->engine_mode, trt->buffers, blob, trt->prob, trt->output_size, trt->prob1, trt->output1_size);
        delete[] blob;
    }
    return (void*)trt;
}

uchar* outputResultPoint;
int outRow, outCol;
int bytesize;
BYTE* boxMask;       //矩形框内mask，节省内存空间和加快速度
void AIDetect(void* h, cv::Mat img, std::vector<OutputSeg> &output, Point point)//std::vector<OutputSeg> outputResult)
{
    Yolov5TRTContext* trt = (Yolov5TRTContext*)h;
    //const std::string input_image_path = "2021042904363803185G-C0500-11.jpg";
    //std::string input_ImagePath = (char*)inputImagePath;
    //std::string input_ImagePath = inputImagePath;
    //cv::Mat img = cv::imread(input_ImagePath);
    //cv::Mat img = cv::Mat(cv::Size(width, height), CV_8UC3, data, stride);
    int img_w = img.cols;
    int img_h = img.rows;
    std::vector<int> padsize;
    cv::Mat pr_img = static_resize(img, padsize);
    std::cout << "blob image" << std::endl;

    float* blob;
    blob = blobFromImage(pr_img, trt->engine_mode, 1);
    float scale = (std::min)(INPUT_W / (img.cols * 1.0), INPUT_H / (img.rows * 1.0));

    // run inference
    //void** buffers;
    //buffers = new void* [trt->engine->getNbBindings()];

    //CHECK(cudaMalloc(&buffers[trt->engine->getBindingIndex(INPUT_BLOB_NAME)], 3 * INPUT_H * INPUT_W * sizeof(float)));
    //CHECK(cudaMalloc(&buffers[trt->engine->getBindingIndex(OUTPUT_BLOB_NAME)], trt->output_size * sizeof(float)));
    //if (trt->engine_mode == "seg") {
    //    CHECK(cudaMalloc(&buffers[trt->engine->getBindingIndex(OUTPUT_BLOB_NAME1)], trt->output1_size * sizeof(float)));
    //}
    //cudaStream_t stream;
    //CHECK(cudaStreamCreate(&stream));
    
    //IExecutionContext* context = trt->engine->createExecutionContext();
    //assert(context != nullptr);
    auto start = std::chrono::system_clock::now();
    doInference(*trt->context[0], trt->stream, *(trt->engine), trt->engine_mode, trt->buffers, blob, trt->prob, trt->output_size, trt->prob1, trt->output1_size);

    auto end = std::chrono::system_clock::now();
    std::cout <<"doInference: "<< std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0 << "ms" << std::endl;

    if (trt->engine_mode == "cls") {
        std::vector<float> vecprob(trt->prob, trt->prob + trt->output_size), vecprob1(vecprob);;
        softmax(vecprob, vecprob1);
        std::cout << arg_max(vecprob1) << std::endl;
        //for (std::vector<float>::const_iterator it = vecprob1.begin(); it != vecprob1.end(); ++it) {
        //    std::cout << *it << " ";
        //}
        //std::cout << std::endl;
        //return 0;
    }
    if (trt->engine_mode == "obj") {
        std::vector<Object> objects;
        nms(objects, trt->prob, BBOX_CONF_THRESH, NMS_THRESH);
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        for (int i = 0; i < objects.size(); i++)
        {
            std::cout << get_rect(img, objects[i].bbox) << " " << objects[i].label << " " << objects[i].prob << std::endl;
        }
        for (size_t j = 0; j < objects.size(); j++) {
            cv::Rect r = get_rect(img, objects[j].bbox);
            cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
            cv::putText(img, std::to_string((int)objects[j].label), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
        }
        //cv::imwrite("./result/" + std::string(input_ImagePath).substr(std::string(input_ImagePath).find_last_of('/ \\') + 1), img);
        //return 0;
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
            for (int i = 0; i < nms_result.size(); ++i) {
                int idx = nms_result[i];
                OutputSeg result;
                result.id = classIds[idx];
                result.confidence = confidences[idx];
                cv::Rect rect = boxes[idx] & holeImgRect;
                result.box[0] = rect.x;
                result.box[1] = rect.y;
                result.box[2] = rect.width;
                result.box[3] = rect.height;
                output.push_back(result);

                temp_mask_proposals.push_back(picked_proposals[idx]);

            }

            // 处理mask
            cv::Mat maskProposals;
            for (int i = 0; i < temp_mask_proposals.size(); ++i)
                //std::cout<< Mat(temp_mask_proposals[i]).t().size();
                maskProposals.push_back(cv::Mat(temp_mask_proposals[i]).t());

            pdata = trt->prob1;
            std::vector<float> mask(pdata, pdata + trt->_segChannels * trt->_segWidth * trt->_segHeight);

            cv::Mat mask_protos = cv::Mat(mask);
            cv::Mat protos = mask_protos.reshape(0, { trt->_segChannels,trt->_segWidth * trt->_segHeight });//将prob1的值 赋给mask_protos


            cv::Mat matmulRes = (maskProposals * protos).t();//n*32 32*25600 A*B是以数学运算中矩阵相乘的方式实现的，要求A的列数等于B的行数时
            cv::Mat masks = matmulRes.reshape(output.size(), { trt->_segWidth, trt->_segHeight });
            //std::cout << protos.size();
            std::vector<cv::Mat> maskChannels;
            split(masks, maskChannels);
            //std::cout << maskChannels.size();
            for (int i = 0; i < output.size(); ++i) {
                cv::Mat dest, masktmp;
                //sigmoid
                cv::exp(-maskChannels[i], dest);
                dest = 1.0 / (1.0 + dest);//160*160

                cv::Rect roi(int((float)padw / INPUT_W * trt->_segWidth), int((float)padh / INPUT_H * trt->_segHeight), int(trt->_segWidth - padw / 2), int(trt->_segHeight - padh / 2));
                //std::cout << roi;
                dest = dest(roi);
                resize(dest, masktmp, img.size(), cv::INTER_NEAREST);


                //crop----截取box中的mask作为该box对应的mask
                //cv::Rect temp_rect = output[i].box;
                masktmp = masktmp(cv::Rect(output[i].box[0], output[i].box[1], output[i].box[2], output[i].box[3])) > MASK_THRESHOLD;

                //Point classIdPoint;
                //double max_class_socre;
                //minMaxLoc(mask, 0, &max_class_socre, 0, &classIdPoint);
                //max_class_socre = (float)max_class_socre;
                //printf("最大值:%.2f", max_class_socre);

                //output[i].boxMask = masktmp;
                //std::vector<int> mask = masktmp.reshape(1, 1);
                //outputmask.insert(outputmask.end(), mask.begin(), mask.end());
                int nBytes = output[i].box[3] * output[i].box[2];//图像总的字节
                BYTE* pImg = new BYTE[nBytes];//new的单位为字节
                memcpy(pImg, masktmp.data, nBytes);
                output[i].bytesize = nBytes;
                output[i].boxMask = pImg;
                output[i].box[0] += point.x;
                output[i].box[1] += point.y;
                //delete[] data;
                //data = nullptr;
            }
        }

        //for c#
        //outputResultLen = output.size();
        //int arraylen = 1;
        //for (int i = 0; i < outputResultLen; ++i) {
        //    outputResult[i*4] = (double)output[i].id;
        //    outputResult[i * 4 + 1] = (double)output[i].confidence;
        //    outputResult[i * 4 + 2] = (double)output[i].box.x;
        //    outputResult[i * 4 + 3] = (double)output[i].box.y;
        //    outputResult[i * 4 + 4] = (double)output[i].box.width;
        //    outputResult[i * 4 + 5] = (double)output[i].box.height;
        //}
        //outputResultPoint = nullptr;
        //if (!output.empty()) {
        //    outputResultPoint = output[0].boxMask.data;
        //    outRow = output[0].boxMask.rows;
        //    outCol = output[0].boxMask.cols;
        //    //std::cout << output[0].boxMask.channels();
        //
        //memcpy(outputResult, &output[0], output.size() * sizeof(OutputSeg));
        //std::cout << output.size() << std::endl;
        //for (int j = 0; j < outputResultLen * 6; j++)
        //    std::cout << outputResult[j] << std::endl;
        //for (int j = 0; j < outCol; j++)
        //    std::cout << (int)output[0].boxMask.at<uchar>(0, j) << " ";
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        //DrawPred(img, output);
        //
        //cv::imwrite("./result/1.1.bmp", img);

    }
    delete blob;
}

void AIDetectInter(void* h, uchar* data, int width, int height, int stride, int &outputResultLen, OutputSeg*& outputResult)
//void AIDetectInter(void* h, Mat img, vector<OutputSeg>& outputtotal, int& outputResultLen)
{
    cv::Mat img = cv::Mat(cv::Size(width, height), CV_8UC3, data, stride);
    int single = 640;
    bool divideflagw = width % single;
    int widthnum = divideflagw ? (width / single) + 1 : (width / single);
    bool divideflagh = height % single;
    int heightnum = divideflagh ? (height / single) + 1 : (height / single);
    vector<vector<OutputSeg>> outputtotaltmp;
    vector<vector<int>> outputtotalmasktmp;
    for (size_t i = 0; i < heightnum; i++) {
        int singleheighttmp = (i == heightnum - 1) && divideflagh ? (height - i * single) : single;
        for (size_t j = 0; j < widthnum; j++) {
            int singlewidthtmp = (j == widthnum - 1) && divideflagw ? (width - j * single) : single;
            Rect rect(j* single, i* single, singlewidthtmp, singleheighttmp);
            Mat singleimg = img(rect);
            Point point(j * single, i * single);
            vector<OutputSeg> outputtmp;
            AIDetect(h, singleimg, outputtmp, point);
            outputtotaltmp.push_back(outputtmp);
        }
    }
    vector<OutputSeg> outputtotal;
    for (size_t i = 0; i < outputtotaltmp.size(); i++) {
        outputtotal.insert(outputtotal.end(), outputtotaltmp[i].begin(), outputtotaltmp[i].end());
    }
    outputResultLen = outputtotal.size();
    outputResult = new OutputSeg[outputResultLen];
    memcpy(outputResult, &outputtotal[0], outputResultLen * sizeof(OutputSeg));
}

uchar* DetectArray(int& rows, int& cols) {
    rows = outRow;
    cols = outCol;
    return outputResultPoint;
}

void Freecuda(void* h) {
    Yolov5TRTContext* trt = (Yolov5TRTContext*)h;
    cudaStreamDestroy(trt->stream);
    cudaFree(trt->buffers[0]);
    cudaFree(trt->buffers[1]);
    if (trt->engine_mode == "seg") {
        cudaFree(trt->buffers[2]);
    }
    trt->context[0]->destroy();
    trt->context[1]->destroy();
    trt->engine->destroy();
    trt->runtime->destroy();
}


void handle(Yolov5TRTContext* trt, std::vector<cv::String> imgLists, int threadid) {

    int outputResultLen = 0;
    auto start = std::chrono::system_clock::now();
    for (auto img : imgLists) {
        std::cout << std::string(img) << std::endl;
        cv::Mat imgdata = cv::imread(img);
        //char* ch1 = (char*)img.c_str();
        std::vector<OutputSeg> output;
        //AIDetect(trt, imgdata, output);
    }
    std::cout << "thread id " <<threadid<<" : " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start).count() << "ms" << std::endl;  
}

//瑕疵检测速度测试
#include "gpucal.h"
using namespace std;
using namespace cv;
int* imgArray;
cv::Mat img;
int* result;
void ReadImgarray(std::string imgfile)
{
    cv::Mat imgreal = cv::imread(imgfile, 0);
    cv::Rect roi(900, 0, 14100, 2000);
    img = imgreal(roi);
    cout << imgreal.rows << "  " << imgreal.cols << endl;
    //img = imgreal;
    imgArray = new int[img.rows * img.cols];
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            imgArray[i * img.cols + j] = int(img.at<uchar>(i, j));
        }
    }
}
std::queue<Mat> imgmat_queue;
//bool finished = false;
//std::atomic_bool finished = ATOMIC_VAR_INIT(false);
void gpucal(cudaStream_t streams[]) {
    ReadImgarray("D://C#//xiaci_pinjie_1209.bmp");
    GPUCAL gpucal(img.rows, img.cols);
    result = new int[RESULTSIZE];
    auto start = std::chrono::system_clock::now();

    for (auto start1 = 0; start1 < 1000; start1++) {
        gpucal.MatCal(streams, imgArray, result, 225, 199);
        //cout << result[0] << endl;
        Mat imgcut;
        //string outpath = "D:\\C#\\DefectCut\\";
        //system("del D:\\C#\\DefectCut\\*.png  /a /s /f /q");
        for (int i = 1; i < result[0] + 1; i = i + 2) {
            //cout << result[i] << "  " << result[i + 1] << endl;
            Rect rect(result[i + 1], result[i], 128, 128);
            imgcut = img(rect);
            imgmat_queue.push(imgcut);
            //imwrite(outpath + to_string(result[i]) + "_" + to_string(result[i + 1]) + ".png", imgcut);
        }
        //std::cout << "插入 线程：" << imgmat_queue.size() << std::endl;
    }
    finished = true;
    delete[] result;
    std::cout << "队列插入线程：" << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start).count() << "ms" << std::endl;
}
void handle1(Yolov5TRTContext* trt, int threadid) {
    auto prestart = std::chrono::system_clock::now();
    while (!imgmat_queue.empty() or !finished) {
        if (!imgmat_queue.empty()) {
            //std::cout << "handle 线程：" << imgmat_queue.size() << std::endl;
            auto start = std::chrono::system_clock::now();
            float* blob;
            blob = blobFromImage(imgmat_queue.front(), trt->engine_mode, 1);
            
            doInference(*trt->context[threadid], trt->stream, *(trt->engine), trt->engine_mode, trt->buffers, blob, trt->prob, trt->output_size, trt->prob1, trt->output1_size);
            auto end = std::chrono::system_clock::now();
            
            if (trt->engine_mode == "cls") {
                std::vector<float> vecprob(trt->prob, trt->prob + trt->output_size), vecprob1(vecprob);;
                softmax(vecprob, vecprob1);
                //std::cout << arg_max(vecprob1) << std::endl;
                //for (std::vector<float>::const_iterator it = vecprob1.begin(); it != vecprob1.end(); ++it) {
                //    std::cout << *it << " ";
                //}
                //std::cout << std::endl;
            }
            imgmat_queue.pop();
            //std::cout << "doInference: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0 << "ms" << std::endl;
            //std::cout << blob[0] << " " << blob[1] << std::endl;
        }
    }
    //blob_queue.quit();
    std::cout << "推理线程：" << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - prestart).count() / 1000.0 << "ms" << std::endl;
}
void main(int argc, char** argv) {

    //std::string s = "D:\\c++\\tensorrt\\tensorrt\\model\\varta640_seg.engine";
    //char* ch = (char*)s.c_str();
    //std::string s2 = "seg";
    //char* ch2 = (char*)s2.c_str();
    //std::string s1 = "D:\\c++\\tensorrt\\tensorrt\\test_x\\1-1.bmp";
    //char* ch1 = (char*)s1.c_str();
    //int outputResultLen=0;
    char engine_filepath[1000] = { 0 };
    char enginemode[100] = { 0 };
    char input_w[100] = { 0 };
    char input_h[100] = { 0 };
    char num_class[100] = { 0 };
    GetConfigValue("engine_file_path", engine_filepath);
    GetConfigValue("engine_mode", enginemode);
    GetConfigValue("INPUT_W", input_w);
    GetConfigValue("INPUT_H", input_h);
    GetConfigValue("NUM_CLASSES", num_class);
    engine_filepath[strlen(engine_filepath) - 1] = 0;
    enginemode[strlen(enginemode) - 1] = 0;
    INPUT_W = atoi(input_w);
    INPUT_H = atoi(input_h);
    NUM_CLASSES = atoi(num_class);
    const int nStreams = 5;

    //Stream的初始化
    cudaStream_t streams[nStreams];
    for (int i = 0; i < nStreams; i++) {
        cudaStreamCreate(&streams[i]);
    }
    //istringstream in(engine_filepath);
    //vector<string> engine_files;
    //string t;
    //while (in >> t) {
    //    engine_files.push_back(t);
    //}
    //vector<Yolov5TRTContext*> trts;
    //for (auto i : engine_files) {
    //    int index = 0;
    //    trts.push_back((Yolov5TRTContext*)AIInit( "./model/"+i, enginemode));
    //    index++;
    //}
    Yolov5TRTContext* trt = (Yolov5TRTContext*) AIInit( engine_filepath, enginemode);
    //Yolov5TRTContext* trt1 = (Yolov5TRTContext*)AIInit(streams[1], engine_filepath, enginemode);
    std::vector<std::thread> mythreads;
    std::string imgDir = "D:\\c++\\tensorrt\\tensorrt\\test_x\\";
    std::vector<cv::String> imgLists;
    cv::glob(imgDir, imgLists, false);
    int nSubVecSize =400;// 每个小vector的容量
    auto start = std::chrono::system_clock::now();
    for (auto img : imgLists) {
        std::cout << std::string(img) << std::endl;
        cv::Mat imgdata = cv::imread(img);
        //char* ch1 = (char*)img.c_str();
        std::vector<OutputSeg> output;
        int outputlen = -1;
        vector<OutputSeg> outputResult;
        //AIDetectInter(trt, imgdata, outputResult, outputlen);
    }
    //for (size_t i = 0; i < imgLists.size(); i += nSubVecSize)
    //{
    //    //Yolov5TRTContext* trt = (Yolov5TRTContext*)AIInit(ch, ch2);
    //    std::vector<cv::String> vecSmall;
    //    auto last = min(imgLists.size(), i + nSubVecSize);
    //    vecSmall.insert(vecSmall.begin(), imgLists.begin() + i, imgLists.begin() + last);
    //    //for (auto j : trts) {
    //    //    mythreads.push_back(std::thread(handle, j, vecSmall, mythreads.size()));
    //    //}
    //    mythreads.push_back(std::thread(handle, trt, vecSmall, mythreads.size()));
    //    std::cout << "thread size:" << mythreads.size() << std::endl;
    //}
    //for (auto it = mythreads.begin(); it != mythreads.end(); ++it)
    //{
    //    it->join();
    //}
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start).count() << "ms" << std::endl;
    //AIDetect(trt, ch1, outputResultLen);


    //std::queue<float*> blob_queue1;
    //auto start1 = std::chrono::system_clock::now();
    //auto pre_start = std::chrono::system_clock::now();
    //for (int i = 0; i < 100000000; i++) {
    //    float* blob = new float[2];
    //    blob[0] = i + 10;
    //    blob[1] = i + 11;
    //    blob_queue1.push(blob);
    //    //printf( "插入第%d组数据",i);
    //}
    //std::cout << "单线程预处理线程：" << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - pre_start).count() << "ms" << std::endl;
    //while (!blob_queue1.empty()) {
    //    float* blob = new float[2];
    //    blob = blob_queue1.front();
    //    blob_queue1.pop();
    //    //std::cout << blob[0] << " " << blob[1] << std::endl;
    //}
    //std::cout << "单线程主程序：" << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start1).count() << "ms" << std::endl;
    
    
    
    ////gpucal();
    ////handle1(trt, 0);
    //std::vector<std::thread> mythreads;
    ////mythreads.push_back(std::thread(preprocess));
    ////mythreads.push_back(std::thread(inference));
    //mythreads.push_back(std::thread(handle1, trt, 0));
    //mythreads.push_back(std::thread(gpucal, streams));

    //auto start = std::chrono::system_clock::now();
    //for (auto it = mythreads.begin(); it != mythreads.end(); ++it) {
    //    it->join();
    //}
    //std::cout <<"多线程主程序：" << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start).count() << "ms" << std::endl;
    //return;
    
}
