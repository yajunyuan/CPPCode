#include <opencv2/opencv.hpp>
#include <iostream>
#include "paddleocr.h"
#include "Common.h"
#include <thread>  
#include <mutex>
#include <chrono>
using namespace cv;
using namespace std;
class VisionAlgorithm {
private:
    double thresholdvalue;
    double iouvalue;
    int numLevels;
    int pixelThresholdValue;
    int typeValue;
    Size patternSize;
    double distanceValue;
    Mat srcImage;
    Mat modelImage;
    Mat mask;
    Mat cameraMatirx, distCoeffs;
    double transforArray[9];
    vector<Point2f> cvPoints;
    PaddleOCR::PPOCR* ocrObj;
public:
    MatchResult* matchresult;
    BlobResult* blobresult;
    int resultlen;
    VisionAlgorithm();
    void GetSrcImageData(uchar* srcdata, int srcwidth, int srcheight, int srcstride);
    Mat OutputSrcImage();
    void GetModelImageData(uchar* modeldata, int modelwidth, int modelheight, int modelstride1);
    void GetModelMaskData(TransPoint<int>* pt, int ptsize);
    void GetHist(int* array);
    void GetMatchParam(double thresholdvalue, double iouvalue, int numLevels);
    void Match();
    void GetBlobParam(int thresh, int type);
    void Blob();
    void GetCalibParam(int type, int rows, int cols, double distance);
    //void GetCalibPoints(TransPoint* pt, int ptsize, TransPoint* pt1, int pt1size);
    void Calib(TransPoint<double>* pt, int ptsize, TransPoint<double>* pt1, int pt1size);
    void CalibUndistort(TransPoint<double>* pt, int ptsize);
    void TransferAxes(TransPoint<int> oldOri, TransPoint<int> newOri, double angle);
    void TransferAxesPoints(TransPoint<int>* pt, int ptsize);
    void OcrInit();
    void Ocr();
    void GetFindShapeData(TransPoint<int>* pt, int ptsize, int type);
    GeometryData FindShape(int findMode, int findDir);
    GeometryData FitTool(TransPoint<double>* pt, int ptsize, int type);
    void GetCalcParam(int type);
    void CalcDistance(GeometryData obj, GeometryData obj2, double& distance);
    void CalcAngle(GeometryData obj, GeometryData obj2, double& angle);
    void BarcodeRec();
    void Release();
};

//灰度形态调整,调用方式
//int operation = 0; // 选择形态学操作
//std::vector<int> shapes = { 0, 1, 6 }; // 选择结构元素形状
//std::vector<int> sizes = { 3, 5, 3 };  // 结构元素大小
//
//// 自定义结构元素（示例）
//cv::Mat customElement = (cv::Mat_<uchar>(3, 3) << 0, 1, 0, 1, 1, 1, 0, 1, 0);
//
//// 创建MorphologyProcessor对象并处理图像
//MorphologyProcessor processor(operation, shapes, sizes, customElement);
//cv::Mat result = processor.process(src);
class MorphologyProcessor {
public:
    MorphologyProcessor(int operation, const std::vector<int>& shapes, const std::vector<int>& sizes, const cv::Mat& customElement = cv::Mat())
        : operation_(operation), shapes_(shapes), sizes_(sizes), customElement_(customElement) {}

    cv::Mat process(const cv::Mat& src) {
        cv::Mat gray;
        if (src.channels() == 3) {
            cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
        }
        else {
            gray = src;
        }
        // 生成结构元素列表
        std::vector<cv::Mat> elements = generateStructuringElements(shapes_, sizes_, customElement_);
        // 组合结构元素
        cv::Mat combinedElement = combineStructuringElements(elements);
        // 应用形态学操作并获取结果图像
        return applyMorphologyOperation(gray, operation_, combinedElement);
    }

private:
    int operation_;
    std::vector<int> shapes_;
    std::vector<int> sizes_;
    cv::Mat customElement_;

    cv::Mat createStructuringElement(int shape, int size, const cv::Mat& customElement = cv::Mat()) {
        cv::Mat element;
        switch (shape) {
        case 0: // nxn方形(奇数)
            element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(size, size));
            break;
        case 1: // nxn菱形
            element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(size, size));
            break;
        case 2: // 1xn水平
            element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(size, 1));
            break;
        case 3: { // 1xn平面45°
            element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(size, size));
            cv::Mat rotationMatrix = cv::getRotationMatrix2D(cv::Point(size / 2, size / 2), 45, 1);
            cv::warpAffine(element, element, rotationMatrix, element.size());
            break;
        }
        case 4: // 1xn平面垂直
            element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, size));
            break;
        case 5: { // 1xn平面135°
            element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(size, size));
            cv::Mat rotationMatrix = cv::getRotationMatrix2D(cv::Point(size / 2, size / 2), 135, 1);
            cv::warpAffine(element, element, rotationMatrix, element.size());
            break;
        }
        case 6: // 自定义结构元素
            if (!customElement.empty()) {
                element = customElement;
            }
            else {
                std::cerr << "Custom element is empty!" << std::endl;
            }
            break;
        default:
            element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(size, size));
            break;
        }
        return element;
    }

    std::vector<cv::Mat> generateStructuringElements(const std::vector<int>& shapes, const std::vector<int>& sizes, const cv::Mat& customElement = cv::Mat()) {
        std::vector<cv::Mat> elements;
        for (size_t i = 0; i < shapes.size(); ++i) {
            elements.push_back(createStructuringElement(shapes[i], sizes[i], customElement));
        }
        return elements;
    }

    cv::Mat combineStructuringElements(const std::vector<cv::Mat>& elements) {
        // 计算复合大小
        int maxWidth = 0, maxHeight = 0;
        for (const auto& element : elements) {
            maxWidth = std::max(maxWidth, element.cols);
            maxHeight = std::max(maxHeight, element.rows);
        }

        // 创建复合结构元素
        cv::Mat combined = cv::Mat::zeros(maxHeight, maxWidth, CV_8U);
        for (const auto& element : elements) {
            int xOffset = (maxWidth - element.cols) / 2;
            int yOffset = (maxHeight - element.rows) / 2;
            cv::Mat roi = combined(cv::Rect(xOffset, yOffset, element.cols, element.rows));
            cv::bitwise_or(roi, element, roi);
        }

        return combined;
    }

    cv::Mat applyMorphologyOperation(const cv::Mat& src, int operation, const cv::Mat& element) {
        cv::Mat result;
        switch (operation) {
        case 0: // 腐蚀
            cv::erode(src, result, element);
            break;
        case 1: // 膨胀
            cv::dilate(src, result, element);
            break;
        case 2: // 开运算
            cv::morphologyEx(src, result, cv::MORPH_OPEN, element);
            break;
        case 3: // 闭运算
            cv::morphologyEx(src, result, cv::MORPH_CLOSE, element);
            break;
        case 4: // 顶帽
            cv::morphologyEx(src, result, cv::MORPH_TOPHAT, element);
            break;
        case 5: // 黑帽
            cv::morphologyEx(src, result, cv::MORPH_BLACKHAT, element);
            break;
        default:
            std::cerr << "Invalid operation selected!" << std::endl;
            return src;
        }
        return result;
    }
};

//丢失像素，实例化： PixelReplacer replacer(PixelReplacer::NEIGHBOR_INTERPOLATION, PixelReplacer::BOTH);
class PixelReplacer {
public:
    PixelReplacer(ReplacementMethod method, Direction direction, GlobalMethod globalMethod = FIXED, uchar fixedValue = 0)
        : method_(method), direction_(direction), globalMethod_(globalMethod), fixedValue_(fixedValue) {}

    cv::Mat process(const cv::Mat& src) {
        // 判断图像是否为灰度图像，如果不是则转换为灰度图像
        cv::Mat gray;
        if (src.channels() == 3) {
            cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
        }
        else {
            gray = src;
        }

        return replaceMissingPixels(gray);
    }

private:
    ReplacementMethod method_;
    Direction direction_;
    GlobalMethod globalMethod_;
    uchar fixedValue_;

    cv::Mat replaceMissingPixels(const cv::Mat& image) {
        cv::Mat result = image.clone();
        cv::Mat temp;
        double minVal, maxVal;
        cv::Scalar meanVal;

        switch (method_) {
        case NEIGHBOR_MIN:
            cv::erode(image, temp, cv::Mat());
            break;
        case NEIGHBOR_MAX:
            cv::dilate(image, temp, cv::Mat());
            break;
        case NEIGHBOR_INTERPOLATION:
            cv::blur(image, temp, cv::Size(3, 3));
            break;
        case GLOBAL:
            switch (globalMethod_) {
            case FIXED:
                break;
            case IMAGE_MIN:
                cv::minMaxLoc(image, &minVal, nullptr);
                fixedValue_ = static_cast<uchar>(minVal);
                break;
            case IMAGE_MAX:
                cv::minMaxLoc(image, nullptr, &maxVal);
                fixedValue_ = static_cast<uchar>(maxVal);
                break;
            case IMAGE_AVERAGE:
                meanVal = cv::mean(image);
                fixedValue_ = static_cast<uchar>(meanVal[0]);
                break;
            }
            for (int i = 0; i < result.rows; ++i) {
                for (int j = 0; j < result.cols; ++j) {
                    processPixel(result, i, j, fixedValue_);
                }
            }
            return result;
        }

        for (int i = 0; i < result.rows; ++i) {
            for (int j = 0; j < result.cols; ++j) {
                if (result.at<uchar>(i, j) == 0) {
                    uchar value = getNeighborValue(temp, i, j, direction_);
                    processPixel(result, i, j, value);
                }
            }
        }

        return result;
    }

    static void processPixel(cv::Mat& result, int i, int j, uchar value) {
        if (result.at<uchar>(i, j) == 0) {
            result.at<uchar>(i, j) = value;
        }
    }

    static uchar getNeighborValue(const cv::Mat& temp, int i, int j, Direction direction) {
        uchar value = 0;
        int count = 0;
        if (direction == HORIZONTAL || direction == BOTH) {
            if (j > 0) { value += temp.at<uchar>(i, j - 1); count++; }
            if (j < temp.cols - 1) { value += temp.at<uchar>(i, j + 1); count++; }
        }
        if (direction == VERTICAL || direction == BOTH) {
            if (i > 0) { value += temp.at<uchar>(i - 1, j); count++; }
            if (i < temp.rows - 1) { value += temp.at<uchar>(i + 1, j); count++; }
        }
        return count > 0 ? value / count : 0;
    }
};
//乘以常数，实例化：ImageMultiplier multiplier(1.5, 1.0, 1.0, 1.0, ImageMultiplier::CLAMP);
class ImageMultiplier {
public:

    ImageMultiplier(double grayConstant, double plane2 = 1.0, double plane1 = 1.0, double plane0 = 1.0, OverflowMode mode = CLAMP)
        : grayConstant_(grayConstant), plane2_(plane2), plane1_(plane1), plane0_(plane0), mode_(mode) {}

    cv::Mat process(const cv::Mat& src) {
        return multiplyImageByConstant(src);
    }

private:
    double grayConstant_;
    double plane2_;
    double plane1_;
    double plane0_;
    OverflowMode mode_;

    cv::Mat multiplyImageByConstant(const cv::Mat& image) {
        cv::Mat result = image.clone();
        if (image.channels() == 1) {
            // 处理灰度图像
            for (int y = 0; y < result.rows; y++) {
                for (int x = 0; x < result.cols; x++) {
                    double newValue = image.at<uchar>(y, x) * grayConstant_;
                    if (mode_ == CLAMP) {
                        result.at<uchar>(y, x) = cv::saturate_cast<uchar>(newValue);
                    }
                    else if (mode_ == WRAP) {
                        result.at<uchar>(y, x) = static_cast<uchar>(newValue);
                    }
                }
            }
        }
        else if (image.channels() == 3) {
            // 处理彩色图像
            for (int y = 0; y < result.rows; y++) {
                for (int x = 0; x < result.cols; x++) {
                    for (int c = 0; c < 3; c++) {
                        double constant = (c == 0) ? plane2_ : (c == 1) ? plane1_ : plane0_;
                        double newValue = image.at<cv::Vec3b>(y, x)[c] * constant;
                        if (mode_ == CLAMP) {
                            result.at<cv::Vec3b>(y, x)[c] = cv::saturate_cast<uchar>(newValue);
                        }
                        else if (mode_ == WRAP) {
                            result.at<cv::Vec3b>(y, x)[c] = static_cast<uchar>(newValue);
                        }
                    }
                }
            }
        }
        return result;
    }
};

//像素映射
//创建表 std::unordered_map<int, int> grayMapping = {{0, 255}, {1, 254}, {2, 253}, /* ... */ {254, 1}, {255, 0}}....
//实例化：PixelMapper mapper(grayMapping, plane0Mapping, plane1Mapping, plane2Mapping);
class PixelMapper {
public:


    PixelMapper(const std::unordered_map<int, int>& grayMapping,
        const std::unordered_map<int, int>& plane0Mapping,
        const std::unordered_map<int, int>& plane1Mapping,
        const std::unordered_map<int, int>& plane2Mapping) {
        grayLUT_ = createLUT(grayMapping);
        plane0LUT_ = createLUT(plane0Mapping);
        plane1LUT_ = createLUT(plane1Mapping);
        plane2LUT_ = createLUT(plane2Mapping);
    }

    static std::vector<uchar> createLUT(const std::unordered_map<int, int>& mapping) {
        std::vector<uchar> lut(256);
        for (int i = 0; i < 256; i++) {
            lut[i] = mapping.at(i);
        }
        return lut;
    }

    cv::Mat mapPixels(const cv::Mat& image) const {
        cv::Mat result = image.clone();
        if (image.channels() == 1) {
            // 处理灰度图像
            for (int y = 0; y < result.rows; y++) {
                for (int x = 0; x < result.cols; x++) {
                    result.at<uchar>(y, x) = grayLUT_[image.at<uchar>(y, x)];
                }
            }
        }
        else if (image.channels() == 3) {
            // 处理彩色图像
            for (int y = 0; y < result.rows; y++) {
                for (int x = 0; x < result.cols; x++) {
                    result.at<cv::Vec3b>(y, x)[0] = plane0LUT_[image.at<cv::Vec3b>(y, x)[0]];
                    result.at<cv::Vec3b>(y, x)[1] = plane1LUT_[image.at<cv::Vec3b>(y, x)[1]];
                    result.at<cv::Vec3b>(y, x)[2] = plane2LUT_[image.at<cv::Vec3b>(y, x)[2]];
                }
            }
        }
        return result;
    }

private:
    std::vector<uchar> grayLUT_;
    std::vector<uchar> plane0LUT_;
    std::vector<uchar> plane1LUT_;
    std::vector<uchar> plane2LUT_;
};

//采样卷积
//定义卷积核，卷积核x，y需设置奇数
//std::vector<float> kernelWeights = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};int kernelSizeX = 5;int kernelSizeY = 3;
//实例化：cv::Mat result = SampleConvolution::sampleAndConvolve(image, kernelWeights, kernelSizeX, kernelSizeY, sigmaX, sigmaY, scaleX, scaleY);
class SampleConvolution {
public:
    static cv::Mat sampleAndConvolve(const cv::Mat& image, std::vector<float> kernelWeights, int kernelSizeX, int kernelSizeY, double sigmaX, double sigmaY, double scaleX, double scaleY) {
        // 归一化卷积核权重
        for (auto& weight : kernelWeights) {
            weight /= (kernelSizeX * kernelSizeY);
        }

        // 创建卷积核
        cv::Mat kernel = cv::Mat(kernelSizeY, kernelSizeX, CV_32F, kernelWeights.data());

        // 先进行卷积操作
        cv::Mat convolved;
        cv::filter2D(image, convolved, -1, kernel);

        // 然后进行降采样操作
        cv::Mat sampled;
        cv::resize(convolved, sampled, cv::Size(), scaleX, scaleY, cv::INTER_LINEAR);

        return sampled;
    }

    static cv::Mat downsample(const cv::Mat& image, double scaleX, double scaleY, bool useAverage) {
        cv::Mat result;
        // 选择插值方法
        int interpolation = useAverage ? cv::INTER_AREA : cv::INTER_LINEAR;

        // 调整图像尺寸
        cv::resize(image, result, cv::Size(), scaleX, scaleY, interpolation);

        return result;
    }
};

cv::Mat addConstantToImage(const cv::Mat& image, int constant);
cv::Mat applyConvolution(const cv::Mat& image, cv::Mat& kernel);
cv::Mat equalizeImage(const cv::Mat& image);
cv::Mat resizeImage(const cv::Mat& image, double scaleX, double scaleY);
cv::Mat flipImage(const cv::Mat& image, int flipway);
cv::Mat rotateImage(const cv::Mat& image, double angle);
cv::Mat gaussianSampling(const cv::Mat& inputImage,double scaleX, double scaleY,int kernelSizeX, int kernelSizeY,double sigmaX,
    double sigmaY);
cv::Mat applyFilter(const cv::Mat& src, FilterType filterType, int kernelWidth, int kernelHeight);
cv::Mat medianFilter(const cv::Mat& inputImage, int kernelSizeX, int kernelSizeY);
cv::Mat quantizeImage(const cv::Mat& image, int levels);