// VisionAlgorithm.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//#include <opencv2/opencv.hpp>
//#include <iostream>
//#include "paddleocr.h"
#include<AlgorithmProcess.h>

extern "C"
{
    __declspec(dllexport) VisionAlgorithm* VisionInit();
    __declspec(dllexport) void VisionObjRelease(VisionAlgorithm* visionAlgorithmobj);
    __declspec(dllexport) void VisionSrcData(VisionAlgorithm* visionAlgorithmobj, uchar* srcdata, int srcwidth, int srcheight, int srcstride);
    __declspec(dllexport) void VisionMatchParam(VisionAlgorithm* visionAlgorithmobj, double threshold, double iou, int numLevel);
    __declspec(dllexport) void VisionMaskData(VisionAlgorithm* visionAlgorithmobj, uchar* modeldata, int modelwidth, int modelheight, int modelstride1, TransPoint<int>* pt, int ptsize);
    __declspec(dllexport) void VisionMatchResult(VisionAlgorithm* visionAlgorithmobj, MatchResult*& result, int& resultlen);
    
    __declspec(dllexport) void VisionGetHist(VisionAlgorithm* visionAlgorithmobj, int* array);
    
    __declspec(dllexport) void VisionBlobParam(VisionAlgorithm* visionAlgorithmobj, int thresh, int type);
    __declspec(dllexport) void VisionBlobResult(VisionAlgorithm* visionAlgorithmobj, BlobResult*& result, int& resultlen);

    __declspec(dllexport) void VisionCalibParam(VisionAlgorithm* visionAlgorithmobj, int type, int rows, int cols, double distance);
    __declspec(dllexport) void VisionCalibResult(VisionAlgorithm* visionAlgorithmobj, TransPoint<int>* pt, int ptsize, TransPoint<int>* pt1, int pt1size);

    __declspec(dllexport) void VisionOcrInit(VisionAlgorithm* visionAlgorithmobj);
    __declspec(dllexport) void VisionOcrResult(VisionAlgorithm* visionAlgorithmobj, MatchResult*& result, int& resultlen);

    __declspec(dllexport) void VisionFindShapeParam(VisionAlgorithm* visionAlgorithmobj, TransPoint<int>* pt, int ptsize, int type);
    __declspec(dllexport) GeometryData VisionFitResult(VisionAlgorithm* visionAlgorithmobj, TransPoint<int>* pt, int ptsize, int type);
    __declspec(dllexport) GeometryData VisionFindShapeResult(VisionAlgorithm* visionAlgorithmobj, int findMode, int findDir);

    __declspec(dllexport) void VisionCalcParam(VisionAlgorithm* visionAlgorithmobj, int type);
    __declspec(dllexport) void VisionCalcDistance(VisionAlgorithm* visionAlgorithmobj, GeometryData obj1, GeometryData obj2, double& distance);
    __declspec(dllexport) void VisionCalcAngle(VisionAlgorithm* visionAlgorithmobj, GeometryData obj1, GeometryData obj2, double& angle);

    __declspec(dllexport) void VisionBarcodeRecResult(VisionAlgorithm* visionAlgorithmobj, MatchResult*& result, int& resultlen);

    __declspec(dllexport) void VisionImgProcessAddConstant(VisionAlgorithm* visionAlgorithmobj, int constant, uchar*& imgData, int& width, int& height,
        int& stride);
    __declspec(dllexport) void VisionImgProcessConv(VisionAlgorithm* visionAlgorithmobj, uchar*& imgData, int& width, int& height, int& stride);
    __declspec(dllexport) void VisionImgProcessEqualize(VisionAlgorithm* visionAlgorithmobj, uchar*& imgData, int& width, int& height, int& stride);
    __declspec(dllexport) void VisionImgProcessResize(VisionAlgorithm* visionAlgorithmobj, double scaleX, double scaleY, uchar*& imgData, int& width,
        int& height, int& stride);
    __declspec(dllexport) void VisionImgProcessFlip(VisionAlgorithm* visionAlgorithmobj, int flipway, uchar*& imgData, int& width,
        int& height, int& stride);
    __declspec(dllexport) void VisionImgProcessRotate(VisionAlgorithm* visionAlgorithmobj, double angle, uchar*& imgData, int& width,
        int& height, int& stride);
    __declspec(dllexport) void VisionImgProcessGaussianSampling(VisionAlgorithm* visionAlgorithmobj, double scaleX, double scaleY, int kernelSizeX,
        int kernelSizeY, double sigmaX, double sigmaY, uchar*& imgData, int& width, int& height, int& stride);
    __declspec(dllexport) void VisionImgProcessFilter(VisionAlgorithm* visionAlgorithmobj, int filterType, int kernelWidth, int kernelHeight,
            uchar*& imgData, int& width, int& height, int& stride);
    __declspec(dllexport) void VisionImgProcessMorphology(VisionAlgorithm* visionAlgorithmobj, int operation, int* shapesArr, int shapesSize,
            int* sizesArr, int sizesSize, uchar*& imgData, int& width, int& height, int& stride);
    __declspec(dllexport) void VisionImgProcessPixelReplacer(VisionAlgorithm* visionAlgorithmobj, int method, int dir,
        uchar*& imgData, int& width, int& height, int& stride);
    __declspec(dllexport) void VisionImgProcessMultiplier(VisionAlgorithm* visionAlgorithmobj, double grayConstant,
        uchar*& imgData, int& width, int& height, int& stride);
    __declspec(dllexport) void VisionImgProcessQuantize(VisionAlgorithm* visionAlgorithmobj, int levels,
        uchar*& imgData, int& width, int& height, int& stride);
    __declspec(dllexport) void VisionImgProcessReleaseImg(uchar*& imgData);
    __declspec(dllexport) void VisionRelease(VisionAlgorithm* visionAlgorithmobj);
}

VisionAlgorithm* VisionInit() {
    return new VisionAlgorithm();
}
void VisionObjRelease(VisionAlgorithm* visionAlgorithmobj) {
    if (visionAlgorithmobj) {
        delete visionAlgorithmobj;
    }
}

void VisionSrcData(VisionAlgorithm* visionAlgorithmobj, uchar* srcdata, int srcwidth, int srcheight, int srcstride) {
    visionAlgorithmobj->GetSrcImageData(srcdata, srcwidth, srcheight, srcstride);
}

//threshold：模版匹配的阈值
//iou：模版匹配框的重合度
//numLevel：模版匹配金字塔水平
void VisionMatchParam(VisionAlgorithm* visionAlgorithmobj, double threshold, double iou, int numLevel) {
    visionAlgorithmobj->GetMatchParam(threshold, iou, numLevel);
}

void VisionMaskData(VisionAlgorithm* visionAlgorithmobj, uchar* modeldata, int modelwidth, int modelheight, int modelstride1, TransPoint<int>* pt, int ptsize) {
    visionAlgorithmobj->GetModelImageData(modeldata, modelwidth, modelheight, modelstride1);
    visionAlgorithmobj->GetModelMaskData(pt, ptsize);
}

void VisionMatchResult(VisionAlgorithm* visionAlgorithmobj, MatchResult*& result, int& resultlen) {
    auto start = std::chrono::high_resolution_clock::now();
    visionAlgorithmobj->Match();
    result = visionAlgorithmobj->matchresult;
    resultlen = visionAlgorithmobj->resultlen;
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start);
    std::cout << "match process time: " << duration.count() << " ms" << std::endl;
}

void VisionGetHist(VisionAlgorithm* visionAlgorithmobj, int* array) {
    visionAlgorithmobj->GetHist(array);
}

//thresh：blob像素阈值
//type：blob模式
//0-白底黑点
//1-黑底白点
void VisionBlobParam(VisionAlgorithm* visionAlgorithmobj, int thresh, int type) {
    visionAlgorithmobj->GetBlobParam(thresh, type);
}

void VisionBlobResult(VisionAlgorithm* visionAlgorithmobj, BlobResult*& result, int& resultlen) {
    auto start = std::chrono::high_resolution_clock::now();
    visionAlgorithmobj->Blob();
    result = visionAlgorithmobj->blobresult;
    resultlen = visionAlgorithmobj->resultlen;
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start);
    std::cout << "blob process time: " << duration.count() << " ms" << std::endl;
}

//type:
//0-棋盘格标定
//1-圆形标定
//2-输入点标定
//rows、cols：棋盘格和圆形格个数尺寸
//distance：单格实际距离
void VisionCalibParam(VisionAlgorithm* visionAlgorithmobj, int type, int rows, int cols, double distance) {
    visionAlgorithmobj->GetCalibParam(type, rows, cols, distance);
}

void VisionCalibResult(VisionAlgorithm* visionAlgorithmobj, TransPoint<double>* pt, int ptsize, TransPoint<double>* pt1, int pt1size) {
    visionAlgorithmobj->Calib(pt, ptsize, pt1, pt1size);
}

//oldOri：原坐标系原点（一般是0,0）
//newOri：新坐标系原点
//angle: 新坐标系相对于旧坐标系顺时针旋转的角度
void VisionTransferAxes(VisionAlgorithm* visionAlgorithmobj, TransPoint<int> oldOri, TransPoint<int> newOri, double angle) {
    visionAlgorithmobj->TransferAxes(oldOri, newOri, angle);
}

void VisionTransferAxesPoints(VisionAlgorithm* visionAlgorithmobj, TransPoint<int>* pt, int ptsize) {
    visionAlgorithmobj->TransferAxesPoints(pt, ptsize);
}

void VisionOcrInit(VisionAlgorithm* visionAlgorithmobj) {
    visionAlgorithmobj->OcrInit();
}

void VisionOcrResult(VisionAlgorithm* visionAlgorithmobj, MatchResult*& result, int& resultlen) {
    visionAlgorithmobj->Ocr();
    result = visionAlgorithmobj->matchresult;
    resultlen = visionAlgorithmobj->resultlen;
}

//type:
//0-线拟合
//1-圆拟合
//2-椭圆拟合
void VisionFindShapeParam(VisionAlgorithm* visionAlgorithmobj, TransPoint<int>* pt, int ptsize, int type) {
    visionAlgorithmobj->GetFindShapeData(pt, ptsize, type);
}

GeometryData VisionFitResult(VisionAlgorithm* visionAlgorithmobj, TransPoint<double>* pt, int ptsize, int type) {
    return visionAlgorithmobj->FitTool(pt, ptsize, type);
}

//findMode：寻找模式
// 0-从暗到亮
// 1-从亮到暗
//findDir：寻找方向
// 0-纵向寻找
// 1-横向寻找
GeometryData VisionFindShapeResult(VisionAlgorithm* visionAlgorithmobj, int findMode, int findDir) {
    return visionAlgorithmobj->FindShape(findMode, findDir);
}

//type：
//0-点点模式
//1-点线模式
//2-点圆模式
//3-线线模式
//4-线圆模式
void VisionCalcParam(VisionAlgorithm* visionAlgorithmobj, int type) {
    visionAlgorithmobj->GetCalcParam(type);
}

void VisionCalcDistance(VisionAlgorithm* visionAlgorithmobj, GeometryData obj1, GeometryData obj2, double& distance) {
    visionAlgorithmobj->CalcDistance(obj1, obj2, distance);
}

void VisionCalcAngle(VisionAlgorithm* visionAlgorithmobj, GeometryData obj1, GeometryData obj2, double& angle) {
    visionAlgorithmobj->CalcAngle(obj1, obj2, angle);
}

void VisionBarcodeRecResult(VisionAlgorithm* visionAlgorithmobj, MatchResult*& result, int& resultlen) {
    visionAlgorithmobj->BarcodeRec();
    result = visionAlgorithmobj->matchresult;
    resultlen = visionAlgorithmobj->resultlen;
}

void VisionImgProcessOutputImg(const Mat& temp, uchar*& imgData, int& width, int& height, int& stride) {
    Mat tempRgb;
    if (temp.channels() == 1) {
        cvtColor(temp, tempRgb, COLOR_GRAY2BGR);
    }
    else {
        tempRgb = temp.clone();
    }
    
    Mat NewImage = cv::Mat::zeros(tempRgb.rows, (tempRgb.cols + 3) / 4 * 4, tempRgb.type());
    tempRgb.copyTo(NewImage(Rect(0, 0, tempRgb.cols, tempRgb.rows)));

    width = NewImage.cols;
    height = NewImage.rows;
    stride = static_cast<int>(NewImage.step);
    //width = temp.cols;
    //height = temp.rows;
    //stride = (int)temp.step;
    imgData = new unsigned char[NewImage.total() * NewImage.elemSize()];
    memcpy(imgData, NewImage.data, NewImage.total() * NewImage.elemSize());
}

//constant：图像加减的常数
void VisionImgProcessAddConstant(VisionAlgorithm* visionAlgorithmobj, int constant, uchar*& imgData, int& width, int& height,
    int& stride) {
    Mat temp = addConstantToImage(visionAlgorithmobj->OutputSrcImage(), constant);
    VisionImgProcessOutputImg(temp, imgData, width, height, stride);
}

void VisionImgProcessConv(VisionAlgorithm* visionAlgorithmobj, uchar*& imgData, int& width, int& height, int& stride) {
    Mat kernel = (Mat_<char>(3, 3) << 1, 1, 1, 1, -8, 1, 1, 1, 1);
    Mat temp = applyConvolution(visionAlgorithmobj->OutputSrcImage(), kernel);
    VisionImgProcessOutputImg(temp, imgData, width, height, stride);
}

void VisionImgProcessEqualize(VisionAlgorithm* visionAlgorithmobj, uchar*& imgData, int& width, int& height, int& stride) {
    Mat temp = equalizeImage(visionAlgorithmobj->OutputSrcImage());
    VisionImgProcessOutputImg(temp, imgData, width, height, stride);
}

//scaleX：图像水平方向缩放因子
//scaleY：图像垂直方向缩放因子
void VisionImgProcessResize(VisionAlgorithm* visionAlgorithmobj, double scaleX, double scaleY, uchar*& imgData, int& width,
    int& height, int& stride) {
    Mat temp = resizeImage(visionAlgorithmobj->OutputSrcImage(), scaleX, scaleY);
    VisionImgProcessOutputImg(temp, imgData, width, height, stride);
}

//flipway：图像翻转方式
//-1- 水平和垂直翻转
//0- 垂直翻转
//1-水平翻转
void VisionImgProcessFlip(VisionAlgorithm* visionAlgorithmobj, int flipway, uchar*& imgData, int& width,
    int& height, int& stride) {
    Mat temp = flipImage(visionAlgorithmobj->OutputSrcImage(), flipway);
    VisionImgProcessOutputImg(temp, imgData, width, height, stride);
}

//angle：图像旋转角度（顺时针）
void VisionImgProcessRotate(VisionAlgorithm* visionAlgorithmobj, double angle, uchar*& imgData, int& width,
    int& height, int& stride) {
    Mat temp = rotateImage(visionAlgorithmobj->OutputSrcImage(), angle);
    VisionImgProcessOutputImg(temp, imgData, width, height, stride);
}

//scaleX，Y：调整X和Y方向的采样因子
//kernelSizeX，Y：平滑参数
//sigmaX，Y:高斯模糊的sigma值
void VisionImgProcessGaussianSampling(VisionAlgorithm* visionAlgorithmobj, double scaleX, double scaleY, int kernelSizeX,
    int kernelSizeY,double sigmaX, double sigmaY, uchar*& imgData, int& width, int& height, int& stride) {
    Mat temp = gaussianSampling(visionAlgorithmobj->OutputSrcImage(), scaleX, scaleY, kernelSizeX, kernelSizeY, sigmaX, sigmaY);
    VisionImgProcessOutputImg(temp, imgData, width, height, stride);
}

//filterType：图像滤波方式
//0-高斯滤波
//1-均值滤波
//2-中值滤波
void VisionImgProcessFilter(VisionAlgorithm* visionAlgorithmobj, int filterType, int kernelWidth, int kernelHeight,
    uchar*& imgData, int& width, int& height, int& stride) {
    Mat temp = applyFilter(visionAlgorithmobj->OutputSrcImage(), FilterType(filterType), kernelWidth, kernelHeight);
    VisionImgProcessOutputImg(temp, imgData, width, height, stride);
}

//operation：图像形态学操作
//0-腐蚀
//1-膨胀
//2-开
//3-闭
//4-顶帽
//5-黑帽
//shapesArr：形态学结构方式
//0- nxn方形(奇数)
//1- nxn菱形
//2- 1xn水平
//3- 1xn平面45°
//4- 1xn平面垂直
//5- 1xn平面135°
//6- 自定义结构
//sizesArr：形态学结构尺寸大小
void VisionImgProcessMorphology(VisionAlgorithm* visionAlgorithmobj, int operation, int* shapesArr, int shapesSize,
    int* sizesArr, int sizesSize, uchar*& imgData, int& width, int& height, int& stride) {

    vector<int> shapes(shapesArr, shapesArr + shapesSize);
    vector<int> sizes(sizesArr, sizesArr + sizesSize);
    Mat customElement = (cv::Mat_<uchar>(3, 3) << 0, 1, 0, 1, 1, 1, 0, 1, 0);
    MorphologyProcessor processor(operation, shapes, sizes, customElement);
    Mat temp = processor.process(visionAlgorithmobj->OutputSrcImage());
    VisionImgProcessOutputImg(temp, imgData, width, height, stride);
}

//void VisionImgProcessMedianFilter(VisionAlgorithm* visionAlgorithmobj, int kernelSizeX, int kernelSizeY,
//    uchar*& imgData, int& width, int& height, int& stride) {
//    Mat temp = medianFilter(visionAlgorithmobj->OutputSrcImage(), kernelSizeX, kernelSizeY);
//    VisionImgProcessOutputImg(temp, imgData, width, height, stride);
//}

void VisionImgProcessPixelReplacer(VisionAlgorithm* visionAlgorithmobj, int method, int dir,
    uchar*& imgData, int& width, int& height, int& stride) {
    PixelReplacer replacer = PixelReplacer(ReplacementMethod(method), Direction(dir));
    //PixelReplacer replacer(PixelReplacer::NEIGHBOR_INTERPOLATION, PixelReplacer::BOTH);
    Mat temp = replacer.process(visionAlgorithmobj->OutputSrcImage());
    VisionImgProcessOutputImg(temp, imgData, width, height, stride);
}

//grayConstant：图像乘法常数
void VisionImgProcessMultiplier(VisionAlgorithm* visionAlgorithmobj, double grayConstant,
    uchar*& imgData, int& width, int& height, int& stride) {
    ImageMultiplier multiplier(grayConstant);
    Mat temp = multiplier.process(visionAlgorithmobj->OutputSrcImage());
    VisionImgProcessOutputImg(temp, imgData, width, height, stride);
}

//levels：图像量化的步长
void VisionImgProcessQuantize(VisionAlgorithm* visionAlgorithmobj, int levels,
    uchar*& imgData, int& width, int& height, int& stride) {
    Mat temp = quantizeImage(visionAlgorithmobj->OutputSrcImage(), levels);
    VisionImgProcessOutputImg(temp, imgData, width, height, stride);
}

void VisionImgProcessReleaseImg(uchar*& imgData) {
    if (imgData) {
        delete imgData;
    }
}

void VisionRelease(VisionAlgorithm* visionAlgorithmobj) {
    visionAlgorithmobj->Release();
    if (visionAlgorithmobj) {
        delete visionAlgorithmobj;
    }
}

void handlematch() {

    std::vector<cv::String> imgLists;
    string path = R"(E:\data\visonpro_demo\02 硬币统计\img)";
    //path = R"(E:\wechatdown\WeChat Files\wxid_t3qhsc9h7g8i12\FileStorage\File\2024-06\visonpro_demo\03 骰子点数统计\img)";
    glob(path, imgLists, true);
    Mat imgtmp = imread(R"(E:\data\visonpro_demo\11 多目标\img3\tmp.png)");
    imgtmp = imread(R"(C:\Users\rs\Desktop\5jiao.png)");
    //imgtmp = imread(R"(E:\test.png)");
    int imgtmp_length = imgtmp.total() * imgtmp.channels();
    unsigned char* imgtmp_array_ptr = new unsigned char[imgtmp_length]();
    memcpy(imgtmp_array_ptr, imgtmp.ptr<unsigned char>(0), imgtmp_length * sizeof(unsigned char));
    VisionAlgorithm* obj = VisionInit();
    for (auto imgpath : imgLists) {
        //Mat img = imread(R"(E:\wechatdown\WeChat Files\wxid_t3qhsc9h7g8i12\FileStorage\File\2024-06\visonpro_demo\02 硬币统计\img\1.bmp)");
        //img = imread(R"(E:\wechatdown\WeChat Files\wxid_t3qhsc9h7g8i12\FileStorage\File\2024-06\visonpro_demo\03 骰子点数统计\img\1.jpg)");
        imgpath = R"(E:\data\visonpro_demo\11 多目标\img3\engine_parts_19.png)";
        imgpath = R"(E:\data\visonpro_demo\05 零件瑕疵检测\img\01_good.bmp)";
        imgpath = R"(E:\data\visonpro_demo\09 车牌识别\img\1.jpg)";
        imgpath = R"(C:\Users\rs\Desktop\test2.jpg)";
        imgpath = R"(E:\20240925091833.bmp)";
        imgpath = R"(E:\data\visonpro_demo\02 硬币统计\img\1.bmp)";
        //imgpath = R"(E:\data\zxing-cpp\test\samples\code39-1\1.png)";
        Mat img = imread(imgpath);
        int img_length = img.total() * img.channels();
        unsigned char* image_array_ptr = new unsigned char[img_length]();
        memcpy(image_array_ptr, img.ptr<unsigned char>(0), img_length * sizeof(unsigned char));
        //CallbackParams params = { img, tempalteimg };
        //namedWindow("原始图", 1);
        ////createTrackbar("方法", "原始图", &MatchMethod, MaxTrackbarNum, TemplateMatching, &params);
        //TemplateMatching(5, &params);

        int resultlen;
        //TransPoint pt[32] = { 273.168f, 164.915f, 283.168f, 164.915f,283.168f, 194.915f ,273.168f, 194.915f ,
        //    289.418f, 164.915f, 299.418f, 164.915f, 299.418f, 194.915f, 289.418f, 194.915f,
        //    305.668f, 164.915f, 315.668f, 164.915f, 315.668f, 194.915f, 305.668f, 194.915f,
        //    321.918f, 164.915f, 331.918f, 164.915f, 331.918f, 194.915f, 321.918f, 194.915f,
        //    338.168f, 164.915f, 348.168f, 164.915f, 348.168f, 194.915f, 338.168f, 194.915f,
        //    354.418f, 164.915f, 364.418f, 164.915f, 364.418f, 194.915f, 354.418f, 194.915f,
        //    370.668f, 164.915f, 380.668f, 164.915f, 380.668f, 194.915f, 370.668f, 194.915f,
        //    386.918f, 164.915f, 396.918f, 164.915f, 396.918f, 194.915f, 386.918f, 194.915f };
        //TransPoint pt[54] = {100,100,100,101,100,102,100,103,100,104,100,105,100,106,100,107,100,108,
        //    101,100,101,101,101,102,101,103,101,104,101,105,101,106,101,107,101,108,
        //    102,100,102,101,102,102,102,103,102,104,102,105,102,106,102,107,102,108,
        //    103,100,103,101,103,102,103,103,103,104,103,105,103,106,103,107,103,108,
        //    104,100,104,101,104,102,104,103,104,104,104,105,104,106,104,107,104,108,
        //    105,100,105,101,105,102,105,103,105,104,105,105,105,106,105,107,105,108
        //};
        Rect rect1(190, 210, 70, 5);
        Rect rect2(200, 225, 70, 5);
        //rect1= Rect(245, 200, 180, 5);
        //rect2 = Rect(245, 215, 180, 5);

        Rect rect3(220, 260, 5, 5);
        Rect rect4(203, 276, 5, 5);
        Rect rect5(220, 297, 5, 5);
        Rect rect6(239, 278, 5, 5);
        vector<Rect> rectvec;
        rectvec.push_back(rect1);
        rectvec.push_back(rect2);
        TransPoint<int> pt1[8];
        for (int j = 0; j < rectvec.size();j++) {
            int i = j * 4;
                pt1[i].x = rectvec[j].tl().x;
                pt1[i].y = rectvec[j].tl().y;
                pt1[i+1].x = rectvec[j].tl().x + rectvec[j].width;
                pt1[i + 1].y = rectvec[j].tl().y;
                pt1[i+2].x = rectvec[j].br().x;
                pt1[i+2].y = rectvec[j].br().y;
                pt1[i+3].x = rectvec[j].br().x - rectvec[j].width;
                pt1[i+3].y = rectvec[j].br().y;
        }
        TransPoint<int> pt2[16];
        rectvec.clear();
        rectvec.push_back(rect3);
        rectvec.push_back(rect4);
        rectvec.push_back(rect5);
        rectvec.push_back(rect6);
        for (int j = 0; j < rectvec.size(); j++) {
            int i = j * 4;
            pt2[i].x = rectvec[j].tl().x;
            pt2[i].y = rectvec[j].tl().y;
            pt2[i + 1].x = rectvec[j].tl().x + rectvec[j].width;
            pt2[i + 1].y = rectvec[j].tl().y;
            pt2[i + 2].x = rectvec[j].br().x;
            pt2[i + 2].y = rectvec[j].br().y;
            pt2[i + 3].x = rectvec[j].br().x - rectvec[j].width;
            pt2[i + 3].y = rectvec[j].br().y;
        }

        MatchResult* result;
        VisionMatchParam(obj, 0.5, 0.2, 0);
        VisionMaskData(obj, imgtmp_array_ptr, imgtmp.cols, imgtmp.rows, imgtmp.step, pt2, 0);
        VisionSrcData(obj, image_array_ptr, img.cols, img.rows, img.step);
        VisionMatchResult(obj, result, resultlen);


        TransPoint<int> pt[20] = {447,441,409,446,403,396,440,391,366,470,342,499,303,467,327,438,327,547,329,584,278,588,276,
        550,350,629,377,656,342,692,315,666,423,674,461,675,461,726,423,725};
        VisionFindShapeParam(obj, pt, 20, 1);
        GeometryData result2 = VisionFindShapeResult(obj, 1, 0);

        VisionCalcParam(obj, 4);
        double dis;
        GeometryData pointtmp{ 514.01, 470.20 };
        GeometryData tmp1{ 1003.20, 489.19, 273.6, 109.4 };
        GeometryData linetmp{1465.68, 1548.4, 0, 0,-10.14};
        GeometryData tmp2{ 2373.4, 1023.45,205.67, 205.67 };
        VisionCalcDistance(obj, linetmp, tmp2, dis);


        //uchar* imgdata;
        //int width, height, stride;
        ////VisionImgProcessAddConstant(obj, 50, imgdata, width, height, stride);
        //VisionImgProcessFlip(obj, 1, imgdata, width, height, stride);
        //Mat srctmp(height, width, CV_8UC3, imgdata);
        //VisionImgProcessReleaseImg(imgdata);


        //VisionBarcodeRecResult(obj,result, resultlen);

        //int arraytest[256];
        //VisionGetHist(obj, arraytest);
        //delete[] image_array_ptr;
        
        //OCR(result, resultlen, image_array_ptr, img.cols, img.rows, img.step);
        //VisionOcrInit(obj);
        //VisionOcrResult(obj, result, resultlen);


        //Mat imgtmp = img.clone();
        //for (int i = 0; i < sizeof(pt1) / sizeof(pt1[0]); ) {
        //    cv::line(imgtmp, Point(pt1[i].x, pt1[i].y), Point(pt1[i + 1].x, pt1[i + 1].y), cv::Scalar(0, 255, 0), 1);
        //    cv::line(imgtmp, Point(pt1[i + 1].x, pt1[i + 1].y), Point(pt1[i + 2].x, pt1[i + 2].y), cv::Scalar(0, 255, 0), 1);
        //    cv::line(imgtmp, Point(pt1[i + 2].x, pt1[i + 2].y), Point(pt1[i + 3].x, pt1[i + 3].y), cv::Scalar(0, 255, 0), 1);
        //    cv::line(imgtmp, Point(pt1[i + 3].x, pt1[i + 3].y), Point(pt1[i].x, pt1[i].y), cv::Scalar(0, 255, 0), 1);
        //    i = i + 4;
        //}
        //for (int i = 0; i < sizeof(pt2) / sizeof(pt2[0]); ) {
        //    cv::line(imgtmp, Point(pt2[i].x, pt2[i].y), Point(pt2[i + 1].x, pt2[i + 1].y), cv::Scalar(0, 255, 0), 1);
        //    cv::line(imgtmp, Point(pt2[i + 1].x, pt2[i + 1].y), Point(pt2[i + 2].x, pt2[i + 2].y), cv::Scalar(0, 255, 0), 1);
        //    cv::line(imgtmp, Point(pt2[i + 2].x, pt2[i + 2].y), Point(pt2[i + 3].x, pt2[i + 3].y), cv::Scalar(0, 255, 0), 1);
        //    cv::line(imgtmp, Point(pt2[i + 3].x, pt2[i + 3].y), Point(pt2[i].x, pt2[i].y), cv::Scalar(0, 255, 0), 1);
        //    i = i + 4;
        //}
        //VisionFindShapeParam(obj, pt1, sizeof(pt1) / sizeof(pt1[0]), 0);
        //GeometryData result1 = VisionFindShapeResult(obj, 1, 1);
        //double radian = int(result1.angle) * CV_PI / 180.0; // 将角度转换为弧度  
        //int endX = static_cast<int>(result1.centerX + 50 * cos(radian));
        //int endY = static_cast<int>(result1.centerY + 50 * sin(radian));
        //cv::Point endPoint(endX, endY);
        //endX = static_cast<int>(result1.centerX - 50 * cos(radian));
        //endY = static_cast<int>(result1.centerY - 50 * sin(radian));
        //Point startPoint(endX, endY);
        ////cv::line(imgtmp, startPoint, endPoint, cv::Scalar(0, 0, 255), 1);
        //GeometryData result2 = VisionFindShapeResult(obj, 0, 1);
        //radian = int(result2.angle) * CV_PI / 180.0; // 将角度转换为弧度  
        //endX = static_cast<int>(result2.centerX + 50 * cos(radian));
        //endY = static_cast<int>(result2.centerY + 50 * sin(radian));
        //cv::Point endPoint1(endX, endY);
        //endX = static_cast<int>(result2.centerX - 50 * cos(radian));
        //endY = static_cast<int>(result2.centerY - 50 * sin(radian));
        //Point startPoint1(endX, endY);
        ////cv::line(imgtmp, startPoint1, endPoint1, cv::Scalar(0, 0, 255), 1);
        //VisionFindShapeParam(obj, pt2, sizeof(pt2) / sizeof(pt2[0]), 1);
        //GeometryData result3 = VisionFindShapeResult(obj, 1, 1);
        //circle(imgtmp, cv::Point(int(result3.centerX), int(result3.centerY)), int(result3.radiusX), cv::Scalar(0, 0, 255));
        //VisionCalcParam(obj, 4);
        //double dis;
        //VisionCalcDistance(obj, result2, result3, dis);
        //putText(imgtmp, to_string(dis), Point(100, 100), FONT_HERSHEY_SIMPLEX, 2, Scalar(0, 0, 255), 4, 8);
        //RotateMatch(result, resultlen, image_array_ptr, img.cols, img.rows, img.step, imgtmp_array_ptr, imgtmp.cols, imgtmp.rows,
        //    imgtmp.step, pt, 0, 0.9, 0.5,0);


        //BlobResult* result;
        //VisionSrcData(obj, image_array_ptr, img.cols, img.rows, img.step);
        //VisionBlobParam(obj, 180, 0);
        //VisionBlobResult(obj, result, resultlen);


        //VisionSrcData(obj, image_array_ptr, img.cols, img.rows, img.step);
        //VisionCalibParam(obj, 0, 8, 8, 10);
        //TransPoint pt10[1] = { 256,93 };
        //VisionCalibResult(obj,pt,0,pt10,0);
        //obj->CalibUndistort(pt10, 1);
        TransPoint<int> a{ 0,0 };
        TransPoint<int> b{ 1,1 };
        VisionTransferAxes(obj, a, b, -90);
        TransPoint<int> c[1]={ 2,2 };
        VisionTransferAxesPoints(obj, c, 1);


    }
    
}

int main()
{

    //FindCircle1();
    //Point_set_fitting();
    handlematch();
    
    //OCRTrain();

    //cv::Mat img = cv::imread(R"(E:\data\visonpro_demo\09 车牌识别\img\5.jpg)", cv::IMREAD_COLOR);
    //int outputResultLen;
    //MatchResult* outputResult;
    //OcrDetect(OcrInit(),img, outputResultLen, outputResult);
}
