#include<AlgorithmProcess.h>
//using namespace xfeatures2d;
void Point_set_fitting() {
    Mat img(500, 500, CV_8UC3, Scalar::all(0));
    RNG& rng = theRNG();
    int i, count = rng.uniform(1, 101);
    vector<Point> points;
    //生成随机点
    for (i = 0; i < count; i++) {
        Point  pt;
        pt.x = rng.uniform(img.cols / 4, img.cols * 3 / 4);
        pt.y = rng.uniform(img.rows / 4, img.rows * 3 / 4);
        points.push_back(pt);
    }
    img = Scalar::all(0);

    Vec4f lines;
    double param = 0;//距离模型中的数值参数C
    double reps = 0.01;//坐标原点与直线之间的距离精度
    double aeps = 0.01;//角度精度
    fitLine(points, lines, DIST_L1, param, reps, aeps);
    double cos_theta = lines[0];
    double sin_theta = lines[1];
    double x0 = lines[2], y0 = lines[3];
    double k = sin_theta / cos_theta;
    double b = y0 - k * x0;
    double x = 0;
    double y = k * x + b;
    double x1 = img.cols;
    double y1 = k * x1 + b;
    line(img, Point(x, y), Point(x1, y1), cv::Scalar(0, 0, 255), 1);

    cv::Point2f center;
    float radius = 0;
    minEnclosingCircle(points, center, radius);

    //在图像中绘制坐标点
    for (i = 0; i < count; i++) {
        circle(img, points[i], 3, Scalar(255, 255, 255), FILLED, LINE_AA);
    }
    circle(img, center, cvRound(radius), Scalar(255, 255, 255), 1, LINE_AA);
    imshow("test", img);
    waitKey(0);
}

int MatchMethod = 0;
int MaxTrackbarNum = 5;
struct Bbox {
    double score;
    Point location;
    double angle;
};
bool sort_score(Bbox box1, Bbox box2)
{
    return (box1.score > box2.score);
}

Mat ImageRotate(Mat image, double angle)
{
    Mat newImg;
    Point2f pt = Point2f((float)image.cols / 2, (float)image.rows / 2);
    Mat M = getRotationMatrix2D(pt, angle, 1.0);
    warpAffine(image, newImg, M, image.size(), INTER_LINEAR, BORDER_REFLECT, image.at<uchar>(0, 0));
    return newImg;
}

vector<Point> GetRotatePoints(Size size, double angle)
{
    // 构建旋转矩阵
    Point2f center(size.width / 2.0, size.height / 2.0);
    Mat rotationMat = getRotationMatrix2D(center, angle, 1.0);

    // 计算旋转后的四个顶点
    vector<Point> points;
    points.push_back(Point(0, 0));
    points.push_back(Point(size.width, 0));
    points.push_back(Point(size.width, size.height));
    points.push_back(Point(0, size.height));

    vector<Point> rotatedPoints;
    for (size_t i = 0; i < points.size(); i++)
    {
        Point2f pt(points[i]);
        Point2f newPt = Point2f(rotationMat.at<double>(0, 0) * pt.x + rotationMat.at<double>(0, 1) * pt.y + rotationMat.at<double>(0, 2),
            rotationMat.at<double>(1, 0) * pt.x + rotationMat.at<double>(1, 1) * pt.y + rotationMat.at<double>(1, 2));
        rotatedPoints.push_back(Point(newPt));
    }

    return rotatedPoints;
}

//Rect 的iou面积计算
float get_iou_value(Rect rect1, Rect rect2)
{
    int xx1, yy1, xx2, yy2;

    xx1 = max(rect1.x, rect2.x);
    yy1 = max(rect1.y, rect2.y);
    xx2 = min(rect1.x + rect1.width - 1, rect2.x + rect2.width - 1);
    yy2 = min(rect1.y + rect1.height - 1, rect2.y + rect2.height - 1);

    int insection_width, insection_height;
    insection_width = max(0, xx2 - xx1 + 1);
    insection_height = max(0, yy2 - yy1 + 1);

    float insection_area, union_area, iou;
    insection_area = float(insection_width) * insection_height;
    union_area = float(rect1.width * rect1.height + rect2.width * rect2.height - insection_area);
    iou = insection_area / union_area;
    return iou;
}

float CalcIOU(RotatedRect rect1, RotatedRect rect2)
{
    float areaRect1 = rect1.size.width * rect1.size.height;
    float areaRect2 = rect2.size.width * rect2.size.height;
    vector<cv::Point2f> vertices;
    int intersectionType = rotatedRectangleIntersection(rect1, rect2, vertices);
    if (vertices.size() == 0)
        return 0.0;
    else {
        vector<cv::Point2f> order_pts;
        cv::convexHull(cv::Mat(vertices), order_pts, true);
        double area = cv::contourArea(order_pts);
        float inner = (float)(area / (areaRect1 + areaRect2 - area + 0.0001));
        return inner;
    }
}

void CalcMeanStd(const cv::Mat& src, Point pts, Size modelsize, double& mean, double& stddev) {
    Mat result = src(Rect(pts, modelsize));
    cv::Scalar mean1, stddev1;
    meanStdDev(result, mean1, stddev1);
    mean = mean1[0];
    stddev = stddev1[0];
}

void cropRotatedRect(const cv::Mat& src, const cv::RotatedRect& rotatedRect, double& mean, double& stddev) {

    Mat rot_mat = getRotationMatrix2D(rotatedRect.center, -rotatedRect.angle, 1.0);
    cv::Point2f vertices[4];
    rotatedRect.points(vertices);
    Mat rot_image;
    Size dst_sz(src.size());
    warpAffine(src, rot_image, rot_mat, dst_sz);
    int left = max(int(rotatedRect.center.x), 0);
    int top = max(int(rotatedRect.center.y), 0);
    Mat result = rot_image(Rect(left, top, rotatedRect.size.width, rotatedRect.size.height));
    cv::Scalar mean1, stddev1;
    meanStdDev(result, mean1, stddev1);
    mean = mean1[0];
    stddev = stddev1[0];
}

Mat cropRotated(const cv::Mat& src, const cv::RotatedRect& rotatedRect) {
    Mat rot_mat = getRotationMatrix2D(rotatedRect.center, rotatedRect.angle, 1.0);
    Mat rot_image;
    Size dst_sz(src.size());
    warpAffine(src, rot_image, rot_mat, dst_sz);
    int left = max(int(rotatedRect.center.x), 0);
    int top = max(int(rotatedRect.center.y), 0);
    Mat result = rot_image(Rect(left, top, rotatedRect.size.width, rotatedRect.size.height));
    return result;
}

vector<vector<Point>>findContour(Mat Image)
{
    //Mat gray;
    //cvtColor(Image, gray, COLOR_BGR2GRAY);

    Mat thresh;
    threshold(Image, thresh, 0, 255, THRESH_BINARY | THRESH_OTSU);

    vector<vector<Point>>contours;
    findContours(thresh, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    vector<vector<Point>>EffectConts;
    for (int i = 0; i < contours.size(); i++)
    {
        double area = contourArea(contours[i]);

        if (area > 100)
        {
            EffectConts.push_back(contours[i]);
        }
    }

    return EffectConts;
}

double CompareContour(const cv::Mat& src, const cv::Mat& tmp) {
    vector<vector<Point>>src_contours;
    vector<vector<Point>>test_contours;
    src_contours = findContour(src);
    test_contours = findContour(tmp);
    if (src_contours.size() != test_contours.size()) {
        return 100.0;
    }
    double dist = 0;
    for (int i = 0; i < test_contours.size(); i++) {
        Moments m_test = moments(test_contours[i]);
        Mat hu_test;
        HuMoments(m_test, hu_test);
        Moments m_src = moments(src_contours[i]);
        Mat hu_src;
        HuMoments(m_src, hu_src);
        dist += matchShapes(hu_test, hu_src, CONTOURS_MATCH_I1, 0);
    }
    return dist = dist / test_contours.size();
}




VisionAlgorithm::VisionAlgorithm() {
    thresholdvalue = 0.9;
    iouvalue = 0.3;
    numLevels = 0;
    pixelThresholdValue = 128;
    typeValue = 0;
    patternSize = Size();
    distanceValue = 0.0;
    matchresult = nullptr;
    blobresult = nullptr;
    resultlen = -1;
    transforArray[9] = { 0.0 };
}

void VisionAlgorithm::GetModelImageData(uchar* modeldata, int modelwidth, int modelheight, int modelstride1) {
    Mat modeltmp = Mat(Size(modelwidth, modelheight), CV_8UC3, modeldata, modelstride1).clone();
    cvtColor(modeltmp, modelImage, COLOR_BGR2GRAY);
}

void VisionAlgorithm::GetSrcImageData(uchar* srcdata, int srcwidth, int srcheight, int srcstride) {
    Mat srctmp = Mat(Size(srcwidth, srcheight), CV_8UC3, srcdata, srcstride).clone();
    cvtColor(srctmp, srcImage, COLOR_BGR2GRAY);
}

Mat VisionAlgorithm::OutputSrcImage() {
    return srcImage;
}

void VisionAlgorithm::GetModelMaskData(TransPoint<int>* pt, int ptsize) {
    if (ptsize > 0) {
        mask = Mat(modelImage.size(), CV_8UC1, Scalar(255));
        for (int i = 0; i < ptsize; ++i) {
            mask.ptr<uchar>(pt[i].y)[pt[i].x] = 0;
            //uchar* p = mask.ptr<uchar>(pt[i].x);
            //p[pt[i].y] = 255;
        }
    }
    else {
        mask = imread(R"(C:\Users\rs\Desktop\5jiaomask1.png)", 0);
        //mask = Mat();
    }
}

void VisionAlgorithm::GetHist(int* array)
{
    const int channels[1] = { 0 }; //通道索引
    float inRanges[2] = { 0,255 };  //像素范围
    const float* ranges[1] = { inRanges };//像素灰度级范围
    const int bins[1] = { 256 };   //直方图的维度
    Mat hist;
    calcHist(&srcImage, 1, channels, Mat(), hist, 1, bins, ranges);
    for (int i = 0; i < hist.rows; i++)
    {
        float* idata = hist.ptr<float>(i);
        array[i] = (int)idata[0];
    }
}

void VisionAlgorithm::GetMatchParam(double threshold, double iou, int numLevel) {
    thresholdvalue = threshold;
    iouvalue = iou;
    numLevels = numLevel;
}

std::mutex vecMutex;
void matchResultsThread(int threadsNum,double threadProNum, double firstStep, const Mat& modeltmp, const Mat& srctmp, const Mat& mask, double thresholdvalue,
    vector<Bbox>& bboxvec) {
    Mat b = Mat::ones(Size(srctmp.size().width - modeltmp.size().width + 1, srctmp.size().height - modeltmp.size().height + 1), CV_32F);
    double startAngle = 0;
    double endAngle = 360;
    int startIndex = threadsNum * threadProNum;
    int endIndex = min((threadsNum + 1) * threadProNum, endAngle);
    Mat rotatedImg, resulttmp;
    //vector<Bbox> bboxvectmp;
    for (int curAngle = startIndex; curAngle < endIndex; curAngle += firstStep) {
        rotatedImg = ImageRotate(modeltmp, curAngle);
        Mat masktmp = mask.clone();
        matchTemplate(srctmp, rotatedImg, resulttmp, TM_SQDIFF_NORMED, masktmp);
        //matchTemplate(srctmp, rotatedImg, resulttmp, TM_CCOEFF_NORMED, masktmp);
        //normalize(resulttmp, resulttmp, 1, 0, CV_MINMAX);
        double minval, maxval;
        Point minloc, maxloc;
        resulttmp = b - resulttmp;
        std::lock_guard<std::mutex> lock(vecMutex);
        while (true) {
            minMaxLoc(resulttmp, &minval, &maxval, &minloc, &maxloc);
            if (maxval > 1.0) {
                rectangle(resulttmp, Point(maxloc.x - 5, maxloc.y - 5), Point(maxloc.x + modeltmp.cols + 5, maxloc.y + modeltmp.rows + 5), Scalar(0), -1);
                continue;
            }
            if (maxval > thresholdvalue)
            {
                Bbox boxtmp;
                boxtmp.location = maxloc;
                boxtmp.score = maxval;
                boxtmp.angle = curAngle;

                bboxvec.push_back(boxtmp);
                rectangle(resulttmp, Point(maxloc.x - 5, maxloc.y - 5), Point(maxloc.x + modeltmp.cols + 5, maxloc.y + modeltmp.rows + 5), Scalar(0), -1);
            }
            else {
                break;
            }
        }

    }
    //if (bboxvectmp.size() > 0) {
    //    std::lock_guard<std::mutex> lock(vecMutex);
    //    bboxvec.insert(bboxvec.end(), bboxvectmp.begin(), bboxvectmp.end());
    //}
}

void VisionAlgorithm::Match() {
    Mat srctmp, modeltmp;
    modeltmp = modelImage.clone();
    copyMakeBorder(srcImage, srctmp, 50, 50, 50, 50, BorderTypes::BORDER_REPLICATE, 0);
    for (int i = 0; i < numLevels; i++) {
        pyrDown(srctmp, srctmp, Size(srctmp.cols / 2, srctmp.rows / 2));
        pyrDown(modeltmp, modeltmp, Size(modeltmp.cols / 2, modeltmp.rows / 2));
        if (!mask.empty()) {
            //pyrDown(mask, mask, Size(mask.cols / 2, mask.rows / 2));
            //cv::Mat masktmp;
            cv::resize(mask, mask, Size(modeltmp.cols, modeltmp.rows), 0, 0, cv::INTER_NEAREST);
            //if (mask.cols != modeltmp.cols || mask.rows != modeltmp.rows) {
            //    cv::resize(mask, mask, cv::Size(modeltmp.cols, modeltmp.rows), 0, 0, cv::INTER_LINEAR);
            //}
        }
    }
    //Scalar mean1, stddev1;
    //meanStdDev(modeltmp, mean1, stddev1);
    //double meantmp = mean1[0];
    //double stddevtmp = stddev1[0];
    Mat b = Mat::ones(Size(srctmp.size().width - modeltmp.size().width + 1, srctmp.size().height - modeltmp.size().height + 1), CV_32F);
    bool isSecond = false;
    Mat rotatedImg, resulttmp;
    vector<Bbox> bboxvec;
    double startAngle = 0;
    double endAngle = 360;
    double firstStep = 5;
    Size modelsize = Size(modeltmp.cols * pow(2, numLevels), modeltmp.rows * pow(2, numLevels));
    double threadProNum = 40;// 40;
    int threadsNum = endAngle / threadProNum;
    std::vector<std::thread> threads;
    for (int i = 0; i < threadsNum; i++) {
        threads.emplace_back(matchResultsThread, i, threadProNum, firstStep, modeltmp, srctmp, mask, thresholdvalue, ref(bboxvec));
    }
    for (auto& t : threads) {
        t.join();
    }
    //for (double curAngle = startAngle; curAngle < endAngle; curAngle += firstStep) {
    //    rotatedImg = ImageRotate(modeltmp, curAngle);
    //    Mat masktmp = mask.clone();
    //    matchTemplate(srctmp, rotatedImg, resulttmp, TM_SQDIFF_NORMED, masktmp);
    //    //normalize(resulttmp, resulttmp, 1, 0, CV_MINMAX);
    //    double minval, maxval;
    //    Point minloc, maxloc;
    //    resulttmp = b - resulttmp;
    //    while (true) {
    //        minMaxLoc(resulttmp, &minval, &maxval, &minloc, &maxloc);
    //        if (maxval > thresholdvalue)
    //        {
    //            Bbox boxtmp;
    //            boxtmp.location = maxloc;
    //            boxtmp.score = maxval;
    //            boxtmp.angle = curAngle;
    //            bboxvec.push_back(boxtmp);
    //            rectangle(resulttmp, Point(maxloc.x - 10, maxloc.y - 10), Point(maxloc.x + modeltmp.cols + 10, maxloc.y + modeltmp.rows + 10), Scalar(0), -1);
    //        }
    //        else {
    //            break;
    //        }
    //    }
    //}   
    if (bboxvec.size() < 1) {
        cout << "无匹配" << endl;
        return;
    }
    sort(bboxvec.begin(), bboxvec.end(), sort_score);

    for (int j = 0; j < bboxvec.size(); j++) {
        auto& item = bboxvec[j];
        Point finalPointtmp = Point(item.location.x, item.location.y);
        for (int i = j + 1; i < bboxvec.size(); i++) {
            Point finalPoint = Point(bboxvec[i].location.x, bboxvec[i].location.y);
            if (CalcIOU(RotatedRect(finalPointtmp, modelsize, bboxvec[j].angle), RotatedRect(finalPoint, modelsize, bboxvec[i].angle)) >
                iouvalue) {
                bboxvec.erase(bboxvec.begin() + i);
                --i;
            }
        }
    }

    //for (int j = 0; j < bboxvec.size(); j++) {
    //    auto& item = bboxvec[j];
    //    Point finalPointtmp = Point(item.location.x, item.location.y);
    //    double mean, stddev;
    //    CalcMeanStd(srctmp, item.location, modelsize, mean, stddev);
    //    //cropRotatedRect(srctmp, RotatedRect(finalPointtmp, modelsize, item.angle), mean, stddev);
    //    if (abs(mean - meantmp) > 20 || abs(stddev - stddevtmp) > 20) {
    //        bboxvec.erase(bboxvec.begin() + j);
    //        --j;
    //    }
    //}

    //const int minHessian = 700;
    //Ptr<SURF>detector = SURF::create(minHessian);
    //vector<KeyPoint>keypoints_object, keypoints_scene;
    //Mat descriptors_object, descriptors_scene;
    //detector->detectAndCompute(modeltmp, cv::noArray(), keypoints_scene, descriptors_scene);
    //FlannBasedMatcher matcher;
    //vector<DMatch>matches;
    //matcher.match(descriptors_object, descriptors_scene, matches);
    //for (int j = 0; j < bboxvec.size(); j++) {
    //    auto& item = bboxvec[j];
    //    Point finalPointtmp = Point(item.location.x, item.location.y);
    //    Mat image_object = srctmp(Rect(item.location, modelsize));
    //    //Mat image_object = cropRotated(srctmp, RotatedRect(finalPointtmp, modelsize, item.angle));
    //    detector->detectAndCompute(image_object, cv::noArray(), keypoints_object, descriptors_object);
    //    vector<DMatch>matches;
    //    matcher.match(descriptors_object, descriptors_scene, matches);
    //    if (matches.size()<20){
    //        bboxvec.erase(bboxvec.begin() + j);
    //        --j;
    //        continue;
    //    }
    //    sort(matches.begin(), matches.end());
    //    vector<DMatch>good_matches;
    //    for (int i = 0; i < 10; i++)
    //    {
    //        double sum=0.0;
    //        sum += matches[i].distance;
    //        if (sum > 2.0)
    //        {
    //            bboxvec.erase(bboxvec.begin() + j);
    //            --j;
    //        }
    //    }
    //}

    //for (int j = 5; j < bboxvec.size(); j++) {
    //    bboxvec.erase(bboxvec.begin() + j);
    //    --j;
    //}


    vector<MatchResult> matchResult;
    for (int i = 0; i < bboxvec.size(); i++) {
        Point finalPoint = Point(bboxvec[i].location.x * pow(2, numLevels), bboxvec[i].location.y * pow(2, numLevels));
        vector<Point> points = GetRotatePoints(modelsize, bboxvec[i].angle);
        MatchResult matchResulttmp;
        for (int j = 0; j < points.size(); j++)
        {
            points[j].x += finalPoint.x - 50;
            points[j].y += finalPoint.y - 50;
            matchResulttmp.point[2 * j] = points[j].x;
            matchResulttmp.point[2 * j + 1] = points[j].y;
        }
        matchResulttmp.angle = bboxvec[i].angle;
        matchResulttmp.score = bboxvec[i].score;
        matchResult.push_back(matchResulttmp);
        //for (int i = 0; i < 4; ++i)
        //{
        //    cout << i << "个：" << points[i % 4] << "   " << points[(i + 1) % 4] << endl;
        //    cv::line(srcImage, points[i % 4], points[(i + 1) % 4], cv::Scalar(255, 0, 255), 1);
        //}


    }

    resultlen = matchResult.size();
    matchresult = new MatchResult[resultlen];
    memcpy(matchresult, &matchResult[0], resultlen * sizeof(MatchResult));
}


void VisionAlgorithm::GetBlobParam(int thresh, int type) {
    pixelThresholdValue = thresh;
    typeValue = type;
}

void VisionAlgorithm::Blob() {
    Mat binaryimg;
    //type:0  >thresh =255; <thresh =0;
    //type:1  >thresh =0;   <thresh=255;
    threshold(srcImage, binaryimg, pixelThresholdValue, 255, typeValue);
    //std::vector<std::vector<cv::Point>> contours;
    //findContours(binaryimg, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    //for (int i = 0; i < contours.size(); i++)
    //{
    //    if (contours[i].size() > 4) {
    //        drawContours(src, contours, i, Scalar(0, 0, 255), 0);
    //    }
    //}
    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    int nLabels = cv::connectedComponentsWithStats(binaryimg, labels, stats, centroids);
    vector<BlobResult> blobResult;
    for (int j = 1; j < nLabels; j++) {
        int area = stats.at<int>(j, cv::CC_STAT_AREA);
        if (area > 10) {
            int x = stats.at<int>(j, cv::CC_STAT_LEFT);
            int y = stats.at<int>(j, cv::CC_STAT_TOP);
            int width = stats.at<int>(j, cv::CC_STAT_WIDTH);
            int height = stats.at<int>(j, cv::CC_STAT_HEIGHT);
            Rect roi(x, y, width, height);
            Mat labelstmp = labels(roi);
            cv::Mat mask = (labelstmp == j);
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            for (int i = 0; i < contours.size(); i++)
            {
                if (contours[i].size() > 4) {
                    BlobResult blobResulttmp;
                    //double area = contourArea(contours[i]);
                    //if (area < 200)continue;
                    //晒选出轮廓面积大于2000的轮廓	
                    double arc_length = arcLength(contours[i], true);
                    double radius = arc_length / (2 * PI);
                    //RotatedRect rect = fitEllipse(contours[i]);
                    //float ratio = float(rect.size.width) / float(rect.size.height);
                    //Point2f vertices[4];
                    //rect.points(vertices);
                    Point vertices[4] = { Point(roi.x ,roi.y),Point(roi.x + width,roi.y),Point(roi.x + width,roi.y + height),Point(roi.x,roi.y + height) };
                    float ratio = width / float(height);
                    for (int j = 0; j < 4; j++) {
                        blobResulttmp.point[2 * j] = vertices[j].x;
                        blobResulttmp.point[2 * j + 1] = vertices[j].y;
                    }
                    blobResulttmp.area = area;
                    blobResulttmp.arclen = arc_length;
                    blobResulttmp.ratio = ratio;
                    blobResult.push_back(blobResulttmp);
                    drawContours(srcImage, contours, i, Scalar(0, 0, 255), -1);
                }
            }
        }
    }
    resultlen = blobResult.size();
    blobresult = new BlobResult[resultlen];
    memcpy(blobresult, &blobResult[0], resultlen * sizeof(BlobResult));
}

//typeValue:
//0：棋盘格标定
//1：圆形标定
//2：输入点标定
//patternSize：棋盘格和圆形格个数尺寸
//distanceValue：单格实际距离
void VisionAlgorithm::GetCalibParam(int type, int rows, int cols, double distance) {
    patternSize = Size(rows, cols);
    typeValue = type;
    distanceValue = distance;
}

void myUndistortPoints(const std::vector<cv::Point2f>& src, std::vector<cv::Point2f>& dst,
    const cv::Mat& cameraMatrix, const cv::Mat& distortionCoeff)
{

    dst.clear();
    double fx = cameraMatrix.at<double>(0, 0);
    double fy = cameraMatrix.at<double>(1, 1);
    double ux = cameraMatrix.at<double>(0, 2);
    double uy = cameraMatrix.at<double>(1, 2);

    double k1 = distortionCoeff.at<double>(0, 0);
    double k2 = distortionCoeff.at<double>(0, 1);
    double p1 = distortionCoeff.at<double>(0, 2);
    double p2 = distortionCoeff.at<double>(0, 3);
    double k3 = distortionCoeff.at<double>(0, 4);
    double k4 = 0;
    double k5 = 0;
    double k6 = 0;

    for (unsigned int i = 0; i < src.size(); i++)
    {
        const cv::Point2f& p = src[i];
        //首先进行坐标转换；
        double xDistortion = (p.x - ux) / fx;
        double yDistortion = (p.y - uy) / fy;

        double xCorrected, yCorrected;

        double x0 = xDistortion;
        double y0 = yDistortion;
        //这里使用迭代的方式进行求解，因为根据2中的公式直接求解是困难的，所以通过设定初值进行迭代，这也是OpenCV的求解策略；
        for (int j = 0; j < 10; j++)
        {
            double r2 = xDistortion * xDistortion + yDistortion * yDistortion;

            double distRadialA = 1 / (1. + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);
            double distRadialB = 1. + k4 * r2 + k5 * r2 * r2 + k6 * r2 * r2 * r2;

            double deltaX = 2. * p1 * xDistortion * yDistortion + p2 * (r2 + 2. * xDistortion * xDistortion);
            double deltaY = p1 * (r2 + 2. * yDistortion * yDistortion) + 2. * p2 * xDistortion * yDistortion;

            xCorrected = (x0 - deltaX) * distRadialA * distRadialB;
            yCorrected = (y0 - deltaY) * distRadialA * distRadialB;

            xDistortion = xCorrected;
            yDistortion = yCorrected;
        }
        //进行坐标变换；
        xCorrected = xCorrected * fx + ux;
        yCorrected = yCorrected * fy + uy;

        dst.push_back(cv::Point2f(xCorrected, yCorrected));
    }

}

void VisionAlgorithm::Calib(TransPoint<double>* pt, int ptsize, TransPoint<double>* pt1, int pt1size) {
    std::vector<cv::Point2f> corners;
    std::vector<std::vector<cv::Point2f>> cornersVect;
    std::vector<cv::Point3f> worldPoints;
    std::vector<std::vector<cv::Point3f>> worldPointsVect;

    if (typeValue == 2) {
        if (ptsize > 0 && pt1size > 0 && pt1size == ptsize) {
            for (int i = 0; i < ptsize; ++i) {
                worldPoints.push_back(Point3f(pt1[i].x, pt1[i].y, 0));
                corners.push_back(Point2f(pt[i].x, pt[i].y));
            }
            cornersVect.push_back(corners);
        }
    }
    else {
        for (int i = 0; i < patternSize.height; i++)
        {
            for (int j = 0; j < patternSize.width; j++)
            {
                worldPoints.push_back(cv::Point3f(j * distanceValue, i * distanceValue, 0));
            }
        }
    }
    worldPointsVect.push_back(worldPoints);
    if (typeValue == 1) {
        SimpleBlobDetector::Params params;
        params.maxArea = 90000;
        params.minArea = 500;
        params.filterByArea = true;
        cv::Ptr<cv::FeatureDetector> blobDetector = cv::SimpleBlobDetector::create(params);
        if (0 != findCirclesGrid(srcImage, patternSize, corners, cv::CALIB_CB_SYMMETRIC_GRID | cv::CALIB_CB_CLUSTERING, blobDetector)) {
            cornersVect.push_back(corners);
        }
        else {
            return;
        }
    }
    else if (typeValue == 0) {
        if (0 != findChessboardCorners(srcImage, patternSize, corners)) {
            TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 40, 0.001);
            cornerSubPix(srcImage, corners, Size(5, 5), Size(-1, -1), criteria);
            //find4QuadCornerSubpix(gray, corners, cv::Size(5, 5));
            drawChessboardCorners(srcImage, patternSize, corners, true);
            cornersVect.push_back(corners);
        }
        else {
            return;
        }
    }
    //cv::Mat cameraMatirx, distCoeffs;
    std::vector<cv::Mat> rvecs, tvecs;
    double rms = calibrateCamera(worldPointsVect, cornersVect, srcImage.size(), cameraMatirx, distCoeffs, rvecs, tvecs);
    Mat dst;
    undistort(srcImage, dst, cameraMatirx, distCoeffs);
    //std::vector<cv::Point2f> corners2;
    //undistortPoints(corners, corners2, cameraMatirx, distCoeffs, cv::Mat(), cameraMatirx);
    // 投影 3D 点到 2D 图像平面
    vector<Point2f> imagePoints;
    Mat projectedPoints;
    projectPoints(worldPoints, rvecs[0], tvecs[0], cameraMatirx, distCoeffs, imagePoints);

    // 绘制投影点
    //for (int i = 0; i < imagePoints.size(); i++) {
    //    Point projPoint(imagePoints[i].x, imagePoints[i].y);
    //    circle(srcImage, projPoint, 5, Scalar(0, 0, 255), 2);
    //}
    cv::line(srcImage, imagePoints[0], imagePoints[8], Scalar(0, 0, 255), 2); // x轴 - 红色
    cv::line(srcImage, imagePoints[0], imagePoints[1], Scalar(0, 255, 0), 2); // y轴 - 绿色
}

void VisionAlgorithm::CalibUndistort(TransPoint<double>* pt, int ptsize) {
    std::vector<cv::Point2f> corners;
    for (int i = 0; i < ptsize; ++i) {
        corners.push_back(Point2f(pt[i].x, pt[i].y));
    }

    std::vector<cv::Point2f> corners2;
    undistortPoints(corners, corners2, cameraMatirx, distCoeffs, cv::Mat(), cameraMatirx);

    std::vector<TransPoint<double>> corners21;
    for (int i = 0; i < ptsize; ++i) {
        pt[i].x = corners2[i].x;
        pt[i].y = corners2[i].y;
    }
    //for (auto corner : corners2) {
    //    TransPoint TransPointtmp;
    //    TransPointtmp.x = corner.x;
    //    TransPointtmp.y = corner.y;
    //    corners21.push_back(TransPointtmp);
    //}
}

void VisionAlgorithm::TransferAxes(TransPoint<int> oldOri, TransPoint<int> newOri, double angle) {
    //angle: 新坐标系相对于旧坐标系顺时针旋转的角度
    double translationArray[9] = {1,0,0,0,1,0,0,0,1};
    double rotationArray[9] = { 1,0,0,0,1,0,0,0,1 };
    translationArray[2] = oldOri.x - newOri.x;
    translationArray[5] = oldOri.y - newOri.y;
    double angletmp = angle* PI / 180.0;
    rotationArray[0] = cos(angletmp);
    rotationArray[1] = -sin(angletmp);
    rotationArray[3] = sin(angletmp);
    rotationArray[4] = cos(angletmp);
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                transforArray[i*3+j] += (rotationArray[i*3+k] * translationArray[k*3+j]);
            }
        }
    }
}

void VisionAlgorithm::TransferAxesPoints(TransPoint<int>* pt, int ptsize) {
    for (int i = 0; i < ptsize; ++i) {
        double pttmp[3] = { pt[i].x,pt[i].y, 1.0 };
        double temp[2] = { 0.0 };
        for (int k = 0; k < 2; k++)
        { 
            for (int j = 0; j < 3; j++)
            {
                temp[k] += transforArray[k * 3 + j] * pttmp[j];
            }

        }
        pt[i].x = round(temp[0]);
        pt[i].y = round(temp[1]);
    }
}

//void VisionAlgorithm::Ocr(PaddleOCR::PPOCR& ocrobj, std::vector<cv::Mat>& img_list, int& outputResultLen) {
void VisionAlgorithm::OcrInit() {
    //PaddleOCR::PPOCR ocrobj1 = PaddleOCR::PPOCR("./ch_PP-OCRv3_det_infer", "./ch_PP-OCRv3_rec_infer");
    ocrObj = new PaddleOCR::PPOCR("./ch_PP-OCRv3_det_infer", "./ch_PP-OCRv3_rec_infer");
    //return new PaddleOCR::PPOCR("./ch_PP-OCRv3_det_infer", "./ch_PP-OCRv3_rec_infer");
}

void VisionAlgorithm::Ocr() {

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
    std::vector<PaddleOCR::OCRPredictResult> ocr_results =
        (*ocrObj).ocr(srcImage, true, true, false);
    std::cout << "doInference: " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - pre_start).count() / 1000.0 << "ms" << std::endl;
    std::vector<MatchResult> ocrresults;
    //for (int i = 0; i < img_list.size(); ++i) {
    //    
    //}
    for (int j = 0; j < ocr_results.size(); ++j) {
        if (ocr_results[j].score < 0.6) {
            continue;
        }
        MatchResult ocrresulttmp;
        std::vector<int> boxtmp;
        for (const auto& box : ocr_results[j].box) {
            boxtmp.insert(boxtmp.end(), box.begin(), box.end());
        }
        //rectangle(img_list[i], Point(boxtmp[0], boxtmp[1]), Point(boxtmp[4], boxtmp[5]), Scalar(0, 255, 255));
        //putText(img_list[i], ocr_results[i][j].text, Point(boxtmp[4], boxtmp[5]), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);
        memcpy(ocrresulttmp.point, &boxtmp[0], 8 * sizeof(int));
        //ocrresulttmp.box[2] = boxtmp[4] - boxtmp[0];
        //ocrresulttmp.box[3] = boxtmp[5] - boxtmp[1];
        strcpy_s(ocrresulttmp.text, ocr_results[j].text.c_str());
        //ocrresulttmp.text = &ocr_results[i][j].text[0];
        //ocrresulttmp.text = ocr_results[i][j].text.c_str();
        ocrresulttmp.textlen = strlen(ocr_results[j].text.c_str());
        cout << ocrresulttmp.text << endl;
        ocrresulttmp.score = ocr_results[j].score;
        ocrresults.push_back(ocrresulttmp);
    }
    resultlen = ocrresults.size();
    matchresult = new MatchResult[resultlen];
    memcpy(matchresult, &ocrresults[0], resultlen * sizeof(MatchResult));
}

//type:
//0:线拟合
//1:圆拟合
//2:椭圆拟合
void VisionAlgorithm::GetFindShapeData(TransPoint<int>* pt, int ptsize, int type) {
    cvPoints.clear();
    for (int i = 0; i < ptsize; ++i) {
        cvPoints.emplace_back(pt[i].x, pt[i].y);
    }
    typeValue = type;
}

GeometryData VisionAlgorithm::FitTool(TransPoint<double>* pt, int ptsize, int type) {
    GeometryData temp;
    vector<Point2f> cvPointstmp;
    for (int i = 0; i < ptsize; ++i) {
        cvPointstmp.emplace_back(pt[i].x, pt[i].y);
    }
    //FitCircle
    if (type == 1) {
        int num_points = cvPointstmp.size();
        Mat A(num_points, 3, CV_32F);
        Mat B(num_points, 1, CV_32F);

        for (int i = 0; i < num_points; ++i) {
            float x = cvPointstmp[i].x;
            float y = cvPointstmp[i].y;
            A.at<float>(i, 0) = -2 * x;
            A.at<float>(i, 1) = -2 * y;
            A.at<float>(i, 2) = 1;
            B.at<float>(i, 0) = -(x * x + y * y);
        }

        Mat X;
        solve(A, B, X, cv::DECOMP_SVD);

        temp.centerX = X.at<float>(0, 0);
        temp.centerY = X.at<float>(1, 0);
        float c = X.at<float>(2, 0);

        temp.radiusX = std::sqrt(temp.centerX * temp.centerX + temp.centerY * temp.centerY - c);
        temp.radiusY = temp.radiusX;
    }
    //FitEllipse
    else if (type == 2) {
        RotatedRect ellipse = fitEllipse(cvPointstmp);
        temp.centerX = ellipse.center.x;
        temp.centerY = ellipse.center.y;
        temp.radiusX = ellipse.size.width / 2.0;
        temp.radiusY = ellipse.size.height / 2.0;
        temp.angle = ellipse.angle;
    }
    //Line Fit
    else {
        Vec4f line;
        fitLine(cvPointstmp, line, cv::DIST_L2, 0, 0.01, 0.01);
        float vx = line[0];
        float vy = line[1];
        float x0 = line[2];
        float y0 = line[3];
        temp.angle = std::atan2(vy, vx) * 180.0 / PI;
        temp.centerX = x0;
        temp.centerY = y0;
    }
    return temp;
}

// 三次多项式拟合函数
cv::Vec4d cubicFit(const std::vector<double>& x, const std::vector<double>& y) {
    int n = x.size();
    double sum_x = 0.0, sum_x2 = 0.0, sum_x3 = 0.0, sum_x4 = 0.0;
    double sum_x5 = 0.0, sum_x6 = 0.0;
    double sum_y = 0.0, sum_xy = 0.0, sum_x2y = 0.0, sum_x3y = 0.0;

    for (int i = 0; i < n; ++i) {
        double xi = x[i];
        double xi2 = xi * xi;
        double xi3 = xi2 * xi;
        sum_x += xi;
        sum_x2 += xi2;
        sum_x3 += xi3;
        sum_x4 += xi2 * xi2;
        sum_x5 += xi3 * xi2;
        sum_x6 += xi3 * xi3;
        sum_y += y[i];
        sum_xy += xi * y[i];
        sum_x2y += xi2 * y[i];
        sum_x3y += xi3 * y[i];
    }

    cv::Matx44d A(sum_x6, sum_x5, sum_x4, sum_x3,
        sum_x5, sum_x4, sum_x3, sum_x2,
        sum_x4, sum_x3, sum_x2, sum_x,
        sum_x3, sum_x2, sum_x, n);
    cv::Vec4d B(sum_x3y, sum_x2y, sum_xy, sum_y);

    return A.inv() * B;  // 返回多项式系数 (a, b, c, d)
}

// 计算梯度变化最剧烈的点
void findMaxGradientChange(const std::vector<cv::Point2f>& linePoints, const std::vector<double>& grayValues, TransPoint<double>& darkToLight, TransPoint<double>& lightToDark) {
    int n = linePoints.size();
    std::vector<double> x(n);
    for (int i = 0; i < n; ++i) {
        x[i] = i;
    }

    cv::Vec4d polyParams = cubicFit(x, grayValues);

    // 三次多项式的一阶导数: dy/dx = 3ax^2 + 2bx + c
    double a = polyParams[0];
    double b = polyParams[1];
    double c = polyParams[2];

    double maxPositiveGradient = -std::numeric_limits<double>::infinity();
    double maxNegativeGradient = std::numeric_limits<double>::infinity();
    int maxPositiveIdx = 0;
    int maxNegativeIdx = 0;

    //for (int i = 0; i < n; ++i) {
    //    double gradient = 3 * a * x[i] * x[i] + 2 * b * x[i] + c;  // 计算每个点的梯度
    //    if (gradient > maxPositiveGradient) {
    //        maxPositiveGradient = gradient;
    //        maxPositiveIdx = i;
    //    }
    //    if (gradient < maxNegativeGradient) {
    //        maxNegativeGradient = gradient;
    //        maxNegativeIdx = i;
    //    }
    //}
    for (int i = 0; i < n - 1; i++) {
        double diff = grayValues[i] - grayValues[i + 1];
        if (diff > maxPositiveGradient) {
            maxPositiveGradient = diff;
            maxNegativeIdx = i;
        }
        if (diff < maxNegativeGradient) {
            maxNegativeGradient = diff;
            maxPositiveIdx = i;
        }
    }
    darkToLight = { linePoints[maxPositiveIdx].x, linePoints[maxPositiveIdx].y };
    lightToDark = { linePoints[maxNegativeIdx].x, linePoints[maxNegativeIdx].y };
    //darkToLight = { linePoints[maxPositiveIdx], maxPositiveGradient };
    //lightToDark = { linePoints[maxNegativeIdx], maxNegativeGradient };
}

double pointToPointDis(Point2f obj1, Point2f obj2) {
    double dx = obj1.x - obj2.x; // 计算x方向上的差值
    double dy = obj1.y - obj2.y; // 计算y方向上的差值
    return std::sqrt(dx * dx + dy * dy);
}

//findMode--寻找模式 0:从暗到亮；1：从亮到暗
//findDir --寻找方向 0:纵向寻找；1：横向寻找
GeometryData VisionAlgorithm::FindShape(int findMode, int findDir) {
    if (cvPoints.size() % 4 != 0) {
        GeometryData temp;
        temp.centerX = -1;
        return temp;
    }
    if (typeValue != 0) {
        Point2f tmp(0.0, 0.0);
        for (int i = 0; i < cvPoints.size(); i = i + 4) {
            tmp += cvPoints[i];
            if (i > 0) {
                tmp = tmp * 0.5;
            }
        }
        for (int i = 0; i < cvPoints.size(); i = i + 4) {
            double distmp1, distmp2, distmp3, distmp4;
            distmp1 = pointToPointDis(tmp, cvPoints[i]);
            distmp2 = pointToPointDis(tmp, cvPoints[i + 1]);
            distmp3 = pointToPointDis(tmp, cvPoints[i + 2]);
            distmp4 = pointToPointDis(tmp, cvPoints[i + 3]);
            if ((distmp1 > distmp3 && distmp2 < distmp4) || (distmp1 < distmp3 && distmp2 > distmp4)) {
                Point2f tmp1 = cvPoints[i];
                cvPoints[i] = cvPoints[i + 3];
                cvPoints[i + 3] = cvPoints[i + 2];
                cvPoints[i + 2] = cvPoints[i + 1];
                cvPoints[i + 1] = tmp1;

            }

        }
    }
    vector<TransPoint<double>> darkToLightPoints, lightToDarkPoints, optimalPoints;
    for (int i = 0; i < cvPoints.size(); ) {
        // 计算两个边的中点
        Point2f M1 = (cvPoints[i] + cvPoints[i + 1]) * 0.5;
        Point2f M2 = (cvPoints[i + 2] + cvPoints[i + 3]) * 0.5;
        if (findDir == 1 && typeValue == 0) {
            M1 = (cvPoints[i] + cvPoints[i + 3]) * 0.5;
            M2 = (cvPoints[i + 1] + cvPoints[i + 2]) * 0.5;
        }
        // 计算中点连线上的点的灰度值
        std::vector<cv::Point2f> linePoints;
        std::vector<double> grayValues;
        for (double t = 0; t <= 1.0; t += 0.001) {
            cv::Point2f pt = M1 * (1.0 - t) + M2 * t;
            linePoints.push_back(pt);
            grayValues.push_back(static_cast<double>(srcImage.at<uchar>(cv::Point(cvRound(pt.x), cvRound(pt.y)))));
        }

        // 找到两个变化最剧烈的点
        TransPoint<double> darkToLightPoint, lightToDarkPoint;
        findMaxGradientChange(linePoints, grayValues, darkToLightPoint, lightToDarkPoint);
        //if (darkToLightPoint.x == cvPoints[i].x || darkToLightPoint.x == cvPoints[i + 1].x ||
        //    darkToLightPoint.y == cvPoints[i].y || darkToLightPoint.y == cvPoints[i + 3].y) {
        //    optimalPoints.push_back(lightToDarkPoint);
        //}
        //else {
        //    optimalPoints.push_back(darkToLightPoint);
        //}
        if ((fabs(darkToLightPoint.x - M1.x) < 1.0 && fabs(darkToLightPoint.y - M1.y) < 1.0) ||
            (fabs(darkToLightPoint.x - M2.x) < 1.0 && fabs(darkToLightPoint.y - M2.y) < 1.0)) {
            optimalPoints.push_back(lightToDarkPoint);
        }
        else {
            optimalPoints.push_back(darkToLightPoint);
        }
        darkToLightPoints.push_back(darkToLightPoint);
        lightToDarkPoints.push_back(lightToDarkPoint);
        cout << "暗 to 亮 Max Change Point: (" << darkToLightPoint.x <<" ,"<<darkToLightPoint.y<<")"<<endl;
        cout << "亮 to 暗 Max Change Point: (" << lightToDarkPoint.x << " ," << lightToDarkPoint.y << ")" << endl;
      
        //cv::line(srcImagetmp, cvPoints[i], cvPoints[(i + 1)], cv::Scalar(0, 255, 0), 1);
        //cv::line(srcImagetmp, cvPoints[i+1], cvPoints[(i + 2)], cv::Scalar(0, 255, 0), 1);
        //cv::line(srcImagetmp, cvPoints[i+2], cvPoints[(i + 3)], cv::Scalar(0, 255, 0), 1);
        //cv::line(srcImagetmp, cvPoints[i + 3], cvPoints[(i)], cv::Scalar(0, 255, 0), 1);

        //// 在图像上绘制中点和连线
        //cv::line(srcImage, M1, M2, cv::Scalar(0, 255, 255), 1);
        //cv::circle(srcImage, M1, 1, cv::Scalar(0, 0, 255), 1);
        //cv::circle(srcImage, M2, 1, cv::Scalar(255, 0, 255), 1);

        //// 在图像上绘制变化最剧烈的点
        //cv::circle(srcImagetmp, cv::Point(darkToLightPoint.x, darkToLightPoint.y), 1, cv::Scalar(0, 255, 255), 1); // 黄色，大小为2
        //cv::circle(srcImagetmp, cv::Point(lightToDarkPoint.x, lightToDarkPoint.y), 1, cv::Scalar(255, 0, 255), 1); // 紫色，大小为2
        i = i + 4;
    }
    Mat srcImagetmp;
    cvtColor(srcImage, srcImagetmp, COLOR_GRAY2BGR);
    for (int i = 0; i < optimalPoints.size(); ++i) {
        cv::line(srcImagetmp, cvPoints[i*4], cvPoints[(i*4 + 1)], cv::Scalar(0, 255, 0), 1);
        cv::line(srcImagetmp, cvPoints[i * 4 +1], cvPoints[(i * 4 + 2)], cv::Scalar(0, 255, 0), 1);
        cv::line(srcImagetmp, cvPoints[i * 4 +2], cvPoints[(i * 4 + 3)], cv::Scalar(0, 255, 0), 1);
        cv::line(srcImagetmp, cvPoints[i * 4 + 3], cvPoints[(i * 4)], cv::Scalar(0, 255, 0), 1);
        cv::circle(srcImagetmp, cv::Point(optimalPoints[i].x, optimalPoints[i].y), 0, cv::Scalar(0, 0, 255), -1); // 黄色，大小为2
        //cv::circle(srcImagetmp, cv::Point(lightToDarkPoint.x, lightToDarkPoint.y), 1, cv::Scalar(255, 0, 255), 1); // 紫色，大小为2
    }

    if (findMode == 1) {
        return FitTool(lightToDarkPoints.data(), lightToDarkPoints.size(), typeValue);
    }
    else {
        return FitTool(darkToLightPoints.data(), darkToLightPoints.size(), typeValue);
    }
    //if (typeValue == 0) {
    //    if (findMode == 1) {
    //        return FitTool(lightToDarkPoints.data(), lightToDarkPoints.size(), typeValue);
    //    }
    //    else {
    //        return FitTool(darkToLightPoints.data(), darkToLightPoints.size(), typeValue);
    //    }
    //}
    //else {
    //    return FitTool(optimalPoints.data(), optimalPoints.size(), typeValue);
    //}


}

void VisionAlgorithm::GetCalcParam(int type) {
    typeValue = type;
}


void pointToLineDistance(GeometryData pointData, GeometryData lineData, double& distance, TransPoint<double>& closestPoint) {
    //TransPoint closestPoint;
    double lineAngleRadians = lineData.angle * PI / 180.0; // 将角度转换为弧度
    double slope;
    if (abs(abs(lineAngleRadians) - PI / 2) < 1e-2) { // 如果直线垂直于x轴
        closestPoint.x = lineData.centerX;
        distance = std::abs(pointData.centerX - closestPoint.x);
        closestPoint.y = pointData.centerY;
    }
    else { // 否则，计算直线的斜率
        slope = std::tan(lineAngleRadians);
        double a = slope;
        double b = -1;
        double c = lineData.centerY - slope * lineData.centerX;
        distance = std::abs(a * pointData.centerX + b * pointData.centerY + c) / std::sqrt(a * a + b * b); // 使用点到直线距离公式计算距离
        double t = std::cos(lineAngleRadians) * (pointData.centerX - lineData.centerX) + std::sin(lineAngleRadians)*(pointData.centerY - lineData.centerY);
        closestPoint.x = lineData.centerX + t * std::cos(lineAngleRadians); // 计算最近点的x坐标
        closestPoint.y = lineData.centerY + t * std::sin(lineAngleRadians); // 计算最近点的y坐标
    }
}

TransPoint<double> closestPointOnEllipse(GeometryData pointLineData, GeometryData ellipseData, int typeValue) {
    double minDistance = std::numeric_limits<double>::max();
    TransPoint<double> closestPoint;
    //椭圆角度以弧度为单位，逆时针方向
    double ellipseRad = ellipseData.angle* PI / 180.0;
    // 通过参数化椭圆方程来寻找最近点  
    TransPoint<double> closestPointInLine;
    for (double theta = 0; theta < 2 * PI; theta += 0.1) { // 以小步长遍历角度  
        double ellipseX = ellipseData.centerX + ellipseData.radiusX * cos(theta) * cos(ellipseRad) - ellipseData.radiusY * sin(theta)*sin(ellipseRad);
        double ellipseY = ellipseData.centerY + ellipseData.radiusX * cos(theta) * sin(ellipseRad) + ellipseData.radiusY * sin(theta) * cos(ellipseRad);
        double distance=-1.0;
        if (typeValue == 2) {
            distance = sqrt(pow(pointLineData.centerX - ellipseX, 2) + pow(pointLineData.centerY - ellipseY, 2));
        }
        else if (typeValue == 4) {
            GeometryData pointTmp;
            pointTmp.centerX = ellipseX;
            pointTmp.centerY = ellipseY;
            pointToLineDistance(pointTmp, pointLineData, distance, closestPointInLine);
        }


        // 更新最小距离和最近点  
        if (distance < minDistance) {
            minDistance = distance;
            closestPoint = { ellipseX, ellipseY };
        }
    }

    return closestPoint;
}

//typeValue：
//0:点点模式
//1:点线模式
//2:点圆模式
//3:线线模式
//4:线圆模式
void VisionAlgorithm::CalcDistance(GeometryData obj1, GeometryData obj2, double& distance) {
    switch (typeValue) {
    case 0: {
        double dx = obj1.centerX - obj2.centerX; // 计算x方向上的差值
        double dy = obj1.centerY - obj2.centerY; // 计算y方向上的差值
        distance = std::sqrt(dx * dx + dy * dy);
        break;
    }
    case 1: {
        GeometryData pointData;
        GeometryData lineData;
        if (obj1.angle >= -180.0 && obj1.angle <= 180.0) {
            lineData = obj1;
            pointData = obj2;
        }
        else {
            lineData = obj2;
            pointData = obj1;
        }
        TransPoint<double> closestPointInLine;
        pointToLineDistance(pointData, lineData, distance, closestPointInLine);
        break;
    }
    case 2: {
        GeometryData pointData = obj1;
        GeometryData circleData = obj2;
        TransPoint<double> closestPoint;
        if (abs(circleData.radiusX - circleData.radiusY) < 1e-2) {
            double distToCenter = std::hypot(pointData.centerX - circleData.centerX, pointData.centerY - circleData.centerY); // 计算点到圆心的距离
            
            if (distToCenter == 0) { // 如果点在圆心上
                distance = circleData.radiusX;
                closestPoint.x = circleData.centerX + circleData.radiusX;
                closestPoint.y = circleData.centerY;
            }
            else if (distToCenter < circleData.radiusX) { // 如果点在圆内
                distance = circleData.radiusX - distToCenter;
                double ratio = circleData.radiusX / distToCenter;
                closestPoint.x = circleData.centerX + (pointData.centerX - circleData.centerX) * ratio;
                closestPoint.y = circleData.centerY + (pointData.centerY - circleData.centerY) * ratio;
            }
            else if (distToCenter == circleData.radiusX) { // 如果点在圆上
                distance = 0;
                closestPoint.x = pointData.centerX;
                closestPoint.y = pointData.centerY;
            }
            else { // 如果点在圆外
                distance = distToCenter - circleData.radiusX;
                double ratio = circleData.radiusX / distToCenter;
                closestPoint.x = circleData.centerX + (pointData.centerX - circleData.centerX) * ratio;
                closestPoint.y = circleData.centerY + (pointData.centerY - circleData.centerY) * ratio;
            }
        }
        else {
            closestPoint = closestPointOnEllipse(pointData, circleData, typeValue);
            distance = sqrt(pow(pointData.centerX - closestPoint.x, 2) + pow(pointData.centerY - closestPoint.y, 2));
        }
        break;
    }

    case 3: {
        double distance1, distance2;
        TransPoint<double> closestPointInLine;
        pointToLineDistance(obj1, obj2, distance1, closestPointInLine);
        pointToLineDistance(obj2, obj1, distance2, closestPointInLine);
        distance = min(distance1, distance2);
        break;
    }

    case 4: {
        GeometryData lineData = obj1;
        GeometryData circleData = obj2;
        if (abs(circleData.radiusX - circleData.radiusY) < 1e-2) {
            double closestXOnLine, closestYOnLine;
            if (abs(lineData.angle - 90.0) < 1e-2) {
                closestXOnLine = lineData.centerX;
                closestYOnLine = circleData.centerY;

            }
            else {
                TransPoint<double> closestPointInLine;
                pointToLineDistance(GeometryData{ circleData.centerX, circleData.centerY}, lineData, distance, closestPointInLine);
                closestXOnLine = closestPointInLine.x;
                closestYOnLine = closestPointInLine.y;
            }
            double distToCenter = std::hypot(closestXOnLine - circleData.centerX, closestYOnLine - circleData.centerY);

            TransPoint<double> closestCirclePoint, closestLinePoint;
            if (distToCenter <= circleData.radiusX) { // 如果直线和圆相交或相切
                distance = circleData.radiusX - distToCenter;
                closestCirclePoint.x = circleData.centerX + (closestXOnLine - circleData.centerX) * (circleData.radiusX / distToCenter);
                closestCirclePoint.y = circleData.centerY + (closestYOnLine - circleData.centerY) * (circleData.radiusX / distToCenter);
            }
            else { // 如果直线和圆不相交
                distance = distToCenter - circleData.radiusX;
                double ratio = circleData.radiusX / distToCenter;
                closestCirclePoint.x = circleData.centerX + (closestXOnLine - circleData.centerX) * ratio;
                closestCirclePoint.y = circleData.centerY + (closestYOnLine - circleData.centerY) * ratio;
            }
            closestLinePoint.x = closestXOnLine; // 直线上的最近点坐标
            closestLinePoint.y = closestYOnLine; // 直线上的最近点坐标
        }
        else {
            TransPoint<double> closestPoint;
            closestPoint = closestPointOnEllipse(lineData, circleData, typeValue);
            GeometryData pointTmp;
            pointTmp.centerX = closestPoint.x;
            pointTmp.centerY = closestPoint.y;
            TransPoint<double> closestPointInLine;
            pointToLineDistance(pointTmp, lineData, distance, closestPointInLine);
        }

        break;
    }
    default:
        distance = -1;

    }
}


void VisionAlgorithm::CalcAngle(GeometryData obj1, GeometryData obj2, double& angle) {
    if (obj1.angle >= -180.0 && obj1.angle <= 180.0 && obj2.angle >= -180.0 && obj2.angle <= 180.0) {
        double lineAngleRadians1 = obj1.angle * PI / 180.0; // 将角度转换为弧度
        double lineAngleRadians2 = obj2.angle * PI / 180.0; // 将角度转换为弧度
        double angleRadians = std::fabs(lineAngleRadians1 - lineAngleRadians2); // 计算两个弧度之间的绝对差值

        if (angleRadians > PI) { // 如果角度差超过180度，则取补角
            angleRadians = 2 * PI - angleRadians;
        }

        angle = angleRadians * 180.0 / PI; // 将弧度转换回角度
    }
    else {
        angle = -100000.0;
    }
}

#include<zxing/LuminanceSource.h>
//#include<zxing/Reader.h>
#include <zxing/common/Counted.h>
#include <zxing/Binarizer.h>
#include <zxing/MultiFormatReader.h>
#include <zxing/Result.h>
#include <zxing/ResultPoint.h>
//#include <zxing/ReaderException.h>
#include <zxing/common/GlobalHistogramBinarizer.h>
//#include <zxing/Exception.h>
#include <zxing/common/IllegalArgumentException.h>
#include <zxing/BinaryBitmap.h>
#include <zxing/DecodeHints.h>
//#include <zxing/qrcode/QRCodeReader.h>
//#include <zxing/MatSource.h>
using namespace zxing;

Point toCvPoint(Ref<ResultPoint> resultPoint) {
    return Point(resultPoint->getX(), resultPoint->getY());
}

class OpenCVLuminanceSource : public LuminanceSource {
public:
    //static zxing::Ref<zxing::LuminanceSource> create(cv::Mat& cvImage) {
    //    return Ref<LuminanceSource>(new OpenCVLuminanceSource(cvImage));
    //}

    OpenCVLuminanceSource(cv::Mat& _cvImage) : LuminanceSource(_cvImage.cols, _cvImage.rows) {
        cvImage = _cvImage.clone();
    }

    ArrayRef<char> getRow(int y, zxing::ArrayRef<char> row) const {

        // Get width
        int width = getWidth();

        if (!row) {

            // Create row
            row = zxing::ArrayRef<char>(width);

        }

        // Get pointer to row
        const char* p = cvImage.ptr<char>(y);

        for (int x = 0; x < width; ++x, ++p) {

            // Set row at index x
            row[x] = *p;

        }

        return row;

    }

    ArrayRef<char> getMatrix() const {

        // Get width and height
        int width = getWidth();
        int height = getHeight();

        // Create matrix
        zxing::ArrayRef<char> matrix = zxing::ArrayRef<char>(width * height);

        for (int y = 0; y < height; ++y) {

            // Get pointer to row
            const char* p = cvImage.ptr<char>(y);

            // Calculate y offset
            int yoffset = y * width;

            for (int x = 0; x < width; ++x, ++p) {

                // Set row at index x with y offset
                matrix[yoffset + x] = *p;

            }

        }

        return matrix;

    }

private:
    cv::Mat cvImage;
    int width;
    int height;
};

void VisionAlgorithm::BarcodeRec() {
    Ref<LuminanceSource> source = Ref<LuminanceSource>(new OpenCVLuminanceSource(srcImage));
    MultiFormatReader reader;
    //Ref<Reader> reader;
    //bool multi = false;
    //if (multi) {
    //    reader.reset(new MultiFormatReader);
    //}
    //else {
    //    reader.reset(new qrcode::QRCodeReader);
    //}
    Ref<Binarizer> binarizer(new GlobalHistogramBinarizer(source));
    Ref<BinaryBitmap> bitmap(new BinaryBitmap(binarizer));
    DecodeHints hints(DecodeHints::DEFAULT_HINT);
    hints.setTryHarder(true);
    //hints.addFormat(BarcodeFormat::CODE_39);
    //hints.addFormat(BarcodeFormat::CODE_128);
    Ref<Result> result = reader.decode(bitmap, hints);
    // 输出找到的条形码  
    int resultPointCount = result->getResultPoints()->size();
    MatchResult resulttmp[1];
    for (int j = 0; j < resultPointCount; j++) {
        resulttmp[0].point[j * 2] = int(result->getResultPoints()[j]->getX());
        resulttmp[0].point[j * 2 + 1] = int(result->getResultPoints()[j]->getY());
        // Draw circle
        circle(srcImage, toCvPoint(result->getResultPoints()[j]), 10, Scalar(110, 220, 0), 2);
    }
    strcpy_s(resulttmp[0].text, result->getText()->getText().c_str());
    resultlen = 1;
    matchresult = new MatchResult[resultlen];
    memcpy(matchresult, &resulttmp[0], resultlen * sizeof(MatchResult));
    if (resultPointCount > 0) {
        // Draw text
        putText(srcImage, result->getText()->getText(), Point(5, 30), FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 255));
    }
    cout << result->getText()->getText();
}

void VisionAlgorithm::Release() {
    if (matchresult) {
        delete[] matchresult;
        matchresult = nullptr;
    }
    if (blobresult) {
        delete[] blobresult;
        blobresult = nullptr;
    }
}

struct CallbackParams {
    Mat srcImage;
    Mat templateImage;
};
void TemplateMatching(int MatchMethod, void* userData)
{
    CallbackParams* params = static_cast<CallbackParams*>(userData);
    Mat result;
    int result_cols = params->srcImage.cols - params->templateImage.cols + 1;
    int result_rows = params->srcImage.rows - params->templateImage.rows + 1;
    if (result_cols < 0 || result_rows < 0)
    {
        cout << "Please input correct image!";
        return;
    }
    result.create(result_cols, result_rows, CV_32FC1);
    //    enum { TM_SQDIFF=0, TM_SQDIFF_NORMED=1, TM_CCORR=2, TM_CCORR_NORMED=3, TM_CCOEFF=4, TM_CCOEFF_NORMED=5 };
    matchTemplate(params->srcImage, params->templateImage, result, MatchMethod);   //最好匹配为1,值越小匹配越差
    double minVal = -1;
    double maxVal;
    Point minLoc;
    Point maxLoc;
    Point matchLoc;
    double matchVal;
    //normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());
    minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
    if (MatchMethod == TM_SQDIFF || MatchMethod == TM_SQDIFF_NORMED)
    {
        matchLoc = minLoc;
        matchVal = minVal;
    }
    else
    {
        matchLoc = maxLoc;
        matchVal = maxVal;
    }


    //取大值(视匹配方法而定)
   //    matchLoc = minLoc;

    //取大值,值越小表示越匹配
    //string str = "Similarity:" + to_string(round(matchVal * 100)) + "%";
    //cout << str << endl;
    Mat mask = params->srcImage.clone(); //绘制最匹配的区域

    double threshold = 0.95; // 设置阈值

    //vector<Point> matches;

    //while (true)
    //{
    //    minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
    //    Scalar x = Scalar(1);
    //    if (MatchMethod == TM_SQDIFF || MatchMethod == CV_TM_SQDIFF_NORMED)
    //    {
    //        matchLoc = minLoc;
    //        matchVal = 1-minVal;
    //    }
    //    else
    //    {
    //        matchLoc = maxLoc;
    //        matchVal = maxVal;
    //        x= Scalar(0);
    //    }
    //    if (matchVal >= threshold)
    //    {
    //        cout << "add one" << endl;
    //        matches.push_back(matchLoc);
    //        rectangle(result, matchLoc, Point(matchLoc.x + params->templateImage.cols, matchLoc.y + params->templateImage.rows), x, -1);
    //        rectangle(mask, matchLoc, Point(matchLoc.x + params->templateImage.cols, matchLoc.y + params->templateImage.rows), Scalar(0, 255, 0), 2, 8, 0);
    //    }
    //    else
    //    {
    //        break;
    //    }
    //}
    if (matchVal >= threshold) {
        rectangle(mask, matchLoc, Point(matchLoc.x + params->templateImage.cols, matchLoc.y + params->templateImage.rows), Scalar(0, 255, 0), 2, 8, 0);
        rectangle(result, matchLoc, Point(matchLoc.x + params->templateImage.cols, matchLoc.y + params->templateImage.rows), Scalar(0, 255, 0), 2, 8, 0);
        imshow("原始图", mask);
        imshow("result", result);
    }
}

uchar* SendImage(int& width, int& height, int& stride)
{
    uchar* imageBuffer;
    Mat frame = imread(R"(C:\Users\rs\Desktop\tmpimg.png)");    // 创建Mat图像，并进行一系列图像处理

    Mat NewImage = cv::Mat::zeros(frame.rows, (frame.cols + 3) / 4 * 4, frame.type());
    frame.copyTo(NewImage(Rect(0, 0, frame.cols, frame.rows)));

    width = NewImage.cols;
    height = NewImage.rows;
    stride = static_cast<int>(NewImage.step);
    // malloc返回类型为void*需要类型转换
    imageBuffer = static_cast<uchar*>(malloc(width * height * 3 * sizeof(uchar)));
    if (imageBuffer != NULL)   // 用于判断内存空间是否申请成功
    {
        for (int i = 0; i < width * height * 3; i++)
        {
            imageBuffer[i] = (uchar)NewImage.data[i];
        }
    }
    return imageBuffer;
}

#include <filesystem>
void OCRTrain() {
    const int ImgWidth = 20;//图片宽
    const int ImgHeight = 30;//图片高
    Mat Train_Chars = imread(R"(C:\Users\rs\Desktop\ocrtrain1.png)");//数据集图像
    if (Train_Chars.empty())
    {
        cout << "can not read the image..." << endl;
        return;
    }
    string filename = "modelnew.xml";//模型文件
    vector<int>ValidChars =
    { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
    'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
    'u', 'v', 'w', 'x', 'y', 'z',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z' };

    //进行图像预处理，提取字符轮廓
    Mat grayImg;
    cvtColor(Train_Chars, grayImg, COLOR_BGR2GRAY);

    Mat blurImg;
    GaussianBlur(grayImg, blurImg, Size(3, 3), 0);

    Mat binImg;
    threshold(blurImg, binImg, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);

    vector<vector<Point>>contours;
    findContours(binImg, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    Mat Train_Data, Train_Label;
    //for (int cnt = 0; cnt < contours.size(); cnt++)
    //{
    //    if (contourArea(contours[cnt]) > 10)
    //    {
    //        Rect rect = boundingRect(contours[cnt]);
    //        rect = rect + Point(-5, -5);
    //        rect = rect + Size(10, 10);
    //        rectangle(Train_Chars, rect, Scalar(0, 255, 0), 2);
    //        Mat ROI = binImg(rect);
    //        imshow("ROI", ROI);
    //        imshow("Training_Chars", Train_Chars);
    //        int charVal = waitKey(0); //将字符通过键盘输入给予标签
    //        if (find(ValidChars.begin(), ValidChars.end(), charVal) != ValidChars.end())
    //        {
    //            Mat resizeRoi;
    //            resize(ROI, resizeRoi, Size(ImgWidth, ImgHeight));
    //            //将图像转成浮点型，因为KNN训练数据集读取的是浮点型数据
    //            Mat RoiFloat;
    //            resizeRoi.convertTo(RoiFloat, CV_32FC1);
    //            Train_Data.push_back(RoiFloat.reshape(0, 1));
    //            Train_Label.push_back(charVal);
    //            cout << charVal << endl;
    //        }
    //    }
    //}
    string pattern = R"(E:\dataset\chardata\)";
    std::vector<cv::Mat> images;
    for (const auto& entry : std::filesystem::directory_iterator(pattern)) {
        if (entry.is_directory()) {
            std::vector<cv::String> imgfiles;
            string labelstring = entry.path().string().substr(entry.path().string().rfind("\\") + 1);
            int label = stoi(labelstring);
            cv::glob(entry.path().string(), imgfiles, false);
            for (auto imgfile : imgfiles) {
                Mat img = imread(imgfile, 0);
                Rect rect(0, 0, img.cols, img.rows);
                rect = rect + Point(5, 5);
                rect = rect + Size(-10, -10);
                Mat roi, resizeRoi;
                roi = img(rect);
                resize(roi, resizeRoi, Size(ImgWidth, ImgHeight));
                Mat RoiFloat;
                resizeRoi.convertTo(RoiFloat, CV_32FC1);
                Train_Data.push_back(RoiFloat.reshape(0, 1));
                Train_Label.push_back(label);
            }
        }
    }


    //进行KNN训练
    Train_Data.convertTo(Train_Data, CV_32FC1);
    Train_Label.convertTo(Train_Label, CV_32FC1);
    const int k = 3;//k取值，基数
    Ptr<ml::KNearest>knn = ml::KNearest::create();//构造KNN模型
    knn->setDefaultK(k);//设定k值
    knn->setIsClassifier(true);//KNN算法可用于分类，回归
    knn->setAlgorithmType(ml::KNearest::BRUTE_FORCE);//字符匹配算法
    knn->train(Train_Data, ml::ROW_SAMPLE, Train_Label);//模型训练
    Mat results;
    knn->predict(Train_Data, results);
    knn->save(filename);//模型保存

    cout << "Model training is complete!" << endl;
}

void OCR(MatchResult*& result, int& resultlen, uchar* srcdata, int srcwidth, int srcheight, int srcstride)
{
    const int ImgWidth = 20;
    const int ImgHeight = 30;
    //准备数据集
    //Mat src = imread(R"(C:\Users\rs\Desktop\ocr2.png)");//测试图像
    Mat src = Mat(Size(srcwidth, srcheight), CV_8UC3, srcdata, srcstride).clone();
    if (src.empty())
    {
        cout << "can not read the image..." << endl;
        return;
    }

    string filename = "modelnew.xml";//模型文件
    fstream fin;
    fin.open(filename, ios::in);
    if (fin.is_open())
    {
        Ptr<ml::KNearest>knn = cv::Algorithm::load<cv::ml::KNearest>(filename);//加载KNN模型

        //图像预处理
        Mat grayImg;
        cvtColor(src, grayImg, COLOR_BGR2GRAY);

        Mat blurImg;
        GaussianBlur(grayImg, blurImg, Size(3, 3), 0);

        Mat binImg;
        threshold(blurImg, binImg, 128, 255, THRESH_BINARY);

        vector<vector<Point>>contours;
        findContours(binImg, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        vector<MatchResult> matchResult;
        for (int cnt = 0; cnt < contours.size(); cnt++)
        {
            if (contourArea(contours[cnt]) > 10)
            {
                Rect rect = boundingRect(contours[cnt]);
                //rectangle(src, rect, Scalar(0, 0, 255), 2);
                Mat ROI = binImg(rect);

                Mat resizeRoi;
                resize(ROI, resizeRoi, Size(ImgWidth, ImgHeight), INTER_NEAREST);

                Mat RoiFloat;//将图像转化成CV_32FC1
                resizeRoi.convertTo(RoiFloat, CV_32FC1);
                RoiFloat = RoiFloat.reshape(0, 1);

                float f = knn->predict(RoiFloat);//进行字符识别预测
                MatchResult matchResulttmp;
                vector<Point> points;
                points.push_back(rect.tl());
                points.push_back(Point(rect.tl().x, rect.br().y));
                points.push_back(rect.br());
                points.push_back(Point(rect.tl().y, rect.br().x));
                for (int j = 0; j < points.size(); j++)
                {
                    matchResulttmp.point[2 * j] = points[j].x;
                    matchResulttmp.point[2 * j + 1] = points[j].y;
                }
                matchResulttmp.score = f;
                matchResult.push_back(matchResulttmp);
                //结果显示
                char text[50];
                sprintf_s(text, "%c", char(int(f)));
                cout << char(int(f)) << " ";//将字符结果float转成char
                double scale = rect.width * 0.05;
                putText(src, text, rect.br(), FONT_HERSHEY_SIMPLEX, scale, Scalar(0, 255, 0), 1);

            }
        }
        resultlen = matchResult.size();
        result = new MatchResult[resultlen];
        memcpy(result, &matchResult[0], resultlen * sizeof(MatchResult));
        cout << endl;
    }
    else
    {
        cout << "can not read the ocr file..." << endl;
        return;
    }
}

cv::Mat addConstantToImage(const cv::Mat& image, int constant) {
    cv::Mat result = image.clone();
    for (int y = 0; y < result.rows; y++) {
        for (int x = 0; x < result.cols; x++) {
            if (result.channels() == 3) {
                for (int c = 0; c < result.channels(); c++) {
                    result.at<cv::Vec3b>(y, x)[c] = cv::saturate_cast<uchar>(result.at<cv::Vec3b>(y, x)[c] + constant);
                }
            }
            else {
                result.at<uchar>(y, x) = cv::saturate_cast<uchar>(result.at<uchar>(y, x) + constant);
            }

        }
    }
    return result;
}
//3*3卷积
//cv::Mat kernel = (cv::Mat_<float>(3, 3) <<
//0, -1, 0,
//-1, 5, -1,
//0, -1, 0);
// 定义一个NxM卷积核
//int N = 5; // 例如，设置N为5
//int M = 7; // 例如，设置M为7
//cv::Mat kernel = cv::Mat::ones(N, M, CV_32F) / (float)(N * M); // 简单的平均滤波器x
cv::Mat applyConvolution(const cv::Mat& image, cv::Mat& kernel) {
    cv::Mat result;
    cv::filter2D(image, result, -1, kernel);
    return result;
}
//均衡
cv::Mat equalizeImage(const cv::Mat& image) {
    cv::Mat result;
    if (image.channels() == 3) {
        cv::Mat ycrcb;
        cv::cvtColor(image, ycrcb, cv::COLOR_BGR2YCrCb);

        std::vector<cv::Mat> channels;
        cv::split(ycrcb, channels);

        cv::equalizeHist(channels[0], channels[0]);

        cv::merge(channels, ycrcb);
        cv::cvtColor(ycrcb, result, cv::COLOR_YCrCb2BGR);
    }
    else {
        cv::equalizeHist(image, result);
    }
    return result;
}
//扩展
cv::Mat resizeImage(const cv::Mat& image, double scaleX, double scaleY) {
    cv::Mat resizedImage;
    cv::resize(image, resizedImage, cv::Size(), scaleX, scaleY, cv::INTER_LINEAR);
    return resizedImage;
}
//翻转
cv::Mat flipImage(const cv::Mat& image, int flipway) {
    cv::Mat result;
    cv::flip(image, result, flipway); // 1表示水平翻转,//0表示垂直翻转，//-1表示水平和垂直翻转
    return result;
}
//顺时针旋转
cv::Mat rotateImage(const cv::Mat& image, double angle) {
    // 计算旋转中心
    cv::Point2f center(image.cols / 2.0, image.rows / 2.0);

    // 获取旋转矩阵
    cv::Mat rotationMatrix = cv::getRotationMatrix2D(center, angle, 1.0);

    // 计算旋转后的图像边界
    cv::Rect2f bbox = cv::RotatedRect(cv::Point2f(), image.size(), angle).boundingRect2f();

    // 调整旋转矩阵
    rotationMatrix.at<double>(0, 2) += bbox.width / 2.0 - image.cols / 2.0;
    rotationMatrix.at<double>(1, 2) += bbox.height / 2.0 - image.rows / 2.0;

    // 执行仿射变换
    cv::Mat result;
    cv::warpAffine(image, result, rotationMatrix, bbox.size());

    return result;
}
//高斯采样
//scaleX，Y：调整X和Y方向的采样因子，kernelSizeX，Y：平滑参数，sigmaX，Y:高斯模糊的sigma值
cv::Mat gaussianSampling(const cv::Mat& inputImage,
    double scaleX, double scaleY,
    int kernelSizeX, int kernelSizeY,
    double sigmaX, double sigmaY) {

    // 确保内核大小为奇数
    if (kernelSizeX % 2 == 0 || kernelSizeY % 2 == 0) {
        std::cerr << "内核宽度和高度必须为奇数" << std::endl;
        return cv::Mat();
    }

    // 检查输入图像是否为空
    if (inputImage.empty()) {
        std::cerr << "输入图像为空" << std::endl;
        return cv::Mat();
    }

    // 高斯模糊
    cv::Mat blurredImage;
    cv::GaussianBlur(inputImage, blurredImage, cv::Size(kernelSizeX, kernelSizeY), sigmaX, sigmaY);

    // 调整图像大小
    cv::Mat outputImage;
    cv::resize(blurredImage, outputImage, cv::Size(), scaleX, scaleY, cv::INTER_LINEAR);

    return outputImage;
}

//高通过滤器
cv::Mat applyFilter(const cv::Mat& src, FilterType filterType, int kernelWidth, int kernelHeight) {
    // 确保内核大小为奇数
    if (kernelWidth % 2 == 0 || kernelHeight % 2 == 0) {
        cerr << "内核宽度和高度必须为奇数" << endl;
        return cv::Mat();
    }
    cv::Mat dst;
    switch (filterType) {
    case GAUSSIAN:
        GaussianBlur(src, dst, cv::Size(kernelWidth, kernelHeight), 0);
        break;
    case MEAN:
        blur(src, dst, cv::Size(kernelWidth, kernelHeight));
        break;
    case MEDIAN:
        medianBlur(src, dst, kernelWidth); // medianBlur 只接受单一尺寸
        break;
    default:
        cerr << "未知的滤波类型" << endl;
        return cv::Mat();
    }

    return dst;
}



//中值操作
cv::Mat medianFilter(const cv::Mat& inputImage, int kernelSizeX, int kernelSizeY) {
    if (kernelSizeX % 2 == 0 || kernelSizeY % 2 == 0) {
        std::cerr << "核大小必须为奇数" << std::endl;
        return cv::Mat();
    }
    cv::Mat outputImage;
    cv::medianBlur(inputImage, outputImage, std::max(kernelSizeX, kernelSizeY));
    return outputImage;
}


//量化
cv::Mat quantizeImage(const cv::Mat& image, int levels) {
    cv::Mat result = image.clone();
    int step = 256 / levels; // 每个量化级别的步长

    for (int y = 0; y < result.rows; y++) {
        for (int x = 0; x < result.cols; x++) {
            if (image.channels() == 1) {
                // 处理灰度图像
                int pixelValue = image.at<uchar>(y, x);
                int quantizedValue = (pixelValue / step) * step + step / 2;
                result.at<uchar>(y, x) = cv::saturate_cast<uchar>(quantizedValue);
            }
            else if (image.channels() == 3) {
                // 处理彩色图像
                for (int c = 0; c < 3; c++) {
                    int pixelValue = image.at<cv::Vec3b>(y, x)[c];
                    int quantizedValue = (pixelValue / step) * step + step / 2;
                    result.at<cv::Vec3b>(y, x)[c] = cv::saturate_cast<uchar>(quantizedValue);
                }
            }
        }
    }
    return result;
};


