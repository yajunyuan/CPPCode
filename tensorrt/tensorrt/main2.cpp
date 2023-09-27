#include<stdio.h>
#include<iostream>
#include<opencv2/opencv.hpp>
#include <time.h>
#include <opencv2\opencv.hpp>
#include "gpucal.h"
#include<mutex>
#include <thread>
using namespace std;
using namespace cv;
int* imgArray;
cv::Mat img;
int* result;
double upper, lower;
void ReadImgarray(std::string imgfile)
{
	cv::Mat imgreal = cv::imread(imgfile, 0);
	//cv::Rect roi(900, 0, 14100, 2000);
	//cv::Mat img;
	//img = imgreal(roi);
	//cv::Mat mask,dst;
	//cv::SimpleBlobDetector::Params params;

	//// Change thresholds
	//params.minThreshold = 10;
	//params.thresholdStep = 10;
	//params.maxThreshold = 100;
	//params.filterByArea = true;
	//params.minArea = 100;
	//Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
	//cv::threshold(img, mask, 200, 1, cv::THRESH_BINARY);
	//cv::bitwise_and(img, mask, dst);
	//cv::imshow("test", img);
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

void pinjieimg(string path)
{
	std::vector<cv::Mat> imgs;
	vector<String> filepaths;
	glob(path, filepaths, false);
	cout << filepaths.size() << endl;
	for(auto x : filepaths) {
		string suffix_str = x.substr(x.find_last_of('.') + 1);
		if ((suffix_str == "JPG") || (suffix_str == "png") || suffix_str == "bmp" || suffix_str == "jpeg")
		{
			cv::Mat img = cv::imread(string(x), 0);
			if (img.cols == 88 && img.rows == 83) {
				imgs.push_back(img);
			}
			else {
				Size dsize = Size(500, 100);
				Mat shrink;
				resize(img, shrink, dsize, 0, 0, INTER_AREA);
				imgs.push_back(shrink);
			}
		}
	}
	cv::Mat output;
	cv::vconcat(imgs, output);
	imwrite(path + "pinjie.png", output);
}
vector<int> cpuresult;
void SearchDefect(int* da, int row_a) {
	int SingleImage_count = 0;
	for (auto i = row_a; i < row_a + 100; i++) {
		for (auto j = 0; j < img.cols-1; j++) {
			int dis = pow((da[i * img.cols + j] - da[i * img.cols + j + 1]), 2);
			if (dis >= 256) {
				//result[i * img.rows+ j] = 1;
				SingleImage_count++;
				//printf("row is %d  %d ", row, col);
			}
			//else {
			//	result[i * img.rows + j] = 0;
			//}
		}
	}
	if (SingleImage_count >= 5)
	{
		mutex mu;
		mu.lock();
		cpuresult.push_back(row_a);
		//cpuresult.push_back(col_a);
		mu.unlock();
	}
}

void cpucal() {
	int* imgarraytmp = new int[img.rows * (img.cols - 1)];
	int threadidsy = int(img.rows / 100);
	int threadidsx = int(img.cols / 100);
	vector<thread> mythreads;
	for (auto i = 0; i < threadidsy; i++) {
		//for (auto j = 0; j < threadidsx; j++) {
			mythreads.push_back(thread(SearchDefect, imgArray, i*100));
/*		}*/	
	}
	auto start = std::chrono::system_clock::now();
	for(auto& mythread : mythreads) {
		mythread.join();
	}
	cout << cpuresult.size() << endl;
	std::cout << "cpu多线程主程序：" << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start).count() << "ms" << std::endl;
	for (auto it = cpuresult.begin(); it != cpuresult.end(); it = it + 2)
		cout << *it << " " << *(it + 1)<< endl;
}

void gpucal() {
	GPUCAL gpucal(img.rows, img.cols);
	result = new int[RESULTSIZE];
	auto start = std::chrono::system_clock::now();
	gpucal.MatCal(imgArray, result, 225, 199);
	cout << result[0] << endl;
	std::cout << "gpu多线程主程序：" << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start).count() << "ms" << std::endl;
	Mat imgcut;
	string outpath = "D:\\C#\\DefectCut\\";
	system("del D:\\C#\\DefectCut\\*.png  /a /s /f /q");
	for (int i = 1; i < result[0]+1; i = i+2) {
		cout << result[i] << "  " << result[i + 1] << endl;
		Rect rect(result[i + 1], result[i], 100, 100);
		imgcut = img(rect);
		imwrite(outpath + to_string(result[i]) +"_"+to_string(result[i+1]) + ".png", imgcut);
	}
	delete[] result;
}

void main22()
{
	ReadImgarray("D://C#//xiaci_pinjie_1209.bmp");//(R"(D:\C#\瑕疵检测资料\清洗出拿来训练的图像\pinjiepinjie.png)");//("D://C#//xiaci_pinjie_1209.bmp");
	//pinjieimg("D:/C#/瑕疵检测资料/清洗出拿来训练的图像/pinjie");
	//cpucal();
	gpucal();
	//double minValue, maxValue;
	//cv::Point  minIdx, maxIdx;
	//cv::Mat imgreal = cv::imread("D://C#//DefectCut//0_0.png", 0);
	//cv::minMaxLoc(imgreal, & minValue, &maxValue, &minIdx, &maxIdx);
	//std::cout << "最大值：" << maxValue << "最小值：" << minValue << std::endl;
	//std::cout << "最大值位置：" << maxIdx << "最小值位置：" << minIdx;
	//vector<Vec4i> hierarchy;
	//vector<vector<Point>> contours;
	//auto start = std::chrono::system_clock::now();
	//findContours(imgreal, contours, hierarchy, CV_RETR_LIST, CHAIN_APPROX_NONE, Point());
	//std::cout << contours.size() << endl;
	//std::cout << "opencv 找轮廓：" << 
	//	std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - 
	//		start).count() << "ms" << std::endl;
}