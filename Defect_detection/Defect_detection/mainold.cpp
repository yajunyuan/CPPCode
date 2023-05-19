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
	auto start = std::chrono::system_clock::now();
	img = imgreal;

	//Mat imgDark;
	//imgDark.create(imgreal.size().height /2, imgreal.size().width, imgreal.type());
	//img.create(imgreal.size().height / 2, imgreal.size().width, imgreal.type());
	//for (auto i = 0; i < imgreal.size().height-1; i++) {
	//	uchar* sdata = imgreal.data + imgreal.step * i;
	//	uchar* ddata = imgDark.data + imgDark.step * (i/2);
	//	uchar* bdata = img.data + img.step * (i / 2);
	//	if (i%2==0) {
	//		for (auto j = 0; j < imgreal.size().width; j++) {
	//			bdata[j] = sdata[j];
	//		}
	//	}
	//	else {
	//		for (auto j = 0; j < imgreal.size().width; j++) {
	//			ddata[j] = sdata[j];
	//		}
	//	}
	//}

	//cv::Rect roi(900, 0, 14100, 2000);
	//img = imgreal(roi);
	//cout << imgreal.rows << "  " << imgreal.cols << endl;

	/*img = imgreal;
	auto start = std::chrono::system_clock::now();
	imgArray = new int[img.rows * img.cols];
	for (int i = 0; i < img.rows; i++) {
		uchar* ptr = img.ptr<uchar>(i);
		for (int j = 0; j < img.cols; j++) {
			imgArray[i * img.cols + j] = int(*(ptr + j));
		}
	}	*/
	std::cout << "图转换：" << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count() / 1000.0 << "ms" << std::endl;
}

void pinjieimg(string path)
{
	std::vector<cv::Mat> imgs;
	vector<String> filepaths;
	glob(path, filepaths, false);
	cout << filepaths.size() << endl;
	for (auto x : filepaths) {
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
		for (auto j = 0; j < img.cols - 1; j++) {
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
struct MyPoint
{
	double x;
	double y;
};

void SearchDefect1(int* da, string path, int& count) {
	auto start = std::chrono::system_clock::now();
	vector<MyPoint> cpuresult1;
	int SingleImage_count = 0;
	for (auto i = 0; i < img.rows; i++) {
		for (auto j = 0; j < img.cols - 5; j++) {
			int dis = pow((da[i * img.cols + j] - da[i * img.cols + j + 5]), 2);
			if (dis >= 144) {
				MyPoint point;
				point.x = i;
				point.y = j + 1;
				cpuresult1.push_back(point);
				//result[i * img.rows+ j] = 1;
				//SingleImage_count++;
				//printf("row is %d  %d ", row, col);
			}
			//else {
			//	result[i * img.rows + j] = 0;
			//}
		}
	}
	std::cout << "cpu找瑕疵：" << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count() / 1000.0 << "ms" << std::endl;
	if (cpuresult1.empty()) {
		count++;
		cout << path << "  未找到" << endl;
		return;
	}
	Mat imgRGB;
	cvtColor(img, imgRGB, COLOR_GRAY2RGB);
	for (auto i : cpuresult1) {
		imgRGB.at<Vec3b>(i.x, i.y) = Vec3b(0, 0, 255);
	}
	size_t  b = path.find_last_of("\\");
	imwrite("D:\\C#\\瑕疵检测资料\\test\\" + path.substr(b) + ".jpg", imgRGB);
}

void cpucal() {
	int* imgarraytmp = new int[img.rows * (img.cols - 1)];
	int threadidsy = int(img.rows / 100);
	int threadidsx = int(img.cols / 100);
	vector<thread> mythreads;
	for (auto i = 0; i < threadidsy; i++) {
		//for (auto j = 0; j < threadidsx; j++) {
		mythreads.push_back(thread(SearchDefect, imgArray, i * 100));
		/*		}*/
	}
	auto start = std::chrono::system_clock::now();
	for (auto& mythread : mythreads) {
		mythread.join();
	}
	cout << cpuresult.size() << endl;
	std::cout << "cpu多线程主程序：" << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start).count() << "ms" << std::endl;
	for (auto it = cpuresult.begin(); it != cpuresult.end(); it = it + 2)
		cout << *it << " " << *(it + 1) << endl;
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
	for (int i = 1; i < result[0] + 1; i = i + 2) {
		cout << result[i] << "  " << result[i + 1] << endl;
		Rect rect(result[i + 1], result[i], 100, 100);
		imgcut = img(rect);
		imwrite(outpath + to_string(result[i]) + "_" + to_string(result[i + 1]) + ".png", imgcut);
	}
	delete[] result;
}

void threshold(Mat src, Mat& dst, int minthreshold, int maxthreshold, int blockSize) {
	Mat mean;
	dst.create(src.size(), src.type());
	//blur(img, mean, Size(blockSize, 1));
	//boxFilter(src, mean, src.type(), Size(blockSize, blockSize), Point(-1, -1), true, BORDER_REPLICATE);
	for (auto i = 0; i < src.size().height; i++) {
		uchar* sdata = src.data + src.step * i;
		uchar* mdata = mean.data + mean.step * i;
		uchar* ddata = dst.data + dst.step * i;

		for (auto j = 0; j < src.size().width; j++) {
			if (sdata[j] < minthreshold || sdata[j] >= maxthreshold) {
				ddata[j] = 255;
				continue;
			}
			bool blurflag = (abs(sdata[j] - minthreshold) < 5) || (abs(sdata[j] - maxthreshold) < 5);
			if (j > 1 && blurflag && abs(sdata[j + 1] - sdata[j - 2]) >= 12) {//abs(sdata[j+1]- sdata[j-2])>=12)   //abs(mdata[j] - mdata[j-1])>=4 )
				//uchar* psrc0= src.data + src.step * (i - 1);
				//uchar* psrc2 = src.data + src.step * (i + 1);
				//psrc0[]
				ddata[j] = 255;
				continue;
			}
			ddata[j] = 0;
		}
	}

}

void findcounter(int threadid, Mat threadimg, vector<double>& outputinfo)
{
	auto starttmp = std::chrono::system_clock::now();
	Mat canny_img;
	auto start = std::chrono::system_clock::now();
	//Scalar mean1 = mean(img);
	//cv::threshold(img, canny_img, 132, 255, THRESH_BINARY);//二值化阈值处理
	//GaussianBlur(img, canny_img, Size(5, 5), 0, 0);
	//threshold(img, canny_img, 115, 135, 3);
	threshold(threadimg, canny_img, 115, 140, 3);
	//Mat dst;
	//Mat kernel = getStructuringElement(0, Size(2, 2));
	//morphologyEx(canny_img, dst, MORPH_OPEN, kernel);//开操作

	std::cout << "  opencv threshold：" <<
		std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() -
			start).count() << "ms" << std::endl;

	start = std::chrono::system_clock::now();
	Mat labels, stats, centroids;
	int num = connectedComponentsWithStats(canny_img, labels, stats, centroids);
	std::cout << "  connect state：" <<
		std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() -
			start).count() << "ms" << std::endl;

	vector<int> totaldata(num);
	start = std::chrono::system_clock::now();
	labels.convertTo(labels, CV_8UC1);
	for (auto i = 0; i < labels.size().height; i++) {
		uchar* labelsdata = labels.data + labels.step * i;
		uchar* imgdata = threadimg.data + threadimg.step * i;
		for (auto j = 0; j < labels.size().width; j++) {
			if (labelsdata[j] == 0) continue;
			totaldata[labelsdata[j]] += imgdata[j];
		}
	}
	std::cout << "  cal total avg gray：" <<
		std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() -
			start).count() << "ms" << std::endl;

	start = std::chrono::system_clock::now();
	vector<double> output;
	output.resize((num - 1) * 8);
	for (int i = 1; i < num; i++) {
		Vec2d pt = centroids.at<Vec2d>(i, 0);
		double x = stats.at<int>(i, CC_STAT_LEFT);
		double y = stats.at<int>(i, CC_STAT_TOP);
		double width = stats.at<int>(i, CC_STAT_WIDTH);
		double height = stats.at<int>(i, CC_STAT_HEIGHT);
		double area = stats.at<int>(i, CC_STAT_AREA);
		//cout << i << "个： len: " << max(width,height) << "  wid:" << min(width, height) << " rate: " << 
		//	max(width, height) / min(width, height) << "  area: " << area << "  avg pixel: "<< totaldata[i]/ area << endl;
		//output[(i - 1) * 8] = double(i);
		//output[(i - 1) * 8 + 1] = x;
		//output[(i - 1) * 8 + 2] = y;
		//output[(i - 1) * 8 + 3] = max(width, height);
		//output[(i - 1) * 8 + 4] = min(width, height);
		//output[(i - 1) * 8 + 5] = max(width, height) / min(width, height);
		//output[(i - 1) * 8 + 6] = area;
		//output[(i - 1) * 8 + 7] = totaldata[i] / area;

		outputinfo.push_back(double(i));
		outputinfo.push_back(x);
		outputinfo.push_back(y);
		outputinfo.push_back(max(width, height));
		outputinfo.push_back(min(width, height));
		outputinfo.push_back(max(width, height) / min(width, height));
		outputinfo.push_back(area);
		outputinfo.push_back(totaldata[i] / area);


		//printf("area : %d, center point(%.2f, %.2f)\n", area, pt[0], pt[1]);//面积信息
		//circle(img, Point(pt[0], pt[1]), 1, 0, -1);//中心点坐标
		//rectangle(img, Rect(x, y, width, height), 255, 1, 8, 0);//外接矩形
	}
	//std::mutex mlock;
	//mlock.lock();
	//outputinfo.insert(outputinfo.end(),output.begin(), output.end());
	//outputInfoLen = num-1;
	//outputinfo = new double[outputInfoLen * 8 * sizeof(double)];
	//memcpy(outputinfo, &output[0], outputInfoLen * 8 * sizeof(double));
	//for (int i = 0; i < outputInfoLen * 8; i++)
	//{
	//	cout << outputinfo[i] << endl;
	//}
	//findContours(canny_img, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE, Point());
	//vector<Rect> boundRect(contours.size());
	//for (int i = 0; i < contours.size(); i++)
	//{
	//	double area = contourArea(contours[i]);
	//	boundRect[i] = boundingRect(Mat(contours[i]));
	//	int lenx = boundRect[i].br().x - boundRect[i].tl().x;
	//	int leny = boundRect[i].br().y - boundRect[i].tl().y;
	//	double len = max(lenx, leny);
	//	double wid = min(lenx, leny);
	//	cout << i << "个： len: " << len << "  wid:" << wid << " rate: " << len / wid << "  area: "<<area << endl;
	//	//rectangle(img, boundRect[i].tl(), boundRect[i].br(), (0, 0, 255), 2, 8, 0);
	//	//rectangle(img, boundRect[i].tl(), boundRect[i].br(), 255, 1, 1, 0);
	//	//rectangle(srcImage,rect,(255, 0, 0), 2, 8, 0);
	//}
	std::cout << "  transfer data：" <<
		std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() -
			start).count() << "ms" << std::endl;
	std::cout << " 瑕疵:" << num - 1 << " 参数信息时间：" <<
		std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() -
			starttmp).count() << "ms" << std::endl;
}

#ifdef _MSC_VER
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT
#endif
extern "C"
{
	DLL_EXPORT void handle(char* imgpath, int threads, double*& outputinfo, int& outputInfoLen);

}

void handle(char* imgpath, int threads, double*& outputinfo, int& outputInfoLen) {
	std::string input_ImagePath = imgpath;
	int singlewidth = 200;
	ReadImgarray(input_ImagePath);//; ("D://C#//xiaci_pinjie_1209.bmp")
	//pinjieimg("D:/C#/瑕疵检测资料/清洗出拿来训练的图像/pinjie");
	//cpucal();
	//gpucal();
	//double minValue, maxValue;
	//cv::Point  minIdx, maxIdx;
	//cv::Mat imgreal = cv::imread("D://C#//DefectCut//0_0.png", 0);
	//cv::minMaxLoc(imgreal, & minValue, &maxValue, &minIdx, &maxIdx);
	//std::cout << "最大值：" << maxValue << "最小值：" << minValue << std::endl;
	//std::cout << "最大值位置：" << maxIdx << "最小值位置：" << minIdx;
	int imgwidth = img.size().width;
	bool divideflag = imgwidth % singlewidth;
	int threads = divideflag ? (imgwidth / singlewidth) + 1 : (imgwidth / singlewidth);
	auto start = std::chrono::system_clock::now();

	std::vector<std::thread> mythreads;
	vector<vector<double>> outputInfotmp(threads);
	for (size_t i = 0; i < threads; i++) {
		int singlewidthtmp = (i == threads - 1) && divideflag ? (imgwidth - (i - 1) * singlewidth) : singlewidth;
		Rect rect(i * singlewidth, 0, singlewidthtmp, img.size().height);
		Mat singleimg = img(rect);
		mythreads.push_back(std::thread(findcounter, i, singleimg, ref(outputInfotmp[i])));
	}
	for (auto it = mythreads.begin(); it != mythreads.end(); ++it) {
		it->join();
	}

	int totallen = 0;
	vector<double> outputInfo;
	for (size_t i = 0; i < threads; i++) {
		outputInfo.insert(outputInfo.end(), outputInfotmp[i].begin(), outputInfotmp[i].end());
	}
	outputInfoLen = outputInfo.size();
	outputinfo = new double[outputInfoLen * sizeof(double)];
	memcpy(outputinfo, &outputInfo[0], outputInfoLen * sizeof(double));
	std::cout << "瑕疵数据输出总时间：" <<
		std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() -
			start).count() << "ms" << std::endl;
	cout << outputInfoLen << endl;
}
void main()
{
	//std::vector<cv::Mat> imgs;
	//cv::Mat img1 = cv::imread(R"(D:\C#\瑕疵检测资料\清洗出拿来训练的图像\pinjie\pinjie-new.png)", 0);
	//cv::Rect roi(0, 0, 17000, 2000);
	//cv::Mat C = img1(roi);;
	////cv::hconcat(imgs, C);
	//imwrite(R"(D:\C#\瑕疵检测资料\清洗出拿来训练的图像\pinjie\pinjie-new1.png)", C);
	string path = R"(D:\C#\瑕疵检测资料\清洗出拿来训练的图像\pinjie\pinjie-new1.png)";//"D://C#//xiaci_pinjie_1209.bmp";
	char* imgpath = (char*)path.c_str();
	int outputInfoLen = -1;
	double* outputinfo;
	handle(imgpath, 50, outputinfo, outputInfoLen);
}

void main1()
{
	std::string folder_path = "D:\\C#\\瑕疵检测资料\\清洗出拿来训练的图像\\漏涂反G4\\*.*"; //path of folder, you can replace "*.*" by "*.jpg" or "*.png"
	std::vector<cv::String> file_names;
	cv::glob(folder_path, file_names);

	for (int i = 0; i < 50; i++) {
		int count = 0;
		ReadImgarray(file_names[i]);
		SearchDefect1(imgArray, file_names[i], count);
	}
}