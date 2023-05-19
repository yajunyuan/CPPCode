
#include<stdio.h>
#include<iostream>
#include<opencv2/opencv.hpp>
#include <time.h>
using namespace std;
using namespace cv;
#include <opencv2\opencv.hpp>
using namespace cv;
#include <Eigen\Dense>
#include "gpucal.h"

using namespace Eigen;
using namespace std;
double roi[5 * 4] = { 1129, 0, 200, 1000 ,
		2700, 0, 200, 1000 ,
		4270, 0, 200, 1000 ,
		5850, 0, 200, 1000,
		7430,0,200,1000
	/*1200, 0, 400, 1000 ,
	2800, 0, 400, 1000 ,
	4370, 0, 400, 1000 ,
	5946, 0, 400, 1000,
	1200, 0, 400, 1000 ,
	2800, 0, 400, 1000 ,
	4370, 0, 400, 1000 ,
	5946, 0, 400, 1000,
	1200, 0, 400, 1000 ,
	2800, 0, 400, 1000 ,
	4370, 0, 400, 1000 ,
	5946, 0, 400, 1000,
	1200, 0, 400, 1000 ,
	2800, 0, 400, 1000 ,
	4370, 0, 400, 1000 ,
	5946, 0, 400, 1000,
	1200, 0, 400, 1000 ,
	2800, 0, 400, 1000 ,
	4370, 0, 400, 1000 ,
	5946, 0, 400, 1000*/
};
class ROICAL
{
public:
	double* xlist, * xdlist, * roi_array, * start_position_array, * roi_param;
	int roinumber;
	ROICAL(int roi_number, double* roiparam);
	~ROICAL();
	void get_roi(Mat image);
	void get_xlist();
	void get_array(double* array, int n, int length);
};
ROICAL::ROICAL(int roi_number,double* roiparam)
{
	xlist = new double[xlength * xhigh];
	xdlist = new double[xhigh * xdlength];
	//roi_param = new double[roi_number*4];
	start_position_array = new double[roi_number];
	roinumber = roi_number;
	roi_param = roiparam;
}
ROICAL::~ROICAL(void)
{
	
	delete []xlist;
	delete []xdlist;
	delete []roi_array;
	delete []start_position_array;
	cout << "free ROICAL memory" << endl;
}
void ROICAL::get_roi(Mat image)
{
	clock_t begin_time = clock();
	//for (int i = 0; i < roinumber; i++) {
	//	start_position_array[i] = roi_param[i * 4];
	//}
	Mat roi_area;
	vector<Mat> vm;
	for (int i = 0; i < roinumber; i++) {
		start_position_array[i] = roi_param[i*4];
		Mat roi_tmp = image(Rect(roi_param[i * 4], roi_param[i * 4+1], roi_param[i * 4+2], roi_param[i * 4+3]));
		vm.push_back(roi_tmp);
		cv::vconcat(vm, roi_area);
	}

	cout << roi_area.size<<endl;
	int inputroirow = roi_param[3];
	int inputroicol = roi_param[2];
	roi_array = new double[roinumber * inputroirow * inputroicol];
	//roi_array = new double[imgrow * imgcol];
	for (int i = 0; i < roinumber * inputroirow; i++) {
		for (int j=0;j< inputroicol;j++)
	//for (int i = 0; i < imgrow; i++) {
	//	for (int j = 0; j < imgcol; j++)
			roi_array[i* inputroicol + j]= int(roi_area.at<uchar>(i, j));
	}
	cout << "get roi：" << float(clock() - begin_time) << "ms" << endl;
	
	
	
	/*int roi_area[ROIRow][ROICol];
	int roiarg1 = roi[i][0];
	int roiarg2 = roi[i][1];
	int roiarg3 = roi[i][2];
	int roiarg4 = roi[i][3];
	for (int i = roiarg2; i < roiarg2 + roiarg4; i++)
	{
		for (int j = roiarg1; j < roiarg1 + roiarg3; j++)
		{
			roi_area[i - roiarg2][j - roiarg1] = image.at<int>(i, j);
		}
	}*/
}



void ROICAL::get_array(double* array, int n, int length)
{
	for (int row = 0; row < n; row++)
	{
		for (int con = 0; con < length; con++)
		{
			double t = con * 2.0 / (length - 1.0) - 1.0;
			array[row * length + con] = cos(row * acos(t));
		}
	}
}

void ROICAL::get_xlist()
{
	clock_t begin_time = clock();
	double xarray[xhigh * xlength];
	get_array(xarray, xhigh, xlength);
	MatrixXd xarraytmp = Map<Matrix<double, xhigh, xlength, RowMajor>>(xarray);
	MatrixXd xarray_T = xarraytmp.transpose();
	MatrixXd xarray_1 = (xarraytmp * xarray_T).inverse();
	MatrixXd xlistresult = (xarray_T * xarray_1).transpose();
	cout << xlistresult.rows()<< xlistresult.cols() <<endl;
	double* xlist_tmp = xlistresult.data();
	//Map<MatrixXd>(xlist, xlistresult.rows(), xlistresult.cols()) = xlistresult;
	memcpy( xlist, xlistresult.data(), sizeof(double) * xlength * xhigh);

	get_array(xdlist, xhigh, xdlength);
	cout << "get xlist：" << float(clock() - begin_time) << "ms" << endl;
}

class TA {
public:
	int m_i;
	int j;
	TA(int i) :m_i(i) {}
	void operator()()//不能带参数，代码从这开始执行
	{
		cout << "我的线程" << m_i << "开始执行了" << endl;
		//...
		cout << "我的线程结束执行了" << endl;
	}
	void out(int j1) {
		j = j1 + 1;
		cout << "线程" << m_i <<"  "<<j << "开始执行了" << endl;
	}
};

//int main1() {
//	vector<thread>mythreads;
//	//创建10个线程，线程入口函数统一使用myprint.
//
//	TA ta1(1);
//	TA ta2(2);
//	mythreads.push_back(thread(&TA::out, &ta1, 1));//创建了10个线程，同时这10个已经开始执行了
//	mythreads.push_back(thread(&TA::out, &ta2, 1));
//	for (auto iter = mythreads.begin(); iter != mythreads.end(); ++iter)
//	{
//		iter->join();//等待着10个线程都返回
//	}
//	//int myi = 6;
//	//for (int i = 0; i < 6; i++) {
//	//	TA ta(i);
//	//	thread my_thread(ta);// ta 可调用对象
//	//	my_thread.detach();
//	//}
//
//	//my_thread.join();//等待子线程执行结束
//
//	cout << "I love China" << endl;
//	return 0;
//}

extern "C" __declspec(dllexport)  GPUCAL* _stdcall gpucalobj(int roinumber, int roi_row, int roi_col);
extern "C" __declspec(dllexport)  ROICAL * _stdcall roicalobj(int roinumber, double* roiparam);
extern "C" __declspec(dllexport)  int _stdcall cal(GPUCAL* gpuobj, ROICAL* roiobj, uchar* data, int width, int height, int stride, double* result);
extern "C" __declspec(dllexport)  void _stdcall gpuinit(GPUCAL* gpuobj, ROICAL* roiobj);
extern "C" __declspec(dllexport)  void _stdcall destroyobject(GPUCAL* gpuobj, ROICAL* roiobj);

GPUCAL* gpucalobj(int roinumber, int roi_row, int roi_col) {

	return new GPUCAL(roinumber, roi_row, roi_col);
}

ROICAL* roicalobj(int roinumber, double* roiparam) {

	return new ROICAL(roinumber, roiparam);
}

void destroyobject(GPUCAL* gpuobj, ROICAL* roiobj) {
	cout << "GPU释放内存" << endl;
	if (roiobj)
	{
		delete roiobj;
		roiobj = nullptr;
	}
	if (gpuobj)
	{
		delete gpuobj;
		gpuobj = nullptr;
	}

}
void _stdcall gpuinit(GPUCAL* gpuobj, ROICAL* roiobj) {
	roiobj->get_xlist();
	gpuobj->initxlist(roiobj->xlist, roiobj->xdlist);
}

int _stdcall cal(GPUCAL* gpuobj, ROICAL* roiobj, uchar* data, int width, int height, int stride, double* result) {
	//ROICAL roscal;
	Mat img = Mat(cv::Size(width, height), CV_8UC1, data, stride);
	roiobj->get_roi(img);
	return gpuobj->matcal(roiobj->roi_array, roiobj->start_position_array, result);
}


int main()
{
	ROICAL roscal(5,roi);
	double* result = new double[(4 * ROIRow - ROIRow)];
	roscal.get_xlist();
	GPUCAL gpucal(5, roscal.roi_param[3], roscal.roi_param[2]);
	clock_t begin_time = clock();
	for (int i = 0; i < 10; i++) {
		string a= "D:/"+to_string(i) + ".bmp";
		Mat image = imread(a, 0);
		
		cout << "read：" << float(clock() - begin_time) << "ms" << endl;
		gpucal.initxlist(roscal.xlist, roscal.xdlist);
		roscal.get_roi(image);
		cout << "bmp: " << a << endl;
		gpucal.matcal(roscal.roi_array,roscal.start_position_array, result);
	}
	cout << "总时间：" << float(clock() - begin_time) << "ms" << endl;

	return 1;

}