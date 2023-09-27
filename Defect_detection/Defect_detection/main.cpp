#include<stdio.h>
#include<iostream>
#include<opencv2/opencv.hpp>
#include <time.h>
#include <opencv2\opencv.hpp>
#include<mutex>
#include <map>
#include <thread>
#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
using namespace std;
using namespace cv;

int PinholeThreshold[2] = { 80, 37 };
int Threshold[2] = { 100,145 };
string ClassName[10]; //class����
bool ClassFlag[5];
struct ClassStr
{
	int classParam[10];
	int gradeParam[5];
};
// vector: class���к�+ѡ����ж�������0-����ȡ�1-���ȡ�2-��ȡ�3-�����4��ƽ�����ȣ�  
//        �ṹ�壺�������ж�������������ֵ �� �ȼ��ж�������
map<vector<int>, ClassStr> ClassMap;

struct DefectData {
	int offsetx; //��Ĥ��ʼλ��x
	int pinholeflag;
	int box[4];    //���ο�
	double ratio; //�����
	double area; //���
	int maxgray; //�������
	int mingray; //��С����
	double avggray; //ƽ������
	double darkavggray; //����ƽ������
	int classid; //覴����
	int grade; //覴õȼ�
};
struct AvgPixelData {
	int avgpixelsize; //ƽ�����صĳ���
	uchar* avgpixel;  //ÿ��ƽ�����ص�����
	int darkavgpixelsize; //����ƽ�����صĳ���
	uchar* darkavgpixel;  //����ÿ��ƽ�����ص�����
	int initPos; //��Ĥ��ͼ�����ʼλ��
	int endPos; //��Ĥ��ͼ��Ľ���λ��
};
void ReadImgarray(std::string imgfile, Mat& img)
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
	std::cout << "ͼת����" << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count()/1000.0 << "ms" << std::endl;
}
template<typename ... Args>
static std::string str_format(const std::string& format, Args ... args)
{
	auto size_buf = std::snprintf(nullptr, 0, format.c_str(), args ...) + 1;
	std::unique_ptr<char[]> buf(new(std::nothrow) char[size_buf]);

	if (!buf)
		return std::string("");

	std::snprintf(buf.get(), size_buf, format.c_str(), args ...);
	return std::string(buf.get(), buf.get() + size_buf - 1);
}

void DetectClassification(DefectData& data) {
	// class
	if (data.pinholeflag!=-1) {
		if (data.pinholeflag == 1) {
			data.classid = 0;
		}
		else {
			data.classid = 1;
		}
		if (ClassMap.size() > 2) {
			for (map<vector<int>, ClassStr>::iterator it = ClassMap.begin(); it != ClassMap.end(); it++) {
				if (it->first[0] == 2) {
					break;
				}
				//grade
				if (data.area < it->second.gradeParam[0]) {
					data.grade = 0;
				}
				else if (data.area < it->second.gradeParam[1]) {
					data.grade = 1;
				}
				else if (data.area < it->second.gradeParam[2]) {
					data.grade = 2;
				}
				else if (data.area < it->second.gradeParam[3]) {
					data.grade = 3;
				}
				else if (data.area < it->second.gradeParam[4]) {
					data.grade = 4;
				}
				else {
					data.grade = 5;
				}
			}
		}
		else {
			cout << "覴÷������Ҫ�������У���ס�͸����" << endl;
		}
	}
	else {
		//ѭ������覴÷�����е�覴��ж�
		for (map<vector<int>, ClassStr>::iterator it = ClassMap.begin(); it != ClassMap.end(); it++)
		{
			//��ס�͸����ִ���ж���
			if (it->first[0] == 0 || it->first[0] == 1) {
				continue;
			}
			//ѭ�������ж������������㷵��覴����к�
			bool flag = true;
			for (int i = 1; i < it->first.size(); i++) {
				bool minvalueJudg = true;
				bool maxvalueJudg = true;
				switch (it->first[i]) {
				case 1:
					minvalueJudg = (it->second.classParam[it->first[i] * 2] != -1) ?
						(data.box[2] >= it->second.classParam[it->first[i] * 2]) : true;
					maxvalueJudg = (it->second.classParam[it->first[i] * 2 + 1] != -1) ?
						(data.box[2] <= it->second.classParam[it->first[i] * 2 + 1]) : true;
					if (minvalueJudg && maxvalueJudg) {
						flag = flag & true;
					}
					else {
						flag = flag & false;
					}
					break;
				case 2:
					minvalueJudg = (it->second.classParam[it->first[i] * 2] != -1) ?
						(data.box[3] >= it->second.classParam[it->first[i] * 2]) : true;
					maxvalueJudg = (it->second.classParam[it->first[i] * 2 + 1] != -1) ?
						(data.box[3] <= it->second.classParam[it->first[i] * 2 + 1]) : true;
					if (minvalueJudg && maxvalueJudg) {
						flag = flag & true;
					}
					else {
						flag = flag & false;
					}
					break;
				case 3:
					minvalueJudg = (it->second.classParam[it->first[i] * 2] != -1) ?
						(data.area >= it->second.classParam[it->first[i] * 2]) : true;
					maxvalueJudg = (it->second.classParam[it->first[i] * 2 + 1] != -1) ?
						(data.area <= it->second.classParam[it->first[i] * 2 + 1]) : true;
					if (minvalueJudg && maxvalueJudg) {
						flag = flag & true;
					}
					else {
						flag = flag & false;
					}
					break;
				case 4:
					minvalueJudg = (it->second.classParam[it->first[i] * 2] != -1) ?
						(data.avggray >= it->second.classParam[it->first[i] * 2]) : true;
					maxvalueJudg = (it->second.classParam[it->first[i] * 2 + 1] != -1) ?
						(data.avggray <= it->second.classParam[it->first[i] * 2 + 1]) : true;
					if (minvalueJudg && maxvalueJudg) {
						flag = flag & true;
					}
					else {
						flag = flag & false;
					}
					break;
				default:
					minvalueJudg = (it->second.classParam[it->first[i] * 2] != -1) ?
						(data.ratio >= it->second.classParam[it->first[i] * 2]) : true;
					maxvalueJudg = (it->second.classParam[it->first[i] * 2 + 1] != -1) ?
						(data.ratio <= it->second.classParam[it->first[i] * 2 + 1]) : true;
					if (minvalueJudg && maxvalueJudg) {
						flag = flag & true;
					}
					else {
						flag = flag & false;
					}
					break;
				}
				//if (!flag) {
				//	break;
				//}	
			}
			if (flag) {
				data.classid = it->first[0];
				//grade
				if (data.area < it->second.gradeParam[0]) {
					data.grade = 0;
				}
				else if (data.area < it->second.gradeParam[1]) {
					data.grade = 1;
				}
				else if (data.area < it->second.gradeParam[2]) {
					data.grade = 2;
				}
				else if (data.area < it->second.gradeParam[3]) {
					data.grade = 3;
				}
				else if (data.area < it->second.gradeParam[4]) {
					data.grade = 4;
				}
				else {
					data.grade = 5;
				}
				break;
			}

		}
	}
}


#ifdef _MSC_VER
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT
#endif
#undef GetObject
extern "C"
{
	DLL_EXPORT void handle(uchar* data, uchar* data1, int width, int height, int stride, DefectData* &outputinfo, int& outputInfoLen, AvgPixelData& avgPixelData);
	DLL_EXPORT void GetDetectJsonParam(char* paramstr);
	DLL_EXPORT void GetClassJsonParam(char* paramstr);
	DLL_EXPORT void release(DefectData*& outputinfo);
}

void GetDetectJsonParam(char* paramstr) {
	rapidjson::Document document;
	if (string(paramstr) == "") {
		std::string paramStr = R"({"DarkGrayVal":114,"BrightGrayVal":145})";
		document.Parse(paramStr.c_str());
	}
	else {
		document.Parse(paramstr);
		cout << "��ȡ������ò���ok" << endl;
	}
	if (document.HasParseError()) {
		fprintf(stderr, "\nError(offset %u): %s\n",(unsigned)document.GetErrorOffset(),GetParseError_En(document.GetParseError()));
		return;
	}
	if (document.HasMember("DarkGrayVal")&& document["DarkGrayVal"].IsInt()) {
		Threshold[0] = document["DarkGrayVal"].GetInt();
	}
	if (document.HasMember("BrightGrayVal") && document["BrightGrayVal"].IsInt()) {
		Threshold[1] = document["BrightGrayVal"].GetInt();
	}
	cout << "����ֵ�� " << Threshold[0] <<"  ����ֵ�� "<< Threshold[1] << endl;
}

void AddClassflag(vector<int>& classFlags, string str) {
	if (str == "") {
		return;
	}
	if (str == "�����") {
		classFlags.push_back(0);
	}
	else if (str == "����") {
		classFlags.push_back(1);
	}
	else if (str == "���") {
		classFlags.push_back(2);
	}
	else if (str == "���") {
		classFlags.push_back(3);
	}
	else {
		classFlags.push_back(4);
	}
}

void GetClassJsonParam(char* paramstr) {
	rapidjson::Document document;
	if (string(paramstr) == "") {
		std::string paramStr = R"([{"ȱ������":"��ȱ��","�ж�����1":"���","�ж�����2":"����","�ж�����3":"",
					"�ж�����4":"","�ж�����5":"","�������mm2":"50","�������mm2":"","��������mm":"5","��������mm":"",
"�������mm":"","�������mm":"","���������":"4","���������":"","ƽ���߶�����":"100","ƽ���߶�����":""},
					{"ȱ������":"��ȱ��","�ж�����1":"���","�ж�����2":"����","�ж�����3":"",
					"�ж�����4":"","�ж�����5":"","�������mm2":"20","�������mm2":"50","��������mm":"2","��������mm":"",
"�������mm":"","�������mm":"","���������":"4","���������":"","ƽ���߶�����":"100","ƽ���߶�����":""},
					{"ȱ������":"Сȱ��","�ж�����1":"���","�ж�����2":"����","�ж�����3":"",
					"�ж�����4":"","�ж�����5":"","�������mm2":"","�������mm2":"20","��������mm":"0","��������mm":"",
"�������mm":"","�������mm":"","���������":"4","���������":"","ƽ���߶�����":"100","ƽ���߶�����":""}])";
		document.Parse(paramStr.c_str());
	}
	else {
		document.Parse(paramstr);
		cout << "��ȡ�������ò���ok" << endl;
	}
	if (!ClassMap.empty()) {
		ClassMap.clear();
	}
	if (document.HasParseError()) {
		fprintf(stderr, "\nError(offset %u): %s\n", (unsigned)document.GetErrorOffset(), GetParseError_En(document.GetParseError()));
		return;
	}
	if (document.IsArray()) {
		rapidjson::Document::Array classobjs = document.GetArray();
		
		for (unsigned int i = 0; i < classobjs.Size(); i++) {
			vector<int> classFlagtmp;
			classFlagtmp.push_back(i);
			ClassStr classStrtmp;
			rapidjson::Document::Object classobj = classobjs[i].GetObject();
			ClassName[i] = classobj["ȱ������"].GetString();
			AddClassflag(classFlagtmp, classobj["�ж�����1"].GetString());
			AddClassflag(classFlagtmp, classobj["�ж�����2"].GetString());
			AddClassflag(classFlagtmp, classobj["�ж�����3"].GetString());
			AddClassflag(classFlagtmp, classobj["�ж�����4"].GetString());
			AddClassflag(classFlagtmp, classobj["�ж�����5"].GetString());
			//�ж�����������
			string limitparam = classobj["���������"].GetString();
			classStrtmp.classParam[0] = (limitparam != "") ? stoi(limitparam) : -1; // double---stod
			limitparam = classobj["���������"].GetString();
			classStrtmp.classParam[1] = (limitparam != "") ? stoi(limitparam) : -1;
			limitparam = classobj["��������mm"].GetString();
			classStrtmp.classParam[2] = (limitparam != "") ? stoi(limitparam) : -1;
			limitparam = classobj["��������mm"].GetString();
			classStrtmp.classParam[3] = (limitparam != "") ? stoi(limitparam) : -1;
			limitparam = classobj["�������mm"].GetString();
			classStrtmp.classParam[4] = (limitparam != "") ? stoi(limitparam) : -1;
			limitparam = classobj["�������mm"].GetString();
			classStrtmp.classParam[5] = (limitparam != "") ? stoi(limitparam) : -1;
			limitparam = classobj["�������mm2"].GetString();
			classStrtmp.classParam[6] = (limitparam != "") ? stoi(limitparam) : -1;
			limitparam = classobj["�������mm2"].GetString();
			classStrtmp.classParam[7] = (limitparam != "") ? stoi(limitparam) : -1;
			limitparam = classobj["ƽ����������"].GetString();
			classStrtmp.classParam[8] = (limitparam != "") ? stoi(limitparam) : -1;
			limitparam = classobj["ƽ����������"].GetString();
			classStrtmp.classParam[9] = (limitparam != "") ? stoi(limitparam) : -1;
			//�ȼ�������
			limitparam = classobj["�ȼ�0"].GetString();
			classStrtmp.gradeParam[0] = (limitparam != "") ? stod(limitparam) : -1.0;
			limitparam = classobj["�ȼ�1"].GetString();
			classStrtmp.gradeParam[1] = (limitparam != "") ? stod(limitparam) : -1.0;
			limitparam = classobj["�ȼ�2"].GetString();
			classStrtmp.gradeParam[2] = (limitparam != "") ? stod(limitparam) : -1.0;
			limitparam = classobj["�ȼ�3"].GetString();
			classStrtmp.gradeParam[3] = (limitparam != "") ? stod(limitparam) : -1.0;
			limitparam = classobj["�ȼ�4"].GetString();
			classStrtmp.gradeParam[4] = (limitparam != "") ? stod(limitparam) : -1.0;
			ClassMap.insert(pair<vector<int>, ClassStr>(classFlagtmp, classStrtmp));
		}
	}
	auto iter = ClassMap.begin();
	PinholeThreshold[iter->first[0]] = (iter->second.classParam[8] != -1) ? iter->second.classParam[8] : PinholeThreshold[iter->first[0]];
	PinholeThreshold[next(iter)->first[0]] = (next(iter)->second.classParam[8] != -1) ? next(iter)->second.classParam[8] : PinholeThreshold[next(iter)->first[0]];
	for (map<vector<int>, ClassStr>::iterator it = ClassMap.begin(); it != ClassMap.end(); it++) {
		for (int i = 1; i < it->first.size(); i++) {
			cout << it->first[i] << "  ";
		}
		cout << endl;
		for (auto i = 0; i < 10; i++) {
			cout << it->second.classParam[i] << "  ";
		}
		cout << endl;
	}
	cout << "���͸����ֵ��" << PinholeThreshold[0] << "    " << PinholeThreshold[1] << endl;
}

bool adjacencyJudge(uchar data, int minthreshold, int maxthreshold)
{
	return (abs(data - minthreshold) < 10) || (abs(data - maxthreshold) < 10);
}

void highHandle(Mat src, Mat darksrc, Mat& dst, int minthreshold, int maxthreshold, int initHigh, int highvalue) {
	for (auto i = initHigh; i < initHigh+ highvalue; i++) {
		uchar* darkdata = darksrc.data + darksrc.step * i;
		uchar* sdata = src.data + src.step * i; //��������
		uchar* ddata = dst.data + dst.step * i; //�ָ�����
		if (i == initHigh) {
			uchar* adata = src.data + src.step * (i + 1); //��������
			for (auto j = 1; j < src.size().width - 1; j++) {
				//����覴�
				if (minthreshold - sdata[j] >= 20 || sdata[j]- maxthreshold>=20) {
					ddata[j] = 255;
					continue;
				}
				else if (sdata[j] < minthreshold || sdata[j] >= maxthreshold) {
					if (abs(sdata[j + 1] - sdata[j]) < 10 || abs(sdata[j - 1] - sdata[j]) < 10) {
						if (abs(sdata[j] - adata[j]) < 10) {
							ddata[j] = 255;
							continue;
						}

					}
				}
				//else if (sdata[j] < minthreshold + 10 || sdata[j] >= maxthreshold - 10) {
				//	if (j > 1 && abs(sdata[j + 1] - sdata[j - 2]) >= 12) {
				//		ddata[j] = 255;
				//		continue;
				//	}
				//}
			}
		}
		else if (i == initHigh + highvalue - 1) {
			uchar* bdata = src.data + src.step * (i - 1); //��������
			for (auto j = 1; j < src.size().width - 1; j++) {
				//����覴�
				if (minthreshold - sdata[j] >= 20 || sdata[j] - maxthreshold >= 20) {
					ddata[j] = 255;
					continue;
				}
				else if (sdata[j] < minthreshold || sdata[j] >= maxthreshold) {
					if (abs(sdata[j + 1] - sdata[j]) < 10 || abs(sdata[j - 1] - sdata[j]) < 10) {
						if (abs(bdata[j] - sdata[j]) < 10) {
							ddata[j] = 255;
							continue;
						}
					}
				}
				//else if (sdata[j] <=minthreshold+10 || sdata[j] >= maxthreshold-10) {
				//	if (j > 1 && abs(sdata[j + 1] - sdata[j - 2]) >= 12) {
				//		ddata[j] = 255;
				//		continue;
				//	}
				//}
			}
		}
		else {
			uchar* adata = src.data + src.step * (i + 1); //��������
			uchar* bdata = src.data + src.step * (i - 1); //��������
			for (auto j = 1; j < src.size().width - 1; j++) {
				//����覴�
				if (minthreshold - sdata[j] >= 20 || sdata[j] - maxthreshold >= 20) {
					ddata[j] = 255;
					continue;
				}
				else if (sdata[j] < minthreshold || sdata[j] >= maxthreshold) {
					if (abs(sdata[j + 1] - sdata[j]) < 10 || abs(sdata[j - 1] - sdata[j]) < 10) {
						if (abs(sdata[j] - adata[j]) < 10 || abs(bdata[j] - sdata[j]) < 10) {
							ddata[j] = 255;
							continue;
						}
					}
				}
				if (darkdata[j] > 60) {
					ddata[j] = 255;
					continue;
				}
				//else if (sdata[j] <=minthreshold + 10 || sdata[j] >= maxthreshold -10) {
				//	if (j > 1 && abs(sdata[j + 1] - sdata[j - 2]) >= 12) {
				//		ddata[j] = 255;
				//		continue;
				//	}
				//}
				bool adjacencyflag = adjacencyJudge(sdata[j], minthreshold, maxthreshold);
				//if (adjacencyflag) {
				//	if (abs(sdata[j + 1] - sdata[j]) < 10) {
				//		ddata[j] = 255;
				//		continue;
				//	}

				//}
				////���覴ñ߽�����
				//if (j > 1 && j < (src.size().width - 1) && adjacencyflag) {
				//	if (abs(sdata[j + 1] - sdata[j - 2]) >= 12) {//abs(sdata[j+1]- sdata[j-2])>=12)   //abs(mdata[j] - mdata[j-1])>=4 )
				//		ddata[j] = 255;
				//		continue;
				//	}
				//	//覴��ڲ�ģ������
				//	else if (abs(sdata[j] - 120) >= 10&& abs(sdata[j-1] - 120) >= 10&& abs(sdata[j+1] - 120) >= 10) {
				//			ddata[j] = 255;
				//			continue;
				//	}
				//}
				ddata[j] = 0;
			}
		}
		
		//�������   ����� 255 255 255 255����� 
		if (i > initHigh+1) {
			uchar* befddata = dst.data + dst.step * (i - 2);
			uchar* curddata = dst.data + dst.step * (i - 1);
			uchar* cursdata = src.data + src.step * (i - 1);
			for (auto k = 1; k < src.size().width - 1; k++) {
				if (curddata[k] == 0) {
					if (befddata[k]==255 && ddata[k]==255) {
						curddata[k] = 255;
						continue;
					}
					if (curddata[k-1] == 255 && curddata[k+1] == 255) {
						curddata[k] = 255;
						continue;
					}
					vector<bool> stat;
					for (auto m = k - 1; m < k + 2; m++) {
						if (befddata[m] == 255) {
							stat.push_back(true);
						}
						if (curddata[m] == 255) {
							stat.push_back(true);
						}
						if (ddata[m] == 255) {
							stat.push_back(true);
						}
						if (stat.size() >= 4) {
							curddata[k] = 255;
							break;
						}
					}
				}
				//else {
				//	vector<bool> stat;
				//	for (auto m = k - 1; m < k + 2; m++) {
				//		if (befddata[m] == 255) {
				//			stat.push_back(true);
				//		}
				//		if (curddata[m] == 255) {
				//			stat.push_back(true);
				//		}
				//		if (ddata[m] == 255) {
				//			stat.push_back(true);
				//		}
				//	}
				//	if (stat.size() < 3) {
				//		curddata[k] = 0;
				//	}
				//}
			}
		}
	}
}

void threshold(Mat src,Mat darksrc, Mat& dst, int minthreshold, int maxthreshold) {
	Mat mean;
	//dst.create(src.size().height / 2, src.size().width, src.type());
	dst.create(src.size().height, src.size().width, src.type());
	int heightjudge = (src.size().height % 2 != 0) ? src.size().height - 1 : src.size().height;
	//for (auto i = 0; i < heightjudge; i=i+2) {
	//	uchar* sdata = src.data + src.step * i; //��������
	//	uchar* ddata = dst.data + dst.step * i / 2;

	//���̱߳�������ͼ��߶�
	std::vector<std::thread> mythreads;
	int threadcount = 5;
	int singleheight = src.size().height/ threadcount;
	int imgheight = src.size().height;
	bool divideflag = imgheight % singleheight;
	int threads = divideflag ? (imgheight / singleheight) + 1 : (imgheight / singleheight);
	for (size_t i = 0; i < threads; i++) {
		int singleheighttmp = (i == threads - 1) && divideflag ? (imgheight - i * singleheight) : singleheight;
		mythreads.push_back(std::thread(highHandle, src, darksrc, ref(dst), minthreshold, maxthreshold, i* singleheight, singleheighttmp));
	}
	for (auto it = mythreads.begin(); it != mythreads.end(); ++it) {
		it->join();
	}
}

std::vector<Rect> findRect(Mat img,int slotnums)
{

	std::vector<Rect> rectVector;
	int startPintx = 0;
	int endPintx = 0;
	uchar* bdata = img.data + img.step * 10;; //��������
	for (auto i = img.size().width - 1; i > 0; i--) {
		if (bdata[i] > 100&& bdata[i]<240) {
			endPintx = i - 10;
			break;
		}
	}
	for (auto j = 0; j < img.size().width; j++) {
		if (bdata[j] > 100 && bdata[j] < 240) {
			startPintx = j + 10;
			break;
		}
	}
	if (slotnums == 0) {
		rectVector.push_back(Rect(startPintx, 0, endPintx - startPintx+1, img.size().height));
	}
	else
	{
		std::vector<int> slotPos;
		slotPos.push_back(startPintx-10);
		for (auto i = startPintx; i < endPintx; i++) {
			//�ж��з�����
			if (bdata[i] <70) {
				slotPos.push_back(i);
				i += 10;
			}
			if (slotPos.size() == slotnums+1) {
				break;
			}
		}
		slotPos.push_back(endPintx+10);
		for (auto i = 0; i < slotPos.size()-1; i++) {
			rectVector.push_back(Rect(slotPos[i]+10, 0, slotPos[i+1] - slotPos[i] -20  + 1, img.size().height));
		}
	}
	return rectVector;
}

void CombineCounter(const vector<DefectData>& outputinfotmp, vector<DefectData>& outputinfo) {
	int recordcount = -1;
	int recordarea = 0;
	int flag = -1;
	int i = 0;
	for (; i < outputinfotmp.size(); ++i) {
		if (outputinfotmp[i].pinholeflag == 1) {
			outputinfo.push_back(outputinfotmp[i]);
			continue;
		}
		int minx = outputinfotmp[i].box[0] - 50;
		int maxx = outputinfotmp[i].box[0] + outputinfotmp[i].box[2] + 50;
		int miny = outputinfotmp[i].box[1];
		int maxy = outputinfotmp[i].box[1] + outputinfotmp[i].box[3] + 50;

		int j = i + 1;
		for (; j < outputinfotmp.size(); ++j) {
			bool judge = outputinfotmp[j].box[0] >= minx && outputinfotmp[j].box[0] <= maxx &&
				outputinfotmp[j].box[1] >= miny && outputinfotmp[j].box[1] <= maxy;
			bool judge1 = outputinfotmp[j].box[0] + outputinfotmp[j].box[2] >= minx &&
				outputinfotmp[j].box[0] + outputinfotmp[j].box[2] <= maxx &&
				outputinfotmp[j].box[1] >= miny && outputinfotmp[j].box[1] <= maxy;
			if (judge || judge1) {
				flag = 1;
				if (outputinfotmp[j].area > outputinfotmp[i].area) {
					if (outputinfotmp[j].area > recordarea) {
						recordarea = outputinfotmp[j].area;
						recordcount = j;
					}
				}
				else {
					if (outputinfotmp[i].area > recordarea) {
						recordarea = outputinfotmp[i].area;
						recordcount = i;
					}
				}
				break;
			}
		}
		if (flag == -1) {
			outputinfo.push_back(outputinfotmp[i]);
		}
		else if (j >= outputinfotmp.size()) {
			outputinfo.push_back(outputinfotmp[recordcount]);
			recordarea = 0;
			recordcount = -1;
			flag = -1;
		}
	}
}

void findcounter(int offsetx, Mat brightImg, Mat darkImg, vector<DefectData>& outputinfo)
{
	auto starttmp = std::chrono::system_clock::now();
	Mat canny_img;
	auto start = std::chrono::system_clock::now();
	//Scalar mean1 = mean(img);
	//Mat canny_img1;
	//cv::threshold(threadimg, canny_img1, 132, 255, THRESH_BINARY);//��ֵ����ֵ����

	//GaussianBlur(img, canny_img, Size(5, 5), 0, 0);

	//threshold(img, canny_img, 115, 135, 3);
	start = std::chrono::system_clock::now();
	threshold(brightImg, darkImg, canny_img, Threshold[0], Threshold[1]);
	//Mat dst;
	//Mat kernel = getStructuringElement(0, Size(2, 2));
	//morphologyEx(canny_img, dst, MORPH_OPEN, kernel);//������

	std::cout << "  opencv threshold��" <<
		std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() -
			start).count() << "ms" << std::endl;

	start = std::chrono::system_clock::now();
	Mat labels, stats, centroids;
	int num = connectedComponentsWithStats(canny_img, labels, stats, centroids);
	std::cout << "  connect state��" <<
		std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() -
			start).count() << "ms" << std::endl;

	vector<int> totaldata(num);
	vector<int> darktotaldata(num);

	//MYSQL mysql;
	//mysql_init(&mysql);
	//mysql_real_connect(&mysql, "localhost", "root", "123456", "Defectdatabase", 0, NULL, 0);
	//mysql_query(&mysql, "create database if not exists Defectdatabase");
	//mysql_query(&mysql, "use Defectdatabase");
	//mysql_query(&mysql, "CREATE TABLE IF NOT EXISTS defectparams (id INT NOT NULL AUTO_INCREMENT, seq FLOAT, x FLOAT, y FLOAT,"
	//	"length FLOAT, width FLOAT, ratio FLOAT, area FLOAT,gray FLOAT, PRIMARY KEY (id))");
	//string queryStr = "INSERT INTO defectparams (seq, x, y, length, width, ratio, area, gray) VALUES ";
	start = std::chrono::system_clock::now();
	vector<DefectData> outputinfotmp;
	for (int i = 1; i < num; i++) {
		int left = stats.at<int>(i, CC_STAT_LEFT);
		int top = stats.at<int>(i, CC_STAT_TOP);
		int width = stats.at<int>(i, CC_STAT_WIDTH);
		int height = stats.at<int>(i, CC_STAT_HEIGHT);
		int area = stats.at<int>(i, CC_STAT_AREA);
		if (area < 2) {
			continue;
		}
		DefectData datatmp;
		datatmp.offsetx = offsetx;
		datatmp.pinholeflag = -1;
		datatmp.classid = -1;
		Vec2d pt = centroids.at<Vec2d>(i, 0);
		datatmp.box[0] = left;
		datatmp.box[1] = top;
		datatmp.box[2] = width;
		datatmp.box[3] = height;
		//datatmp.area = area/1.028357; //����궨
		datatmp.area = area/1.005594;
		datatmp.ratio = max(width, height )/ min(width, height);
		Rect rect(left, top, width, height); //����roi
		Mat imgtmp = brightImg(rect);
		Mat_<ushort> labeltmp = labels(rect);
		//Rect rectimg(datatmp.box[0], datatmp.box[1], datatmp.box[2], datatmp.box[3]); //��ͼroi
		
		Mat darimgtmp = darkImg(rect);
		labeltmp.convertTo(labeltmp, CV_16UC1);
		//imgtmp.convertTo(imgtmp, CV_8UC1);
		datatmp.maxgray = 0;
		datatmp.mingray = 255;
		for (auto k = 0; k < labeltmp.size().height; k++) {
			ushort* labelsdata = (ushort*)(labeltmp.data + labeltmp.step * k);
			//uchar* imgdata = imgtmp.data + imgtmp.step * k*2; // ��������
			uchar* imgdata = imgtmp.data + imgtmp.step * k; // ��������
			uchar* darkdata = darimgtmp.data + darimgtmp.step * k; // ��������
			int edgeindex = -1;
			bool firstindex = true;
			for (auto j = 0; j < labeltmp.size().width; j++) {
				if (labelsdata[j] == 0) continue;
				if (imgdata[j] > datatmp.maxgray) {
					datatmp.maxgray = imgdata[j];
				}
				if (imgdata[j] < datatmp.mingray) {
					datatmp.mingray = imgdata[j];
				}
				totaldata[labelsdata[j]] += imgdata[j];
				darktotaldata[labelsdata[j]] += darkdata[j];
				
				//�ж����-覴���һ��������Ϊ0����Ϊ255����Ϊpinhole�����ڸ�Ϊ覴�����ƽ��ֵ��Ϊ0����Ϊ255
				if (datatmp.pinholeflag == -1|| datatmp.pinholeflag == 2) {
					if (firstindex) {
						edgeindex = j;
						firstindex = false;
					}
					if (imgdata[j] < 128) {
						if (imgdata[j] < Threshold[0] - 10) {
							//if (darkdata[j] >= 80) {
							//	datatmp.pinholeflag = 1;
							//}
							if (darkdata[j] >= PinholeThreshold[0]) {
								if (abs(j - edgeindex) < 3) {
									datatmp.pinholeflag = 1;
								}
								else {
									datatmp.pinholeflag = 2;
								}
							}
						}
						else {
							if (darkdata[j] >= PinholeThreshold[1]) {
								datatmp.pinholeflag = 2;
							}
						}
					}
					//if (imgdata[j] < 100) {
					//	//uchar* darkdata = imgtmp.data + imgtmp.step * (k * 2+1); // ��������
					//	//�ж���͸��������
					//	if (edgeindex == -1) {
					//		edgeindex = j;
					//	}
					//	if (darkdata[j] > 200) {
					//		if (darkdata[edgeindex]>200) {
					//			datatmp.pinholeflag = 1;
					//		}
					//		else {
					//			datatmp.pinholeflag = 2;
					//		}
					//	}
					//	else if (darkdata[j] > 100) {
					//		datatmp.pinholeflag = 2;
					//	}
					//}
				}
				//else if (datatmp.pinholeflag == 2) {
				//	if (imgdata[j] < 40) {
				//		uchar* darkdata = darimgtmp.data + darimgtmp.step * k; // ��������
				//		if (darkdata[j] > 200) {
				//			datatmp.pinholeflag = 1;
				//		}
				//	}
				//}
			}
			//覴�ƽ���Ҷȱ���Χ�Ҷȴ�ȥ����Ե����Ҷȵ͵�����
			if (k == labeltmp.size().height - 1) {
				uchar* briimgdata = brightImg.data + brightImg.step * (k+ top);
				if (left < 100 || left > labels.size().width - 100) {
					if (abs(totaldata[i] / datatmp.area - briimgdata[left+edgeindex - 5])<=10 ||
						abs(totaldata[i] / datatmp.area - briimgdata[left + edgeindex - 4]) <=10) {
						datatmp.pinholeflag = -2;
					}
				}
			}
		}
		//if (datatmp.area < 10 && datatmp.pinholeflag != 1&& abs(datatmp.avggray-120)<10) {
		//	continue;
		//}
		if (datatmp.pinholeflag == -2) {
			continue;
		}
		if (datatmp.ratio<3	&& width * height/datatmp.area>5) {
			datatmp.ratio = 5;
		}
		//��ͼλ��
		datatmp.box[0] += offsetx;
		datatmp.avggray = totaldata[i] / datatmp.area;
		datatmp.darkavggray = darktotaldata[i] / datatmp.area;
		//	cout << i << "���� len: " << max(width,height) << "  wid:" << min(width, height) << " rate: " << 
		//max(width, height) / min(width, height) << "  area: " << area << "  avg pixel: "<< totaldata[i]/ area << endl;

			// todo  ������Ϣ����
		DetectClassification(datatmp);
		//output[(i - 1) * 8] = double(i);
		//output[(i - 1) * 8 + 1] = x;
		//output[(i - 1) * 8 + 2] = y;
		//output[(i - 1) * 8 + 3] = max(width, height);
		//output[(i - 1) * 8 + 4] = min(width, height);
		//output[(i - 1) * 8 + 5] = max(width, height) / min(width, height);
		//output[(i - 1) * 8 + 6] = area;
		//output[(i - 1) * 8 + 7] = totaldata[i] / area;
		//queryStr = queryStr +str_format(" (%d, %f, %f, %f, %f, %f, %f, %f)", i, datatmp.box[0], datatmp.box[1],
		//	max(datatmp.box[2], datatmp.box[3]), min(datatmp.box[2], datatmp.box[3]), max(datatmp.box[2], datatmp.box[3])/ 
		//	min(datatmp.box[2], datatmp.box[3]), datatmp.area, datatmp.avggray);
		//if (i < num-1) {
		//	queryStr = queryStr + ",";
		//}
		outputinfotmp.push_back(datatmp);

		//printf("area : %d, center point(%.2f, %.2f)\n", area, pt[0], pt[1]);//�����Ϣ
		//circle(img, Point(pt[0], pt[1]), 1, 0, -1);//���ĵ�����
		//rectangle(img, Rect(x, y, width, height), 255, 1, 8, 0);//��Ӿ���
	}
	//mysql_query(&mysql, queryStr.c_str());

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
	//	cout << i << "���� len: " << len << "  wid:" << wid << " rate: " << len / wid << "  area: "<<area << endl;
	//	//rectangle(img, boundRect[i].tl(), boundRect[i].br(), (0, 0, 255), 2, 8, 0);
	//	//rectangle(img, boundRect[i].tl(), boundRect[i].br(), 255, 1, 1, 0);
	//	//rectangle(srcImage,rect,(255, 0, 0), 2, 8, 0);
	//}
	CombineCounter(outputinfotmp, outputinfo);
	std::cout << "  transfer data��" <<
		std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() -
			start).count() << "ms" << std::endl;
	std::cout << " 覴�:" << num - 1 << " ������Ϣʱ�䣺" <<
		std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() -
			starttmp).count() << "ms" << std::endl;
}

void CalAvgPixel(Mat brightImg, Mat darkImg, std::vector<Rect> rectVector, AvgPixelData& avgPixelData)
{
	uchar* adata = new uchar[brightImg.size().width-20]; //ÿ��ƽ����������
	uchar* ddata = new uchar[darkImg.size().width - 20]; //ÿ��ƽ����������
	vector<int> sumpixel(brightImg.size().width - 20, 0);
	vector<int> darksumpixel(darkImg.size().width - 20, 0);
	for (auto i = 0; i < brightImg.size().height; i=i+10) {
		uchar* sdata = brightImg.data + brightImg.step * i; //��������
		for (auto j = 10; j < brightImg.size().width-10; j++) {
			sumpixel[j - 10] += sdata[j];
		}
	}
	for (auto i = 0; i < darkImg.size().height; i = i + 10) {
		uchar* kdata = darkImg.data + darkImg.step * i; //��������
		for (auto j = 10; j < darkImg.size().width - 10; j++) {
			darksumpixel[j - 10] += kdata[j];
		}
	}
	int startPos = rectVector[0].x + rectVector[0].width / 3;
	int endPos = rectVector[0].x + 2 * rectVector[0].width / 3;
	int mindata = 255;
	int maxdata = 100;
	for (auto j = 0; j < sumpixel.size(); j++) {
		adata[j] = int(1.0 * sumpixel[j] / brightImg.size().height*10);
		if (j > startPos && j < endPos) {
			if (adata[j] > maxdata) {
				maxdata = adata[j];
			}
			if (adata[j] < mindata && adata[j]>100) {
				mindata = adata[j];
			}
		}
	}
	Threshold[0] = mindata - 10;
	Threshold[1] = maxdata + 10;
	cout << "����Ҷ������ޣ�" << Threshold[0] << "   " << Threshold[1] << endl;
	for (auto j = 0; j < darksumpixel.size(); j++) {
		ddata[j] = int(1.0*darksumpixel[j] / darkImg.size().height * 10);
	}
 	avgPixelData.avgpixelsize = brightImg.size().width - 20;
	avgPixelData.avgpixel = adata;
	avgPixelData.darkavgpixelsize = darkImg.size().width - 20;
	avgPixelData.darkavgpixel = ddata;
	cout << "cal avgpixel over" <<endl;
}
void handle(char* imgpath, char* imgpath1, DefectData* &outputinfo, int& outputInfoLen,  AvgPixelData& avgPixelData) {
	std::string input_ImagePath = imgpath;
	std::string input_ImagePath1 = imgpath1;
	Mat img1;
	Mat darimg1;
	ReadImgarray(input_ImagePath, img1);//; ("D://C#//xiaci_pinjie_1209.bmp")
	ReadImgarray(input_ImagePath1, darimg1);
//void handle(uchar* data, uchar* data1, int width, int height, int stride, DefectData*& outputinfo, int& outputInfoLen, AvgPixelData& avgPixelData){
//	//c# ͼ�����
//	cv::Mat img01 = cv::Mat(cv::Size(width, height), CV_8UC3, data, stride);
//	Mat img1;
//	cvtColor(img01, img1, COLOR_BGR2GRAY);
//	cv::Mat img2 = cv::Mat(cv::Size(width, height), CV_8UC3, data1, stride);
//	Mat darimg1;
//	cvtColor(img2, darimg1, COLOR_BGR2GRAY);
	auto start = std::chrono::system_clock::now();
	auto starttmp = std::chrono::system_clock::now();

	//// ��ȶ��̴߳���ͼ��
	//int singlewidth =  img.size().width;
	//int imgwidth = img.size().width;
	//bool divideflag = imgwidth % singlewidth;
	//int threads = divideflag ? (imgwidth / singlewidth) + 1 : (imgwidth / singlewidth);
	//std::vector<std::thread> mythreads;
	//vector<vector<DefectData>> outputInfotmp(threads);
	//for (size_t i = 0; i < threads; i++) {
	//	int singlewidthtmp = (i == threads - 1) && divideflag ? (imgwidth - i * singlewidth) : singlewidth;
	//	Rect rect(i* singlewidth, 0, singlewidthtmp, img.size().height);
	//	Mat singleimg = img(rect);
	//	Mat singledarimg = darimg(rect);
	//	mythreads.push_back(std::thread(findcounter, i * singlewidth, singleimg, singledarimg, ref(outputInfotmp[i])));
	//}
	//mythreads.push_back(std::thread(CalAvgPixel, img, ref(avgPixelData)));
	//for (auto it = mythreads.begin(); it != mythreads.end(); ++it) {
	//	it->join();
	//}
	//vector<DefectData> outputInfo;
	//for (size_t i = 0; i < threads; i++) {
	//	outputInfo.insert(outputInfo.end(), outputInfotmp[i].begin(), outputInfotmp[i].end());
	//}

	//Rect initrect = findRect(img1);
	//Mat img = img1(initrect);
	//Mat darimg = darimg1(initrect);
	//std::thread threadcal(CalAvgPixel, img1, ref(avgPixelData));
	//threadcal.detach();
	//threadcal.join();
	vector<DefectData> outputInfo;

	//�����ߴ���
	std::vector<Rect> rectVector = findRect(img1, 0);
	//std::vector<Rect> rectVector;
	//rectVector.push_back(Rect(13360, 0, 17000, 2000));
	std::vector<std::thread> mythreads;
	mythreads.push_back(std::thread(CalAvgPixel, img1, darimg1, rectVector, ref(avgPixelData)));
	if (rectVector.size() == 0) {
		avgPixelData.initPos = rectVector[0].x - 10;
		avgPixelData.endPos = rectVector[0].x+ rectVector[0].width+20;
	}
	else {
		avgPixelData.initPos = rectVector[0].x - 10;
		avgPixelData.endPos = rectVector[rectVector.size()-1].x + rectVector[rectVector.size() - 1].width + 20;
	}
	vector<vector<DefectData>> outputInfotmp(rectVector.size());
	for (size_t i = 0; i < rectVector.size(); i++) {
		Mat singleimg = img1(rectVector[i]);
		Mat singledarimg = darimg1(rectVector[i]);
		mythreads.push_back(std::thread(findcounter, rectVector[i].x, singleimg, singledarimg, ref(outputInfotmp[i])));
	}
	
	for (auto it = mythreads.begin(); it != mythreads.end(); ++it) {
		it->join();
	}
	for (size_t i = 0; i < rectVector.size(); i++) {
		outputInfo.insert(outputInfo.end(), outputInfotmp[i].begin(), outputInfotmp[i].end());
	}


	//findcounter(initrect.x, img, darimg, outputInfo);
	outputInfoLen = outputInfo.size();
	outputinfo = new DefectData[outputInfoLen];
    memcpy(outputinfo, &outputInfo[0], outputInfoLen * sizeof(DefectData));
	std::cout << outputInfoLen<<" ��覴����������ʱ�䣺" <<
		std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() -
			start).count() << "ms" << std::endl;
}

void release(DefectData*& outputinfo) 
{
	if (NULL != outputinfo) {
		delete[] outputinfo;
	}
}

void main()
{
	string path = R"(D:\c++\image\0919\1��.bmp)";//R"(D:\c++\Defect_detection\test.bmp)";
	char* imgpath = (char*)path.c_str();
	string path1 = R"(D:\c++\image\0919\1��.bmp)";//R"(D:\c++\Defect_detection\test.bmp)";
	char* imgpath1 = (char*)path1.c_str();

	string detectparam = R"({"DarkGrayVal":110,"BrightGrayVal":140})";

	string classparam = R"([{"ȱ������":"���","�ж�����1":"ƽ������","�ж�����2":"","�ж�����3":"","�ж�����4":"","�ж�����5":"",
"ͼ������":"","ͼ����ɫ":"","�������mm2":"0.040","�������mm2":"0.049","��������mm":"","��������mm":"","�������mm":"","�������mm":"",
"���������":"","���������":"","ƽ����������":"80","ƽ����������":"","�ȼ�0":"","�ȼ�1":"","�ȼ�2":"","�ȼ�3":"","�ȼ�4":""},
{"ȱ������":"͸��","�ж�����1":"ƽ������","�ж�����2":"","�ж�����3":"","�ж�����4":"","�ж�����5":"","ͼ������":"","ͼ����ɫ":"",
"�������mm2":"0.040","�������mm2":"0.049","��������mm":"","��������mm":"","�������mm":"","�������mm":"","���������":"","���������":"",
"ƽ����������":"37","ƽ����������":"","�ȼ�0":"","�ȼ�1":"","�ȼ�2":"","�ȼ�3":"","�ȼ�4":""}])";
	
	GetDetectJsonParam((char*)detectparam.c_str());
	GetClassJsonParam((char*)classparam.c_str());
	int outputInfoLen = -1;
	DefectData* outputinfo;
	AvgPixelData outputpixeldata;
	for (auto i = 0; i < 2; i++) {
		handle(imgpath, imgpath1, outputinfo, outputInfoLen, outputpixeldata);
	}
	//release(outputinfo);
}
