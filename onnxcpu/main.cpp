#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <onnxruntime_cxx_api.h>	 // C��c++��api
#include <math.h>
#include <future>
#include <thread>
#include<direct.h> 
#pragma warning(disable:4996)
// �����ռ�
using namespace std;
using namespace cv;
using namespace Ort;

struct RecResult {
	char imgname[100];
	int reallabel;
	int id;             //������id
	double confidence;   //������Ŷ�
	int box[4];       //���ο�
	int bytesize;
	//BYTE* boxMask;
};

// �Զ������ýṹ
struct Configuration
{
public:
	float confThreshold; // Confidence threshold
	float nmsThreshold;  // Non-maximum suppression threshold
	float objThreshold;  //Object Confidence threshold
	string modelpath;
	string modelmode;
};

// ����BoxInfo�ṹ����
typedef struct BoxInfo
{
	float x1;
	float y1;
	float x2;
	float y2;
	float score;
	int label;
} BoxInfo;

// int endsWith(string s, string sub) {
// 	return s.rfind(sub) == (s.length() - sub.length()) ? 1 : 0;
// }

// const float anchors_640[3][6] = { {10.0,  13.0, 16.0,  30.0,  33.0,  23.0},
// 								 {30.0,  61.0, 62.0,  45.0,  59.0,  119.0},
// 								 {116.0, 90.0, 156.0, 198.0, 373.0, 326.0} };

// const float anchors_1280[4][6] = { {19, 27, 44, 40, 38, 94},{96, 68, 86, 152, 180, 137},{140, 301, 303, 264, 238, 542},
// 					   {436, 615, 739, 380, 925, 792} };

class YOLO
{
public:
	YOLO(Configuration config);
	void detect(Mat& frame, string imgname, std::vector<RecResult>& output);
private:
	float confThreshold;
	float nmsThreshold;
	float objThreshold;
	string recmode;
	int mask_num = 32;
	int inpWidth;
	int inpHeight;
	int nout;
	int yolomode;
	int num_proposal;
	int num_classes;
	int segMaskWidth;
	int segMaskHeight;
	int segMaskChannel;
	const bool keep_ratio = true;
	vector<float> input_image_;		// ����ͼƬ
	void GetConfigValue(const char* keyName, char* keyValue);
	void normalize_(Mat img, int channels);		// ��һ������
	void nms(vector<BoxInfo>& input_boxes);
	Mat resize_image(Mat srcimg, int* newh, int* neww, int* top, int* left, string recmode);

	
	Session* ort_session = nullptr;    // ��ʼ��Sessionָ��ѡ��
	Ort::Env env;
	SessionOptions sessionOptions = SessionOptions();  //��ʼ��Session����
	//SessionOptions sessionOptions;
	vector<char*> input_names;  // ����һ���ַ�ָ��vector
	vector<char*> output_names; // ����һ���ַ�ָ��vector
	vector<vector<int64_t>> input_node_dims; // >=1 outputs  ����άvector
	vector<vector<int64_t>> output_node_dims; // >=1 outputs ,int64_t C/C++��׼
};

YOLO::YOLO(Configuration config)
{
	this->confThreshold = config.confThreshold;
	this->nmsThreshold = config.nmsThreshold;
	this->objThreshold = config.objThreshold;
	//char engine_filepath[1000] = { 0 };
	//char enginemode[100] = { 0 };
	//GetConfigValue("engine_file_path", engine_filepath);
	//GetConfigValue("engine_mode", enginemode);
	//engine_filepath[strlen(engine_filepath) - 1] = 0;
	recmode = config.modelmode;
	string model_pathtmp = config.modelpath;
	//std::wstring widestr = std::wstring(model_path.begin(), model_path.end());  //����UTF-16������ַ�

	////gpu, https://blog.csdn.net/weixin_44684139/article/details/123504222
	////CUDA���ٿ���
	//OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);

	//sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);  //����ͼ�Ż�����
	//ort_session = new Session(env, widestr.c_str(), sessionOptions);  // �����Ự����ģ�ͼ��ص��ڴ���
	//ort_session = new Session(env, (const ORTCHAR_T*)model_path.c_str(), sessionOptions); // �����Ự����ģ�ͼ��ص��ڴ���
	env = Env(ORT_LOGGING_LEVEL_ERROR, "onnxtest"); // ��ʼ������
	wchar_t* model_path = new wchar_t[model_pathtmp.size()];
	swprintf(model_path, 100, L"%S", model_pathtmp.c_str());
	//const wchar_t* model_path = L"hole_cls.onnx";
	ort_session = new Session(env, model_path, sessionOptions);
	size_t numInputNodes = ort_session->GetInputCount();  //��������ڵ�����                         
	size_t numOutputNodes = ort_session->GetOutputCount();
	AllocatorWithDefaultOptions allocator;   // ������������ڵ��ڴ�
	for (int i = 0; i < numInputNodes; i++)
	{
		input_names.push_back(ort_session->GetInputName(i, allocator));		// �ڴ�
		Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);   // ����
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();  // 
		auto input_dims = input_tensor_info.GetShape();    // ����shape
		input_node_dims.push_back(input_dims);	// ����
	}
	for (int i = 0; i < numOutputNodes; i++)
	{
		output_names.push_back(ort_session->GetOutputName(i, allocator));
		Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		auto output_dims = output_tensor_info.GetShape();
		output_node_dims.push_back(output_dims);
	}
	vector<int64_t> outdims;
	vector<int64_t> outmaskdims;
	if (output_node_dims.size() > 1) {
		for (int i = 0; i < output_node_dims.size(); i++) {
			if (output_node_dims[i].size() == 4) {
				outmaskdims = output_node_dims[i];
			}
			if (output_node_dims[i].size() == 3) {
				outdims = output_node_dims[i];
			}
		}
	}
	else {
		outdims = output_node_dims[0];
	}
	if (outdims.size() == 2) { //clsģ���޷�������ά������v5��v8
		nout = outdims[0] * outdims[1];
	}
	else {
		if (outdims[1] < outdims[2]) {
			this->yolomode = 1;
			this->nout = outdims[1];      // 4+classes
			this->num_proposal = outdims[2];  // pre_box
			cout << "the onnx is yolov8" << endl;
		}
		else {
			this->yolomode = 0;
			this->nout = outdims[2];      // 5+classes
			this->num_proposal = outdims[1];  // pre_box
			cout << "the onnx is yolov5" << endl;
		}
	}
	if (recmode == "cls") {
		num_classes = outdims[1];
	}
	else if (recmode == "obj") {
		if (yolomode == 1) {
			num_classes = outdims[1]-4;
		}
		else {
			num_classes = outdims[2] - 5;
		}	
	}
	else {
		if (yolomode == 1) {
			num_classes = outdims[1] - mask_num- 4;
		}
		else {
			num_classes = outdims[2] - mask_num- 5;
		}
		segMaskWidth = outmaskdims[3];
		segMaskHeight = outmaskdims[2];
		segMaskChannel = outmaskdims[1];
	}
	this->inpHeight = input_node_dims[0][2];
	this->inpWidth = input_node_dims[0][3];
}

void YOLO::GetConfigValue(const char* keyName, char* keyValue)
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

Mat YOLO::resize_image(Mat srcimg, int* newh, int* neww, int* top, int* left, string recmode)
{
	int srch = srcimg.rows, srcw = srcimg.cols;
	*newh = this->inpHeight;
	*neww = this->inpWidth;
	Mat dstimg;
	if (recmode =="cls") { //���ࡪ�����Ĳü�
		int minSize = std::min(srch, srcw);
		int topPoint = (srch - minSize) / 2;
		int leftPoint = (srcw - minSize) / 2;
		resize(srcimg(Rect(leftPoint, topPoint, minSize, minSize)), dstimg, Size(*neww, *newh), INTER_LINEAR);
	}
	else {//��⡢�ָ���ȱ����ü�
		if (this->keep_ratio && srch != srcw) {
			float hw_scale = (float)srch / srcw;
			if (hw_scale > 1) {
				*newh = this->inpHeight;
				*neww = int(this->inpWidth / hw_scale);
				resize(srcimg, dstimg, Size(*neww, *newh), INTER_LINEAR);
				*left = int((this->inpWidth - *neww) * 0.5);
				copyMakeBorder(dstimg, dstimg, 0, 0, *left, this->inpWidth - *neww - *left, BORDER_CONSTANT, cv::Scalar(114, 114, 114));
			}
			else {
				*newh = (int)this->inpHeight * hw_scale;
				*neww = this->inpWidth;
				resize(srcimg, dstimg, Size(*neww, *newh), INTER_LINEAR);
				*top = (int)(this->inpHeight - *newh) * 0.5;
				copyMakeBorder(dstimg, dstimg, *top, this->inpHeight - *newh - *top, 0, 0, BORDER_CONSTANT, cv::Scalar(114, 114, 114));
			}
		}
		else {
			resize(srcimg, dstimg, Size(*neww, *newh), INTER_LINEAR);
		}
	}
	return dstimg;
}

float MeanArray[3] = { 0.485,0.456,0.406 };
float StdArray[3] = { 0.229, 0.224, 0.225 };
void YOLO::normalize_(Mat img, int channels)
{
	//    img.convertTo(img, CV_32F);

	int row = img.rows;
	int col = img.cols;
	this->input_image_.resize(row * col * img.channels());  // vector��С
	for (int c = 0; c < channels; c++)  // bgr
	{
		for (int i = 0; i < row; i++)  // ��
		{
			for (int j = 0; j < col; j++)  // ��
			{
				float pix = img.ptr<uchar>(i)[j * 3 + 2 - c];  // Mat���ptr������������һ�����ص��׵�ַ,2-c:��ʾrgb
				this->input_image_[c * row * col + i * col + j] = pix / 255.0;
				if (recmode == "cls") {
					this->input_image_[c * row * col + i * col + j] = (this->input_image_[c * row * col + i * col + j] - MeanArray[c]) / StdArray[c];
				}
			}
		}
	}
}

void YOLO::nms(vector<BoxInfo>& input_boxes)
{
	sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; }); // ��������
	vector<float> vArea(input_boxes.size());
	for (int i = 0; i < input_boxes.size(); ++i)
	{
		vArea[i] = (input_boxes[i].x2 - input_boxes[i].x1 + 1)
			* (input_boxes[i].y2 - input_boxes[i].y1 + 1);
	}
	// ȫ��ʼ��Ϊfalse��������Ϊ��¼�Ƿ�����Ӧ������pre_box�ı�־vector
	vector<bool> isSuppressed(input_boxes.size(), false);
	for (int i = 0; i < input_boxes.size(); ++i)
	{
		if (isSuppressed[i]) { continue; }
		for (int j = i + 1; j < input_boxes.size(); ++j)
		{
			if (isSuppressed[j]) { continue; }
			float xx1 = max(input_boxes[i].x1, input_boxes[j].x1);
			float yy1 = max(input_boxes[i].y1, input_boxes[j].y1);
			float xx2 = min(input_boxes[i].x2, input_boxes[j].x2);
			float yy2 = min(input_boxes[i].y2, input_boxes[j].y2);

			float w = max(0.0f, xx2 - xx1 + 1);
			float h = max(0.0f, yy2 - yy1 + 1);
			float inter = w * h;	// ����
			if (input_boxes[i].label == input_boxes[j].label)
			{
				float ovr = inter / (vArea[i] + vArea[j] - inter);  // ����iou
				if (ovr >= this->nmsThreshold)
				{
					isSuppressed[j] = true;
				}
			}
		}
	}
	// return post_nms;
	int idx_t = 0;
	// remove_if()���� remove_if(beg, end, op) //�Ƴ�����[beg,end)��ÿһ�������ж�ʽ:op(elem)���true����Ԫ��
	input_boxes.erase(remove_if(input_boxes.begin(), input_boxes.end(), [&idx_t, &isSuppressed](const BoxInfo& f) { return isSuppressed[idx_t++]; }), input_boxes.end());
	// ��һ��д��
	// sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; }); // ��������
	// vector<bool> remove_flags(input_boxes.size(),false);
	// auto iou = [](const BoxInfo& box1,const BoxInfo& box2)
	// {
	// 	float xx1 = max(box1.x1, box2.x1);
	// 	float yy1 = max(box1.y1, box2.y1);
	// 	float xx2 = min(box1.x2, box2.x2);
	// 	float yy2 = min(box1.y2, box2.y2);
	// 	// ����
	// 	float w = max(0.0f, xx2 - xx1 + 1);
	// 	float h = max(0.0f, yy2 - yy1 + 1);
	// 	float inter_area = w * h;
	// 	// ����
	// 	float union_area = max(0.0f,box1.x2-box1.x1) * max(0.0f,box1.y2-box1.y1)
	// 					   + max(0.0f,box2.x2-box2.x1) * max(0.0f,box2.y2-box2.y1) - inter_area;
	// 	return inter_area / union_area;
	// };
	// for (int i = 0; i < input_boxes.size(); ++i)
	// {
	// 	if(remove_flags[i]) continue;
	// 	for (int j = i + 1; j < input_boxes.size(); ++j)
	// 	{
	// 		if(remove_flags[j]) continue;
	// 		if(input_boxes[i].label == input_boxes[j].label && iou(input_boxes[i],input_boxes[j])>=this->nmsThreshold)
	// 		{
	// 			remove_flags[j] = true;
	// 		}
	// 	}
	// }
	// int idx_t = 0;
	// // remove_if()���� remove_if(beg, end, op) //�Ƴ�����[beg,end)��ÿһ�������ж�ʽ:op(elem)���true����Ԫ��
	// input_boxes.erase(remove_if(input_boxes.begin(), input_boxes.end(), [&idx_t, &remove_flags](const BoxInfo& f) { return remove_flags[idx_t++]; }), input_boxes.end());
}

template <typename T, typename A>
int arg_max(std::vector<T, A> const& vec) {
	return static_cast<int>(std::distance(vec.begin(), max_element(vec.begin(), vec.end())));
}

vector<float> softmax(vector<float> input)
{
	double total = 0;
	float MAX = input[0];
	for (auto x : input)
	{
		MAX = std::max(x, MAX);
	}
	for (auto x : input)
	{
		total += exp(x);
	}
	vector<float> result;
	for (auto x : input)
	{
		result.push_back(exp(x) / total);
	}
	return result;
}
string labelstr[2] = { {"cat"},{"dog"} };

void YOLO::detect(Mat& frame, string imgname, std::vector<RecResult>& output)
{
	int newh = 0, neww = 0, padh = 0, padw = 0;
	Mat dstimg = this->resize_image(frame, &newh, &neww, &padh, &padw, recmode);
	this->normalize_(dstimg, frame.channels());
	// ����һ���������int64_t��������Ϊ�������ʱ������
	array<int64_t, 4> input_shape_{ 1, frame.channels(), this->inpHeight, this->inpWidth };

	//��������tensor
	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input_image_.data(),
		input_image_.size(), input_shape_.data(), input_shape_.size());

	// ��ʼ����
	vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr }, &input_names[0], &input_tensor_, 1,
		output_names.data(), output_names.size());   // ��ʼ����
	/////generate proposals
	vector<BoxInfo> generate_boxes;  // BoxInfo�Զ���Ľṹ��
	float ratioh = (float)frame.rows / newh, ratiow = (float)frame.cols / neww;
	float* prob = ort_outputs[0].GetTensorMutableData<float>(); // GetTensorMutableData
	if (recmode == "cls") {
		vector<float> vecprobtmp(prob, prob + nout);
		vector<float> vecprob;
		if (find_if(vecprobtmp.begin(), vecprobtmp.end(), [](float i) { return i > 1; }) != vecprobtmp.end()) {
			vecprob = softmax(vecprobtmp);
		}
		else {
			vecprob = vecprobtmp;
		}
		RecResult result;
		result.id = arg_max(vecprob);
		result.confidence = vecprob[arg_max(vecprob)];
		strcpy(result.imgname, imgname.c_str());
		output.push_back(result);
		string label = format("label: %s score: %.2f", labelstr[result.id],result.confidence);
		putText(frame, label, Point(10, 30), FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255));
		std::cout << "label: " << arg_max(vecprob)<<"   score: " <<vecprob[arg_max(vecprob)] << std::endl;
	}
	if (recmode == "obj") {
		int numbox = -1, boxscore = 0;
		float* pdata;
		if (yolomode == 1) {
			numbox = num_proposal;
			boxscore = 4;
			cv::Mat outputtrans = cv::Mat(num_classes + 4, numbox, CV_32F, prob).t();
			pdata = outputtrans.ptr<float>();
		}else {
			numbox = 1000;
			boxscore = 5;
			pdata = prob;
		}
		for (int i = 0; i < numbox; ++i) // �������е�num_pre_boxes
		{
			int index = i * nout;      // prob[b*num_pred_boxes*(classes+5)]  
			float obj_conf = pdata[index + 4];  // ���Ŷȷ���
			if (yolomode == 0 && obj_conf < this->objThreshold) continue;
			int class_idx = 0;
			float max_class_socre = 0;
			for (int k = 0; k < this->num_classes; ++k)
			{
				if (pdata[k + index + boxscore] > max_class_socre)
				{
					max_class_socre = pdata[k + index + boxscore];
					class_idx = k;
				}
			}
			if (yolomode == 0) {
				max_class_socre *= obj_conf;   // ����������*���Ŷ�
			}
			if (max_class_socre > this->confThreshold) // �ٴ�ɸѡ
			{
				//const int class_idx = classIdPoint.x;
				float cx = pdata[index];  //x
				float cy = pdata[index + 1];  //y
				float w = pdata[index + 2];  //w
				float h = pdata[index + 3];  //h

				float xmin = (cx - padw - 0.5 * w) * ratiow;
				float ymin = (cy - padh - 0.5 * h) * ratioh;
				float xmax = (cx - padw + 0.5 * w) * ratiow;
				float ymax = (cy - padh + 0.5 * h) * ratioh;

				generate_boxes.push_back(BoxInfo{ xmin, ymin, xmax, ymax, max_class_socre, class_idx });
			}
		}

		// Perform non maximum suppression to eliminate redundant overlapping boxes with
		// lower confidences
		nms(generate_boxes);
		for (size_t i = 0; i < generate_boxes.size(); ++i)
		{
			int xmin = int(generate_boxes[i].x1);
			int ymin = int(generate_boxes[i].y1);
			RecResult result;
			strcpy(result.imgname, imgname.c_str());
			result.id = generate_boxes[i].label;
			result.confidence = generate_boxes[i].score;
			result.box[0] = int(generate_boxes[i].x1);
			result.box[1] = int(generate_boxes[i].y1);
			result.box[2] = int(generate_boxes[i].x2 - generate_boxes[i].x1);
			result.box[3] = int(generate_boxes[i].y2 - generate_boxes[i].y1);
			output.push_back(result);
			rectangle(frame, Point(xmin, ymin), Point(int(generate_boxes[i].x2), int(generate_boxes[i].y2)), Scalar(0, 0, 255), 2);
			string label = format("%.2f", generate_boxes[i].score);
			cout << (generate_boxes[i].x1 + generate_boxes[i].x2) / 2 / frame.cols << endl;
			cout << (generate_boxes[i].y1 + generate_boxes[i].y2) / 2 / frame.rows << endl;
			cout << (generate_boxes[i].x2 - generate_boxes[i].x1) / frame.cols << endl;
			cout << (generate_boxes[i].y2 - generate_boxes[i].y1) / frame.rows << endl;
			cout << generate_boxes[i].label << endl;
			cout << generate_boxes[i].score << endl;
			putText(frame, label, Point(xmin, ymin - 5), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);
		}
	}

	if (recmode == "seg") {
		float* prob1 = ort_outputs[1].GetTensorMutableData<float>();
		int numbox = -1, boxscore = 0;
		float* pdata;
		if (yolomode == 1) {
			numbox = num_proposal;
			boxscore = 4;
			cv::Mat outputtrans = cv::Mat(num_classes + 4 + segMaskChannel, numbox, CV_32F, prob).t();
			pdata = outputtrans.ptr<float>();
		}
		else {
			numbox = 1000;
			boxscore = 5;
			pdata = prob;
		}
		for (int i = 0; i < numbox; ++i) // �������е�num_pre_boxes
		{
			int index = i * nout;      // prob[b*num_pred_boxes*(classes+5)]  
			float obj_conf = pdata[index + 4];  // ���Ŷȷ���
			if (yolomode == 0 && obj_conf < this->objThreshold) continue;
			int class_idx = 0;
			float max_class_socre = 0;
			for (int k = 0; k < this->num_classes; ++k)
			{
				if (pdata[k + index + boxscore] > max_class_socre)
				{
					max_class_socre = pdata[k + index + boxscore];
					class_idx = k;
				}
			}
			if (yolomode == 0) {
				max_class_socre *= obj_conf;   // ����������*���Ŷ�
			}
			if (max_class_socre > this->confThreshold) // �ٴ�ɸѡ
			{
				//const int class_idx = classIdPoint.x;
				float cx = pdata[index];  //x
				float cy = pdata[index + 1];  //y
				float w = pdata[index + 2];  //w
				float h = pdata[index + 3];  //h

				float xmin = MAX((cx - padw - 0.5 * w) * ratiow,0);
				float ymin = MAX((cy - padh - 0.5 * h) * ratioh,0);
				float xmax = (cx - padw + 0.5 * w) * ratiow;
				float ymax = (cy - padh + 0.5 * h) * ratioh;

				generate_boxes.push_back(BoxInfo{ xmin, ymin, xmax, ymax, max_class_socre, class_idx });
			}
		}

		// Perform non maximum suppression to eliminate redundant overlapping boxes with
		// lower confidences
		nms(generate_boxes);
		for (size_t i = 0; i < generate_boxes.size(); ++i)
		{
			int xmin = int(generate_boxes[i].x1);
			int ymin = int(generate_boxes[i].y1);
			RecResult result;
			strcpy(result.imgname, imgname.c_str());
			result.id = generate_boxes[i].label;
			result.confidence = generate_boxes[i].score;
			result.box[0] = int(generate_boxes[i].x1);
			result.box[1] = int(generate_boxes[i].y1);
			result.box[2] = int(generate_boxes[i].x2 - generate_boxes[i].x1);
			result.box[3] = int(generate_boxes[i].y2 - generate_boxes[i].y1);
			output.push_back(result);
			rectangle(frame, Point(xmin, ymin), Point(int(generate_boxes[i].x2), int(generate_boxes[i].y2)), Scalar(0, 0, 255), 2);
			string label = format("%.2f", generate_boxes[i].score);
			cout << generate_boxes[i].x1 << endl;
			cout << generate_boxes[i].y1 << endl;
			cout << generate_boxes[i].x2 - generate_boxes[i].x1 << endl;
			cout << generate_boxes[i].y2 - generate_boxes[i].y1 << endl;
			cout << generate_boxes[i].label << endl;
			cout << generate_boxes[i].score << endl;
			putText(frame, label, Point(xmin, ymin - 5), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);
		}
	}
}

extern "C"
{
	__declspec(dllexport) void* AiCPUInit(const char* modelpath, const char* modelmode);
	__declspec(dllexport) void AiCPUDetect(void* h, cv::Mat img);
	__declspec(dllexport) void AiCPUDetectImg(void* h);
	__declspec(dllexport) void AiCPUDetectPath(void* h, const char* imgpath, RecResult*& output, int& outlen);
}

void* AiCPUInit(const char* modelpath, const char* modelmode) {
	string model_path, model_mode;
	model_path = modelpath;
	model_mode = modelmode;
	Configuration yolo_nets = { 0.25, 0.5, 0.25,model_path,model_mode };
	YOLO* yolo_modelptr = new YOLO(yolo_nets);
	return (void*)yolo_modelptr;
}

void AiCPUDetect(void* h, string image,cv::Mat img) {
	YOLO* modelptr = (YOLO*)h;
	std::vector<RecResult> output;
	modelptr->detect(img,image, output);
}

void AiCPUDetectPath(void* h, const char* imgpath, RecResult*& result, int& outlen) {
	YOLO* modelptr = (YOLO*)h;
	std::vector<cv::String> imgLists;
	string path;
	path = imgpath;
	cv::glob(path, imgLists, true);
	size_t pos = path.rfind("\\");
	string temp = "./result/" + path.substr(pos+1);
	//mkdir(temp.c_str());
	std::vector<RecResult> output;
	for (auto img : imgLists) {
		std::cout << std::string(img) << std::endl;
		cv::Mat srcimg = cv::imread(img);
		string imgname = img.substr(path.size()+1);
		modelptr->detect(srcimg,imgname,output);
		pos = img.rfind("\\");
		imwrite(temp+ img.substr(pos), srcimg);
	}
	outlen = output.size();
	result = new RecResult[outlen];
	memcpy(result, &output[0], outlen * sizeof(RecResult));
}

vector<string> split(string str, string sep)
{
	vector<string> result; // �洢�ָ����Ӵ�
	int start = 0; // ��ʼλ��
	int end = 0; // ����λ��
	while ((end = str.find(sep, start)) != string::npos) // ���ҷָ���
	{
		result.push_back(str.substr(start, end - start)); // ��ȡ�Ӵ�����������
		start = end + sep.size(); // ������ʼλ��
	}
	result.push_back(str.substr(start)); // �������һ���Ӵ�
	return result;
}

void find_circle(void* init, string image)
{
	Mat img = imread(image);
	Mat img_gray;
	cvtColor(img, img_gray, COLOR_BGR2GRAY);

	// ������ֵ�����ֵ
	double thresh = 150;
	double maxval = 255;
	Mat kernel = getStructuringElement(MORPH_RECT, Size(31, 31)); // ���νṹԪ��
	// ʹ��cv::threshold()����������ֵ�ָ�
	Mat img_binary;
	Mat roi(img_gray, Rect(300, 300, 5000, 2500));
	threshold(roi, img_binary, thresh, maxval, THRESH_BINARY);
	Mat img_close;
	morphologyEx(img_binary, img_close, MORPH_OPEN, kernel, Point(-1, -1), 3);
	Mat img_binary1;
	threshold(img_close, img_binary1, thresh, maxval, THRESH_BINARY_INV);

	// ʹ��cv::findContours()��������ͼ���е�����
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(img_binary1, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	//string sep = "\\";
	//vector<std::string> result = split(image, sep);
	//string sep1 = ".";
	//vector<std::string> result1 = split(result[2], sep1);
	// �������ҵ�������
	for (size_t i = 0; i < contours.size(); i++)
	{
		if (contours[i].size() < 20) {
			continue;
		}
		// ʹ��cv::minEnclosingCircle()����������������С���Բ
		Point2f center;
		float radius;
		minEnclosingCircle(contours[i], center, radius);

		// ������������С���Բ
		drawContours(img_binary1, contours, i, Scalar(0, 255, 0), 2, 8, hierarchy);
		circle(img_binary1, center, radius, Scalar(0, 0, 255), 2, 8, 0);

		// ������Ӿ��ε�����
		int x = center.x - radius;
		int y = center.y - radius;
		int w = radius * 2;
		int h = radius * 2;
		Mat roi1(roi, Rect(x - 50, y - 50, w + 100, h + 60));
		Mat roiinput;
		cvtColor(roi1, roiinput, COLOR_GRAY2BGR);
		cout << "image: " << image <<"  ��" << i+1<<"������"<< endl;
		AiCPUDetect(init, image, roiinput);
		//imwrite(path + result1[0] + "-" + to_string(i) + ".jpg", roi1);

		// ������Ӿ���
		//rectangle(roi, Rect(x, y, w, h), Scalar(255, 0, 0), 2, 8, 0);
	}
}

void AiCPUDetectImg(void* h) {
	string path = R"(./img/)";
	//string save_path = R"(D:\shuju\small\)";
	string pattern = "*.tif";
	vector<cv::String> filenames;
	glob(path + pattern, filenames, false);
	for (auto filename : filenames) {
		find_circle(h, filename);
	}
	// ����һ�� std::future ������
	//vector<std::future<void>> futures;

	//for (size_t i = 0; i < filenames.size(); i++) {
	//	// ʹ�� std::async �첽ִ�� find_circle �������������ص� std::future ������ӵ�������
	//	futures.push_back(std::async(std::launch::async, find_circle, h, filenames[i]));
	//}

	//// �����������ȴ�ÿ���̵߳����
	//for (auto& f : futures) {
	//	f.get();
	//}
}

int main(int argc, char* argv[])
{
	clock_t startTime, endTime; //����ʱ��
	string img= R"(D:\C#\xumeng31\Glide4NetDemo\Glide4NetDemo\bin\Debug\data\catsdogs\test)";
	const char* p = img.c_str();
	string modelpathstr = "D:\\c++\\onnxcpu\\model-2023-4/best.onnx";
	const char* modelpath = modelpathstr.c_str();
	string modelmodestr = "cls";
	const char* modelmode = modelmodestr.c_str();
	RecResult* output;
	int outlen = 0;
	AiCPUDetectPath(AiCPUInit(modelpath,modelmode), p, output, outlen);

	//cv::Mat srcimg = cv::imread(img);
	//AiCPUDetect(AiCPUInit(), srcimg);
	//AiCPUDetectImg(AiCPUInit());
	//std::string imgDir = "D:\\test\\good\\";
	//std::vector<cv::String> imgLists;
	//cv::glob(imgDir, imgLists, false);
	//for (auto img : imgLists) {
	//	std::cout << std::string(img) << std::endl;
	//	cv::Mat srcimg = cv::imread(img);
	//	double timeStart = (double)getTickCount();
	//	startTime = clock();//��ʱ��ʼ	
	//	yolo_modelptr->detect(srcimg);
	//	endTime = clock();//��ʱ����
	//	cout << "clock_running time is:" << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
	//}

	return 0;
}
