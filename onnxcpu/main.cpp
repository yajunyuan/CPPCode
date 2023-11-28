#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <onnxruntime_cxx_api.h>	 // C��c++��api
#include <math.h>
// �����ռ�
using namespace std;
using namespace cv;
using namespace Ort;

// �Զ������ýṹ
struct Configuration
{
public:
	float confThreshold; // Confidence threshold
	float nmsThreshold;  // Non-maximum suppression threshold
	float objThreshold;  //Object Confidence threshold
	string modelpath;
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

class YOLOv5
{
public:
	YOLOv5(Configuration config);
	void detect(Mat& frame, string engine_mode);
private:
	float confThreshold;
	float nmsThreshold;
	float objThreshold;
	int inpWidth;
	int inpHeight;
	int nout;
	int num_proposal;
	int num_classes;
	string classes[80] = { "person", "bicycle", "car", "motorbike", "aeroplane", "bus",
							"train", "truck", "boat", "traffic light", "fire hydrant",
							"stop sign", "parking meter", "bench", "bird", "cat", "dog",
							"horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
							"backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
							"skis", "snowboard", "sports ball", "kite", "baseball bat",
							"baseball glove", "skateboard", "surfboard", "tennis racket",
							"bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
							"banana", "apple", "sandwich", "orange", "broccoli", "carrot",
							"hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant",
							"bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
							"remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
							"sink", "refrigerator", "book", "clock", "vase", "scissors",
							"teddy bear", "hair drier", "toothbrush" };

	const bool keep_ratio = true;
	vector<float> input_image_;		// ����ͼƬ
	void normalize_(Mat img);		// ��һ������
	void nms(vector<BoxInfo>& input_boxes);
	Mat resize_image(Mat srcimg, int* newh, int* neww, int* top, int* left, string engine_mode);

	
	Session* ort_session = nullptr;    // ��ʼ��Sessionָ��ѡ��
	Ort::Env env;
	SessionOptions sessionOptions = SessionOptions();  //��ʼ��Session����
	//SessionOptions sessionOptions;
	vector<char*> input_names;  // ����һ���ַ�ָ��vector
	vector<char*> output_names; // ����һ���ַ�ָ��vector
	vector<vector<int64_t>> input_node_dims; // >=1 outputs  ����άvector
	vector<vector<int64_t>> output_node_dims; // >=1 outputs ,int64_t C/C++��׼
};

YOLOv5::YOLOv5(Configuration config)
{
	this->confThreshold = config.confThreshold;
	this->nmsThreshold = config.nmsThreshold;
	this->objThreshold = config.objThreshold;
	this->num_classes = 2;//sizeof(this->classes) / sizeof(this->classes[0]);  // �������
	this->inpHeight = 640;
	this->inpWidth = 640;

	string model_path1 = config.modelpath;
	//std::wstring widestr = std::wstring(model_path.begin(), model_path.end());  //����UTF-16������ַ�

	////gpu, https://blog.csdn.net/weixin_44684139/article/details/123504222
	////CUDA���ٿ���
	//OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);

	//sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);  //����ͼ�Ż�����
	//ort_session = new Session(env, widestr.c_str(), sessionOptions);  // �����Ự����ģ�ͼ��ص��ڴ���
	//ort_session = new Session(env, (const ORTCHAR_T*)model_path.c_str(), sessionOptions); // �����Ự����ģ�ͼ��ص��ڴ���
	env = Env(ORT_LOGGING_LEVEL_ERROR, "onnxtest"); // ��ʼ������
	const wchar_t* model_path = L"hole_cls.onnx";
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
	this->inpHeight = input_node_dims[0][2];
	this->inpWidth = input_node_dims[0][3];
	this->nout = output_node_dims[0][2];      // 5+classes
	this->num_proposal = output_node_dims[0][1];  // pre_box

}

Mat YOLOv5::resize_image(Mat srcimg, int* newh, int* neww, int* top, int* left, string engine_mode)
{
	int srch = srcimg.rows, srcw = srcimg.cols;
	*newh = this->inpHeight;
	*neww = this->inpWidth;
	Mat dstimg;
	if (engine_mode=="cls") { //���ࡪ�����Ĳü�
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
void YOLOv5::normalize_(Mat img)
{
	//    img.convertTo(img, CV_32F);
	int row = img.rows;
	int col = img.cols;
	this->input_image_.resize(row * col * img.channels());  // vector��С
	for (int c = 0; c < 3; c++)  // bgr
	{
		for (int i = 0; i < row; i++)  // ��
		{
			for (int j = 0; j < col; j++)  // ��
			{
				float pix = img.ptr<uchar>(i)[j * 3 + 2 - c];  // Mat���ptr������������һ�����ص��׵�ַ,2-c:��ʾrgb
				this->input_image_[c * row * col + i * col + j] = pix / 255.0;
				this->input_image_[c * row * col + i * col + j] = (this->input_image_[c * row * col + i * col + j] - MeanArray[c]) / StdArray[c];
			}
		}
	}
}

void YOLOv5::nms(vector<BoxInfo>& input_boxes)
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

void YOLOv5::detect(Mat& frame, string engine_mode)
{
	int newh = 0, neww = 0, padh = 0, padw = 0;
	Mat dstimg = this->resize_image(frame, &newh, &neww, &padh, &padw, engine_mode);
	this->normalize_(dstimg);
	// ����һ���������int64_t��������Ϊ�������ʱ������
	array<int64_t, 4> input_shape_{ 1, 3, this->inpHeight, this->inpWidth };

	//��������tensor
	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());

	// ��ʼ����
	vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr }, &input_names[0], &input_tensor_, 1, output_names.data(), output_names.size());   // ��ʼ����
	/////generate proposals
	vector<BoxInfo> generate_boxes;  // BoxInfo�Զ���Ľṹ��
	float ratioh = (float)frame.rows / newh, ratiow = (float)frame.cols / neww;
	float* pdata = ort_outputs[0].GetTensorMutableData<float>(); // GetTensorMutableData
	if (engine_mode == "cls") {
		vector<float> vecprobtmp(pdata, pdata + num_proposal);
		vector<float> vecprob = softmax(vecprobtmp);
		std::cout << vecprob[0] << "  " << vecprob[1] << " " << vecprob[2] << " " << arg_max(vecprob) << std::endl;
	}
	if (engine_mode == "obj") {
		for (int i = 0; i < num_proposal; ++i) // �������е�num_pre_boxes
		{
			int index = i * nout;      // prob[b*num_pred_boxes*(classes+5)]  
			float obj_conf = pdata[index + 4];  // ���Ŷȷ���
			if (obj_conf > this->objThreshold)  // ������ֵ
			{
				int class_idx = 0;
				float max_class_socre = 0;
				for (int k = 0; k < this->num_classes; ++k)
				{
					if (pdata[k + index + 5] > max_class_socre)
					{
						max_class_socre = pdata[k + index + 5];
						class_idx = k;
					}
				}
				max_class_socre *= obj_conf;   // ����������*���Ŷ�
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
		}

		// Perform non maximum suppression to eliminate redundant overlapping boxes with
		// lower confidences
		nms(generate_boxes);
		for (size_t i = 0; i < generate_boxes.size(); ++i)
		{
			int xmin = int(generate_boxes[i].x1);
			int ymin = int(generate_boxes[i].y1);
			rectangle(frame, Point(xmin, ymin), Point(int(generate_boxes[i].x2), int(generate_boxes[i].y2)), Scalar(0, 0, 255), 2);
			string label = format("%.2f", generate_boxes[i].score);
			label = this->classes[generate_boxes[i].label] + ":" + label;
			cout << (generate_boxes[i].x1 + generate_boxes[i].x2) / 2 / frame.cols << endl;
			cout << (generate_boxes[i].y1 + generate_boxes[i].y2) / 2 / frame.rows << endl;
			cout << (generate_boxes[i].x2 - generate_boxes[i].x1) / frame.cols << endl;
			cout << (generate_boxes[i].y2 - generate_boxes[i].y1) / frame.rows << endl;
			cout << generate_boxes[i].label << endl;
			cout << generate_boxes[i].score << endl;
			putText(frame, label, Point(xmin, ymin - 5), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);
		}
	}
}

int main(int argc, char* argv[])
{
	clock_t startTime, endTime; //����ʱ��
	Configuration yolo_nets = { 0.3, 0.5, 0.3,"hole_cls.onnx" };
	YOLOv5 yolo_model(yolo_nets);
	std::string imgDir = "D:\\test\\good\\";
	std::vector<cv::String> imgLists;
	//string imgpath = R"(D:\my_yolov5\data\C0\JPEGImages\20200404051008-C0910-5.jpg)";
//string imgpath = R"(D:\test\bad\BottomLeft3185045-0.jpg)";
	//Mat srcimg = imread(imgpath);
	cv::glob(imgDir, imgLists, false);
	for (auto img : imgLists) {
		std::cout << std::string(img) << std::endl;
		cv::Mat srcimg = cv::imread(img);
		double timeStart = (double)getTickCount();
		startTime = clock();//��ʱ��ʼ	
		yolo_model.detect(srcimg, "cls");
		endTime = clock();//��ʱ����
		cout << "clock_running time is:" << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
	}



	// static const string kWinName = "Deep learning object detection in ONNXRuntime";
	// namedWindow(kWinName, WINDOW_NORMAL);
	// imshow(kWinName, srcimg);
	//imwrite("restult_ort.jpg", srcimg);
	// waitKey(0);
	// destroyAllWindows();
	return 0;
}
