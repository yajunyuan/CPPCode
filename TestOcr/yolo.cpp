#include"yolo.h"
using namespace std;
using namespace cv;
using namespace cv::dnn;

void Yolov5::LetterBox(const cv::Mat& image, cv::Mat& outImage, cv::Vec4d& params, const cv::Size& newShape,
	bool autoShape, bool scaleFill, bool scaleUp, int stride, const cv::Scalar& color)
{
	if (false) {
		int maxLen = MAX(image.rows, image.cols);
		outImage = Mat::zeros(Size(maxLen, maxLen), CV_8UC3);
		image.copyTo(outImage(Rect(0, 0, image.cols, image.rows)));
		params[0] = 1;
		params[1] = 1;
		params[3] = 0;
		params[2] = 0;
	}

	cv::Size shape = image.size();
	float r = std::min((float)newShape.height / (float)shape.height,
		(float)newShape.width / (float)shape.width);
	if (!scaleUp)
		r = std::min(r, 1.0f);

	float ratio[2]{ r, r };
	int new_un_pad[2] = { (int)std::round((float)shape.width * r),(int)std::round((float)shape.height * r) };

	auto dw = (float)(newShape.width - new_un_pad[0]);
	auto dh = (float)(newShape.height - new_un_pad[1]);

	if (autoShape)
	{
		dw = (float)((int)dw % stride);
		dh = (float)((int)dh % stride);
	}
	else if (scaleFill)
	{
		dw = 0.0f;
		dh = 0.0f;
		new_un_pad[0] = newShape.width;
		new_un_pad[1] = newShape.height;
		ratio[0] = (float)newShape.width / (float)shape.width;
		ratio[1] = (float)newShape.height / (float)shape.height;
	}

	dw /= 2.0f;
	dh /= 2.0f;

	if (shape.width != new_un_pad[0] && shape.height != new_un_pad[1])
	{
		cv::resize(image, outImage, cv::Size(new_un_pad[0], new_un_pad[1]));
	}
	else {
		outImage = image.clone();
	}

	int top = int(std::round(dh - 0.1f));
	int bottom = int(std::round(dh + 0.1f));
	int left = int(std::round(dw - 0.1f));
	int right = int(std::round(dw + 0.1f));
	params[0] = ratio[0];
	params[1] = ratio[1];
	params[2] = left;
	params[3] = top;
	cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
}



bool Yolov5::readModel(Net& net, string& netPath, bool isCuda = false) {
	try {

		net = readNetFromONNX(netPath);
#if CV_VERSION_MAJOR==4 &&CV_VERSION_MINOR==7&&CV_VERSION_REVISION==0
		net.enableWinograd(false);  //bug of opencv4.7.x in AVX only platform ,https://github.com/opencv/opencv/pull/23112 and https://github.com/opencv/opencv/issues/23080 
		//net.enableWinograd(true);		//If your CPU supports AVX2, you can set it true to speed up
#endif
	}
	catch (const std::exception&) {
		return false;
	}
	////cuda
	//if (isCuda) {
	//	net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
	//	net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
	//}
	//cpu
	net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
	net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
	return true;
}
void Yolov5::Detect(Mat& SrcImg, Net& net, int& outputResultLen, RecResult*& outputResult) {
	Mat blob;
	int col = SrcImg.cols;
	int row = SrcImg.rows;
	int maxLen = MAX(col, row);
	Mat netInputImg = SrcImg.clone();
	Vec4d params;
	LetterBox(SrcImg, netInputImg, params, cv::Size(_netWidth, _netHeight));

	blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(_netWidth, _netHeight), cv::Scalar(0, 0, 0), true, false);
	//�������������û�����������µ��ǽ��ƫ��ܴ󣬿��Գ������������������
	//blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(_netWidth, _netHeight), cv::Scalar(104, 117, 123), true, false);
	//blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(_netWidth, _netHeight), cv::Scalar(114, 114,114), true, false);
	net.setInput(blob);
	std::vector<cv::Mat> netOutputImg;
	//vector<string> outputLayerName{"345","403", "461","output" };
	//net.forward(netOutputImg, outputLayerName[3]); //��ȡoutput�����

	std::vector<int> unconnected_layers = net.getUnconnectedOutLayers();
	std::vector<String>layer_names = net.getLayerNames();
	std::vector<String>output_layer_names;
	for (auto layer : unconnected_layers) {
		output_layer_names.push_back(layer_names[layer - 1]);
	}
	//output_layer_names = [layer_names[i[0] - 1] for i in unconnected_layers]
	net.forward(netOutputImg, output_layer_names);
	std::vector<int> classIds;//���id����
	std::vector<float> confidences;//���ÿ��id��Ӧ���Ŷ�����
	std::vector<cv::Rect> boxes;//ÿ��id���ο�

	//int net_out_width = netOutputImg[0].size[2];
	int net_width = netOutputImg[0].size[2];  //������������������+5
	int classnum = net_width - 5;
	float* pdata = (float*)netOutputImg[0].data;
	int net_height = netOutputImg[0].size[1];
	for (int r = 0; r < net_height; ++r) {
		float box_score = pdata[4]; ;//��ȡÿһ�е�box���к���ĳ������ĸ���
		if (box_score >= _classThreshold) {
			cv::Mat scores(1, classnum, CV_32FC1, pdata + 5);
			Point classIdPoint;
			double max_class_socre;
			minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
			max_class_socre = max_class_socre * box_score;
			if (max_class_socre >= _classThreshold) {
				//rect [x,y,w,h]
				std::cout << pdata[0] << "  " << pdata[1] << "  " << pdata[2]
					<< "  " << pdata[3] << "  " << pdata[4] << std::endl;
				float x = (pdata[0] - params[2]) / params[0];
				float y = (pdata[1] - params[3]) / params[1];
				float w = pdata[2] / params[0];
				float h = pdata[3] / params[1];
				int left = MAX(round(x - 0.5 * w + 0.5), 0);
				int top = MAX(round(y - 0.5 * h + 0.5), 0);
				classIds.push_back(classIdPoint.x);
				confidences.push_back(max_class_socre);
				boxes.push_back(Rect(left, top, int(w + 0.5), int(h + 0.5)));
			}
		}
		pdata += net_width;//��һ��

	}

	//ִ�з�����������������нϵ����Ŷȵ������ص���NMS��
	vector<int> nms_result;
	NMSBoxes(boxes, confidences, _classThreshold, _nmsThreshold, nms_result);
	outputResultLen = nms_result.size();
	outputResult = new RecResult[outputResultLen];
	for (int i = 0; i < outputResultLen; i++) {
		int idx = nms_result[i];
		//Output result;
		//result.id = classIds[idx];
		//result.confidence = confidences[idx];
		//result.box = boxes[idx];
		outputResult[i].label = classIds[idx];
		outputResult[i].score = confidences[idx];
		outputResult[i].box[0] = boxes[idx].x;
		outputResult[i].box[1] = boxes[idx].y;
		outputResult[i].box[2] = boxes[idx].width;
		outputResult[i].box[3] = boxes[idx].height;
		//output.push_back(result);
		std::cout << outputResult[i].box[0] << "  " << outputResult[i].box[1] << "  " << outputResult[i].box[2]
			<< "  " << outputResult[i].box[3] << "  " << outputResult[i].score << "  " << outputResult[i].label << std::endl;
	}
}

void Yolov5::drawPred(Mat& img, vector<Output> result, vector<Scalar> color) {
	for (int i = 0; i < result.size(); i++) {
		int left, top;
		left = result[i].box.x;
		top = result[i].box.y;
		int color_num = i;
		rectangle(img, result[i].box, color[result[i].id], 2, 8);

		string label = _className[result[i].id] + ":" + to_string(result[i].confidence);

		int baseLine;
		Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
		top = max(top, labelSize.height);
		//rectangle(frame, Point(left, top - int(1.5 * labelSize.height)), Point(left + int(1.5 * labelSize.width), top + baseLine), Scalar(0, 255, 0), FILLED);
		putText(img, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 1, color[result[i].id], 2);
	}
	imshow("1", img);
	//imwrite("out.bmp", img);
	waitKey();
	//destroyAllWindows();

}
