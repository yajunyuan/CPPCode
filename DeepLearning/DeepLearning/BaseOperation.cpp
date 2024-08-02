#include "BaseOperation.h"

void BaseOperation::GetConfigValue(const char* keyName, char* keyValue)
{
    std::string config_file = "../config/recconfig.conf";
    char buff[300] = { 0 };
    FILE* file = fopen(config_file.c_str(), "r");
    while (fgets(buff, 300, file))
    {
        char* tempKeyName = strtok(buff, "=");
        if (!tempKeyName) continue;
        char* tempKeyValue = strtok(NULL, "=");

        if (!strcmp(tempKeyName, keyName))
            strcpy(keyValue,tempKeyValue);
    }
    fclose(file);
}

void BaseOperation::doInference(IExecutionContext& context, cudaStream_t& stream, ICudaEngine& engine, std::string engine_mode, void** buffers, float* input, float* output, const int output_size, float* output1, const int output1_size, Yolov5TRTContext* trt) {
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host

    //CHECK(cudaMemcpyAsync(buffers[engine.getBindingIndex(INPUT_BLOB_NAME)], input, 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(buffers[trt->inputindex], input, 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    //context.executeV2(buffers);
    context.enqueueV2(buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[trt->outputindex], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    if (engine_mode == "seg") {
        CHECK(cudaMemcpyAsync(output1, buffers[trt->output1index], output1_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    }
    cudaStreamSynchronize(stream);
}

cv::Mat BaseOperation::static_resize(cv::Mat img, std::vector<int>& padsize, std::string engine_mode) {
    if (engine_mode == "cls") {
        int crop_size = std::min(img.cols, img.rows);
        int  left = (img.cols - crop_size) / 2, top = (img.rows - crop_size) / 2;
        cv::Mat crop_image = img(cv::Rect(left, top, crop_size, crop_size));
        cv::Mat out;
        cv::resize(crop_image, out, cv::Size(INPUT_W, INPUT_H));
        return out;
    }
    else {
        int w, h, x, y;
        float r_w = INPUT_W / (img.cols * 1.0);
        float r_h = INPUT_H / (img.rows * 1.0);
        if (r_h > r_w) {
            w = INPUT_W;
            h = r_w * img.rows;
            x = 0;
            y = (INPUT_H - h) / 2;
        }
        else {
            w = r_h * img.cols;
            h = INPUT_H;
            x = (INPUT_W - w) / 2;
            y = 0;
        }
        cv::Mat re(h, w, CV_8UC3);
        cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
        cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(114, 114, 114));
        re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
        padsize.push_back(h);
        padsize.push_back(w);
        padsize.push_back(y);
        padsize.push_back(x);// int newh = padsize[0], neww = padsize[1], padh = padsize[2], padw = padsize[3];

        return out;
        //float r = (std::min)(INPUT_W / (img.cols*1.0), INPUT_H / (img.rows*1.0));
        //// r = std::min(r, 1.0f);
        //int unpad_w = r * img.cols;
        //int unpad_h = r * img.rows;
        //cv::Mat re(unpad_h, unpad_w, CV_8UC3);
        //cv::resize(img, re, re.size());
        //cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(114, 114, 114));
        //re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
        //return out;
    }
}

float* BaseOperation::blobFromImage(cv::Mat img, std::string engine_mode, int channels) {
    //std::cout << img.total() << std::endl;
    float* blob = new float[img.total() * channels];
    //int channels = 3;
    int img_h = img.rows;
    int img_w = img.cols;
    for (size_t c = 0; c < channels; c++) {
        for (int h = 0; h < img.rows; ++h)
        {
            //获取第i行首像素指针
            cv::Vec3b* p1 = img.ptr<cv::Vec3b>(h);
            for (int w = 0; w < img.cols; ++w)
            {
                //将img的bgr转为image的rgb 
                blob[c * img_w * img_h + h * img_w + w] = (float)(p1[w][2 - c] / 255.0);
                if (engine_mode == "cls") {
                    // RGB mean(0.485, 0.456, 0.406) std(0.229, 0.224, 0.225)
                    if (c == 0) {
                        blob[c * img_w * img_h + h * img_w + w] = (blob[c * img_w * img_h + h * img_w + w] - 0.485) / 0.229;
                    }
                    else if (c == 1) {
                        blob[c * img_w * img_h + h * img_w + w] = (blob[c * img_w * img_h + h * img_w + w] - 0.456) / 0.224;
                    }
                    else {
                        blob[(c)*img_w * img_h + h * img_w + w] = (blob[c * img_w * img_h + h * img_w + w] - 0.406) / 0.225;
                    }
                }
            }
        }
    }
    //for (size_t c = 0; c < channels; c++)
    //{
    //    for (size_t h = 0; h < img_h; h++)
    //    {
    //        for (size_t w = 0; w < img_w; w++)
    //        {
    //            blob[c * img_w * img_h + h * img_w + w] =
    //                (float)(img.at<cv::Vec3b>(h, w)[c] / 255.0);
    //            if (engine_mode == "cls") {
    //                if (c == 0) {
    //                    blob[c * img_w * img_h + h * img_w + w] = (blob[c * img_w * img_h + h * img_w + w] - 0.406) / 0.225;
    //                }
    //                else if (c == 1) {
    //                    blob[c * img_w * img_h + h * img_w + w] = (blob[c * img_w * img_h + h * img_w + w] - 0.456) / 0.224;
    //                }
    //                else {
    //                    blob[(c) * img_w * img_h + h * img_w + w] = (blob[c * img_w * img_h + h * img_w + w] - 0.485) / 0.229;
    //                }
    //            }
    //        }
    //    }
    //}
    return blob;
}

bool BaseOperation::cmp(const Object& a, const Object& b) {
    return a.prob > b.prob;
}

float BaseOperation::iou(float lbox[4], float rbox[4]) {
    float interBox[] = {
        (std::max)(lbox[0] , rbox[0]), //left
        (std::min)(lbox[0] + lbox[2] , rbox[0] + rbox[2]), //right
        (std::max)(lbox[1] , rbox[1]), //top
        (std::min)(lbox[1] + lbox[3] , rbox[1] + rbox[3]), //bottom
    };

    if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS = (interBox[1] - interBox[0]+1) * (interBox[3] - interBox[2]+1);
    //for (int i = 0; i < 4; i++) {
    //    std::cout << lbox[i] << "  "<< rbox[i]<<"   " << interBox[i] << std::endl;
    //}
    //std::cout << interBoxS << std::endl;
    return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
}

void BaseOperation::Radian(cv::Mat img, const Object& input_box) {
    std::vector<cv::Point> pts;
    int width = int(input_box.bbox[2]);
    int height = int(input_box.bbox[3]);
    int center_x = int(input_box.bbox[0] + input_box.bbox[2] / 2);
    int center_y = int(input_box.bbox[1] + input_box.bbox[3] / 2);
    float cos_value = cos(input_box.radian);
    float sin_value = sin(input_box.radian);
    float vec1[2] = { width / 2 * cos_value, width / 2 * sin_value };
    float vec2[2] = { -height / 2 * sin_value, height / 2 * cos_value };
    pts.push_back(cv::Point(int(center_x + vec1[0] + vec2[0]), int(center_y + vec1[1] + vec2[1])));
    pts.push_back(cv::Point(int(center_x + vec1[0] - vec2[0]), int(center_y + vec1[1] - vec2[1])));
    pts.push_back(cv::Point(int(center_x - vec1[0] - vec2[0]), int(center_y - vec1[1] - vec2[1])));
    pts.push_back(cv::Point(int(center_x - vec1[0] + vec2[0]), int(center_y - vec1[1] + vec2[1])));
    cv::Mat imgtmp = img.clone();
    cv::polylines(imgtmp, pts, true, cv::Scalar(0, 255, 0), 2, cv::LINE_4);
    std::string label = cv::format("%.2f", input_box.prob);
    putText(imgtmp, label, cv::Point(pts[0].x - 5, pts[0].y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 0), 1);
}

void BaseOperation::ObjPostprocess(std::string engine_mode, std::vector<Object>& res, float* prob, int num_box, float conf_thresh, float nms_thresh, int yolomode) {
    
    int numbox = -1,boxscore=0;
    float* output;
    if (yolomode == 1) {
        numbox = num_box;
        boxscore = 4;
        cv::Mat outputtrans;
        if (engine_mode=="obj") {
            outputtrans = cv::Mat(NUM_CLASSES + 4, numbox, CV_32F, prob).t();
        }
        else {
            outputtrans = cv::Mat(NUM_CLASSES + 5, numbox, CV_32F, prob).t();
        }
        output = outputtrans.ptr<float>();
    }
    else {
        numbox = num_box;
        boxscore = 5;
        output = prob;
    }
    int mi = NUM_CLASSES + boxscore;
    if (engine_mode == "obb") {
        mi = mi + 1;
    }
    int det_size = sizeof(Object) / sizeof(float);
    //for (int batch = 0; batch < 2; batch++) {
    //    output += batch * numbox * mi;
     //   Batchres.push_back(res);
    //std::vector<Object> res;
    std::map<float, std::vector<Object>> m;

    for (int i = 0; i < numbox; i++) {
        if (yolomode == 0 && output[mi * i + 4] <= conf_thresh) continue;
        float tmp = 0.0;
        float labeltmp = -1;
        for (int j = 0; j < NUM_CLASSES; j++) {
            if (yolomode == 0) {
                output[mi * i + j + boxscore] *= output[mi * i + 4];
            }
            if (output[mi * i + j + boxscore] > tmp) {
                tmp = output[mi * i + j + boxscore];
                labeltmp = j; //yolov8
                //output[mi * i + 5] = j - 5; //yolov5
            }
        }
        if (tmp < conf_thresh) continue;
        output[mi * i + 4] = tmp;
        output[mi * i + 5] = labeltmp;
        Object det;
        memcpy(&det, &output[mi * i], 6 * sizeof(float));
        if (engine_mode == "obb") {
            det.radian = output[mi * i + mi - 1];
        }
        if (m.count(det.label) == 0) m.emplace(det.label, std::vector<Object>());
        m[det.label].push_back(det);
    }
    for (auto it = m.begin(); it != m.end(); it++) {
        //std::cout << it->second[0].class_id << " --- " << std::endl;
        auto& dets = it->second;
        std::sort(dets.begin(), dets.end(), [this](auto a, auto b) { return this->cmp(a, b); });
        for (size_t m = 0; m < dets.size(); ++m) {
            auto& item = dets[m];
            res.push_back(item);
            for (size_t n = m + 1; n < dets.size(); ++n) {
                if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
                    dets.erase(dets.begin() + n);
                    --n;
                }
            }
        }
    }
        
        
    
}

void BaseOperation::ObjUniqueprocess(std::vector<Object>& res, float nms_thresh)
{
    //for (auto res : Batchres) {
        for (int i = 0; i < res.size(); i++) {
            auto& item = res[i];
            for (int n = i + 1; n < res.size(); ++n) {
                if (iou(item.bbox, res[n].bbox) > nms_thresh) {
                    if (item.prob >= res[n].prob) {
                        res.erase(res.begin() + n);
                        --n;
                    }
                    else {
                        res.erase(res.begin() + i);
                        --i;
                        break;
                    }
                }
            }
        }
    
}

float BaseOperation::SigmoidFunction(float a)
{
    float b = 1. / (1. + exp(-a));
    return b;
}

void BaseOperation::SegPostprocess(std::vector<Object>& res, float* prob, float* prob1, cv::Mat img, const std::vector<int>& padsize,
    const std::vector<int>& segMaskParam, int yolomode){
    std::vector<std::vector<float>> picked_proposals;  //存储output0[:,:, 5 + _className.size():net_width]用以后续计算mask
    int newh = padsize[0], neww = padsize[1], padh = padsize[2], padw = padsize[3];
    //printf("newh:%d,neww:%d,padh:%d,padw:%d", newh, neww, padh, padw);
    float ratio_h = (float)img.rows / newh;
    float ratio_w = (float)img.cols / neww;

    int numbox = -1, boxscore = 0;
    float* pdata;
    if (yolomode == 1) {
        boxscore = 4;
        cv::Mat outputtrans = cv::Mat(NUM_CLASSES + 4 + segMaskParam[1], segMaskParam[0], CV_32F, prob).t();
        //pdata = outputtrans.ptr<float>();
        int img_length = outputtrans.total() * outputtrans.channels();
        pdata = new float[img_length];
        std::memcpy(pdata, outputtrans.ptr<float>(0), img_length * sizeof(float));
    }
    else {
        numbox = 1000;
        boxscore = 5;
        pdata = prob;
    }

    // 处理box
    int net_width = NUM_CLASSES + boxscore + segMaskParam[1];
    //float* pdata = trt->prob;
    std::vector<Object> boxpams;
    for (int j = 0; j < segMaskParam[0]; ++j) {
        //float box_score = pdata[4]; ;//获取每一行的box框中含有某个物体的概率
        if (yolomode == 0 && pdata[4] < BBOX_CONF_THRESH) continue;
        int class_idx = 0;
        float max_class_socre = 0;
        for (int k = 0; k < NUM_CLASSES; ++k)
        {
            if (pdata[k + boxscore] > max_class_socre)
            {
                max_class_socre = pdata[k + boxscore];
                class_idx = k;
            }
        }
        if (yolomode == 0) {
            max_class_socre *= pdata[4];   // 最大的类别分数*置信度
        }
        if (max_class_socre > BBOX_CONF_THRESH) // 再次筛选
        {
            Object boxpamtmp;
            std::vector<float> temp_proto(pdata + boxscore + NUM_CLASSES, pdata + net_width);

            float x = (pdata[0] - padw) * ratio_w;  //x
            float y = (pdata[1] - padh) * ratio_h;  //y
            float w = pdata[2] * ratio_w;  //w
            float h = pdata[3] * ratio_h;  //h

            float left = MAX((x - 0.5 * w), 0);
            float top = MAX((y - 0.5 * h), 0);
            boxpamtmp.label = class_idx;
            if (yolomode == 1) {
                boxpamtmp.prob = max_class_socre;
            }
            else {
                boxpamtmp.prob = max_class_socre * pdata[4];
            }
            //cv::Rect rect = cv::Rect(left, top, w, h)& holeImgRect;
            boxpamtmp.bbox[0] = left;
            boxpamtmp.bbox[1] = top;
            boxpamtmp.bbox[2] = w;
            boxpamtmp.bbox[3] = h;
            boxpamtmp.picked_proposals = temp_proto;
            boxpams.push_back(boxpamtmp);
            //memcpy(&boxpamtmp.bbox, &pdata, 4 * sizeof(float));
            //boxes.push_back(cv::Rect(left, top, int(w), int(h)))
        }

        //cv::Mat scores(1, NUM_CLASSES, CV_32FC1, pdata + boxscore);
        //cv::Point classIdPoint;
        //double max_class_socre;
        //minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
        //max_class_socre = (float)max_class_socre;
        //if (max_class_socre >= BBOX_CONF_THRESH) {
        //    Object boxpamtmp;
        //    std::vector<float> temp_proto(pdata + boxscore + NUM_CLASSES, pdata + net_width);

        //    float x = (pdata[0] - padw) * ratio_w;  //x
        //    float y = (pdata[1] - padh) * ratio_h;  //y
        //    float w = pdata[2] * ratio_w;  //w
        //    float h = pdata[3] * ratio_h;  //h

        //    float left = MAX((x - 0.5 * w), 0);
        //    float top = MAX((y - 0.5 * h), 0);
        //    boxpamtmp.label = classIdPoint.x;
        //    if (yolomode == 1) {
        //        boxpamtmp.prob = max_class_socre;
        //    }
        //    else {
        //        boxpamtmp.prob = max_class_socre * pdata[4];
        //    }
        //    //cv::Rect rect = cv::Rect(left, top, w, h)& holeImgRect;
        //    boxpamtmp.bbox[0] = left;
        //    boxpamtmp.bbox[1] = top;
        //    boxpamtmp.bbox[2] = w;
        //    boxpamtmp.bbox[3] = h;
        //    boxpamtmp.picked_proposals = temp_proto;
        //    boxpams.push_back(boxpamtmp);
        //    //memcpy(&boxpamtmp.bbox, &pdata, 4 * sizeof(float));
        //    //boxes.push_back(cv::Rect(left, top, int(w), int(h)));
        //}
        pdata += net_width;//下一行
    }

    std::sort(boxpams.begin(), boxpams.end(), [this](auto a, auto b) { return this->cmp(a, b); });

    for (int i = 0; i < boxpams.size(); i++)
    {
        res.push_back(boxpams[i]);
        for (int j = i + 1; j < boxpams.size(); j++)
        {
            if (iou(boxpams[i].bbox, boxpams[j].bbox) > NMS_THRESH)
            {
                boxpams.erase(boxpams.begin() + j);
                --j;
            }
        }
    }
    if (!res.empty()) {
        std::vector < cv::Scalar > color;
        for (int i = 0; i < NUM_CLASSES; i++) {
            int b = rand() % 256;
            int g = rand() % 256;
            int r = rand() % 256;
            color.push_back(cv::Scalar(b, g, r));
        }
        cv::Mat mask1(segMaskParam[1], segMaskParam[2] * segMaskParam[3], CV_32F, prob1);
        for (size_t i = 0; i < res.size(); ++i) {
            cv::Mat mask_protos = cv::Mat(res[i].picked_proposals).reshape(1, 1);
            cv::Mat m = mask_protos * mask1;
            for (int col = 0; col < m.cols; col++) {
                m.at<float>(0, col) = SigmoidFunction(m.at<float>(0, col));
            }
            cv::Mat m1 = m.reshape(1, segMaskParam[3]);
            // 将mask roi映射到inpWidth*inpHeight大小内
            cv::Rect roi(int((float)padw / INPUT_W * segMaskParam[2]), int((float)padh / INPUT_H * segMaskParam[3]), int(segMaskParam[2] - padw / 2), int(segMaskParam[3] - padh / 2));
            cv::Mat mask_roi = m1(roi);
            cv::Mat masktmp;
            resize(mask_roi, masktmp, cv::Size(img.cols, img.rows));
            masktmp = masktmp(cv::Rect(res[i].bbox[0], res[i].bbox[1], res[i].bbox[2], res[i].bbox[3])) > MASK_THRESHOLD;
            cv::Mat mask = img.clone();
            mask(cv::Rect(res[i].bbox[0], res[i].bbox[1], res[i].bbox[2], res[i].bbox[3])).setTo(color[res[i].label], masktmp);
            cv::Mat imgtmp;
            addWeighted(img, 0.5, mask, 0.5, 0, imgtmp);
            uchar* pImg = new uchar[res[i].bbox[2]* res[i].bbox[3]];
            memcpy(pImg, masktmp.data, res[i].bbox[2] * res[i].bbox[3]);
            res[i].boxMask = pImg;
        }
    }
}