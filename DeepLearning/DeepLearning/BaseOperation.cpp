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
    std::cout << img.total() << std::endl;
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
    for (int i = 0; i < 4; i++) {
        std::cout << lbox[i] << "  "<< rbox[i]<<"   " << interBox[i] << std::endl;
    }
    std::cout << interBoxS << std::endl;
    return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
}

void BaseOperation::ObjPostprocess(std::vector<Object>& res, float* prob, float conf_thresh, float nms_thresh, int yolomode) {
    
    int numbox = -1,boxscore=0;
    float* output;
    if (yolomode == 1) {
        numbox = 8400;
        boxscore = 4;
        cv::Mat outputtrans = cv::Mat(NUM_CLASSES + 4, numbox, CV_32F, prob).t();
        output = outputtrans.ptr<float>();
    }
    else {
        numbox = 1000;
        boxscore = 5;
        output = prob;
    }
    int mi = NUM_CLASSES + boxscore;
    int det_size = sizeof(Object) / sizeof(float);
    std::map<float, std::vector<Object>> m;
    for (int i = 0; i < numbox; i++) {
        if (yolomode == 0&&output[mi * i + 4] <= conf_thresh) continue;
        float tmp = 0.0;
        float labeltmp = -1;
        for (int j = boxscore; j < mi; j++) {
            if (yolomode == 0) {
                output[mi * i + j] *= output[mi * i + 4];
            }
            if (output[mi * i + j] > tmp) {
                tmp = output[mi * i + j];
                labeltmp = j - boxscore; //yolov8
                //output[mi * i + 5] = j - 5; //yolov5
            }
        }
        output[mi * i + 4] = tmp;
        output[mi * i + 5] = labeltmp;

        if (tmp < conf_thresh) continue;
        Object det;
        memcpy(&det, &output[mi * i], det_size * sizeof(float));
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

void BaseOperation::SegPostprocess(std::vector<Object>& res, float* prob, const std::vector<int>& imgSize, const std::vector<int>& padsize,
    const std::vector<int>& segMaskParam, int yolomode){
    std::vector<std::vector<float>> picked_proposals;  //存储output0[:,:, 5 + _className.size():net_width]用以后续计算mask
    int newh = padsize[0], neww = padsize[1], padh = padsize[2], padw = padsize[3];
    //printf("newh:%d,neww:%d,padh:%d,padw:%d", newh, neww, padh, padw);
    float ratio_h = (float)imgSize[1] / newh;
    float ratio_w = (float)imgSize[0] / neww;

    int numbox = -1, boxscore = 0;
    float* pdata;
    if (yolomode == 1) {
        boxscore = 4;
        cv::Mat outputtrans = cv::Mat(NUM_CLASSES + 4 + segMaskParam[0], segMaskParam[1], CV_32F, prob).t();
        pdata = outputtrans.ptr<float>();
    }
    else {
        numbox = 1000;
        boxscore = 5;
        pdata = prob;
    }

    // 处理box
    int net_width = NUM_CLASSES + boxscore + segMaskParam[0];
    //float* pdata = trt->prob;
    std::vector<Object> boxpams;
    cv::Rect holeImgRect(0, 0, imgSize[0], imgSize[1]);
    for (int j = 0; j < segMaskParam[1]; ++j) {
        //float box_score = pdata[4]; ;//获取每一行的box框中含有某个物体的概率
        if (yolomode == 0 && pdata[4] < BBOX_CONF_THRESH) continue;
        cv::Mat scores(1, NUM_CLASSES, CV_32FC1, pdata + boxscore);
        cv::Point classIdPoint;
        double max_class_socre;
        minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
        max_class_socre = (float)max_class_socre;
        if (max_class_socre >= BBOX_CONF_THRESH) {
            Object boxpamtmp;
            std::vector<float> temp_proto(pdata + boxscore + NUM_CLASSES, pdata + net_width);
            picked_proposals.push_back(temp_proto);

            float x = (pdata[0] - padw) * ratio_w;  //x
            float y = (pdata[1] - padh) * ratio_h;  //y
            float w = pdata[2] * ratio_w;  //w
            float h = pdata[3] * ratio_h;  //h

            float left = MAX((x - 0.5 * w), 0);
            float top = MAX((y - 0.5 * h), 0);
            boxpamtmp.label = classIdPoint.x;
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
            boxpams.push_back(boxpamtmp);
            //memcpy(&boxpamtmp.bbox, &pdata, 4 * sizeof(float));
            //classIds.push_back(classIdPoint.x);
            //if (yolomode == 1) {
            //    confidences.push_back(max_class_socre);
            //}
            //else {
            //    confidences.push_back(max_class_socre * pdata[4]);
            //}
            //boxes.push_back(cv::Rect(left, top, int(w), int(h)));
        }
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
    //if (!classIds.empty()) {
    //    //执行非最大抑制以消除具有较低置信度的冗余重叠框（NMS）
    //    cv::dnn::NMSBoxes(boxes, confidences, BBOX_CONF_THRESH, NMS_THRESH, res);
    //    std::vector<std::vector<float>> temp_mask_proposals;
    //}

}