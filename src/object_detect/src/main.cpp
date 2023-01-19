#include<iostream>
#include<fstream>
#include<chrono>
#include <thread>

#include <ros/ros.h>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include "tf2_geometry_msgs/tf2_geometry_msgs.h"
#include "geometry_msgs/TransformStamped.h"


#include "../include/cv_bridge.h"

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <mutex>

using namespace std;
using namespace cv;
using namespace cv::dnn;

ros::Publisher pubDetectedPoint;
//confThreshold为置信度阈值
float confThreshold = 0, nmsThreshold;
mutex is_ready_mutex;
std::vector<std::string> classes;
Mat currentImage;
bool isReady = false;
bool isDetectSuccess = false;

inline void preprocess(const Mat& frame, Net& net, Size inpSize, bool swapRB);
void doDetect();
void doGrabImage();
void postprocess(Mat& frame, const std::vector<Mat>& outs, Net& net, int backend,std::vector<Rect>& boxes);
void extractFeature(const cv::Mat& input,cv::Mat& result,const std::vector<Rect>& boxes);
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);


class ImageGrabber
{
public:
    ImageGrabber(){}

    void GrabRGBD(const sensor_msgs::ImageConstPtr& img_color,const sensor_msgs::ImageConstPtr& img_depth);
    
};

int process(cv::Mat img_left,cv::Mat img_right, cv::Mat& output);

int main(int argc, char **argv)
{
    
    ros::init(argc, argv, "object_detect");
    ros::start();
    //检测线程
    thread detectThread(doDetect);
    //获取图像线程
    thread ImageGrabberThread(doGrabImage);
    ImageGrabberThread.join();
    detectThread.join();
    ros::shutdown();

    return 0;
}

void doGrabImage(){
    ros::NodeHandle nh;
    ImageGrabber igb;
    message_filters::Subscriber<sensor_msgs::Image> left_sub(nh, "/camera/rgb/image_raw", 1);
    message_filters::Subscriber<sensor_msgs::Image> right_sub(nh, "/camera/depth/image", 1);
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> sync_pol;
    message_filters::Synchronizer<sync_pol> sync(sync_pol(10), left_sub,right_sub);
    sync.registerCallback(boost::bind(&ImageGrabber::GrabRGBD,&igb,_1,_2));

    ros::Rate r(30);
    while (ros::ok())
    {
        r.sleep();
        ros::spinOnce();
    }
}

void doDetect(){
    
    cv::namedWindow("result",0);
    cv::resizeWindow("result", 640, 480);
    Net net = readNet(
    "src/object_detect/src/yolov3/yolov3.weights", 
    "src/object_detect/src/yolov3/yolov3.cfg"
    
    );
    
    int backend = cv::dnn::DNN_BACKEND_DEFAULT;
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    std::vector<String> outNames = net.getUnconnectedOutLayersNames();

    
    std::string file = "src/object_detect/src/coco.names";
    std::ifstream ifs(file.c_str());
    if (!ifs.is_open())
        CV_Error(Error::StsError, "File " + file + " not found");
    std::string line;
    while (std::getline(ifs, line))
    {
        classes.push_back(line);
    }

    while(ros::ok()){
        {
            // std::cout << isReady <<std::endl;
            std::lock_guard<std::mutex> guard(is_ready_mutex);
            if(isReady){
                preprocess(currentImage, net, Size(416,416),true);

                std::vector<Mat> outs;
                net.forward(outs, outNames);
                std::vector<Rect> boxes;
                postprocess(currentImage, outs, net, backend,boxes);

                // Put efficiency information.
                std::vector<double> layersTimes;
                double freq = getTickFrequency() / 1000;
                double t = net.getPerfProfile(layersTimes) / freq;
                std::string label = format("Inference time: %.2f ms", t);
        
                //提取特征
                // cv::Mat result = currentImage.clone();
                // extractFeature(currentImage,result,boxes);
                std::cout <<"检测到的目标数量：" << boxes.size()<<endl;
                cv::imshow("result",currentImage);
                waitKey(25);
            }
            isReady = false;
        }
        

    }
}

void ImageGrabber::GrabRGBD(const sensor_msgs::ImageConstPtr& img_color,const sensor_msgs::ImageConstPtr& img_depth)
{
    
    // Copy the ros image message to cv::Mat.
    cv_bridge::CvImagePtr pImgColor;
    try
    {
        pImgColor = cv_bridge::toCvCopy(img_color);
    }
    catch (cv_bridge::Exception &e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    cv_bridge::CvImagePtr pDepthImg;
    try
    {
        pDepthImg = cv_bridge::toCvCopy(img_depth);
    }
    catch (cv_bridge::Exception &e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }


    cv::cvtColor(pImgColor->image,pImgColor->image,cv::COLOR_BGR2RGB);
    
    currentImage = pImgColor->image.clone();
    {
        std::lock_guard<std::mutex> guard(is_ready_mutex);
        isReady = true;
    }
   
}
inline void preprocess(const Mat& frame, Net& net, Size inpSize,bool swapRB)
{
    static Mat blob;
    // Create a 4D blob from a frame.
    if (inpSize.width <= 0) inpSize.width = frame.cols;
    if (inpSize.height <= 0) inpSize.height = frame.rows;

    blobFromImage(frame, blob, 0.00392, inpSize, Scalar(), true, false);
    // Run a model.
    net.setInput(blob);
    if (net.getLayer(0)->outputNameToIndex("im_info") != -1)  // Faster-RCNN or R-FCN
    {
        resize(frame, frame, inpSize);
        Mat imInfo = (Mat_<float>(1, 3) << inpSize.height, inpSize.width, 1.6f);
        net.setInput(imInfo, "im_info");
    }
}

void postprocess(Mat& frame, const std::vector<Mat>& outs, Net& net, int backend,std::vector<Rect>& boxes)
{
    static std::vector<int> outLayers = net.getUnconnectedOutLayers();
    static std::string outLayerType = net.getLayer(outLayers[0])->type;

    std::vector<int> classIds;
    std::vector<float> confidences;
    
    if (outLayerType == "DetectionOutput")
    {
        // Network produces output blob with a shape 1x1xNx7 where N is a number of
        // detections and an every detection is a vector of values
        // [batchId, classId, confidence, left, top, right, bottom]
        CV_Assert(outs.size() > 0);
        for (size_t k = 0; k < outs.size(); k++)
        {
            float* data = (float*)outs[k].data;
            for (size_t i = 0; i < outs[k].total(); i += 7)
            {
                float confidence = data[i + 2];
                if (confidence > confThreshold)
                {
                    int left   = (int)data[i + 3];
                    int top    = (int)data[i + 4];
                    int right  = (int)data[i + 5];
                    int bottom = (int)data[i + 6];
                    int width  = right - left + 1;
                    int height = bottom - top + 1;
                    if (width <= 2 || height <= 2)
                    {
                        left   = (int)(data[i + 3] * frame.cols);
                        top    = (int)(data[i + 4] * frame.rows);
                        right  = (int)(data[i + 5] * frame.cols);
                        bottom = (int)(data[i + 6] * frame.rows);
                        width  = right - left + 1;
                        height = bottom - top + 1;
                    }
                    classIds.push_back((int)(data[i + 1]) - 1);  // Skip 0th background class id.
                    boxes.push_back(Rect(left, top, width, height));
                    confidences.push_back(confidence);
                }
            }
        }
    }
    else if (outLayerType == "Region")
    {
        for (size_t i = 0; i < outs.size(); ++i)
        {
            // Network produces output blob with a shape NxC where N is a number of
            // detected objects and C is a number of classes + 4 where the first 4
            // numbers are [center_x, center_y, width, height]
            float* data = (float*)outs[i].data;
            for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
            {
                Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
                Point classIdPoint;
                double confidence;
                minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                if (confidence > confThreshold)
                {
                    int centerX = (int)(data[0] * frame.cols);
                    int centerY = (int)(data[1] * frame.rows);
                    int width = (int)(data[2] * frame.cols);
                    int height = (int)(data[3] * frame.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    classIds.push_back(classIdPoint.x);
                    confidences.push_back((float)confidence);
                    boxes.push_back(Rect(left, top, width, height));
                }
            }
        }
    }
    else
        CV_Error(Error::StsNotImplemented, "Unknown output layer type: " + outLayerType);

    // NMS is used inside Region layer only on DNN_BACKEND_OPENCV for another backends we need NMS in sample
    // or NMS is required if number of outputs > 1
    if (outLayers.size() > 1 || (outLayerType == "Region" && backend != DNN_BACKEND_OPENCV))
    {
        std::map<int, std::vector<size_t> > class2indices;
        for (size_t i = 0; i < classIds.size(); i++)
        {
            if (confidences[i] >= confThreshold)
            {
                class2indices[classIds[i]].push_back(i);
            }
        }
        std::vector<Rect> nmsBoxes;
        std::vector<float> nmsConfidences;
        std::vector<int> nmsClassIds;
        for (std::map<int, std::vector<size_t> >::iterator it = class2indices.begin(); it != class2indices.end(); ++it)
        {
            std::vector<Rect> localBoxes;
            std::vector<float> localConfidences;
            std::vector<size_t> classIndices = it->second;
            for (size_t i = 0; i < classIndices.size(); i++)
            {
                localBoxes.push_back(boxes[classIndices[i]]);
                localConfidences.push_back(confidences[classIndices[i]]);
            }
            std::vector<int> nmsIndices;
            NMSBoxes(localBoxes, localConfidences, confThreshold, nmsThreshold, nmsIndices);
            for (size_t i = 0; i < nmsIndices.size(); i++)
            {
                size_t idx = nmsIndices[i];
                localBoxes[idx].x -= 30;
                localBoxes[idx].y -= 30;
                localBoxes[idx].width += 30;
                localBoxes[idx].height += 30;
                nmsBoxes.push_back(localBoxes[idx]);
                nmsConfidences.push_back(localConfidences[idx]);
                nmsClassIds.push_back(it->first);
            }
        }
        boxes = nmsBoxes;
        classIds = nmsClassIds;
        confidences = nmsConfidences;
    }

    if(boxes.size() >= 1){
        isDetectSuccess = true;
    }

    for (size_t idx = 0; idx < boxes.size(); ++idx)
    {
        Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx], box.x, box.y,
                 box.x + box.width, box.y + box.height, frame);
    }
}
/**
 * @brief 用于提取动态特征点并剔除
 * 
 * @param input 
 * @param result 
 * @param boxes 
 *//*
void extractFeature(const cv::Mat& input,cv::Mat& result,const std::vector<Rect>& boxes){
    bool isRemoveDynamicPoint = true;
    cv::Mat output;
    cv::cvtColor(input,output,CV_RGB2GRAY);
    std::vector<cv::KeyPoint> kp; //特征点容器
    cv::FAST(output, kp, 10);
    const float r = 5;
    for(auto p : kp){
        bool flag = false;
        cv::Point2f pt1,pt2;
        pt1.x=p.pt.x-r;
        pt1.y=p.pt.y-r;
        pt2.x=p.pt.x+r;
        pt2.y=p.pt.y+r;
        for(auto box : boxes){
            if(box.contains(p.pt)){
                flag = true;
                break;
            }
        }
        if(isRemoveDynamicPoint){
            if(!flag){
                cv::rectangle(result,pt1,pt2,cv::Scalar(0,255,0));
                cv::circle(result,p.pt,2,cv::Scalar(0,255,0),-1);
            }
        }else{
            cv::rectangle(result,pt1,pt2,cv::Scalar(0,255,0));
            cv::circle(result,p.pt,2,cv::Scalar(0,255,0),-1);
        }
        
    }
}
*/
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0));

    std::string label = format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ": " + label;
    }

    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - labelSize.height),
              Point(left + labelSize.width, top + baseLine), Scalar::all(255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar());
}
