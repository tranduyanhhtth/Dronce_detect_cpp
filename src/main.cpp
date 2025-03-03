#include <iostream>
#include <opencv2/opencv.hpp>
#include "yolov11_onnx.h"
#include "bbox.h"

int main()
{
    Yolov11_Onnx detector("/home/danz/Downloads/Drone_detect/src/best.onnx");

    string image_path = "/home/danz/Downloads/Drone_detect/Data_bla/DroneTestDataset/Drone_TestSet/VS_P7637.jpg";

    vector<Detection> detections = detector.detect(image_path);

    cv::Mat result_frame = Bbox::draw_box(image_path, detections);

    cv::imshow("Detection Result", result_frame);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}