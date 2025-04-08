#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
using namespace std;
using namespace cv;

// defining boundaries for red, yellow and blue
/*
Scalar lowRed1(0, 50, 60);
Scalar highRed1(20, 255, 255);
Scalar lowRed2(170, 70, 50);
Scalar highRed2(180, 255, 255);
Scalar lowYellow(57, 35, 100);
Scalar highYellow(60, 100, 100);
Scalar lowBlue(100, 50, 50);
Scalar highBlue(130, 255, 255);
*/

Scalar lowRed1(0, 135, 135);
Scalar lowRed2(15, 255, 255);
Scalar upRed1(159, 135, 80);
Scalar upRed2(179, 255, 255);

Scalar lowBlue(100, 100, 50);
Scalar upBlue(130, 255, 255); 

int main() {

    string imagePath1 = "/Users/andreaboarini/driverless_perception/frame_1.png";
    string imagePath2 = "/Users/andreaboarini/driverless_perception/frame_2.png";
    
    Mat read1 = imread(imagePath1, IMREAD_COLOR);
    Mat read2 = imread(imagePath2, IMREAD_COLOR);

    Mat gray, canny, hsv, colorTreshOutput, output = read1.clone();
    cvtColor(read1, hsv, COLOR_BGR2HSV);

    Mat treshLow, treshUp, treshold, treshBlue, smoothed;
    Mat kernel;
    inRange(hsv, lowRed1, lowRed2, treshLow);
    inRange(hsv, upRed1, upRed2, treshUp);
    inRange(hsv, lowBlue, upBlue, treshBlue);

    kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    bitwise_or(treshLow, treshUp, treshold);
    bitwise_or(treshold, treshBlue, treshold);

    dilate(treshold, treshold, kernel, Point(-1, -1), 11, MORPH_ELLIPSE);
    erode(treshold, treshold, kernel, Point(-1, -1), 8, MORPH_ELLIPSE);
    Canny(treshold, smoothed, 30, 100, 7);

    vector<vector<Point>> contours;
    findContours(smoothed, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);

    for(const auto& cnt : contours) {
        vector<Point> approx;
        Rect boundingRect = cv::boundingRect(cnt);
        approxPolyDP(cnt, approx, 0.08*arcLength(cnt, true), true);
        if (approx.size() == 3) {
            int x = boundingRect.x;
            int y = boundingRect.y;
            int w = boundingRect.width;
            int h = boundingRect.height;
            cv::rectangle(output, cv::Point(x, y), cv::Point(x + w, y + h), cv::Scalar(0, 255, 0), 3);
        }
    }

    imshow("original", read1);
    imshow("smoothed", smoothed);
    imshow("contourned", output);
    waitKey(0);

    return 0;
}