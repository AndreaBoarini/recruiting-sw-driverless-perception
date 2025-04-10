#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
using namespace std;
using namespace cv;

// defining boundaries for red, yellow and blue
Scalar lowBlue(100, 100, 50);
Scalar upBlue(130, 255, 255);

Scalar lowRed1(0, 160, 160);
Scalar highRed1(10, 255, 255);
Scalar lowRed2(160, 160, 160);
Scalar highRed2(179, 255, 255);

Scalar lowYellow(10, 150, 150);
Scalar highYellow(25, 255, 255);

// defining label colours
Scalar redLabel(0, 0, 255);
Scalar blueLabel(255, 0, 0);
Scalar yellowLabel(0, 255, 255);

void printBoundaries(Mat& cannyMat, Scalar colorRect, string labelName, Mat& outputMat, vector<vector<Point>>& contours) {
    findContours(cannyMat, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
    for(const auto& cnt : contours) {
        vector<Point> approx;
        Rect boundingRect = cv::boundingRect(cnt);
        approxPolyDP(cnt, approx, 0.08*arcLength(cnt, true), true);
        if (approx.size() == 3) {
            int x = boundingRect.x;
            int y = boundingRect.y;
            int w = boundingRect.width;
            int h = boundingRect.height;
            rectangle(outputMat, Point(x, y), Point(x + w, y + h), colorRect, 2);
            putText(outputMat, labelName, Point(x, y), FONT_HERSHEY_PLAIN, 1, colorRect, 2);
        }
    }
    contours.clear();
}

void processColorMask(Mat& colorTreshold, Mat& kernel, int kSizeA, int kSizeB, int dilateIterations, int erodeIterations, int cannyAperture, Mat& outputCanny) {
    kernel = getStructuringElement(MORPH_RECT, Size(kSizeA, kSizeB));
    dilate(colorTreshold, colorTreshold, kernel, Point(-1, -1), dilateIterations, MORPH_ELLIPSE);
    erode(colorTreshold, colorTreshold, kernel, Point(-1, -1), erodeIterations, MORPH_ELLIPSE);
    Canny(colorTreshold, outputCanny, 30, 100, cannyAperture);
}

void extractColorMask(Mat& src, Scalar lower, Scalar higher, Mat& outputColor, Scalar lower2 = Scalar(0,0,0), Scalar higher2 = Scalar(0,0,0)) {
    Mat hsv, primary, secondary;
    cvtColor(src, hsv, COLOR_BGR2HSV);
    inRange(hsv, lower, higher, primary);
    inRange(hsv, lower2, higher2, secondary);
    bitwise_or(primary, secondary, outputColor);
}

int main() {

    string imagePath1 = "/Users/andreaboarini/driverless_perception/frame_1.png";
    string imagePath2 = "/Users/andreaboarini/driverless_perception/frame_2.png";
    
    Mat read1 = imread(imagePath1, IMREAD_COLOR);
    Mat read2 = imread(imagePath2, IMREAD_COLOR);

    vector<vector<Point>> contours;
    Mat treshold, kernel, smoothed, output = read1.clone();

    extractColorMask(read1, lowYellow, highYellow, treshold);
    processColorMask(treshold, kernel, 3, 3, 11, 8, 7, smoothed);
    printBoundaries(smoothed, yellowLabel, "Yellow", output, contours);

    extractColorMask(read1, lowRed1, highRed1, treshold, lowRed2, highRed2);
    processColorMask(treshold, kernel, 3, 3, 11, 8, 7, smoothed);
    printBoundaries(smoothed, redLabel, "Red", output, contours);

    extractColorMask(read1, lowBlue, upBlue, treshold);
    processColorMask(treshold, kernel, 3, 3, 11, 8, 7, smoothed);
    printBoundaries(smoothed, blueLabel,"Blue", output, contours);

    imshow("smoothed", smoothed);
    imshow("contourned", output);
    waitKey(0);

    return 0;
}