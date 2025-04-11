#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
using namespace std;
using namespace cv;

// defining boundaries for red, yellow and blue
Scalar lowBlue(100, 100, 50);
Scalar highBlue(130, 255, 255);
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

// defining a structure to iterate color ranges
class RangeColor {
public:
    string label;
    Scalar valueLow; // hsv coded
    Scalar valueUp;
    Scalar valueLow2; // always =0 if the range doesn't involve red
    Scalar valueUp2;
    Scalar color; // BGR coded

    RangeColor(){
        label = "";
        valueLow = Scalar(0, 0, 0);
        valueUp = Scalar(0, 0, 0);
        valueLow2 = Scalar(-1, -1, -1);
        valueUp2 = Scalar(-1, -1, -1);
        color = Scalar(0, 0, 0);
    }
    RangeColor(string l, Scalar vL, Scalar vU, Scalar clr) {
        label = l;
        valueLow = vL;
        valueUp = vU;
        valueLow2 = Scalar(-1, -1, -1);
        valueUp2 = Scalar(-1, -1, -1);
        color = clr;
    }
    RangeColor(string l, Scalar vL, Scalar vU, Scalar vL2, Scalar vU2, Scalar clr) {
        label = l;
        valueLow = vL;
        valueUp = vU;
        valueLow2 = vL2;
        valueUp2 = vU2;
        color = clr;
    }
    ~RangeColor() {}
};
class Cone {
public:
    Scalar colorClass;
    string classifiedAs;
    Rect boundaries;
    Cone() {
        classifiedAs = "";
        boundaries = Rect();
        colorClass = Scalar(255, 255, 255);
    }
    Cone(Rect bound, string classAs, Scalar cc) {
        classifiedAs = classAs;
        boundaries = bound;
        colorClass = cc;
    }
    ~Cone() {}
};

void findBoundaries(Mat& cannyMat, Scalar colorRect, string labelName, vector<vector<Point>>& contours, vector<Cone>& coneVector) {
    findContours(cannyMat, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
    for(const auto& cnt : contours) {
        vector<Point> approx;
        Rect r = boundingRect(cnt);
        approxPolyDP(cnt, approx, 0.08*arcLength(cnt, true), true);
        if (approx.size() == 3) {
            coneVector.push_back(Cone(Rect(r.x, r.y, r.width, r.height), labelName, colorRect));
        }
    }
    contours.clear();
}
void printBoundaries(vector<Cone>& v, Mat& outputMat) {
    for(const auto& item : v) {
        int x = item.boundaries.x;
        int y = item.boundaries.y;
        int w = item.boundaries.width;
        int h = item.boundaries.height;
        rectangle(outputMat, Point(x, y), Point(x + w, y + h), item.colorClass, 2);
        putText(outputMat, item.classifiedAs, Point(x, y), FONT_HERSHEY_PLAIN, 1, item.colorClass, 2);
    }
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
    if (lower2 != Scalar(-1, -1, -1) && higher2 != Scalar(-1, -1, -1)) {
        inRange(hsv, lower2, higher2, secondary);
        bitwise_or(primary, secondary, outputColor);
    } else {
        outputColor = primary.clone();
    }
}
int main() {

    RangeColor red("Red", lowRed1, highRed1, lowRed2, highRed2, redLabel);
    RangeColor yellow("Yellow", lowYellow, highYellow, yellowLabel);
    RangeColor blue("Blue", lowBlue, highBlue, blueLabel);
    vector<RangeColor> rc = {red, yellow, blue};
    vector<vector<Point>> contours;
    vector<Cone> conesFound;
    const int tolerance = 10; // defining tolerance for duplicated cones

    string imagePath1 = "/Users/andreaboarini/driverless_perception/frame_1.png";
    string imagePath2 = "/Users/andreaboarini/driverless_perception/frame_2.png";
    
    Mat read1 = imread(imagePath1, IMREAD_COLOR);
    Mat read2 = imread(imagePath2, IMREAD_COLOR);

    Mat treshold, kernel, smoothed, output = read1.clone();

    for(const auto& range : rc) {
        extractColorMask(read1, range.valueLow, range.valueUp, treshold, range.valueLow2, range.valueUp2);
        processColorMask(treshold, kernel, 3, 3, 11, 8, 7, smoothed);
        findBoundaries(smoothed, range.color, range.label, contours, conesFound);
    }

    // sorting the cones' vector
    sort(conesFound.begin(), conesFound.end(), [](const Cone& c1, const Cone& c2) {
        return (c1.boundaries.x < c2.boundaries.x);
    });

    // removing all the cones that refere to the same object in the image
    auto toRemove = unique(conesFound.begin(), conesFound.end(), [tolerance](const Cone& c1, const Cone& c2) {
        return (abs(c1.boundaries.x - c2.boundaries.x) < tolerance &&
            abs(c1.boundaries.y - c2.boundaries.y) < tolerance &&
            abs(c1.boundaries.width - c2.boundaries.width) < tolerance &&
            abs(c1.boundaries.height - c2.boundaries.height) < tolerance);
    });

    conesFound.erase(toRemove, conesFound.end());
   
    printBoundaries(conesFound, output);

    imshow("smoothed", smoothed);
    imshow("contourned", output);
    waitKey(0);

    return 0;
}