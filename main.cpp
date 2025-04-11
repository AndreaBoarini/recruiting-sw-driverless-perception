#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
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

// defining the given matrix with camera parameters
Mat intrinsicMatrix = (Mat_<double>(3, 3) <<
    387.3502807617188, 0,                 317.7719116210938,
    0,                 387.3502807617188, 242.4875946044922,
    0,                 0,                 1
);

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
    Point center;
    bool isRight;
    Cone() {
        classifiedAs = "";
        boundaries = Rect();
        colorClass = Scalar(255, 255, 255);
        center = Point(0, 0);
        isRight = false;
    }
    Cone(Rect bound, string classAs, Scalar cc) {
        classifiedAs = classAs;
        boundaries = bound;
        colorClass = cc;
        center = Point(bound.x + bound.width/2, bound.y + bound.height/2);
        isRight = (classAs == "Yellow" || (classAs == "Red" && center.x > 0));
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
// Function to determine if a red cone is on the right or left side of the track
int assignRedConeSides(vector<Cone>& v, const Mat& input) {
    int imageWidth = 0;
    for (const auto& item : v) {
        if (item.center.x + item.boundaries.width > imageWidth) {
            imageWidth = item.center.x + item.boundaries.width;
        }
    }
    int centerX = imageWidth/2;
    for (auto& item : v) {
        if (item.classifiedAs == "Red") {
            item.isRight = (item.center.x >= centerX);
        }
    }
    return centerX;
}
void drawTrackBoundaries(vector<Cone>& v, const Mat& input, Mat& output) {
    vector<Cone> rightBoundaryCones;
    vector<Cone> leftBoundaryCones;
    int centerX = assignRedConeSides(v, input);
    
    for (const auto& item : v) {
        if (item.classifiedAs == "Yellow" || (item.classifiedAs == "Red" && item.isRight)) {
            rightBoundaryCones.push_back(item);
        } else if (item.classifiedAs == "Blue" || (item.classifiedAs == "Red" && !item.isRight)) {
            leftBoundaryCones.push_back(item);
        }
    }

    auto sortByProximity = [centerX](const Cone& a, const Cone& b) {
        return (a.center.y > b.center.y);
    };

    sort(rightBoundaryCones.begin(), rightBoundaryCones.end(), sortByProximity);
    sort(leftBoundaryCones.begin(), leftBoundaryCones.end(), sortByProximity);
    
    if (rightBoundaryCones.size() > 1) {
        for (int i = 0; i < rightBoundaryCones.size() - 1; i++) {
            Scalar lineColor = Scalar(0, 255, 40);
            line(output, rightBoundaryCones[i].center, rightBoundaryCones[i+1].center, lineColor, 2);
        }
    }
    
    if (leftBoundaryCones.size() > 1) {
        for (int i = 0; i < leftBoundaryCones.size() - 1; i++) {
            Scalar lineColor = Scalar(0, 255, 40);
            line(output, leftBoundaryCones[i].center, leftBoundaryCones[i+1].center, lineColor, 2);
        }
    }
}
// Function that implement features detection using ORB
void featuresMatcher(const Mat& first, const Mat& second, Mat& output) {
    Mat grayFirst, graySecond, descriptorFirst, descriptorSecond, essentialMat, mask, rotationMat, translationMat;
    cvtColor(first, grayFirst, COLOR_BGR2GRAY);
    cvtColor(second, graySecond, COLOR_BGR2GRAY);
    Ptr<ORB> orb = ORB::create(10); // smart pointer to manage deletes automatically
    vector<KeyPoint> keyPointsFirst, keyPointsSecond;

    // find keypoints and compute their descriptors on the entire image
    orb->detectAndCompute(grayFirst, noArray(), keyPointsFirst, descriptorFirst);
    orb->detectAndCompute(graySecond, noArray(), keyPointsSecond, descriptorSecond);

    // match the descriptors already computed with BruteForce
    BFMatcher match(NORM_HAMMING);
    vector<vector<DMatch>> knn;
    match.knnMatch(descriptorFirst, descriptorSecond, knn, 2);

    // apply the Lowe's ratio test
    vector<DMatch> matchesFound;
    for (int i = 0; i < knn.size(); i++) {
        if (knn[i].size() > 1) {
            if (knn[i][0].distance < 0.75f * knn[i][1].distance) {
                matchesFound.push_back(knn[i][0]);
            }
        }
    }
    
    // extract the 2D cooridnates from the matches
    vector<Point2f> pointsFirst, pointsSecond;
    for (int i = 0; i < matchesFound.size(); i++) {
        pointsFirst.push_back(keyPointsFirst[matchesFound[i].queryIdx].pt);
        pointsSecond.push_back(keyPointsSecond[matchesFound[i].trainIdx].pt);
    }

    if(pointsFirst.size() >= 5) {
        essentialMat = findEssentialMat(pointsFirst, pointsSecond, intrinsicMatrix, RANSAC, 0.999, 1.0, mask);
        recoverPose(essentialMat, pointsFirst, pointsSecond, intrinsicMatrix, rotationMat, translationMat, mask);
    }

    // visualization of matches
    drawMatches(first, keyPointsFirst, second, keyPointsSecond, matchesFound, output);

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
    cout << (read1.cols - 1) << endl;

    Mat treshold, kernel, smoothed, output = read1.clone();
    Mat odometry;

    for(const auto& range : rc) {
        extractColorMask(read1, range.valueLow, range.valueUp, treshold, range.valueLow2, range.valueUp2);
        processColorMask(treshold, kernel, 3, 3, 11, 8, 7, smoothed);
        findBoundaries(smoothed, range.color, range.label, contours, conesFound);
    }

    // sorting the cones vector
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
    drawTrackBoundaries(conesFound, read1, output);
    featuresMatcher(read1, read2, odometry);

    imshow("smoothed", smoothed);
    imshow("contourned", output);
    imshow("odometry", odometry);
    waitKey(0);

    return 0;
}