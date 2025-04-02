#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;

int main() {

    // defining images' path name
    string imagePath1 = "/Users/andreaboarini/driverless_perception/frame_1.png";
    string imagePath2 = "/Users/andreaboarini/driverless_perception/frame_2.png";
    
    // reading the paths
    Mat read1 = imread(imagePath1, IMREAD_COLOR);
    Mat read2 = imread(imagePath2, IMREAD_COLOR);

    // displaying the images
    imshow("display image", read1);
    waitKey(0);

    return 0;
}