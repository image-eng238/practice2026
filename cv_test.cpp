#include <iostream>
#include <opencv2/opencv.hpp>

int main(void) {
    std::cout << "cv_test" << std::endl;
    cv::Mat image = cv::imread("H:/Documents/image_eng/practice2026/barbara.ppm", cv::IMREAD_ANYCOLOR);

    std::cout << "width = " << image.cols << ", height = " << image.rows << std::endl;

    cv::imshow("loaded image", image);

    cv::waitKey(0);

    return 0;
}