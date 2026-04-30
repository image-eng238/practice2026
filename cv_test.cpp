#include <iostream>
#include <opencv2/opencv.hpp>

int main(void) {
    std::cout << "cv_test" << std::endl;
    cv::Mat image = cv::imread("H:/Documents/image_eng/practice2026/barbara.ppm", cv::IMREAD_ANYCOLOR);
    if (image.empty()) return EXIT_FAILURE;

    const int width   = image.cols;
    const int height  = image.rows;
    const int num_chn = image.channels();
    const int stride  = image.step;
    std::cout << "width = " << width << ", height = " << height << std::endl;

    auto pixel = image.data;
    auto clamp = [](int val) {
        if (val > 255) { val = 255; }
        if (val < 0) { val = 0; }
        return val;
    };
    auto pic = [&stride, &num_chn](const int& h, const int& w, const int& c) { return h * stride + w * num_chn + c; };
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int B = pixel[i * stride + j * num_chn + 0];
            int G = pixel[i * stride + j * num_chn + 1];
            int R = pixel[i * stride + j * num_chn + 2];

            pixel[pic(i, j, 0)] = clamp(B * 4);
            pixel[pic(i, j, 1)] = clamp(B * 4);
            pixel[pic(i, j, 2)] = clamp(B * 4);
        }
    }

    cv::imshow("loaded image", image);

    cv::waitKey(0);
    cv::destroyAllWindows();
    return EXIT_SUCCESS;
}