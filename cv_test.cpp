#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

auto clamp = [](int val) {
    if (val > 255) { val = 255; }
    if (val < 0) { val = 0; }
    return val;
};

cv::Mat rgb2YCbCr(cv::Mat img) {
    // RGB -> YCbCr, img はシャローコピー
    const int width   = img.cols;
    const int height  = img.rows;
    const int num_chn = img.channels();
    const int stride  = img.step;
    uint8_t* pixel    = img.data;

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int B = pixel[i * stride + j * num_chn + 0];
            int G = pixel[i * stride + j * num_chn + 1];
            int R = pixel[i * stride + j * num_chn + 2];

            int Y, Cb, Cr;
            Y  = 0.299 * R + 0.587 * G + 0.114 * B;
            Cb = -0.1687 * R + -0.3313 * G + 0.5 * B;
            Cr = 0.5 * R + -0.4187 * G + -0.0813 * B;
            Cb += 128;
            Cr += 128;

            pixel[i * stride + j * num_chn + 0] = clamp(Y);
            pixel[i * stride + j * num_chn + 1] = clamp(Cr);
            pixel[i * stride + j * num_chn + 2] = clamp(Cb);
        }
    }
    return img;
}

int main(void) {
    std::cout << "cv_test" << std::endl;
    cv::Mat image = cv::imread("H:/Documents/image_eng/practice2026/barbara.ppm", cv::IMREAD_ANYCOLOR);
    if (image.empty()) return EXIT_FAILURE;

    const int width   = image.cols;
    const int height  = image.rows;
    const int num_chn = image.channels();
    const int stride  = image.step;
    std::cout << "width = " << width << ", height = " << height << std::endl;

    image = rgb2YCbCr(image);
    std::vector<cv::Mat> ycrcb;
    cv::split(image, ycrcb);                              //[0] -> Y, [1] -> Cr, [2] -> Cb
    cv::resize(ycrcb[1], ycrcb[1], cv::Size(), 0.5, 0.5); // 444 -> 420
    cv::resize(ycrcb[2], ycrcb[2], cv::Size(), 0.5, 0.5); // 444 -> 420

    for (int y = 0; y < height; y += 8) {
        for (int x = 0; x < width; x += 8) {
            cv::Mat blk;
            ycrcb[0](cv::Rect(x, y, 8, 8)).convertTo(blk, CV_32F);
            blk -= 128.0f;
            // DCT
            // 量子化
            // エントロピー符号化
        }
    }

    cv::imshow("loaded image", ycrcb[0]);

    cv::waitKey(0);
    cv::destroyAllWindows();
    return EXIT_SUCCESS;
}