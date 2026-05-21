#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

#include "qtable.hpp"

constexpr size_t FWD = 0;
constexpr size_t INV = 1;

template <typename T>
// auto clamp = [](T val) {
T clamp(T val) {
    if (val > 255) { val = 255; }
    if (val < 0) { val = 0; }
    return val;
}

cv::Mat
rgb2YCbCr(cv::Mat img) {
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

template <size_t X = 0>
void quantize(cv::Mat& blk, const float* const qtable, const int quality = 75) {
    const int width    = blk.cols;
    const int height   = blk.rows;
    const int num_chn  = blk.channels();
    const int stride   = blk.step / sizeof(float);
    float* const pixel = reinterpret_cast<float*>(blk.data);

    for (size_t i = 0; i < height; ++i) {
        for (size_t j = 0; j < width; ++j) {
            const auto val      = pixel[i * stride + j];
            const auto stepsize = clamp(qtable[i * stride + j]);
            if constexpr (X == 0) { // 量子化
                pixel[i * stride + j] = roundf(val / stepsize);
            } else { // 逆量子化
                pixel[i * stride + j] = val * stepsize;
            }
        }
    }
}

int main(void) {
    std::cout << "cv_test" << std::endl;
    cv::Mat image = cv::imread("H:/Documents/image_eng/practice2026/barbara.ppm", cv::IMREAD_ANYCOLOR);
    if (image.empty()) return EXIT_FAILURE;

    const int num_chn = image.channels();

    image = rgb2YCbCr(image);
    std::vector<cv::Mat> ycrcb;
    cv::split(image, ycrcb);                              //[0] -> Y, [1] -> Cr, [2] -> Cb
    cv::resize(ycrcb[1], ycrcb[1], cv::Size(), 0.5, 0.5); // 444 -> 420
    cv::resize(ycrcb[2], ycrcb[2], cv::Size(), 0.5, 0.5); // 444 -> 420

    for (int c = 0; c < num_chn; ++c) {
        const int width  = ycrcb[c].cols;
        const int height = ycrcb[c].rows;
        for (int y = 0; y < height; y += 8) {
            for (int x = 0; x < width; x += 8) {
                auto tmp = ycrcb[c](cv::Rect(x, y, 8, 8));
                cv::Mat blk;
                tmp.convertTo(blk, CV_32F);
                blk -= 128.0f;
                // Forward DCT
                cv::dct(blk, blk, FWD);
                // 量子化
                quantize<FWD>(blk, qmatrix[c > 0]);
                // 逆量子化
                quantize<INV>(blk, qmatrix[c > 0]);

                // Inverse IDCT
                cv::dct(blk, blk, 1);
                blk += 128.0f;
                blk.convertTo(tmp, CV_8U);

                // エントロピー符号化
            }
        }
    }

    cv::resize(ycrcb[1], ycrcb[1], cv::Size(), 1 / 0.5, 1 / 0.5); // 420 -> 444
    cv::resize(ycrcb[2], ycrcb[2], cv::Size(), 1 / 0.5, 1 / 0.5); // 420 -> 444
    cv::merge(ycrcb, image);

    cv::cvtColor(image, image, cv::COLOR_YCrCb2BGR);
    cv::imshow("loaded image", image);

    cv::waitKey(0);
    cv::destroyAllWindows();
    return EXIT_SUCCESS;
}