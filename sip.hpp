#pragma once
#include <opencv2/core.hpp>

enum { FWD = 0,
       INV = 1
};

template <typename T>
// auto clamp = [](T val) {
T clamp(T val) {
    if (val > 255) { val = 255; }
    if (val < 0) { val = 0; }
    return val;
}

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
            Y  = .299 * R + .587 * G + .114 * B;
            Cb = -.1687 * R - .3313 * G + .5 * B;
            Cr = .5 * R - .4187 * G - .0813 * B;
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
void quantize(cv::Mat& blk, const int* const qtable, const float scale) {

    const int width    = blk.cols;
    const int height   = blk.rows;
    const int num_chn  = blk.channels();
    const int stride   = blk.step / sizeof(float);
    float* const pixel = reinterpret_cast<float*>(blk.data);

    for (size_t i = 0; i < height; ++i) {
        for (size_t j = 0; j < width; ++j) {
            const auto val = pixel[i * stride + j];
            auto stepsize  = clamp<int>(qtable[i * stride + j]);
            if constexpr (X == FWD) { // 量子化
                pixel[i * stride + j] = roundf(val / stepsize);
            } else { // 逆量子化
                pixel[i * stride + j] = val * stepsize;
            }
        }
    }
}