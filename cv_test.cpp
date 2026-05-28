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
void quantize(cv::Mat& blk, const float* const qtable, const float scale = 75) {

    const int width    = blk.cols;
    const int height   = blk.rows;
    const int num_chn  = blk.channels();
    const int stride   = blk.step / sizeof(float);
    float* const pixel = reinterpret_cast<float*>(blk.data);

    for (size_t i = 0; i < height; ++i) {
        for (size_t j = 0; j < width; ++j) {
            const auto val      = pixel[i * stride + j];
            const auto stepsize = clamp(qtable[i * stride + j] * scale);
            if constexpr (X == 0) { // 量子化
                pixel[i * stride + j] = roundf(val / stepsize);
            } else { // 逆量子化
                pixel[i * stride + j] = val * stepsize;
            }
        }
    }
}

int main(int argc, char* argv[]) {
    std::cout << "cv_test" << std::endl;
    cv::Mat image = cv::imread("H:/Documents/image_eng/practice2026/barbara.ppm", cv::IMREAD_ANYCOLOR);
    if (image.empty()) return EXIT_FAILURE;

    const int num_chn = image.channels();

    image = rgb2YCbCr(image);
    std::vector<cv::Mat> ycrcb;
    cv::split(image, ycrcb); //[0] -> Y, [1] -> Cr, [2] -> Cb

    int dH = 2, dV = 2;
    cv::resize(ycrcb[1], ycrcb[1], cv::Size(), 1.0f / dH, 1.0f / dV); // 444 -> 420
    cv::resize(ycrcb[2], ycrcb[2], cv::Size(), 1.0f / dH, 1.0f / dV); // 444 -> 420

    int quality = 75;
    if (argc > 1) {
        try {
            quality = std::stoi(argv[1]);
        } catch (...) {
            std::cout << "the argument of 1 is image quality" << std::endl;
            exit(1);
        }
    }

    auto QF = [](int q) {
        // assert(0 <= q && q <= 100);
        q = q ? q : 1;
        q = q > 100 ? 100 : q;
        int qf;
        if (q <= 50) {
            qf = static_cast<int>(5000 / q);
        } else {
            qf = 200 - (2 * q);
        }
        qf = qf ? qf : 1;
        return qf;
    };

    float scale = QF(quality) / 100.0f;

    auto blkproc = [](cv::Mat& tmp, const float* qmatrix, const float scale) {
        cv::Mat blk;
        tmp.convertTo(blk, CV_32F);
        blk -= 128.0f;
        // Forward DCT
        cv::dct(blk, blk, FWD);
        // 量子化
        quantize<FWD>(blk, qmatrix, scale);
        // 逆量子化
        quantize<INV>(blk, qmatrix, scale);

        // Inverse IDCT
        cv::dct(blk, blk, 1);
        blk += 128.0f;
        blk.convertTo(tmp, CV_8U);

        // エントロピー符号化
    };

    // MCU 単位での処理
    const int width  = ycrcb[0].cols;
    const int height = ycrcb[0].rows;
    for (int y = 0, cy = 0; y < height; y += 8 * dV, cy += 8) {
        for (int x = 0, cx = 0; x < width; x += 8 * dH, cx += 8) {
            for (int i = 0; i < dV; ++i) {
                for (int j = 0; j < dH; ++j) {
                    auto tmpY = ycrcb[0](cv::Rect(x + j * 8, y + i * 8, 8, 8)); // Y
                    blkproc(tmpY, qmatrix[0], scale);
                }
                auto tmpCr = ycrcb[1](cv::Rect(cx, cy, 8, 8)); // Cr
                blkproc(tmpCr, qmatrix[1], scale);
                auto tmpCy = ycrcb[2](cv::Rect(cx, cy, 8, 8)); // Cb
                blkproc(tmpCr, qmatrix[1], scale);
            }
        }
    }

    cv::resize(ycrcb[1], ycrcb[1], cv::Size(), dH, dV); // 420 -> 444
    cv::resize(ycrcb[2], ycrcb[2], cv::Size(), dH, dV); // 420 -> 444
    cv::merge(ycrcb, image);

    cv::cvtColor(image, image, cv::COLOR_YCrCb2BGR);
    cv::imshow("loaded image", image);

    cv::waitKey(0);
    cv::destroyAllWindows();
    return EXIT_SUCCESS;
}