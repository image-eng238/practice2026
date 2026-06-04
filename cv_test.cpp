#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

#include "qtable.hpp"
#include "zigzag.hpp"
#include "huffman_tables.hpp"

enum {
    FWD    = 0,
    INV    = 1,
    DC     = 0,
    AC     = 1,
    Luma   = 0,
    Chroma = 1,
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

template <size_t Type>
void encode(int run, int val, const unsigned int* const t_cwd, const unsigned int* const t_len) {
    int s     = 0; // 何ビット入るか
    int uval  = (val < 0) ? -val : val;
    int bound = 1;
    while (uval >= bound) {
        bound += bound;
        ++s;
    }
    if constexpr (Type == DC) {
        // DC
        // t_cwd[s] を　t_len[s] ビットの値としてビットストリームに書き込む
    } else {
        // AC
        // t_cwd[(run << 4) + s] を　t_len[(run << 4) + s] ビットの値としてビットストリームに書き込む
    }
    if (s != 0) {
        val = (val < 0) ? val - 1 : val;
    }
    // val を sビットの値としてビットストリームに書き込む
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
    /*YY|C.*/
    /*YY|..*/

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

    auto blkproc = [](cv::Mat& tmp, const float* qmatrix, const float scale, int& prev_dc, const int c) {
        cv::Mat blk;
        tmp.convertTo(blk, CV_32F);
        blk -= 128.0f;
        // Forward DCT
        cv::dct(blk, blk, FWD);
        // 量子化
        quantize<FWD>(blk, qmatrix, scale);

        auto p   = reinterpret_cast<float*>(blk.data);
        // 現在のブロックのDC成分から前のブロックのDC成分を引く
        int diff = p[0] - prev_dc;
        prev_dc  = p[0]; // prev_dcを更新
        encode<DC>(0, diff, DC_cwd[c], DC_len[c]);

        // AC成分のためのジグザクスキャン・ハフマン符号
        int zero_run = 0; // 0の連続数
        for (int i = 1; i < 64; ++i) {
            auto ac = static_cast<int>(p[zigzag_scan[i]]);
            if (ac == 0) {
                ++zero_run;
            } else {
                while (zero_run > 15) {
                    encode<AC>(0xf, 0x0, AC_cwd[c], AC_len[c]);
                    zero_run -= 16;
                }
                encode<AC>(zero_run, ac, AC_cwd[c], AC_len[c]);
                zero_run = 0;
            }
        }
        if (zero_run != 0) {
            // EOB
            encode<AC>(0x0, 0x0, AC_cwd[c], AC_len[c]);
        }

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
    // prev_dc[] には前のブロックのDC成分値が入る
    int prev_dc[3]   = {};

    for (int y = 0, cy = 0; y < height; y += 8 * dV, cy += 8) {
        for (int x = 0, cx = 0; x < width; x += 8 * dH, cx += 8) {
            for (int i = 0; i < dV; ++i) {
                for (int j = 0; j < dH; ++j) {
                    auto tmpY = ycrcb[0](cv::Rect(x + j * 8, y + i * 8, 8, 8)); // Y
                    blkproc(tmpY, qmatrix[Luma], scale, prev_dc[0], Luma);
                }
                auto tmpCr = ycrcb[1](cv::Rect(cx, cy, 8, 8)); // Cr
                blkproc(tmpCr, qmatrix[Chroma], scale, prev_dc[1], Chroma);
                auto tmpCy = ycrcb[2](cv::Rect(cx, cy, 8, 8)); // Cb
                blkproc(tmpCr, qmatrix[Chroma], scale, prev_dc[2], Chroma);
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