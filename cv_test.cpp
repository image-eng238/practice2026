#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

#include "coding.hpp"
#include "sip.hpp"

#include "bitstream.hpp"
#include "qtable.hpp"
#include "jpgheaders.hpp"
#include "ycctype.hpp"
#include "jpgmarkers.hpp"

#include "measure.hpp"

enum {
    Luma   = 0,
    Chroma = 1,
};

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("usage: %s input output <quality> <444, 420, ...\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    cv::Mat image = cv::imread(argv[1], cv::IMREAD_ANYCOLOR);
    if (image.empty()) {
        exit(EXIT_FAILURE);
    }
    cv::Mat original = image.clone(); // PSNR 算出のため
    bitstream encbuf;

    const int num_chn = image.channels();

    auto pixel = image.data;
    Measure::tictoc(Measure::type::rgb2YCbCr, [&] { image = rgb2YCbCr(image); });
    std::vector<cv::Mat> ycrcb;
    cv::split(image, ycrcb); //[0] -> Y, [1] -> Cr, [2] -> Cb

    int YCCtype = YUV420;
    if (argc > 4) {
        if (std::string_view{"444"} == argv[4]) {
            YCCtype = YUV444;
        } else if (std::string_view{"422"} == argv[4]) {
            YCCtype = YUV422;
        } else if (std::string_view{"411"} == argv[4]) {
            YCCtype = YUV411;
        } else if (std::string_view{"440"} == argv[4]) {
            YCCtype = YUV440;
        } else if (std::string_view{"420"} == argv[4]) {
            YCCtype = YUV420;
        } else if (std::string_view{"410"} == argv[4]) {
            YCCtype = YUV410;
        } else if (std::string_view{"GRAY"} == argv[4]) {
            YCCtype = GRAY;
        } else {
            std::cout << "YCC type error" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    int dH = YCC_HV[YCCtype][0] >> 4, dV = YCC_HV[YCCtype][0] & 0xF;
    cv::resize(ycrcb[1], ycrcb[1], cv::Size(), 1.0f / dH, 1.0f / dV); // 444 -> 420
    cv::resize(ycrcb[2], ycrcb[2], cv::Size(), 1.0f / dH, 1.0f / dV); // 444 -> 420

    const int width  = ycrcb[0].cols;
    const int height = ycrcb[0].rows;
    /*YY|C.*/
    /*YY|..*/

    int QF;
    int quality;
    if (argc < 4) {
        quality = 75;
    } else {
        try {
            quality = std::stoi(argv[3]);
        } catch (...) {
            std::cout << "the argument of 1 is image quality" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    quality = (quality == 0) ? 1 : quality;
    quality = (quality > 100) ? 100 : quality;
    if (quality <= 50) {
        QF = 5000 / quality;
    } else {
        QF = 200 - 2 * quality;
    }
    QF          = (QF == 0) ? 1 : QF;
    float scale = QF / 100.0f;

    int qtable[2][64];
    for (int i = 0; i < 64; ++i) {
        if (QF != 1) {
            qtable[0][i] = static_cast<int>(clamp<float>(qmatrix[0][i] * scale));
            qtable[1][i] = static_cast<int>(clamp<float>(qmatrix[1][i] * scale));
            if (qtable[0][i] == 0) {
                qtable[0][i] = 1;
            }
            if (qtable[1][i] == 0) {
                qtable[1][i] = 1;
            }
        } else {
            qtable[0][i] = qtable[1][i] = 1;
        }
    }

    // printf("encoding...\n");
    create_mainheader(width, height, num_chn, qtable[Luma], qtable[Chroma], YCCtype, encbuf);

    // MCU 単位での処理
    // prev_dc[] には前のブロックのDC成分値が入る
    int prev_dc[3] = {};

    for (int y = 0, cy = 0; y < height; y += 8 * dV, cy += 8) {
        for (int x = 0, cx = 0; x < width; x += 8 * dH, cx += 8) {
            for (int i = 0; i < dV; ++i) {
                for (int j = 0; j < dH; ++j) {
                    auto tmpY = ycrcb[0](cv::Rect(x + j * 8, y + i * 8, 8, 8)); // Y
                    blkproc(tmpY, qtable[Luma], scale, prev_dc[0], Luma, encbuf);
                }
            }
            auto tmpCr = ycrcb[2](cv::Rect(cx, cy, 8, 8)); // Cr
            blkproc(tmpCr, qtable[Chroma], scale, prev_dc[2], Chroma, encbuf);
            auto tmpCb = ycrcb[1](cv::Rect(cx, cy, 8, 8)); // Cb
            blkproc(tmpCb, qtable[Chroma], scale, prev_dc[1], Chroma, encbuf);
        }
    }

    auto length = encbuf.finalize();

    // std::cout << "codestream size = " << length << ", ";
    std::cout << quality << "," << static_cast<double>(length) * 8.0 / (width * height) << ",";
    FILE* fp = fopen(argv[2], "wb");
    if (fp == nullptr) {
        printf("cant open file '%s'\n", argv[2]);
        exit(EXIT_FAILURE);
    }

    fwrite(encbuf.get_data(), sizeof(uint8_t), length, fp);
    fclose(fp);

    cv::resize(ycrcb[1], ycrcb[1], cv::Size(), dH, dV); // 420 -> 444
    cv::resize(ycrcb[2], ycrcb[2], cv::Size(), dH, dV); // 420 -> 444
    cv::merge(ycrcb, image);
    cv::cvtColor(image, image, cv::COLOR_YCrCb2BGR);

    double mse = 0.0;
    for (size_t i = 0; i < height; ++i) {
        for (size_t j = 0; j < width; ++j) {
            for (size_t c = 0; c < num_chn; ++c) {
                int v0 = original.data[i * original.step + j * num_chn + c];
                int v1 = image.data[i * image.step + j * num_chn + c];
                mse += (v0 - v1) * (v0 - v1);
            }
        }
    }
    mse /= (width * height * num_chn);
    double psnr = 10 * log10(255 * 255 / mse);
    std::cout << psnr << "," << Measure::getstr(",") << std::endl;
    // cv::imshow("loaded image", image);
    // cv::waitKey(0);
    // cv::destroyAllWindows();

    return EXIT_SUCCESS;
}