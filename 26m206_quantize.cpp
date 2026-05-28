template <size_t X = 0>
void quantize(cv::Mat& blk, const float* const qtable, const int quality = 75) {
    auto QF = [](int q) {
        assert(0 <= q && q <= 100);
        q = q ? q : 1;
        double qf;
        if (q <= 50) {
            qf = static_cast<int>(5000 / q);
        } else {
            qf = 200 - (2 * q);
        }
        qf = qf ? qf : 1;
        return qf;
    };

    float Scale        = QF(75) / 100;
    const int width    = blk.cols;
    const int height   = blk.rows;
    const int num_chn  = blk.channels();
    const int stride   = blk.step / sizeof(float);
    float* const pixel = reinterpret_cast<float*>(blk.data);

    for (size_t i = 0; i < height; ++i) {
        for (size_t j = 0; j < width; ++j) {
            const auto val      = pixel[i * stride + j];
            const auto stepsize = clamp(qtable[i * stride + j] * Scale); // clamp を関数テンプレートで定義することでテンプレート引数を推論
            if constexpr (X == 0) {                                      // 量子化
                pixel[i * stride + j] = roundf(val / stepsize);
            } else { // 逆量子化
                pixel[i * stride + j] = val * stepsize;
            }
        }
    }
}