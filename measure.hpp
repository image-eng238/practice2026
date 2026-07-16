#pragma once
#include <chrono>
#include <vector>
#include <iostream>

class Measure {
public:
    enum type : size_t {
        rgb2YCbCr,
        dct,
        quantize,
        encode
    };

private:
    Measure() = default;
    inline static std::vector<std::chrono::high_resolution_clock::duration> time{4};
    static auto toms(type t) {
        return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(time[t]).count();
    };

public:
    template <typename F>
    static void tictoc(type t, F f) {
        const auto tic = std::chrono::high_resolution_clock::now();
        f();
        time[t] += std::chrono::high_resolution_clock::now() - tic;
    }

    static void print() {
        std::cout << "rgb2YCbCr = " << toms(rgb2YCbCr) << "ms" << std::endl;
        std::cout << "dct = " << toms(dct) << "ms" << std::endl;
        std::cout << "quantize = " << toms(quantize) << "ms" << std::endl;
        std::cout << "encode = " << toms(encode) << "ms" << std::endl;
    }

    static auto getstr(std::string_view s = " ") {
        return (std::stringstream() << toms(rgb2YCbCr) << s << toms(dct) << s << toms(quantize) << s << toms(encode)).str();
    }
};
