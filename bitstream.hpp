#pragma once

#include <cstdint>
#include <vector>

class bitstream {
private:
    uint8_t tmp;
    int32_t bits;
    std::vector<uint8_t> stream;

public:
    bitstream() : tmp{}, bits{}, stream{} {
        stream.reserve(32768);
    }
    ~bitstream() {}

    inline void put_bits(const uint32_t cwd, uint32_t len) {
        uint8_t b;
        while (len > 0) {
            b   = static_cast<uint8_t>((cwd >> (len - 1)) & 1);
            tmp = (tmp << 1) | b;
            ++bits;
            --len;
            if (bits == 8) {
                put_byte(tmp);
                if (tmp == 0xFF) {
                    // バイトスタッフィング処理
                    put_byte(0x00);
                }
                tmp  = 0;
                bits = 0;
            }
        }
    }
    inline void put_byte(const uint8_t val) {
        stream.push_back(val);
    }
    inline void put_word(const uint16_t val) {
        put_byte(val >> 8);
        put_byte(val & 0xFF);
    }

    inline void flush() {
        tmp <<= 8 - bits;    // 上位へ寄せる
        tmp |= 0xFF >> bits; // 下位を1で埋める
        put_byte(tmp);
        if (tmp == 0xFF) {
            // バイトスタッフィング処理
            put_byte(0x00);
        }
    }
    size_t finalize() {
        flush();
        return stream.size();
    }
    auto get_data() const { return stream.data(); }
};