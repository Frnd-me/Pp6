#pragma once

#include <omp.h>
#include <immintrin.h>

#include <random>

struct alignas(32) S {
    __m256i s0, s1;
};

inline auto fastRand(S &s) noexcept -> __m256d {
    __m256i x = s.s0, y = s.s1;
    s.s0 = y; // State evolution

    // Xorshift128+
    x = _mm256_xor_si256(x, _mm256_slli_epi64(x, 23));
    s.s1 = _mm256_xor_si256(_mm256_xor_si256(x, y),
                            _mm256_xor_si256(_mm256_srli_epi64(x, 17), _mm256_srli_epi64(y, 26)));

    // Generate raw 64-bit result by adding the two states
    const auto res = _mm256_add_epi64(s.s1, y);

    // Mask to keep only the bottom 52 bits (fractional part)
    const auto mask = _mm256_set1_epi64x(0x000FFFFFFFFFFFFFLL);
    const auto mantissa = _mm256_and_si256(res, mask);

    // Set Exponent bits to 0x3FF (1.0)
    const auto one = _mm256_set1_epi64x(0x3FF0000000000000LL);

    // Combine results for a double in range [1.0, 2.0)
    const auto bits = _mm256_or_si256(mantissa, one);

    // Cast bits to double and subtract 1.0 to shift range to [0.0, 1.0) (i.e., (double)rand / RAND_MAX)
    return _mm256_sub_pd(_mm256_castsi256_pd(bits), _mm256_set1_pd(1.0));
}

inline auto optimized(long n) noexcept -> double {
    auto inside = 0l;

    const auto _n = (n >= 4) ? (n / 4) : 0;

#pragma omp parallel reduction(+:inside)
    {
        // Unique seeding per thread using random device and tread number
        auto rd = std::random_device{};
        const auto seed = rd() ^ omp_get_thread_num();

        // Initialize 4 RNG streams
        auto s = S{
            _mm256_set_epi64x(seed ^ 0x9E3779B97F4A7C15LL, seed ^ 0x85EBCA6BDB357158LL,
                              seed ^ 0xC4CEB9FE1A85EC53LL, seed ^ 0x123456789ABCDEF0LL),
            _mm256_set_epi64x(seed ^ 0xBB67AE8584CAA73BLL, seed ^ 0x3C6EF372FE94F82BLL,
                              seed ^ 0xA54FF53A5F1D36F1LL, seed ^ 0xFEDCBA9876543210LL)
        };

        auto _hits = 0l;

#pragma omp for nowait
        for (auto i = 0l; i < _n; ++i) {
            // Four random points
            const auto x = fastRand(s);
            const auto y = fastRand(s);

            // x^2 + y^2 <= 1.0
            const auto m = _mm256_cmp_pd(_mm256_add_pd(_mm256_mul_pd(x, x), _mm256_mul_pd(y, y)), _mm256_set1_pd(1.0),
                                         _CMP_LE_OQ);

            // Count the set bits
            _hits += __builtin_popcount(_mm256_movemask_pd(m));
        }
        inside = _hits;
    }

    return (n == 0) ? 0.0 : 4.0 * static_cast<double>(inside) / n;
}
