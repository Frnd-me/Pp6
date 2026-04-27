#pragma once

#include <omp.h>

#include <random>

inline auto atomic(long n) noexcept -> double {
    auto count = 0l;
#pragma omp parallel
    {
        auto gen = std::mt19937{std::random_device{}() ^ omp_get_thread_num()};
        auto dist = std::uniform_real_distribution{0.0, 1.0};

#pragma omp for
        for (auto i = 0l; i < n; ++i) {
            const auto x = dist(gen);
            const auto y = dist(gen);

            if (x * x + y * y <= 1.0) {
#pragma omp atomic
                ++count;
            }
        }
    }
    return 4.0 * static_cast<double>(count) / n;
}
