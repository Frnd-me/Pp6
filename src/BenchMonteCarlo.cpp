#include "Atomic.h"
#include "Critical.h"
#include "Optimized.h"
#include "Reduction.h"

#include "benchmark/registration.h"
#include "benchmark/state.h"
#include "benchmark/utils.h"

static void BM_Atomic(benchmark::State &state) {
    const auto n = state.range(0);

    for (auto _: state) {
        auto result = atomic(n);

        benchmark::DoNotOptimize(result);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * n);
}

BENCHMARK(BM_Atomic)
  ->RangeMultiplier(64)
  ->Range(1, 1 << 30)
  ->UseRealTime();

static void BM_Critical(benchmark::State &state) {
    const auto n = state.range(0);

    for (auto _: state) {
        auto result = critical(n);

        benchmark::DoNotOptimize(result);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * n);
}

BENCHMARK(BM_Critical)
  ->RangeMultiplier(64)
  ->Range(1, 1 << 30)
  ->UseRealTime();

static void BM_Reduction(benchmark::State &state) {
    const auto n = state.range(0);

    for (auto _: state) {
        auto result = reduction(n);

        benchmark::DoNotOptimize(result);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * n);
}

BENCHMARK(BM_Reduction)
  ->RangeMultiplier(64)
  ->Range(1, 1 << 30)
  ->UseRealTime();

static void BM_Optimized(benchmark::State &state) {
    const auto n = state.range(0);

    for (auto _: state) {
        auto result = optimized(n);

        benchmark::DoNotOptimize(result);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * n);
}

BENCHMARK(BM_Optimized)
  ->RangeMultiplier(64)
  ->Range(1, 1 << 30)
  ->UseRealTime();

BENCHMARK_MAIN();
