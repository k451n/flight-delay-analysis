#include "../include/aviation_analysis.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <mpi.h>

void test_simd_preprocessing() {
    std::cout << "[TEST] Running SIMD preprocessing & gap filling test..." << std::endl;

    const int size = 8;
    double missing_val = -999.0;
    
    // Выделяем память с выравниванием в 64 байта
    double* raw = (double*)_aligned_malloc(size * sizeof(double), 64);
    double* clean = (double*)_aligned_malloc(size * sizeof(double), 64);

    // Имитируем мили видимости с пропусками
    raw[0] = 5.0; raw[1] = -999.0; raw[2] = 5.0; raw[3] = 10.0;
    raw[4] = 10.0; raw[5] = -999.0; raw[6] = -999.0; raw[7] = 5.0;

    preprocess_weather_vectors(raw, clean, size, missing_val);

    // Проверяем перевод миль в километры: 5 миль * 1.60934 = 8.0467 км
    assert(std::abs(clean[0] - 8.0467) < 1e-3);
    // Пропуск на индексе 1 должен интерполироваться между 0 и 2 индексами
    assert(std::abs(clean[1] - 8.0467) < 1e-3);
    
    _aligned_free(raw);
    _aligned_free(clean);

    std::cout << "[SUCCESS] SIMD preprocessing test passed!" << std::endl;
}

void test_rolling_correlation() {
    std::cout << "[TEST] Running rolling correlation test..." << std::endl;

    const int size = 10;
    const int window = 5;
    
    std::vector<WeatherRecord> weather(size);
    std::vector<double> delays(size);
    std::vector<double> results(size - window + 1, 0.0);

    // Заполняем строгой линейной зависимостью (коэффициент Пирсона должен быть равен 1.0)
    for (int i = 0; i < size; ++i) {
        weather[i].temperature = static_cast<double>(i);
        delays[i] = static_cast<double>(i) * 2.0 + 10.0;
    }

    compute_rolling_correlation(weather.data(), delays.data(), size, window, results.data());

    int mpi_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    if (mpi_rank == 0) {
        for (size_t i = 0; i < results.size(); ++i) {
            // Проверяем, что корреляция близка к идеальной единице
            assert(std::abs(results[i] - 1.0) < 1e-4);
        }
        std::cout << "[SUCCESS] Rolling correlation test passed!" << std::endl;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        std::cout << "=== RUNNING HPC KERNEL UNIT TESTS ===" << std::endl;
    }

    test_simd_preprocessing();
    test_rolling_correlation();

    if (rank == 0) {
        std::cout << "=== ALL C++ TESTS PASSED SUCCESSFULLY ===" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
