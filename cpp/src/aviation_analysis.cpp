#include "aviation_analysis.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <omp.h>
#include <mpi.h>
#include <cstdint>

extern "C" {

// SIMD-векторизованная предобработка данных с выравниванием адресов
void preprocess_weather_vectors(
    const double* raw_data,
    double* clean_data,
    int size,
    double missing_marker
) {
    if (size <= 0) return;

    const double* r_ptr = raw_data;
    double* c_ptr = clean_data;

    // 1. Перевод миль видимости в км и замена маркеров на временный -1.0
    #pragma omp simd
    for (int i = 0; i < size; ++i) {
        if (r_ptr[i] == missing_marker || std::isnan(r_ptr[i])) {
            c_ptr[i] = -1.0; 
        } else {
            c_ptr[i] = r_ptr[i] * 1.60934; 
        }
    }

    // 2. Безопасное линейное заполнение пропусков без выхода за границы массива
    for (int i = 0; i < size; ++i) {
        if (c_ptr[i] < 0.0) { // Нашли пропуск
            double prev = (i > 0) ? c_ptr[i - 1] : 6.21 * 1.60934; // Дефолт, если пропуск на 0-м индексе
            double next = prev;
            bool found_next = false;

            for (int j = i + 1; j < size; ++j) {
                if (raw_data[j] != missing_marker && !std::isnan(raw_data[j])) {
                    next = raw_data[j] * 1.60934;
                    found_next = true;
                    break;
                }
            }
            
            // Если до конца массива больше нет чистых данных, берем предыдущее значение
            if (!found_next) {
                next = prev;
            }

            c_ptr[i] = (prev + next) * 0.5;
        }
    }
}

// Вычисление скользящей корреляции Пирсона: Гибрид MPI + OpenMP
void compute_rolling_correlation(
    const WeatherRecord* weather,
    const double* delays,
    int size,
    int window_size,
    double* out_correlations
) {
    if (size < window_size) return;

    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);

    int rank = 0, num_procs = 1;
    if (mpi_initialized) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    }

    int total_windows = size - window_size + 1;

    // Распределение окон по процессам MPI (декомпозиция по доменам)
    int windows_per_proc = total_windows / num_procs;
    int start_window = rank * windows_per_proc;
    int end_window = (rank == num_procs - 1) ? total_windows : start_window + windows_per_proc;

    // OpenMP распараллеливание внутри каждого MPI-процесса
    #pragma omp parallel for schedule(dynamic)
    for (int w = start_window; w < end_window; ++w) {
        double sum_x = 0.0, sum_y = 0.0;
        double sum_x_sq = 0.0, sum_y_sq = 0.0;
        double sum_xy = 0.0;

        // Внутренний цикл вычисления статических метрик окна (векторизуется компилятором при -O3)
        #pragma omp simd reduction(+:sum_x,sum_y,sum_x_sq,sum_y_sq,sum_xy)
        for (int i = 0; i < window_size; ++i) {
            double x = weather[w + i].temperature; // Считаем корреляцию по температуре
            double y = delays[w + i];

            sum_x += x;
            sum_y += y;
            sum_x_sq += x * x;
            sum_y_sq += y * y;
            sum_xy += x * y;
        }

        double num = (window_size * sum_xy) - (sum_x * sum_y);
        double den_sq = ((window_size * sum_x_sq) - (sum_x * sum_x)) * 
                        ((window_size * sum_y_sq) - (sum_y * sum_y));

        if (den_sq <= 1e-9) {
            out_correlations[w] = 0.0;
        } else {
            out_correlations[w] = num / std::sqrt(den_sq);
        }
    }

    // Сбор результатов со всех MPI узлов на мастер-процесс (выполняется только если реально запущено несколько процессов MPI)
    if (mpi_initialized && num_procs > 1) {
        if (rank == 0) {
            for (int p = 1; p < num_procs; ++p) {
                int p_start = p * windows_per_proc;
                int p_count = (p == num_procs - 1) ? (total_windows - p_start) : windows_per_proc;
                MPI_Recv(&out_correlations[p_start], p_count, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        } else {
            int my_count = end_window - start_window;
            MPI_Send(&out_correlations[start_window], my_count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
        MPI_Bcast(out_correlations, total_windows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        #pragma omp parallel for schedule(dynamic)
        for (int w = 0; w < total_windows; ++w) {
            double sum_x = 0.0, sum_y = 0.0;
            double sum_x_sq = 0.0, sum_y_sq = 0.0;
            double sum_xy = 0.0;

            #pragma omp simd reduction(+:sum_x,sum_y,sum_x_sq,sum_y_sq,sum_xy)
            for (int i = 0; i < window_size; ++i) {
                double x = weather[w + i].temperature;
                double y = delays[w + i];
                sum_x += x; sum_y += y;
                sum_x_sq += x * x; sum_y_sq += y * y;
                sum_xy += x * y;
            }

            double num = (window_size * sum_xy) - (sum_x * sum_y);
            double den_sq = ((window_size * sum_x_sq) - (sum_x * sum_x)) * 
                            ((window_size * sum_y_sq) - (sum_y * sum_y));

            if (den_sq <= 1e-9) {
                out_correlations[w] = 0.0;
            } else {
                out_correlations[w] = num / std::sqrt(den_sq);
            }
        }
    }
}

} // extern "C"
