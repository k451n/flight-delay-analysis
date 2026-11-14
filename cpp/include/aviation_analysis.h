#ifndef AVIATION_ANALYSIS_H
#define AVIATION_ANALYSIS_H

#ifdef __cplusplus
extern "C" {
#endif

// Структура для передачи часовых срезов метеоданных из Python
struct WeatherRecord {
    double temperature;
    double humidity;
    double precipitation;
    double wind_speed;
    double cloud_cover;
    double visibility;
};

/**
 * Высокопроизводительный конвейер обработки данных (OpenMP + MPI + SIMD)
 * @param weather Массив метеоструктур
 * @param delays Массив средних задержек рейсов (в минутах)
 * @param size Размер векторов данных (количество часов)
 * @param window_size Размер скользящего окна в часах (24/168)
 * @param out_correlations Выходной массив для записи результатов (размер: size - window_size + 1)
 */
__declspec(dllexport) void compute_rolling_correlation(
    const WeatherRecord* weather,
    const double* delays,
    int size,
    int window_size,
    double* out_correlations
);

/**
 * SIMD-векторизованная функция нормализации и интерполяции пропусков
 * Демонстрирует явное выравнивание адресов
 */
__declspec(dllexport) void preprocess_weather_vectors(
    const double* raw_data,
    double* clean_data,
    int size,
    double missing_marker
);

#ifdef __cplusplus
}
#endif

#endif // AVIATION_ANALYSIS_H
