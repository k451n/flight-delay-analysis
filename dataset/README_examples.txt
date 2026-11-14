Эти файлы - синтетический пример (не реальные данные): с их помощью можно попробовать режим «Анализ».

Вариант А (проще): example_hourly.csv - один файл, строка = час.
Вариант Б: example_weather.csv + example_flights.csv (+ example_metar.csv) - исходные таблицы.

Импорт своих данных из CSV в почасовой датасет, с которым работает FDA_main_analyzer.py.

Поддерживаются два способа подготовки данных.

А) Готовый почасовой CSV (один файл, строка = час):
     hour_utc    - время начала часа (UTC; допускаются time / datetime / timestamp)
     n_flights, n_completed, n_cancelled, n_delayed15, mean_delay_min    - рейсы за час
     temperature_2m, relative_humidity_2m, precipitation, snowfall,
     wind_speed_10m, wind_gusts_10m, cloud_cover    - погода (км/ч, °C, мм, %)
     n_diverted, median_delay_min, vsby_mi (или visibility_km), ceiling_ft, n_metar    - необязательно
   Это формат файлов hourly_dataset_*.csv, которые создаёт FDA_main_analyzer.py.

Б) Исходные таблицы (как в коллекциях сборщиков FDA_*):
     weather: время + признаки Open-Meteo (формат hourly_weather или выгрузки Open-Meteo)
     flights: по строке на рейс: время (scheduled_time_utc / hour_utc), delay_min,
              is_cancelled или status; необязательно direction, codeshare_status, is_cargo, base_airport
     metar (необязательно): выгрузка Iowa Mesonet - valid, vsby, skyc1-3, skyl1-3 ('M' = нет данных)

Общие правила чтения: разделитель , ; или табуляция определяется автоматически (для ; десятичная запятая),
строки-комментарии '#' в начале пропускаются, 'M', 'NA', 'null' и пустые ячейки = пропуск, кодировка UTF-8 или CP1251.
Время с часовым поясом (+03:00) переводится в UTC; время без пояса считается UTC (naive_tz="UTC") или
московским (naive_tz="MSK").
