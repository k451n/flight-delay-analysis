cmake_minimum_required(VERSION 3.20)
project(aviation_analysis LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Поиск пакетов OpenMP и MPI
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

# Флаги оптимизации и автовекторизации компилятора
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -ffast-math -ftree-vectorize")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /openmp:experimental")
    add_compile_definitions("aligned_alloc(a,s)=_aligned_malloc(s,a)")
    add_compile_definitions("free(p)=_aligned_free(p)")
endif()

# Определение заголовочных файлов
include_directories(include ${MPI_CXX_INCLUDE_DIRS})

# Создание динамической библиотеки для вызова из Python головной программы
add_library(aviation_analysis SHARED src/aviation_analysis.cpp)
target_link_libraries(aviation_analysis PRIVATE OpenMP::OpenMP_CXX ${MPI_CXX_LIBRARIES})

# Настройка выходного имени
set_target_properties(aviation_analysis PROPERTIES OUTPUT_NAME "aviation_analysis")

# Создание исполняемого файла юнит-тестов для CI/CD-проверки
enable_testing()
add_executable(test_correlations tests/test_correlations.cpp src/aviation_analysis.cpp)
target_link_libraries(test_correlations PRIVATE OpenMP::OpenMP_CXX ${MPI_CXX_LIBRARIES})

# Регистрация теста в CTest
add_test(NAME HpcKernelTest COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:test_correlations>)
