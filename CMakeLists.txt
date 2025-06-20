cmake_minimum_required(VERSION 3.20)
project(CourseWork)

# Базовые настройки
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Настройка выходных директорий
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Поиск OpenCV
find_package(OpenCV REQUIRED)
find_package(nlohmann_json REQUIRED)

# Сообщаем о найденной версии OpenCV
message(STATUS "Found OpenCV ${OpenCV_VERSION} at ${OpenCV_DIR}")

# Добавление поддиректории с библиотекой
add_subdirectory(sdvprp)

# Создание исполняемого файла
add_executable(cw "cw.cpp")
target_link_libraries(cw sdvprp ${OpenCV_LIBS} nlohmann_json::nlohmann_json)

# Установка только исполняемого файла
install(TARGETS cw
    RUNTIME DESTINATION bin
    BUNDLE DESTINATION bin
)

# Установка заголовочных файлов библиотеки (опционально)
install(FILES sdvprp/sdvprp.hpp
    DESTINATION include/sdvprp
)