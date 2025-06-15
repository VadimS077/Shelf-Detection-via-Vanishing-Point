add_library(sdvprp sdvprp.hpp sdvprp.cpp)
target_link_libraries(sdvprp ${OpenCV_LIBS})

target_include_directories(sdvprp PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)