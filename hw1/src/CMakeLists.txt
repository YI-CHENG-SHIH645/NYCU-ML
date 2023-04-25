file(GLOB HEADER_LIST ${CMAKE_SOURCE_DIR}/src/*.h)
file(GLOB CPP_LIST ${CMAKE_SOURCE_DIR}/src/*.cpp)

add_library(hw1 ${HEADER_LIST} ${CPP_LIST})
