set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(stlstrtok_SOURCES stlStrTok.cpp)
add_library(stlstrtok STATIC ${stlstrtok_SOURCES})

set(random_SOURCES Random.cpp)
add_library(random STATIC ${random_SOURCES})

set(parameter_SOURCES Parameter.cpp)
add_library(parameter STATIC ${parameter_SOURCES})
target_link_libraries(parameter gsl gslcblas)
