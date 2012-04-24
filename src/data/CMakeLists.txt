set(CMAKE_INCLUDE_CURRENT_DIR ON)

include_directories(${CMAKE_SOURCE_DIR}/src/Framework)

set(data_SOURCES Data.cpp)
add_library(data STATIC ${data_SOURCES})

set(PosteriorReader_SOURCES PosteriorReader.cpp)
add_library(PosteriorReader STATIC ${PosteriorReader_SOURCES}) 