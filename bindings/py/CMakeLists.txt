# =================================================================
# Img2Num Python Bindings
# =================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(PyImg2Num
  LANGUAGES CXX
  VERSION 0.0.0
)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PYBIND11_FINDPYTHON ON)
# find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NumPy_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# 3. Print it out to your cibuildwheel logs so you can verify it worked
message(STATUS "Found NumPy include directory: ${NumPy_INCLUDE_DIR}")

find_package(pybind11 REQUIRED)

file(GLOB BINDING_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)
pybind11_add_module(_img2num MODULE ${BINDING_SRC})

target_link_libraries(_img2num PRIVATE Img2Num)

target_include_directories(_img2num PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    {NumPy_INCLUDE_DIR}
)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/packages/py/img2num/
    DESTINATION img2num
    FILES_MATCHING PATTERN "*.py"
)

install(TARGETS _img2num
    LIBRARY DESTINATION img2num
    RUNTIME DESTINATION img2num
)
