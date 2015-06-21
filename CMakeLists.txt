# Check cmake version.
cmake_minimum_required (VERSION 2.8)


# Project specification.
project (KS)
set (KS_VERSION_MAJOR 1)
set (KS_VERSION_MINOR 1)
set (KS_VERSION_PATCH 0)
set (KS_VERSION ${KS_VERSION_MAJOR}.${KS_VERSION_MINOR}.${KS_VERSION_PATCH})


# Configure the path structure.
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set (CMAKE_RELEASE_POSTFIX "")
set (CMAKE_DEBUG_POSTFIX "-debug")


# Turn on testing.
ENABLE_TESTING()


# Decide architecture
#set (KS_ARCH_MAJOR "x86_64")
set (KS_ARCH_MAJOR $ENV{KS_ARCH_MAJOR})
#set (KS_ARCH_MINOR "sandybridge")
set (KS_ARCH_MINOR $ENV{KS_ARCH_MINOR})
set (KS_ARCH ${KS_ARCH_MAJOR}/${KS_ARCH_MINOR})


# Compiler Options (KS must use Intel compiler.)
set (CMAKE_C_COMPILER   icc )
set (CMAKE_CXX_COMPILER icpc)
set (KS_CFLAGS          "-O3 -openmp -mavx -Werror -Wall -pedantic")
set (CMAKE_C_FLAGS      "${CMAKE_C_FLAGS} ${KS_CFLAGS}")
set (CMAKE_CXX_FLAGS    "${CMAKE_CXX_FLAGS} ${KS_CFLAGS}")


# Software dependencies.
#set (MKL_DIR /opt/intel/mkl)
set (MKL_DIR $ENV{KS_MKL_DIR})


# Linking infos.
set (CMAKE_EXE_LINKER_FLAGS "-lpthread -lm -mkl=parallel")


# Headers.
include_directories ("include" "${MKL_DIR}/include")
file (GLOB KS_HDRS ${CMAKE_SOURCE_DIR}/include/*)
install (FILES ${KS_HDRS} DESTINATION ${CMAKE_BINARY_DIR}/include)


# Library dependencies
file (GLOB FRAME_CC_SRC  ${CMAKE_SOURCE_DIR}/frame/*.c)
file (GLOB FRAME_CXX_SRC ${CMAKE_SOURCE_DIR}/frame/*.cpp)
file (GLOB KERNEL_SRC ${CMAKE_SOURCE_DIR}/micro_kernel/${KS_ARCH}/*.c)


# Build the static library.
add_library (ks ${FRAME_CC_SRC} ${FRAME_CXX_SRC} ${KERNEL_SRC})


# Build the executable files.
add_executable (test_dgsks.x ${CMAKE_SOURCE_DIR}/test/test_dgsks.c)
target_link_libraries(test_dgsks.x ks)
add_executable (test_dgsks_list.x ${CMAKE_SOURCE_DIR}/test/test_dgsks_list.cpp)
target_link_libraries(test_dgsks_list.x ks)


# Install shell script
file (GLOB KS_SCRIPTS ${CMAKE_SOURCE_DIR}/test/*.sh)
install (FILES ${KS_SCRIPTS} DESTINATION ${CMAKE_BINARY_DIR}/bin)


# Report
message ("===================================================")
message ("Version      =${KS_VERSION}")
message ("Source       =${CMAKE_SOURCE_DIR}")
message ("Target       =${CMAKE_BINARY_DIR}")
message ("KS_ARCH      =${KS_ARCH}")
message ("CC           =${CMAKE_C_COMPILER}")
message ("CFLAGS       =${CMAKE_C_FLAGS}")
message ("CXX          =${CMAKE_CXX_COMPILER}")
message ("CXXFLAGS     =${CMAKE_CXX_FLAGS}")
message ("Linker       =${CMAKE_EXE_LINKER_FLAGS}")
message ("MKLDIR       =${MKL_DIR}")
message ("===================================================")
