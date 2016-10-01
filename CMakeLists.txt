# Check cmake version.
cmake_minimum_required (VERSION 2.8)


# Project specification.
project (GSKS)
set (GSKS_VERSION_MAJOR 1)
set (GSKS_VERSION_MINOR 1)
set (GSKS_VERSION_PATCH 0)
set (GSKS_VERSION ${GSKS_VERSION_MAJOR}.${GSKS_VERSION_MINOR}.${GSKS_VERSION_PATCH})


# Configure the path structure.
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set (CMAKE_RELEASE_POSTFIX "")
set (CMAKE_DEBUG_POSTFIX "-debug")


# Turn on testing.
ENABLE_TESTING()


# Decide architecture
set (GSKS_ARCH_MAJOR $ENV{GSKS_ARCH_MAJOR})
set (GSKS_ARCH_MINOR $ENV{GSKS_ARCH_MINOR})
set (GSKS_ARCH ${GSKS_ARCH_MAJOR}/${GSKS_ARCH_MINOR})


# Compiler Options (GSKS can use Intel or GNU compilers.)
if ($ENV{GSKS_USE_INTEL} MATCHES "true")
  set (CMAKE_C_COMPILER   icc )
  set (CMAKE_CXX_COMPILER icpc)
  set (GSKS_CFLAGS          "-O3 -qopenmp -mavx -Werror -Wall -pedantic")
  set (CMAKE_EXE_LINKER_FLAGS "-lpthread -qopenmp -lm")
else ($ENV{GSKS_USE_INTEL} MATCHES "true")
  set (CMAKE_C_COMPILER   gcc)
  set (CMAKE_CXX_COMPILER g++)
  set (GSKS_CFLAGS          "-O3 -fopenmp -mavx")
  set (CMAKE_EXE_LINKER_FLAGS "-lpthread -fopenmp -lm")
endif ($ENV{GSKS_USE_INTEL} MATCHES "true")


if ($ENV{GSKS_USE_VML} MATCHES "true")
  if ($ENV{GSKS_USE_INTEL} MATCHES "true")
    set (GSKS_CFLAGS          "${GSKS_CFLAGS} -DUSE_VML")
  endif ($ENV{GSKS_USE_INTEL} MATCHES "true")
endif ($ENV{GSKS_USE_VML} MATCHES "true")

if ($ENV{GSKS_MIC_AVX512} MATCHES "true")
  if ($ENV{GSKS_USE_INTEL} MATCHES "true")
	set (GSKS_CFLAGS          "${GSKS_CFLAGS} -xMIC-AVX512 -DGSKS_MIC_AVX512")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lmemkind")
  endif ($ENV{GSKS_USE_INTEL} MATCHES "true")
endif ($ENV{GSKS_MIC_AVX512} MATCHES "true")



if ($ENV{GSKS_USE_BLAS} MATCHES "true")
  set (GSKS_CFLAGS          "${GSKS_CFLAGS} -DUSE_BLAS")
endif ($ENV{GSKS_USE_BLAS} MATCHES "true")
set (CMAKE_C_FLAGS      "${CMAKE_C_FLAGS} ${GSKS_CFLAGS}")
set (CMAKE_CXX_FLAGS    "${CMAKE_CXX_FLAGS} ${GSKS_CFLAGS}")


# Linking infos.
if (($ENV{GSKS_USE_BLAS} MATCHES "true") OR ($ENV{GSKS_USE_VML} MATCHES "true"))
  if ($ENV{GSKS_USE_INTEL} MATCHES "true")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -mkl=parallel")
  else ($ENV{GSKS_USE_INTEL} MATCHES "true")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lblas")
  endif ($ENV{GSKS_USE_INTEL} MATCHES "true")
endif (($ENV{GSKS_USE_BLAS} MATCHES "true") OR ($ENV{GSKS_USE_VML} MATCHES "true"))


# Software dependencies.
set (MKL_DIR $ENV{GSKS_MKL_DIR})


# Headers.
include_directories ("include" "${CMAKE_SOURCE_DIR}/micro_kernel/${GSKS_ARCH}" "${MKL_DIR}/include")
file (GLOB GSKS_HDRS ${CMAKE_SOURCE_DIR}/include/*)
install (FILES ${GSKS_HDRS} DESTINATION ${CMAKE_BINARY_DIR}/include)


# Library dependencies
file (GLOB FRAME_CC_SRC  ${CMAKE_SOURCE_DIR}/frame/*.c)
file (GLOB FRAME_CXX_SRC ${CMAKE_SOURCE_DIR}/frame/*.cpp)
file (GLOB KERNEL_SRC ${CMAKE_SOURCE_DIR}/micro_kernel/${GSKS_ARCH}/*.c)


# Build the static library.
add_library (gsks ${FRAME_CC_SRC} ${FRAME_CXX_SRC} ${KERNEL_SRC})


# Build the executable files.
add_executable (test_dgsks.x ${CMAKE_SOURCE_DIR}/test/test_dgsks.c)
target_link_libraries(test_dgsks.x gsks)
add_executable (test_dgsks_list.x ${CMAKE_SOURCE_DIR}/test/test_dgsks_list.cpp)
target_link_libraries(test_dgsks_list.x gsks)


# Install shell script
file (GLOB GSKS_SCRIPTS ${CMAKE_SOURCE_DIR}/test/*.sh)
install (FILES ${GSKS_SCRIPTS} DESTINATION ${CMAKE_BINARY_DIR}/bin)


# Report
message ("===================================================")
message ("Version      =${GSKS_VERSION}")
message ("Source       =${CMAKE_SOURCE_DIR}")
message ("Target       =${CMAKE_BINARY_DIR}")
message ("GSKS_ARCH    =${GSKS_ARCH}")
message ("CC           =${CMAKE_C_COMPILER}")
message ("CFLAGS       =${CMAKE_C_FLAGS}")
message ("CXX          =${CMAKE_CXX_COMPILER}")
message ("CXXFLAGS     =${CMAKE_CXX_FLAGS}")
message ("Linker       =${CMAKE_EXE_LINKER_FLAGS}")
message ("MKLDIR       =${MKL_DIR}")
message ("===================================================")
