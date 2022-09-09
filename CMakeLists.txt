cmake_minimum_required(VERSION 3.16.9) # "CMake should be at least as new as your compiler"
project(LEMMA VERSION 1.0.3)
cmake_policy(SET CMP0110 OLD)

set(CMAKE_CXX_STANDARD 11)

### Boost Dependency (used in include/)
find_package(Boost REQUIRED COMPONENTS chrono date_time filesystem system thread timer iostreams)
if(Boost_FOUND)
    message("-- Boost found: ${Boost_VERSION_STRING}")
else()
    message(FATAL_ERROR "Boost not found!")
endif()

add_subdirectory(include)

### Compilation flags
set(LOC_COMPILE_OPTS -Wno-deprecated -Wno-parentheses)
if(CMAKE_BUILD_TYPE STREQUAL Debug)
    list(APPEND LOC_COMPILE_OPTS -g -DDEBUG)
else()
    list(APPEND LOC_COMPILE_OPTS -O3)
endif()

if(DEFINED STATIC_BUILD)
    list(APPEND LOC_COMPILE_OPTS -static -static-libgcc -static-libstdc++)
endif()

### BGEN Dependency [required]
    if(APPLE)
    message(STATUS "assuming build on OSX")
        list(APPEND LOC_COMPILE_OPTS -DOSX)
    else()
    message(STATUS "assuming build on Linux")
        list(APPEND LOC_COMPILE_OPTS -lz -fopenmp)
        list(APPEND LIBRARIES rt)
        set(LINKER_OPTS -O3 -fopenmp -ffloat-store)
    endif()


list(APPEND LIBRARIES dl pthread z)

### MPI Dependency [required]
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
string(APPEND LIB_DIRS " ${MPI_LINK_FLAGS}")
list(APPEND LIBRARIES ${MPI_LIBRARIES})
list(APPEND LOC_COMPILE_OPTS "${MPI_COMPILE_FLAGS}")


### INTEL MKL Dependency [optional]
if(DEFINED MKL_ROOT)
    set(MKLROOT ${MKL_ROOT})
    if (APPLE)
        message(STATUS "Setting MKLROOT=${MKL_ROOT} (assuming build on OSX)")
        list(APPEND LINKER_OPTS -L${MKLROOT}/lib -Wl,-rpath,${MKLROOT}/lib -lmkl_intel_lp64 -lmkl_sequential -lmkl_core     -lpthread -lm -ldl)
        list(APPEND LOC_COMPILE_OPTS -m64 -I${MKLROOT}/include -DEIGEN_USE_MKL_ALL)
    elseif(UNIX)
        message(STATUS "Setting MKLROOT=${MKL_ROOT} (assuming build on Linux)")
        list(APPEND LINKER_OPTS -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl)
        list(APPEND LOC_COMPILE_OPTS -m64 -I${MKLROOT}/include -DEIGEN_USE_MKL_ALL)
    else()
        message(WARNING "Unexpected OS, not using the Intel MKL library")
    endif()
else()
    message(STATUS "Optional dependancy Intel MKL Library not provided")
endif()

### Compile LEMMA
set(TARGET lemma_${PROJECT_VERSION_MAJOR}_${PROJECT_VERSION_MINOR}_${PROJECT_VERSION_PATCH})

list(APPEND LOC_COMPILE_OPTS -DVERSION_MAJOR=${PROJECT_VERSION_MAJOR})
list(APPEND LOC_COMPILE_OPTS -DVERSION_MINOR=${PROJECT_VERSION_MINOR})
list(APPEND LOC_COMPILE_OPTS -DVERSION_PATCH=${PROJECT_VERSION_PATCH})


set(SOURCES src/hyps.cpp src/variational_parameters.cpp src/genotype_matrix.cpp src/mpi_utils.cpp)
list(APPEND SOURCES src/parse_arguments.cpp src/file_utils.cpp src/eigen_utils.cpp src/rhe_reg.cpp)
list(APPEND SOURCES src/rhe_reg_component.cpp src/nelder_mead.cpp src/stats_tests.cpp)

include_directories(${PROJECT_SOURCE_DIR})
add_executable(${TARGET} src/main.cpp ${SOURCES})
target_compile_options(${TARGET} PRIVATE -msse2 ${LOC_COMPILE_OPTS})
target_include_directories(${TARGET} PRIVATE ${INCL_DIRS})
set_target_properties(${TARGET} PROPERTIES LINK_FLAGS "${LIB_DIRS}")
target_link_libraries(${TARGET} Boost::boost Boost::iostreams Boost::filesystem ${LIBRARIES} ${LINKER_OPTS})

### Compile Catch Unit Tests
set(TSOURCES unit/test_main.cpp unit/test_hyps.cpp unit/test_gplemma.cpp)
list(APPEND TSOURCES unit/test_resume_inference.cpp unit/test_eigen.cpp)
list(APPEND TSOURCES unit/test_data.cpp unit/test_snptests.cpp)

add_executable(tests unit/main.cpp ${TSOURCES} ${SOURCES})
target_link_libraries(tests Boost::boost Boost::iostreams Boost::filesystem ${LIBRARIES} ${LINKER_OPTS})
target_compile_options(tests PRIVATE -msse2 ${LOC_COMPILE_OPTS})
target_include_directories(tests PRIVATE ${INCL_DIRS})
set_target_properties(tests PROPERTIES LINK_FLAGS "${LIB_DIRS}")

include(CTest)
include(include/ParseAndAddCatchTests.cmake)
set(OptionalCatchTestLauncher ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2)
ParseAndAddCatchTests(tests)
