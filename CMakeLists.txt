cmake_minimum_required(VERSION 3.7.2)
project(LEMMA VERSION 0.11.6)

set(CMAKE_CXX_STANDARD 11)

##Compilation flags
if(CMAKE_CURRENT_SOURCE_DIR MATCHES /Users/kerin)
    message("Laptop build (OSX)")
    set(BGEN /Users/kerin/software/bgen)
    if(CMAKE_BUILD_TYPE STREQUAL Release)
        set(LOC_COMPILE_OPTS -DOSX -Wno-deprecated -Wno-parentheses -O3 -g)
        set(LINKER_OPTS -O3 -g)
    else()
        set(LOC_COMPILE_OPTS -DOSX -Wno-deprecated -Wno-parentheses -DDEBUG -g)
        set(LINKER_OPTS -g)
    endif()
    set(LIB-DIRS "-L${BGEN}/build/3rd_party/boost_1_55_0 -L/usr/local/opt/boost@1.55/lib")
elseif(CMAKE_CURRENT_SOURCE_DIR MATCHES /well/marchini/kebl4230)
    # LINUX File System - Rescomp
    message("Rescomp build (Linux)")
    set(BGEN /well/marchini/kebl4230/software/bgen/)
    set(LIBS rt)
    set(LOC_COMPILE_OPTS -Wno-deprecated -Wno-parentheses -O3 -static -static-libgcc -static-libstdc++ -lz -fopenmp -ffloat-store)
    set(LINKER_OPTS -O3 -fopenmp -ffloat-store)
    set(LIB-DIRS "-L${BGEN}/build/3rd_party/boost_1_55_0 -L/well/marchini/kebl4230/software/boost_1_62_0/stage/lib")
elseif(CMAKE_CURRENT_SOURCE_DIR MATCHES /homes/kerin)
    # LINUX File System - Garganey
    message("Garganey build (Linux)")
    set(BGEN /homes/kerin/projects/bgen/)
    set(LIBS rt)
    set(LOC_COMPILE_OPTS -Wno-deprecated -Wno-parentheses -O3 -fopenmp -g)
    set(LINKER_OPTS -O3 -fopenmp -g)
    set(LIB-DIRS "-L${BGEN}/build/3rd_party/boost_1_55_0")
else()
    message( FATAL_ERROR "Unexpected root directory structure. Where is the BGEN lib located?")
endif()

##Add INTEL MKL lib
if(CMAKE_CURRENT_SOURCE_DIR MATCHES /Users/kerin)
    # OSX File System - Laptop
    set(MKLROOT /opt/intel/compilers_and_libraries_2019.1.144/mac/mkl)
    list(APPEND LINKER_OPTS -L${MKLROOT}/lib -Wl,-rpath,${MKLROOT}/lib -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl)
    list(APPEND LOC_COMPILE_OPTS -m64 -I${MKLROOT}/include)
elseif(CMAKE_CURRENT_SOURCE_DIR MATCHES /well/marchini/kebl4230)
    # LINUX File System - Rescomp
    set(MKLROOT /users/marchini/kebl4230/intel/compilers_and_libraries_2019.0.117/linux/mkl)
    list(APPEND LINKER_OPTS -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl)
    list(APPEND LOC_COMPILE_OPTS -m64 -I${MKLROOT}/include)
elseif(CMAKE_CURRENT_SOURCE_DIR MATCHES /homes/kerin)
    # LINUX File System - Garganey w/ Intel MKL
    set(MKLROOT /homes/kerin/intel/compilers_and_libraries_2019.1.144/linux/mkl)
    list(APPEND LINKER_OPTS -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl)
    list(APPEND LOC_COMPILE_OPTS -m64 -I${MKLROOT}/include)
else()
    message( FATAL_ERROR "Unexpected root directory structure. Where is the MKL lib located?")
endif()
message(${LINKER_OPTS})

##Add project version number to preprocessor macros
list(APPEND LOC_COMPILE_OPTS -DVERSION_MAJOR=${PROJECT_VERSION_MAJOR})
list(APPEND LOC_COMPILE_OPTS -DVERSION_MINOR=${PROJECT_VERSION_MINOR})
list(APPEND LOC_COMPILE_OPTS -DVERSION_PATCH=${PROJECT_VERSION_PATCH})

set(SOURCES src/hyps.cpp src/variational_parameters.cpp src/genotype_matrix.cpp src/parse_arguments.cpp src/file_utils.cpp src/eigen_utils.cpp src/mpi_utils.cpp)
set(BGEN_INCL_DIRS ${BGEN}/genfile/include ${BGEN}/3rd_party/zstd-1.1.0/lib ${BGEN}/db/include ${BGEN}/3rd_party/sqlite3 ${BGEN}/3rd_party/boost_1_55_0)

find_package(MPI REQUIRED)
INCLUDE_DIRECTORIES(${MPI_INCLUDE_PATH})
list(APPEND LOC_COMPILE_OPTS "${MPI_COMPILE_FLAGS}")
string(APPEND LIB-DIRS " ${MPI_LINK_FLAGS}")


###Platform invariant variassbles
find_library(BGEN_LIB bgen HINTS ${BGEN}/build)
find_library(DB_LIB db HINTS ${BGEN}/build/db)
find_library(SQLITE_LIB sqlite3 HINTS ${BGEN}/build/3rd_party/sqlite3)
find_library(ZSTD_LIB zstd HINTS ${BGEN}/build/3rd_party/zstd-1.1.0)

set(LIBS boost dl pthread boost_iostreams z ${MPI_LIBRARIES} ${LIBS})

##Defining compilation commands
#set(TARGET bgen_prog)
set(TARGET bgen_prog_${PROJECT_VERSION_MAJOR}_${PROJECT_VERSION_MINOR}_${PROJECT_VERSION_PATCH})

message("Compile options:")
message(${LOC_COMPILE_OPTS})
message("Lib-dirs: ")
message(${LIB-DIRS})


include_directories(${PROJECT_SOURCE_DIR})
add_executable(${TARGET} src/bgen_prog.cpp ${SOURCES})
target_compile_options(${TARGET} PRIVATE -msse2 ${LOC_COMPILE_OPTS})
include_directories(${TARGET} PRIVATE ${BGEN_INCL_DIRS})
set_target_properties(${TARGET} PROPERTIES LINK_FLAGS "${LIB-DIRS}")
target_link_libraries(${TARGET} ${BGEN_LIB} ${DB_LIB} ${SQLITE_LIB} ${ZSTD_LIB} ${LIBS} ${LINKER_OPTS})

add_executable(${TARGET}_Xf src/bgen_prog.cpp ${SOURCES})
target_compile_options(${TARGET}_Xf PRIVATE -DDATA_AS_FLOAT -msse2 ${LOC_COMPILE_OPTS})
target_include_directories(${TARGET}_Xf PRIVATE ${BGEN_INCL_DIRS})
set_target_properties(${TARGET}_Xf PROPERTIES LINK_FLAGS "${LIB-DIRS}")
target_link_libraries(${TARGET}_Xf ${BGEN_LIB} ${DB_LIB} ${SQLITE_LIB} ${ZSTD_LIB} ${LIBS} ${LINKER_OPTS})


## Catch Unit Tests
set(TSOURCES unit/test_pve.cpp unit/test_main.cpp unit/test_hyps.cpp unit/test_resume_inference.cpp)
add_executable(tests unit/main.cpp ${TSOURCES} ${SOURCES})
target_link_libraries(tests ${BGEN_LIB} ${DB_LIB} ${SQLITE_LIB} ${ZSTD_LIB} ${LIBS} ${LINKER_OPTS})
target_compile_options(tests PRIVATE -msse2 ${LOC_COMPILE_OPTS})
target_include_directories(tests PRIVATE ${BGEN_INCL_DIRS})
set_target_properties(tests PROPERTIES LINK_FLAGS "${LIB-DIRS}")

add_executable(tests_Xf unit/main.cpp unit/test_pve.cpp)
target_link_libraries(tests_Xf ${BGEN_LIB} ${DB_LIB} ${SQLITE_LIB} ${ZSTD_LIB} ${LIBS} ${LINKER_OPTS})
target_compile_options(tests_Xf PRIVATE -DDATA_AS_FLOAT -msse2 ${LOC_COMPILE_OPTS})
target_include_directories(tests_Xf PRIVATE ${BGEN_INCL_DIRS})
set_target_properties(tests_Xf PROPERTIES LINK_FLAGS "${LIB-DIRS}")
#
### Profiling by hand
#add_executable(profile_read_cols examples/profile_read_cols.cpp examples/genotype_matrix_multithread.cpp)
#target_link_libraries(profile_read_cols boost_iostreams boost_thread ${LIBS} ${LINKER_OPTS})
#target_compile_options(profile_read_cols PRIVATE -Wno-deprecated -msse2 ${LOC_COMPILE_OPTS})
#target_include_directories(profile_read_cols PRIVATE ${BGEN}/3rd_party/boost_1_55_0)
#set_target_properties(profile_read_cols PROPERTIES LINK_FLAGS "${LIB-DIRS}")
