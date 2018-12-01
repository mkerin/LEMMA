cmake_minimum_required(VERSION 3.7.2)
project(bgen_prog VERSION 0.10.4)

set(CMAKE_CXX_STANDARD 11)

##Compilation flags
if(CMAKE_CURRENT_SOURCE_DIR MATCHES /Users/kerin)
    message("Laptop build (OSX)")
    set(BGEN /Users/kerin/software/bgen)
    if(CMAKE_BUILD_TYPE STREQUAL Release)
        message("Release build")
        set(LOC_COMPILE_OPTS -DOSX -Wno-deprecated -Wno-parentheses -O3 -g)
        set(LINKER_OPTS -O3 -g)
    else()
        message("Debug build")
        set(LOC_COMPILE_OPTS -DOSX -Wno-deprecated -Wno-parentheses -DDEBUG -pg -Og -g)
        set(LINKER_OPTS -pg -Og -g)
    endif()
    set(LIB-DIRS "-L${BGEN}/build/3rd_party/boost_1_55_0")
elseif(CMAKE_CURRENT_SOURCE_DIR MATCHES /well/marchini/kebl4230)
    # LINUX File System - Rescomp
    message("Rescomp build (Linux)")
    set(BGEN /well/marchini/kebl4230/software/bgen/)
    set(LIBS rt)
    set(LOC_COMPILE_OPTS -Wno-deprecated -Wno-parentheses -O3 -static -static-libgcc -static-libstdc++ -lz -fopenmp)
    set(LINKER_OPTS -O3 -fopenmp)
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
    set(MKLROOT NA)
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
message(${LOC_COMPILE_OPTS})

##Add project version number to preprocessor macros
list(APPEND LOC_COMPILE_OPTS -DVERSION_MAJOR=${PROJECT_VERSION_MAJOR})
list(APPEND LOC_COMPILE_OPTS -DVERSION_MINOR=${PROJECT_VERSION_MINOR})
list(APPEND LOC_COMPILE_OPTS -DVERSION_PATCH=${PROJECT_VERSION_PATCH})

set(SOURCES src/bgen_prog.cpp)



##Platform invariant variables
find_library(BGEN_LIB bgen HINTS ${BGEN}/build)
find_library(DB_LIB db HINTS ${BGEN}/build/db)
find_library(SQLITE_LIB sqlite3 HINTS ${BGEN}/build/3rd_party/sqlite3)
find_library(ZSTD_LIB zstd HINTS ${BGEN}/build/3rd_party/zstd-1.1.0)

set(LIBS boost dl pthread boost_iostreams z ${LIBS})

##Defining compilation commands
#set(TARGET bgen_prog)
set(TARGET bgen_prog_${PROJECT_VERSION_MAJOR}_${PROJECT_VERSION_MINOR}_${PROJECT_VERSION_PATCH})

add_executable(${TARGET} ${SOURCES})
target_compile_options(${TARGET} PRIVATE -msse2 ${LOC_COMPILE_OPTS})
target_include_directories(${TARGET} PRIVATE ${BGEN}/genfile/include ${BGEN}/3rd_party/zstd-1.1.0/lib ${BGEN}/db/include ${BGEN}/3rd_party/sqlite3 ${BGEN}/3rd_party/boost_1_55_0)
set_target_properties(${TARGET} PROPERTIES LINK_FLAGS "${LIB-DIRS}")
target_link_libraries(${TARGET} ${BGEN_LIB} ${DB_LIB} ${SQLITE_LIB} ${ZSTD_LIB} ${LIBS} ${LINKER_OPTS})

add_executable(${TARGET}_Xf ${SOURCES})
target_compile_options(${TARGET}_Xf PRIVATE -DDATA_AS_FLOAT -msse2 ${LOC_COMPILE_OPTS})
target_include_directories(${TARGET}_Xf PRIVATE ${BGEN}/genfile/include ${BGEN}/3rd_party/zstd-1.1.0/lib ${BGEN}/db/include ${BGEN}/3rd_party/sqlite3 ${BGEN}/3rd_party/boost_1_55_0)
set_target_properties(${TARGET}_Xf PROPERTIES LINK_FLAGS "${LIB-DIRS}")
target_link_libraries(${TARGET}_Xf ${BGEN_LIB} ${DB_LIB} ${SQLITE_LIB} ${ZSTD_LIB} ${LIBS} ${LINKER_OPTS})


## Catch Unit Tests
add_executable(catch_tests tests/tests-main.cpp)
target_link_libraries(catch_tests ${BGEN_LIB} ${DB_LIB} ${SQLITE_LIB} ${ZSTD_LIB} ${LIBS} ${LINKER_OPTS})
target_compile_options(catch_tests PRIVATE -msse2 ${LOC_COMPILE_OPTS})
target_include_directories(catch_tests PRIVATE ${BGEN}/genfile/include ${BGEN}/3rd_party/zstd-1.1.0/lib ${BGEN}/db/include ${BGEN}/3rd_party/sqlite3 ${BGEN}/3rd_party/boost_1_55_0)
set_target_properties(catch_tests PROPERTIES LINK_FLAGS "${LIB-DIRS}")

add_executable(catch_tests_Xf tests/tests-main.cpp)
target_link_libraries(catch_tests_Xf ${BGEN_LIB} ${DB_LIB} ${SQLITE_LIB} ${ZSTD_LIB} ${LIBS} ${LINKER_OPTS})
target_compile_options(catch_tests_Xf PRIVATE -DDATA_AS_FLOAT -msse2 ${LOC_COMPILE_OPTS})
target_include_directories(catch_tests_Xf PRIVATE ${BGEN}/genfile/include ${BGEN}/3rd_party/zstd-1.1.0/lib ${BGEN}/db/include ${BGEN}/3rd_party/sqlite3 ${BGEN}/3rd_party/boost_1_55_0)
set_target_properties(catch_tests_Xf PROPERTIES LINK_FLAGS "${LIB-DIRS}")
#
### Profiling by hand
#add_executable(profile_read_cols examples/profile_read_cols.cpp examples/genotype_matrix_multithread.cpp)
#target_link_libraries(profile_read_cols boost_iostreams boost_thread ${LIBS} ${LINKER_OPTS})
#target_compile_options(profile_read_cols PRIVATE -Wno-deprecated -msse2 ${LOC_COMPILE_OPTS})
#target_include_directories(profile_read_cols PRIVATE ${BGEN}/3rd_party/boost_1_55_0)
#set_target_properties(profile_read_cols PROPERTIES LINK_FLAGS "${LIB-DIRS}")
