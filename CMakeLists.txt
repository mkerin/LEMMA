cmake_minimum_required(VERSION 3.7.2)
project(bgen_prog VERSION 0.10.1)

set(CMAKE_CXX_STANDARD 11)

#Bring the headers, such as Student.h into the project
#include_directories(include)

## Laptop build
#Compilation flags
if(CMAKE_CURRENT_SOURCE_DIR MATCHES /Users/kerin)
    # OSX File System - Laptop
    set(BGEN /Users/kerin/software/bgen)
    set(LOC_COMPILE_OPTS -DOSX -DDEBUG)
elseif(CMAKE_CURRENT_SOURCE_DIR MATCHES /well/marchini/kebl4230)
    # LINUX File System - Rescomp
    set(BGEN /well/marchini/kebl4230/software/bgen/)
    set(LIBS rt)
    set(LOC_COMPILE_OPTS -O3 -static -static-libgcc -static-libstdc++ -lz -fopenmp)
elseif(CMAKE_CURRENT_SOURCE_DIR MATCHES /homes/kerin)
    # LINUX File System - Garganey
    set(BGEN /homes/kerin/projects/bgen/)
    set(LIBS rt)
    set(LOC_COMPILE_OPTS -O3 -fopenmp)
else()
    message( FATAL_ERROR "Unexpected root directory structure. Where is the BGEN lib located?")
endif()

set(SOURCES src/bgen_prog.cpp)

##Platform invariant variables
find_library(BGEN_LIB bgen HINTS ${BGEN}/build)
find_library(DB_LIB db HINTS ${BGEN}/build/db)
find_library(SQLITE_LIB sqlite3 HINTS ${BGEN}/build/3rd_party/sqlite3)
find_library(ZSTD_LIB zstd HINTS ${BGEN}/build/3rd_party/zstd-1.1.0)

set(LIB-DIRS "-L${BGEN}/build/3rd_party/boost_1_55_0")
set(LIBS boost dl pthread boost_iostreams z ${LIBS})

##Defining compilation commands
set(TARGET bgen_prog)
#set(TARGET bgen_prog_${PROJECT_VERSION_MAJOR}_${PROJECT_VERSION_MINOR}_${PROJECT_VERSION_PATCH})
add_executable(${TARGET} ${SOURCES})

target_compile_options(${TARGET} PRIVATE -Wno-deprecated -msse2 ${LOC_COMPILE_OPTS})

target_include_directories(${TARGET} PRIVATE ${BGEN}/genfile/include ${BGEN}/3rd_party/zstd-1.1.0/lib ${BGEN}/db/include ${BGEN}/3rd_party/sqlite3 ${BGEN}/3rd_party/boost_1_55_0)

#Set search paths to 3rd party libs
set_target_properties(${TARGET} PROPERTIES LINK_FLAGS "${LIB-DIRS}")

#Link libs
target_link_libraries(${TARGET} ${BGEN_LIB} ${DB_LIB} ${SQLITE_LIB} ${ZSTD_LIB} ${LIBS})

## Catch Unit Tests
add_executable(catch_tests tests/tests-main.cpp)
target_link_libraries(catch_tests ${BGEN_LIB} ${DB_LIB} ${SQLITE_LIB} ${ZSTD_LIB} ${LIBS})
target_compile_options(catch_tests PRIVATE -Wno-deprecated -msse2 ${LOC_COMPILE_OPTS})
target_include_directories(catch_tests PRIVATE ${BGEN}/genfile/include ${BGEN}/3rd_party/zstd-1.1.0/lib ${BGEN}/db/include ${BGEN}/3rd_party/sqlite3 ${BGEN}/3rd_party/boost_1_55_0)
set_target_properties(catch_tests PROPERTIES LINK_FLAGS "${LIB-DIRS}")
