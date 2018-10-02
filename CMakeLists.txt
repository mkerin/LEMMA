cmake_minimum_required(VERSION 3.7.2)
project(bgen_prog VERSION 0.9.0)

set(CMAKE_CXX_STANDARD 11)

#Bring the headers, such as Student.h into the project
#include_directories(include)

## Laptop build
#Compilation flags
set(BGEN /Users/kerin/software/bgen)
set(SOURCES src/bgen_prog.cpp)

##Platform invariant variables
find_library(BGEN_LIB bgen HINTS ${BGEN}/build)
find_library(DB_LIB db HINTS ${BGEN}/build/db)
find_library(SQLITE_LIB sqlite3 HINTS ${BGEN}/build/3rd_party/sqlite3)
find_library(ZSTD_LIB zstd HINTS ${BGEN}/build/3rd_party/zstd-1.1.0)

set(LIB-DIRS "-L${BGEN}/build/3rd_party/boost_1_55_0")
set(LIBS boost dl pthread boost_iostreams z)

##Defining compilation commands
add_executable(bgen_prog ${SOURCES})

target_compile_options(bgen_prog PRIVATE -Wno-deprecated -msse2 -DOSX)

target_include_directories(bgen_prog PRIVATE ${BGEN}/genfile/include ${BGEN}/3rd_party/zstd-1.1.0/lib ${BGEN}/db/include ${BGEN}/3rd_party/sqlite3 ${BGEN}/3rd_party/boost_1_55_0)

#Set search paths to 3rd party libs
set_target_properties(bgen_prog PROPERTIES LINK_FLAGS "${LIB-DIRS}")

#Link libs
target_link_libraries(bgen_prog ${BGEN_LIB} ${DB_LIB} ${SQLITE_LIB} ${ZSTD_LIB} ${LIBS})

## Catch Unit Tests
add_executable(catch_tests tests/tests-main.cpp)
target_link_libraries(catch_tests ${BGEN_LIB} ${DB_LIB} ${SQLITE_LIB} ${ZSTD_LIB} ${LIBS})
target_compile_options(catch_tests PRIVATE -Wno-deprecated -msse2 -DOSX)
target_include_directories(catch_tests PRIVATE ${BGEN}/genfile/include ${BGEN}/3rd_party/zstd-1.1.0/lib ${BGEN}/db/include ${BGEN}/3rd_party/sqlite3 ${BGEN}/3rd_party/boost_1_55_0)
set_target_properties(catch_tests PROPERTIES LINK_FLAGS "${LIB-DIRS}")
