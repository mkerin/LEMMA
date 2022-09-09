add_executable(test_bgen main.cpp ProbSetCheck.cpp test_bgen_snp_format.cpp test_variant_data_block.cpp
                main.cpp test_little_endian.cpp test_bgen_header_format.cpp test_utils.cpp)
target_link_libraries(test_bgen PRIVATE bgen ${ZLIB_LIBRARIES} zstd)
target_include_directories(test_bgen PRIVATE
                          "${CMAKE_CURRENT_SOURCE_DIR}"
                          ${ZLIB_INCLUDE_DIRS}
                          )
