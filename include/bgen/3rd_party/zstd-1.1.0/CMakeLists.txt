
set(SOURCES lib/common/zstd_common.c lib/common/entropy_common.c lib/common/fse_decompress.c lib/common/xxhash.c)
list(APPEND SOURCES lib/decompress/huf_decompress.c lib/decompress/zbuff_decompress.c lib/decompress/zstd_decompress.c)
list(APPEND SOURCES lib/compress/fse_compress.c lib/compress/huf_compress.c lib/compress/zbuff_compress.c)
list(APPEND SOURCES lib/compress/zstd_compress.c)


add_library(zstd ${SOURCES})
set_property(TARGET zstd PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(zstd PRIVATE
                            ${CMAKE_CURRENT_SOURCE_DIR}/lib/common 
                            ${CMAKE_CURRENT_SOURCE_DIR}/lib/compress 
                            ${CMAKE_CURRENT_SOURCE_DIR}/lib/decompress)
target_include_directories(zstd PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/lib)


