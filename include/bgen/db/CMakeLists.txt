add_library(db src/Connection.cpp src/Error.cpp src/SQLStatement.cpp src/SQLite3Connection.cpp src/SQLite3Statement.cpp)
target_include_directories(db PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(db PRIVATE ${Boost_LIBRARIES})
target_link_libraries(db PUBLIC  sqlite3)