TARGET := bin/bgen_prog
SRCDIR := src
FLAGS = -g -std=c++11 -Ibuild/genfile/include/ -I3rd_party/zstd-1.1.0/lib/ -Ibuild/db/include/ -I3rd_party/sqlite3 -I3rd_party/boost_1_55_0  -Lbuild/ -Lbuild/3rd_party/zstd-1.1.0 -Lbuild/db -Lbuild/3rd_party/sqlite3 -Lbuild/3rd_party/boost_1_55_0 -Lbuild/ -lbgen  -ldb -lsqlite3 -lboost -lz -ldl -lrt -lpthread -lzstd

$(TARGET) : $(SRCDIR)/bgen_prog.cpp
	g++ -o $@ $(SRCDIR)/bgen_prog.cpp $(FLAGS)

# tests
UNITTESTS := tests-parse-arguments.cpp
UNITTESTS := $(addprefix test/,$(UNITTESTS))

test/tests: test/tests-main.o $(UNITTESTS)
	g++ $< $(UNITTESTS) -o $@  $(FLAGS) && ./$@ -r compact

test/tests-main.o: test/tests-main.cpp
	g++ test/tests-main.cpp -c -o $@ $(FLAGS)

# cleaning
clean :
	rm $(TARGET)
