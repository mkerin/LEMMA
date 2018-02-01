TARGET := bin/bgen_prog
SRCDIR := src
FLAGS = -g3 -std=c++11 -Ibuild/genfile/include/ -I3rd_party/zstd-1.1.0/lib/ \
        -Ibuild/db/include/ -I3rd_party/sqlite3 -I3rd_party/boost_1_55_0  \
        -Lbuild/ -Lbuild/3rd_party/zstd-1.1.0 -Lbuild/db -Lbuild/3rd_party/sqlite3 \
        -Lbuild/3rd_party/boost_1_55_0 -Lbuild/ -lbgen  -ldb -lsqlite3 -lboost \
        -lz -ldl -lrt -lpthread -lzstd -Wno-deprecated

HEADERS := parse_arguments.hpp data.hpp class.h bgen_parser.hpp
HEADERS := $(addprefix $(SRCDIR)/,$(HEADERS))

$(TARGET) : $(SRCDIR)/bgen_prog.cpp $(HEADERS)
	g++ -o $@ $< $(FLAGS)

# UnitTests
# Note: this uses the Catch library to Unit Test the cpp source code. If we want
# to test input/output of the executable we do that directly with the Makefile
# and `diff` command.
UNITTESTS := test-data.cpp tests-parse-arguments.cpp 
UNITTESTS := $(addprefix test/,$(UNITTESTS))

test/tests: test/tests-main.o $(UNITTESTS)
	g++ $< -o $@  $(FLAGS) && ./$@ -r compact

test/tests-main.o: test/tests-main.cpp
	g++ test/tests-main.cpp -c -o $@ $(FLAGS)

# IO Tests
# Files in data/out are regarded as 'true', and we check that the equivalent
# file in data/io_test is identical after making changes to the executable.
.PHONY: testIO
IOfiles := t1_range t2_lm
IOfiles := $(addprefix data/io_test/,$(addsuffix /attempt.out,$(IOfiles)))

testIO: $(IOfiles)
	@

# Test of --range command
data/io_test/t1_range/attempt.out: data/io_test/example.v11.bgen ./bin/bgen_prog
	./bin/bgen_prog --convert_to_vcf --bgen $< --range 01 1 3000 --out $@
	diff data/io_test/t1_range/answer.out $@

# Small regression analysis; subset to complete cases, normalising, 1dof interaction model
data/io_test/t2_lm/attempt.out: data/io_test/example.v11.bgen ./bin/bgen_prog
	./bin/bgen_prog --lm \
	    --bgen $< \
	    --pheno data/io_test/t2_lm/t2.pheno \
	    --covar data/io_test/t2_lm/t2.covar \
	    --range 01 2000 2000 --out $@
	diff data/io_test/t2_lm/answer.out $@

# Clean dir
cleanIO:
	rm $(IOfiles)

clean:
	rm $(TARGET)
	rm $(IOfiles)
