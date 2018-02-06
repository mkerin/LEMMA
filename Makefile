TARGET := bin/bgen_prog
SRCDIR := src
INCLUDES = -Ibuild/genfile/include/ -I3rd_party/zstd-1.1.0/lib/ \
           -Ibuild/db/include/ -I3rd_party/sqlite3 -I3rd_party/boost_1_55_0
LIBS =     -Lbuild/ -Lbuild/3rd_party/zstd-1.1.0 -Lbuild/db -Lbuild/3rd_party/sqlite3 \
           -Lbuild/3rd_party/boost_1_55_0 -lbgen -ldb -lsqlite3 -lboost \
           -lz -ldl -lrt -lpthread -lzstd
FLAGS = -g3 -std=c++11 -Wno-deprecated $(LIBS) $(INCLUDES)

HEADERS := parse_arguments.hpp data.hpp class.h bgen_parser.hpp
HEADERS := $(addprefix $(SRCDIR)/,$(HEADERS))

rescomp: CXX = /apps/well/gcc/7.2.0/bin/g++
rescomp: $(TARGET)

garganey: CXX = g++
garganey: $(TARGET)

$(TARGET) : $(SRCDIR)/bgen_prog.cpp $(HEADERS)
	$(info $$CXX is [${CXX}])
	$(info $$CFLAGS is [${CFLAGS}])
	$(info $$LDFLAGS is [${LDFLAGS}])
	$(CXX) -o $@ $< $(FLAGS)

# UnitTests
# Note: this uses the Catch library to Unit Test the cpp source code. If we want
# to test input/output of the executable we do that directly with the Makefile
# and `diff` command.
# UNITTESTS := test-data.cpp tests-parse-arguments.cpp 
# UNITTESTS := $(addprefix test/,$(UNITTESTS))

test/tests: test/tests-main.o
	g++ $< -o $@  $(FLAGS) && ./$@ -r compact

test/tests-main.o: test/tests-main.cpp
	g++ test/tests-main.cpp -c -o $@ $(FLAGS)

# IO Tests
# Files in data/out are regarded as 'true', and we check that the equivalent
# file in data/io_test is identical after making changes to the executable.
.PHONY: testIO
IOfiles := t1_range t2_lm t3_lm_two_chunks tr_lm_2dof
IOfiles := $(addprefix data/io_test/,$(addsuffix /attempt.out,$(IOfiles)))
IOfiles += $(addprefix data/io_test/t2_lm/attempt, $(addsuffix .out,B C D))

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
	    --pheno $(dir $@)t2.pheno \
	    --covar $(dir $@)t2.covar \
	    --range 01 2000 2001 --out $@
	diff $(dir $@)answer.out $@

# Same regression analysis as t2_lm but with chunk exact right size
data/io_test/t2_lm/attemptB.out: data/io_test/example.v11.bgen ./bin/bgen_prog
	./bin/bgen_prog --lm \
	    --bgen $< \
	    --pheno $(dir $@)t2.pheno \
	    --covar $(dir $@)t2.covar \
	    --chunk 2 \
	    --range 01 2000 2001 --out $@
	diff $(dir $@)answer.out $@

# Same regression analysis as t2_lm but split into two chunks
data/io_test/t2_lm/attemptC.out: data/io_test/example.v11.bgen ./bin/bgen_prog
	./bin/bgen_prog --lm \
	    --bgen $< \
	    --pheno $(dir $@)t2.pheno \
	    --covar $(dir $@)t2.covar \
	    --chunk 1 \
	    --range 01 2000 2001 --out $@
	diff $(dir $@)answer.out $@

# Same regression analysis as t2_lm but explicitly naming interaction column
data/io_test/t2_lm/attemptD.out: data/io_test/example.v11.bgen ./bin/bgen_prog
	./bin/bgen_prog --lm \
	    --bgen $< \
	    --pheno $(dir $@)t2.pheno \
	    --covar $(dir $@)t2.covar \
	    --interaction covar1 \
	    --range 01 2000 2001 --out $@
	diff $(dir $@)answer.out $@

data/io_test/t4_lm_2dof/attempt.out: data/io_test/example.v11.bgen ./bin/bgen_prog
	./bin/bgen_prog --lm \
	    --bgen $< \
	    --pheno $(dir $@)t4.pheno \
	    --covar $(dir $@)t4.covar \
	    --range 01 3000 3001 --out $@
	diff $(dir $@)answer.out $@

# Clean dir
cleanIO:
	rm $(IOfiles)

clean:
	rm $(TARGET)
	rm $(IOfiles)
