
RSCRIPT := Rscript

TARGET := bin/bgen_prog
SRCDIR := src
INCLUDES = -Ibuild/genfile/include/ -I3rd_party/zstd-1.1.0/lib/ \
           -Ibuild/db/include/ -I3rd_party/sqlite3 -I3rd_party/boost_1_55_0
LIBS =     -Lbuild/ -Lbuild/3rd_party/zstd-1.1.0 -Lbuild/db -Lbuild/3rd_party/sqlite3 \
           -Lbuild/3rd_party/boost_1_55_0
FLAGS = -std=c++11 -Wno-deprecated $(LIBS) $(INCLUDES)

HEADERS := parse_arguments.hpp data.hpp class.h bgen_parser.hpp vbayes.hpp vbayes_x.hpp utils.hpp vbayes_x2.hpp
HEADERS := $(addprefix $(SRCDIR)/,$(HEADERS))

rescomp: CXX = /apps/well/gcc/7.2.0/bin/g++
rescomp: FLAGS += -O3 -lbgen -ldb -lsqlite3 -lboost -lz -ldl -lrt -lpthread -lzstd
rescomp: $(TARGET)

rescomp-debug: CXX = /apps/well/gcc/7.2.0/bin/g++
rescomp-debug: FLAGS += -g3 -lbgen -ldb -lsqlite3 -lboost -lz -ldl -lrt -lpthread -lzstd
rescomp-debug: $(TARGET)

garganey: CXX = g++
garganey: FLAGS += -lbgen -ldb -lsqlite3 -lboost -lz -ldl -lrt -lpthread -lzstd
garganey: $(TARGET)

# Cutting out -lrt, -lz
laptop: CXX = /usr/local/Cellar/gcc/7.3.0/bin/x86_64-apple-darwin17.4.0-g++-7
laptop: LD_LIBRARY_PATH = $(ls -d /usr/local/Cellar/gcc/* | tail -n1)/lib
laptop: FLAGS += -lbgen -ldb -lsqlite3 -lboost -ldl -lpthread -lzstd -L$(LD_LIBRARY_PATH)
laptop: $(TARGET)

$(TARGET) : $(SRCDIR)/bgen_prog.cpp $(HEADERS)
	$(info $$CXX is [${CXX}])
	$(info $$CFLAGS is [${CFLAGS}])
	$(info $$LDFLAGS is [${LDFLAGS}])
	$(CXX) -o $@ $< $(FLAGS)

file_parse : file_parse.cpp
	$(CXX) -o $@ $< $(FLAGS)

# UnitTests
.PHONY: testIO, testUNIT

# Note: this uses the Catch library to Unit Test the cpp source code. If we want
# to test input/output of the executable we do that directly with the Makefile
# and `diff` command.
# UNITTESTS := test-data.cpp tests-parse-arguments.cpp 
# UNITTESTS := $(addprefix test/,$(UNITTESTS))

testUNIT: test/tests-main

test/tests-main: test/tests-main.o
	$(CXX) $< -o $@  $(FLAGS) && ./$@

test/tests-main.o: test/tests-main.cpp $(SRCDIR)/bgen_prog.cpp $(HEADERS)
	$(CXX) test/tests-main.cpp -c -o $@ $(FLAGS)

# IO Tests
# Files in data/out are regarded as 'true', and we check that the equivalent
# file in data/io_test is identical after making changes to the executable.
IOfiles := t1_range t2_lm t3_lm_two_chunks t4_lm_2dof t5_joint_model t6_lm2 t7_varbvs_constrained
IOfiles := $(addprefix data/io_test/,$(addsuffix /attempt.out,$(IOfiles)))
IOfiles += $(addprefix data/io_test/t2_lm/attempt, $(addsuffix .out,B C D))
IOfiles += $(addprefix data/io_test/t1_range/attempt, $(addsuffix .out,B))

testIO: $(IOfiles)
	@

# Test of --range command
data/io_test/t1_range/attempt.out: data/io_test/example.v11.bgen ./bin/bgen_prog
	./bin/bgen_prog --convert_to_vcf --bgen $< --range 01 1 3000 --out $@
	diff $(dir $@)answer.out $@

# Test of --incl_rsids on same test case
data/io_test/t1_range/attemptB.out: data/io_test/example.v11.bgen ./bin/bgen_prog
	./bin/bgen_prog --convert_to_vcf --bgen $< --incl_rsids $(dir $@)t1_variants.txt --out $@
	diff $(dir $@)answer.out $@

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

# # Joint model test
# data/io_test/t5_joint_model/attempt.out: data/io_test/example.v11.bgen ./bin/bgen_prog
# 	./bin/bgen_prog --joint_model \
# 	    --bgen $< \
# 	    --pheno $(dir $@)t2.pheno \
# 	    --covar $(dir $@)t2.covar \
# 	    --range 01 2000 2001 --out $@
# 	diff $(dir $@)answer.out $@

# linear regression with full gene-env-covariates model
data/io_test/t6_lm2/attempt.out: data/io_test/example.v11.bgen ./bin/bgen_prog
	./bin/bgen_prog --full_lm \
	    --bgen $< \
	    --pheno $(dir $@)t6.pheno \
	    --covar $(dir $@)t6.covar \
	    --interaction covar1 \
	    --genetic_confounders covar2 \
	    --range 01 2000 2001 \
	    --out $@
	diff $(dir $@)answer.out $@

# When sigma_b == sigma_g and lambda_b == lambda_g then we can compare with varbvs
t7_dir     := data/io_test/t7_varbvs_constrained
t7_context := $(t7_dir)/hyperpriors_gxage.txt $(t7_dir)/answer.rds

$(t7_dir)/hyperpriors_gxage.txt: R/test7/gen_hyps.R
	$(RSCRIPT) $<

$(t7_dir)/answer.rds: R/test7/run_VBayesR.R $(t7_dir)/hyperpriors_gxage.txt
	$(RSCRIPT) $<

data/io_test/t7_varbvs_constrained/attempt.out: $(t7_dir)/n50_p100.bgen ./bin/bgen_prog $(t7_context)
	./bin/bgen_prog --mode_vb --verbose \
	    --bgen $< \
	    --interaction x \
	    --covar $(dir $@)age.txt \
	    --pheno $(dir $@)pheno.txt \
	    --hyps_grid $(dir $@)hyperpriors_gxage.txt \
	    --hyps_probs $(dir $@)hyperpriors_gxage_probs.txt \
	    --vb_init $(dir $@)answer_init.txt \
	    --out $@
	# $(RSCRIPT) R/plot_vbayes_pip.R $@ $(dir $@)plots/t7_pip_$(notdir $(basename $@)).pdf
	$(RSCRIPT) R/test7/check_output.R > $(dir $@)attempt.log
	diff $(dir $@)answer.log $(dir $@)attempt.log



# comparison with the varbvs R package
data/io_test/t8_varbvs/attempt.out: data/io_test/t8_varbvs/n500_p1000.bgen ./bin/bgen_prog
	./bin/bgen_prog --mode_vb --verbose \
	    --bgen $< \
	    --interaction x \
	    --covar $(dir $@)age.txt \
	    --pheno $(dir $@)pheno.txt \
	    --hyps_grid $(dir $@)hyperpriors_gxage_v1.txt \
	    --hyps_probs $(dir $@)hyperpriors_gxage_v1_probs.txt \
	    --vb_init $(dir $@)cpp_inference_inits.out \
	    --out $@
	# $(RSCRIPT) R/plot_vbayes_pip.R $@ $(dir $@)plots/t8_pip_$(notdir $(basename $@)).pdf
	diff $(dir $@)answer.out $@

# Clean dir
cleanIO:
	rm $(IOfiles)

clean:
	rm $(TARGET)
	rm $(IOfiles)
