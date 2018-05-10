
RSCRIPT := Rscript
CP      := cp
PRINTF  := printf

TARGET := bin/bgen_prog
SRCDIR := src
INCLUDES = -Ibuild/genfile/include/ -I3rd_party/zstd-1.1.0/lib/ \
           -Ibuild/db/include/ -I3rd_party/sqlite3 -I3rd_party/boost_1_55_0
LIBS =     -Lbuild/ -Lbuild/3rd_party/zstd-1.1.0 -Lbuild/db -Lbuild/3rd_party/sqlite3 \
           -Lbuild/3rd_party/boost_1_55_0 -lbgen -ldb -lsqlite3 -lboost -lz -ldl -lrt -lpthread -lzstd
FLAGS := -std=c++11 -Wno-deprecated $(LIBS) $(INCLUDES)

HEADERS := parse_arguments.hpp data.hpp class.h bgen_parser.hpp vbayes.hpp vbayes_x.hpp utils.hpp vbayes_x2.hpp genotype_matrix.hpp
HEADERS := $(addprefix $(SRCDIR)/,$(HEADERS))

rescomp: CXX = /apps/well/gcc/7.2.0/bin/g++
rescomp: FLAGS += -O3
rescomp: $(TARGET)

rescomp-optim: CXX = /apps/well/gcc/7.2.0/bin/g++
rescomp-optim: FLAGS += -O3 -fno-rounding-math -fno-signed-zeros -fprefetch-loop-arrays -flto
rescomp-optim: $(TARGET)

rescomp-debug: CXX = /apps/well/gcc/7.2.0/bin/g++
rescomp-debug: FLAGS += -g3
rescomp-debug: $(TARGET)

garganey: CXX = g++
garganey: FLAGS += -g3
garganey: $(TARGET)

garganey-optim: CXX = g++
garganey-optim: FLAGS += -O3
garganey-optim: $(TARGET)

# Cutting out -lrt, -lz
laptop: CXX = /usr/local/Cellar/gcc/7.3.0/bin/x86_64-apple-darwin17.4.0-g++-7
laptop: LD_LIBRARY_PATH = $(ls -d /usr/local/Cellar/gcc/* | tail -n1)/lib
laptop: FLAGS += -g3 -lbgen -ldb -lsqlite3 -lboost -ldl -lpthread -lzstd -L$(LD_LIBRARY_PATH)
laptop: $(TARGET)

$(TARGET) lastest_compile_branch.txt : $(SRCDIR)/bgen_prog.cpp $(HEADERS)
	$(info $$CXX is [${CXX}])
	$(info $$CFLAGS is [${CFLAGS}])
	$(info $$LDFLAGS is [${LDFLAGS}])
	git rev-parse --abbrev-ref HEAD > latest_compile_branch.txt
	$(PRINTF) "\n\nCompilation flags:\n$(FLAGS)" >> latest_compile_branch.txt
	$(CXX) -o $@ $< $(FLAGS)

file_parse : file_parse.cpp
	$(CXX) -o $@ $< $(FLAGS)

# UnitTests
.PHONY: testIO, testUNIT

# Note: this uses the Catch library to Unit Test the cpp source code. If we want
# to test input/output of the executable we do that directly with the Makefile
# and `diff` command.
# UNITTESTS := test-data.cpp tests-parse-arguments.cpp 
# UNITTESTS := $(addprefix tests/,$(UNITTESTS))

testUNIT: tests/tests-main

tests/tests-main: tests/tests-main.o
	$(CXX) $< -o $@  $(FLAGS) && ./$@

tests/tests-main.o: tests/tests-main.cpp $(SRCDIR)/bgen_prog.cpp $(HEADERS)
	$(CXX) tests/tests-main.cpp -c -o $@ $(FLAGS)

# IO Tests
# Files in data/out are regarded as 'true', and we check that the equivalent
# file in data/io_test is identical after making changes to the executable.
IOfiles := t1_range t2_lm t3_lm_two_chunks t4_lm_2dof t5_joint_model t6_lm2 \
           t7_varbvs_constrained t8_varbvs t10_varbvs_without_init
IOfiles := $(addprefix data/io_test/,$(addsuffix /attempt.out,$(IOfiles)))
IOfiles += $(addprefix data/io_test/t2_lm/attempt, $(addsuffix .out,B C D))
IOfiles += $(addprefix data/io_test/t1_range/attempt, $(addsuffix .out,B))

testIO: $(IOfiles)
	@

# # Test of --range command
# data/io_test/t1_range/attempt.out: data/io_test/example.v11.bgen ./bin/bgen_prog
# 	./bin/bgen_prog --convert_to_vcf --bgen $< --range 01 1 3000 --out $@
# 	diff $(dir $@)answer.out $@
# 
# # Test of --incl_rsids on same test case
# data/io_test/t1_range/attemptB.out: data/io_test/example.v11.bgen ./bin/bgen_prog
# 	./bin/bgen_prog --convert_to_vcf --bgen $< --incl_rsids $(dir $@)t1_variants.txt --out $@
# 	diff $(dir $@)answer.out $@

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

# TEST 7
# Restricted model with sigma_b == sigma_g and lambda_b == lambda_g
# This is equivalent to the original carbonetto model on H = (X, Z)
t7_dir     := data/io_test/t7_varbvs_constrained
t7_context := $(t7_dir)/hyperpriors_gxage.txt $(t7_dir)/answer.rds
data/io_test/t7_varbvs_constrained/attempt.out: data/io_test/n50_p100.bgen ./bin/bgen_prog $(t7_context)
	./bin/bgen_prog --mode_vb --verbose \
	    --bgen $< \
	    --interaction x \
	    --covar $(dir $@)age.txt \
	    --pheno $(dir $@)pheno.txt \
	    --hyps_grid $(dir $@)hyperpriors_gxage.txt \
	    --hyps_probs $(dir $@)hyperpriors_gxage_probs.txt \
	    --vb_init $(dir $@)answer_init.txt \
	    --out $@
	$(RSCRIPT) R/vbayes_x_tests/check_output.R $(dir $@) > $(dir $@)attempt.log
	diff $(dir $@)answer.log $(dir $@)attempt.log

$(t7_dir)/hyperpriors_gxage.txt: R/t7/gen_hyps.R
	$(RSCRIPT) $<

$(t7_dir)/answer.rds: R/vbayes_x_tests/run_VBayesR.R $(t7_dir)/hyperpriors_gxage.txt
	# $(RSCRIPT) $< $(dir $@)
	Rscript R/vbayesr_commandline.R run \
	  --pheno $(dir $@)pheno.txt \
	  --covar $(dir $@)age.txt \
	  --vcf $(dir $@)n50_p100.vcf.gz \
	  --out $@ \
	  --hyps_grid $(dir $@)hyperpriors_gxage.txt \
	  --hyps_probs $(dir $@)hyperpriors_gxage_probs.txt \
	  --vb_init $(dir $@)answer_init.txt

# TEST 8
# Unrestricted model; comparison with R implementation
t8_dir     := data/io_test/t8_varbvs
t8_context := $(t8_dir)/hyperpriors_gxage.txt $(t8_dir)/answer.rds
data/io_test/t8_varbvs/attempt.out: data/io_test/n50_p100.bgen ./bin/bgen_prog $(t8_context)
	./bin/bgen_prog --mode_vb --verbose \
	    --bgen $< \
	    --interaction x \
	    --covar $(dir $@)age.txt \
	    --pheno $(dir $@)pheno.txt \
	    --hyps_grid $(dir $@)hyperpriors_gxage.txt \
	    --hyps_probs $(dir $@)hyperpriors_gxage_probs.txt \
	    --vb_init $(dir $@)answer_init.txt \
	    --out $@
	$(RSCRIPT) R/vbayes_x_tests/check_output.R $(dir $@) > $(dir $@)attempt.log
	diff $(dir $@)answer.log $(dir $@)attempt.log

$(t8_dir)/hyperpriors_gxage.txt: R/t8/gen_hyps.R
	$(RSCRIPT) $<

$(t8_dir)/answer.rds: R/vbayes_x_tests/run_VBayesR.R $(t8_dir)/hyperpriors_gxage.txt
	# $(RSCRIPT) $< $(dir $@)
	Rscript R/vbayesr_commandline.R run \
	  --pheno $(dir $@)pheno.txt \
	  --covar $(dir $@)age.txt \
	  --vcf $(dir $@)n50_p100.vcf.gz \
	  --out $(dir $@)answer.rds \
	  --hyps_grid $(dir $@)hyperpriors_gxage.txt \
	  --hyps_probs $(dir $@)hyperpriors_gxage_probs.txt \
	  --vb_init $(dir $@)answer_init.txt

# NEED TO MAKE THIS RUN WITH VBAYESR IS WANT TO KEEP
# TEST 9
# Tests when h_g is zero (ie collapse down to original carbonetto model on X)
# t9_dir     := data/io_test/t9_varbvs_zero_hg
# t9_context := $(t9_dir)/hyperpriors_gxage.txt $(t9_dir)/answer.rds
# data/io_test/t9_varbvs_zero_hg/attempt.out: data/io_test/n50_p100.bgen ./bin/bgen_prog $(t9_context)
# 	./bin/bgen_prog --mode_vb --verbose \
# 	    --bgen $< \
# 	    --interaction x \
# 	    --covar $(dir $@)age.txt \
# 	    --pheno $(dir $@)pheno.txt \
# 	    --hyps_grid $(dir $@)hyperpriors_gxage.txt \
# 	    --hyps_probs $(dir $@)hyperpriors_gxage_probs.txt \
# 	    --vb_init $(dir $@)answer_init.txt \
# 	    --out $@
# 	$(RSCRIPT) R/vbayes_x_tests/check_output.R $(dir $@) > $(dir $@)attempt.log
# 	diff $(dir $@)answer.log $(dir $@)attempt.log
# 
# $(t9_dir)/answer.rds: R/t9/t9_run_vbayesr.R $(t9_dir)/hyperpriors_gxage.txt
# 	$(RSCRIPT) $< $(dir $@)

# TEST 10
# Test when runnign from random start point.. this may be unstable
t10_dir     := data/io_test/t10_varbvs_without_init
t10_context := $(t10_dir)/hyperpriors_gxage.txt
$(t10_dir)/attempt.out: $(t10_dir)/n50_p100.bgen ./bin/bgen_prog $(t10_context)
	./bin/bgen_prog --mode_vb --verbose \
	    --bgen $< \
	    --interaction x \
	    --covar $(dir $@)age.txt \
	    --pheno $(dir $@)pheno.txt \
	    --hyps_grid $(dir $@)hyperpriors_gxage.txt \
	    --hyps_probs $(dir $@)hyperpriors_gxage_probs.txt \
	    --out $@
	$(CP) $(t10_dir)/attempt_inits.out $(t10_dir)/answer_init.txt
	# $(RSCRIPT) R/vbayes_x_tests/run_VBayesR.R $(dir $@)
	$(RSCRIPT) R/vbayesr_commandline.R run \
	  --pheno $(dir $@)pheno.txt \
	  --covar $(dir $@)age.txt \
	  --vcf $(dir $@)n50_p100.vcf.gz \
	  --out $(dir $@)answer.rds \
	  --hyps_grid $(dir $@)hyperpriors_gxage.txt \
	  --hyps_probs $(dir $@)hyperpriors_gxage_probs.txt \
	  --vb_init $(dir $@)answer_init.txt
	$(RSCRIPT) R/t10/check_output.R $(dir $@) > $(dir $@)attempt.log
	diff $(dir $@)answer.log $(dir $@)attempt.log

$(t10_dir)/hyperpriors_gxage.txt: R/t10/gen_hyps.R
	$(RSCRIPT) $<

# TEST 11
# Multi-thread; yeahhh boi!
t11_dir     := data/io_test/t11_varbvs_multithread
t11_context := $(t11_dir)/hyperpriors_gxage.txt $(t11_dir)/answer.rds
$(t11_dir)/attempt.out: $(t11_dir)/n50_p100.bgen ./bin/bgen_prog $(t11_context)
	./bin/bgen_prog --mode_vb --verbose \
	    --bgen $< \
	    --interaction x \
	    --covar $(dir $@)age.txt \
	    --pheno $(dir $@)pheno.txt \
	    --hyps_grid $(dir $@)hyperpriors_gxage.txt \
	    --hyps_probs $(dir $@)hyperpriors_gxage_probs.txt \
	    --vb_init $(dir $@)answer_init.txt \
	    --threads 2 \
	    --out $@
	$(RSCRIPT) R/vbayes_x_tests/check_output.R $(dir $@) > $(dir $@)attempt.log
	diff $(dir $@)answer.log $(dir $@)attempt.log

$(t11_dir)/hyperpriors_gxage.txt: R/t8/gen_hyps.R
	$(RSCRIPT) $<

$(t11_dir)/answer.rds: R/vbayes_x_tests/run_VBayesR.R $(t11_dir)/hyperpriors_gxage.txt
	# $(RSCRIPT) $< $(dir $@)
	$(RSCRIPT) R/vbayesr_commandline.R run \
	  --pheno $(dir $@)pheno.txt \
	  --covar $(dir $@)age.txt \
	  --vcf $(dir $@)n50_p100.vcf.gz \
	  --out $(dir $@)answer.rds \
	  --hyps_grid $(dir $@)hyperpriors_gxage.txt \
	  --hyps_probs $(dir $@)hyperpriors_gxage_probs.txt \
	  --vb_init $(dir $@)answer_init.txt


# Clean dir
cleanIO:
	rm $(IOfiles)

clean:
	rm $(TARGET)
	rm $(IOfiles)
