
RSCRIPT := Rscript
CP      := cp
PRINTF  := printf

TARGET := bin/bgen_prog
SRCDIR := src
FLAGS = -std=c++11 -Wno-deprecated

rescomp: CXX = /apps/well/gcc/7.2.0/bin/g++
rescomp: FLAGS += -O3 -static -static-libgcc -static-libstdc++ -lrt
rescomp: $(TARGET)

rescomp-optim: CXX = /apps/well/gcc/7.2.0/bin/g++
rescomp-optim: FLAGS += -O3 -fno-rounding-math -fno-signed-zeros -fprefetch-loop-arrays -flto -lrt
rescomp-optim: $(TARGET)

rescomp-debug: CXX = /apps/well/gcc/7.2.0/bin/g++
rescomp-debug: FLAGS += -g3 -lrt
rescomp-debug: $(TARGET)

garganey: CXX = g++
garganey: FLAGS += -g3 -lrt
garganey: $(TARGET)

garganey-optim: CXX = g++
garganey-optim: FLAGS += -O3 -msse2
garganey-optim: $(TARGET)

laptop: BGEN=/Users/kerin/software/bgen/
laptop: LDFLAGS += -L/usr/local/opt/llvm/lib
laptop: CPPFLAGS += -I/usr/local/opt/llvm/include
laptop: CXX = g++
laptop: LD_LIBRARY_PATH = $(ls -d /usr/local/Cellar/gcc/* | tail -n1)/lib
laptop: LIBS += -L/usr/local/Cellar/boost@1.55/1.55.0_1 -lboost_iostreams
laptop: INCLUDES += -I/usr/local/Cellar/boost@1.55/1.55.0_1
laptop: FLAGS += -DOSX -lz -msse2
laptop: $(TARGET)
laptop: examples/test_matrix_matrix_mult


# INCLUDES += -I$(BGEN)build/genfile/include/ -I$(BGEN)3rd_party/zstd-1.1.0/lib/ \
#             -I$(BGEN)build/db/include/ -I$(BGEN)3rd_party/sqlite3 -I$(BGEN)3rd_party/boost_1_55_0
# LIBS +=     -Lbuild/ -Lbuild/3rd_party/zstd-1.1.0 -Lbuild/db -Lbuild/3rd_party/sqlite3 \
#             -Lbuild/3rd_party/boost_1_55_0 -lbgen -ldb -lsqlite3 -lboost -lz -ldl -lrt -lpthread -lzstd

LIBS += -L$(BGEN)build/ -L$(BGEN)build/3rd_party/zstd-1.1.0 -L$(BGEN)build/db \
       -L$(BGEN)build/3rd_party/sqlite3 -L$(BGEN)build/3rd_party/boost_1_55_0 \
       -lbgen -ldb -lsqlite3 -lboost -lz -ldl -lpthread -lzstd
INCLUDES += -I$(BGEN)genfile/include/ -I$(BGEN)3rd_party/zstd-1.1.0/lib/ \
           -I$(BGEN)db/include/ -I$(BGEN)3rd_party/sqlite3 -I$(BGEN)3rd_party/boost_1_55_0
FLAGS += $(LIBS) $(INCLUDES)

HEADERS := vbayes_x2.hpp genotype_matrix.hpp vbayes_tracker.hpp \
           parse_arguments.hpp data.hpp class.h bgen_parser.hpp utils.hpp \
           variational_parameters.hpp
HEADERS := $(addprefix $(SRCDIR)/,$(HEADERS))

$(TARGET) lastest_compile_branch.txt : $(SRCDIR)/bgen_prog.cpp $(HEADERS)
	$(info $$CXX is [${CXX}])
	$(info $$CFLAGS is [${CFLAGS}])
	$(info $$LDFLAGS is [${LDFLAGS}])
	echo "*********************************" >> latest_compile_branch.txt
	echo "Run on host: `hostname`" >> latest_compile_branch.txt
	echo "Git branch: `git rev-parse --abbrev-ref HEAD`" >> latest_compile_branch.txt
	echo "Started at: `date`" >> latest_compile_branch.txt
	echo "*********************************" >> latest_compile_branch.txt
	$(PRINTF) "\n\nCompilation flags:\n$(FLAGS)\n" >> latest_compile_branch.txt
	$(CXX) -o $@ $< ${FLAGS}

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

# Examples
examples/test_updates: examples/test_updates.cpp
	$(CXX) $< -o $@ $(FLAGS) -Ofast -I /homes/kerin/local/boost_1_67_0 -L /homes/kerin/local/boost_1_67_0/stage/lib

examples/test_compression_storage: examples/test_compression_storage.cpp
	$(CXX) $< -o $@ $(FLAGS) -O3 -I /homes/kerin/local/boost_1_67_0 -L /homes/kerin/local/boost_1_67_0/stage/lib

examples/test_vector_matrix_mult: examples/test_vector_matrix_mult.cpp
		$(CXX) $< -o $@ $(FLAGS) -O3 -I /homes/kerin/local/boost_1_67_0 -L /homes/kerin/local/boost_1_67_0/stage/lib

examples/check_my_timer: examples/check_my_timer.cpp
	$(CXX) $< -o $@ $(FLAGS) -I /homes/kerin/local/boost_1_67_0 -L /homes/kerin/local/boost_1_67_0/stage/lib

examples/check_nested_classes: examples/check_nested_classes.cpp
	$(CXX) -o $@ $< $(FLAGS)

examples/check_eigen: examples/check_eigen.cpp
	$(CXX) -o $@ $< $(FLAGS)

examples/test_matrix_matrix_mult: examples/test_matrix_matrix_mult.cpp
	$(CXX) $< -o $@ $(FLAGS)  -fopenmp -O3


# IO Tests
# Files in data/out are regarded as 'true', and we check that the equivalent
# file in data/io_test is identical after making changes to the executable.
# IOfiles := t8_mog_prior
IOfiles := t7_varbvs  t9_vb_covar_update t12_custom_init
IOfiles := $(addprefix data/io_test/,$(addsuffix /attempt.out,$(IOfiles)))
IOfiles += $(addprefix data/io_test/t7_varbvs/attempt, $(addsuffix .out,_multithread _alt_updates))
# IOfiles += $(addprefix data/io_test/t1_range/attempt, $(addsuffix .out,B))

testIO: $(IOfiles)
	@

# TEST 7; TODO: check hyps not rescricted
t7_dir     := data/io_test/t7_varbvs
t7_context := $(t7_dir)/hyperpriors_gxage.txt $(t7_dir)/answer.rds
data/io_test/t7_varbvs/attempt.out: data/io_test/n50_p100.bgen ./bin/bgen_prog $(t7_context)
	./bin/bgen_prog --mode_vb --verbose \
	    --keep_constant_variants \
	    --bgen $< \
	    --interaction x \
	    --covar data/io_test/age.txt \
	    --pheno data/io_test/pheno.txt \
	    --hyps_grid data/io_test/hyperpriors_gxage.txt \
	    --hyps_probs data/io_test/hyperpriors_gxage_probs.txt \
	    --vb_init data/io_test/answer_init.txt \
	    --out $@
	$(RSCRIPT) R/vbayes_x_tests/check_output.R $(dir $@) > $(dir $@)attempt.log
	diff $(dir $@)answer.log $(dir $@)attempt.log

data/io_test/t7_varbvs/attempt_multithread.out: data/io_test/n50_p100.bgen ./bin/bgen_prog $(t7_context)
	./bin/bgen_prog --mode_vb --verbose \
	    --keep_constant_variants \
	    --threads 2 \
	    --bgen $< \
	    --interaction x \
	    --covar data/io_test/age.txt \
	    --pheno data/io_test/pheno.txt \
	    --hyps_grid data/io_test/hyperpriors_gxage.txt \
	    --hyps_probs data/io_test/hyperpriors_gxage_probs.txt \
	    --vb_init data/io_test/answer_init.txt \
	    --out $@
	$(RSCRIPT) R/vbayes_x_tests/check_output.R $(dir $@) > $(dir $@)attempt_multithread.log
	diff $(dir $@)answer.log $(dir $@)attempt_multithread.log

data/io_test/t7_varbvs/attempt_alt_updates.out: data/io_test/n50_p100.bgen ./bin/bgen_prog $(t7_context)
	./bin/bgen_prog --mode_vb --verbose \
	    --keep_constant_variants \
	    --mode_alternating_updates \
	    --bgen $< \
	    --interaction x \
	    --covar data/io_test/age.txt \
	    --pheno data/io_test/pheno.txt \
	    --hyps_grid data/io_test/hyperpriors_gxage.txt \
	    --hyps_probs data/io_test/hyperpriors_gxage_probs.txt \
	    --vb_init data/io_test/answer_init.txt \
	    --out $@
	$(RSCRIPT) R/vbayes_x_tests/check_output.R $(dir $@) $(notdir $@) > $(dir $@)attempt_alt_updates.log
	diff $(dir $@)answer.log $(dir $@)attempt_alt_updates.log

$(t7_dir)/hyperpriors_gxage.txt: R/t7/gen_hyps.R
	$(RSCRIPT) $<

$(t7_dir)/answer.rds: R/vbayes_x_tests/run_VBayesR.R $(t7_dir)/hyperpriors_gxage.txt
	# $(RSCRIPT) $< $(dir $@)
	Rscript R/vbayesr_commandline.R run \
	  --pheno data/io_test/pheno.txt \
	  --covar data/io_test/age.txt \
	  --vcf $(dir $@)n50_p100.vcf.gz \
	  --out $@ \
	  --hyps_grid data/io_test/hyperpriors_gxage.txt \
	  --hyps_probs data/io_test/hyperpriors_gxage_probs.txt \
	  --vb_init data/io_test/answer_init.txt

# TEST 8;
t8_dir     := data/io_test/t8_mog_prior
t8_context := $(t8_dir)/hyperpriors_gxage.txt
data/io_test/t8_mog_prior/attempt.out: data/io_test/n50_p100.bgen ./bin/bgen_prog $(t8_context)
	./bin/bgen_prog --mode_vb --verbose \
	    --keep_constant_variants \
	    --effects_prior_mog \
	    --bgen $< \
	    --interaction x \
	    --covar data/io_test/age.txt \
	    --pheno data/io_test/pheno.txt \
	    --hyps_grid data/io_test/hyperpriors_gxage.txt \
	    --hyps_probs data/io_test/hyperpriors_gxage_probs.txt \
	    --vb_init data/io_test/answer_init.txt \
	    --out $@
	cut -f2 -d" " $(dir $@)answer.out > $(dir $@)answer_reformated.out
	cut -f2 -d" " $(dir $@)attempt.out> $(dir $@)attempt_reformated.out
	diff $(dir $@)answer_reformated.out $(dir $@)attempt_reformated.out

# TEST 9;
t9_dir     := data/io_test/t9_vb_covar_update
t9_context := data/io_test/hyperpriors_gxage.txt
data/io_test/t9_vb_covar_update/attempt.out: data/io_test/n50_p100.bgen ./bin/bgen_prog $(t9_context)
	./bin/bgen_prog --mode_vb --verbose \
	    --keep_constant_variants \
	    --use_vb_on_covars \
	    --bgen $< \
	    --interaction x \
	    --covar data/io_test/age.txt \
	    --pheno data/io_test/pheno.txt \
	    --hyps_grid data/io_test/hyperpriors_gxage.txt \
	    --hyps_probs data/io_test/hyperpriors_gxage_probs.txt \
	    --vb_init data/io_test/answer_init.txt \
	    --out $@
	cut -f2 -d" " $(dir $@)answer.out > $(dir $@)answer_reformated.out
	cut -f2 -d" " $(dir $@)attempt.out> $(dir $@)attempt_reformated.out

# diff $(dir $@)answer_reformated.out $(dir $@)attempt_reformated.out


# TEST 10
# Test when runnign from random start point.. this may be unstable
t10_dir     := data/io_test/t10_varbvs_without_init
t10_context := $(t10_dir)/hyperpriors_gxage.txt
$(t10_dir)/attempt.out: $(t10_dir)/n50_p100.bgen ./bin/bgen_prog $(t10_context)
	./bin/bgen_prog --mode_vb --verbose \
	    --keep_constant_variants \
	    --bgen $< \
	    --interaction x \
	    --covar data/io_test/age.txt \
	    --pheno data/io_test/pheno.txt \
	    --hyps_grid data/io_test/hyperpriors_gxage.txt \
	    --hyps_probs data/io_test/hyperpriors_gxage_probs.txt \
	    --out $@
	$(CP) $(t10_dir)/attempt_inits.out $(t10_dir)/answer_init.txt
	# $(RSCRIPT) R/vbayes_x_tests/run_VBayesR.R $(dir $@)
	$(RSCRIPT) R/vbayesr_commandline.R run \
	  --pheno data/io_test/pheno.txt \
	  --covar data/io_test/age.txt \
	  --vcf $(dir $@)n50_p100.vcf.gz \
	  --out $(dir $@)answer.rds \
	  --hyps_grid data/io_test/hyperpriors_gxage.txt \
	  --hyps_probs data/io_test/hyperpriors_gxage_probs.txt \
	  --vb_init data/io_test/answer_init.txt
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
	    --keep_constant_variants \
	    --bgen $< \
	    --interaction x \
	    --covar data/io_test/age.txt \
	    --pheno data/io_test/pheno.txt \
	    --hyps_grid data/io_test/hyperpriors_gxage.txt \
	    --hyps_probs data/io_test/hyperpriors_gxage_probs.txt \
	    --vb_init data/io_test/answer_init.txt \
	    --threads 2 \
	    --out $@
	$(RSCRIPT) R/vbayes_x_tests/check_output.R $(dir $@) > $(dir $@)attempt.log
	diff $(dir $@)answer.log $(dir $@)attempt.log

$(t11_dir)/hyperpriors_gxage.txt: R/t8/gen_hyps.R
	$(RSCRIPT) $<

$(t11_dir)/answer.rds: R/vbayes_x_tests/run_VBayesR.R $(t11_dir)/hyperpriors_gxage.txt
	# $(RSCRIPT) $< $(dir $@)
	$(RSCRIPT) R/vbayesr_commandline.R run \
	  --pheno data/io_test/pheno.txt \
	  --covar data/io_test/age.txt \
	  --vcf $(dir $@)n50_p100.vcf.gz \
	  --out $(dir $@)answer.rds \
	  --hyps_grid data/io_test/hyperpriors_gxage.txt \
	  --hyps_probs data/io_test/hyperpriors_gxage_probs.txt \
	  --vb_init data/io_test/answer_init.txt

# TEST 12
# custom start;
data/io_test/t12_custom_init/answer.out: data/io_test/n50_p100.bgen ./bin/bgen_prog
	./bin/bgen_prog --mode_vb --verbose --low_mem \
	    --keep_constant_variants \
	    --bgen $< \
	    --interaction x \
	    --covar data/io_test/age.txt \
	    --pheno data/io_test/pheno.txt \
	    --hyps_grid data/io_test/hyperpriors_gxage.txt \
	    --hyps_probs data/io_test/hyperpriors_gxage_probs.txt \
	    --vb_init data/io_test/t12_custom_init/vb_init.txt \
	    --out $@

data/io_test/t12_custom_init/attempt.out: data/io_test/n50_p100.bgen ./bin/bgen_prog data/io_test/t12_custom_init/answer.out
	./bin/bgen_prog --mode_vb --verbose --low_mem \
	    --keep_constant_variants \
	    --bgen $< \
	    --interaction x \
	    --covar data/io_test/age.txt \
	    --pheno data/io_test/pheno.txt \
	    --hyps_grid data/io_test/hyperpriors_gxage.txt \
	    --hyps_probs data/io_test/hyperpriors_gxage_probs.txt \
	    --vb_init data/io_test/t12_custom_init/vb_init_scrambled.txt \
	    --out $@
	cut -f2 -d" " $(dir $@)answer.out > $(dir $@)answer_reformated.out
	cut -f2 -d" " $(dir $@)attempt.out> $(dir $@)attempt_reformated.out
	diff $(dir $@)answer_reformated.out $(dir $@)attempt_reformated.out



# Clean dir
cleanIO:
	rm $(IOfiles)

clean:
	rm $(TARGET)
	rm $(IOfiles)
