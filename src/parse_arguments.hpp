// parse_arguments
#ifndef PARSE_ARGUMENTS_HPP
#define PARSE_ARGUMENTS_HPP

#include "parameters.hpp"
#include "my_timer.hpp"

#include "tools/eigen3.3/Dense" // For vectorise profiling

#include <iostream>
#include <iomanip>
#include <set>
#include <cstring>
#include <boost/filesystem.hpp>
#include <regex>
#include <stdexcept>

void check_counts(const std::string& in_str, int i, int num, int argc);
void parse_arguments(parameters &p, int argc, char *argv[]);
void check_file_exists(const std::string& filename);


#endif
