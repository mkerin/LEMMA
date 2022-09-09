// parse_arguments
#ifndef PARSE_ARGUMENTS_HPP
#define PARSE_ARGUMENTS_HPP

#include "parameters.hpp"

#include <string>

void parse_arguments(parameters &p, int argc, char *argv[]);
void check_file_exists(const std::string& filename);

#endif
