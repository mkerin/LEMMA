// parse_arguments
#ifndef PARSE_ARGUMENTS_HPP
#define PARSE_ARGUMENTS_HPP

#include <iostream>
#include <set>
#include <cstring>
#include "class.h"
#include "version.h"

void check_counts(std::string in_str, int i, int num, int argc);
void parse_arguments(parameters &p, int argc, char *argv[]);

void check_counts(std::string in_str, int i, int num, int argc) {
	if (i + num >= argc) {
		if (num == 1) {
			std::cout << "ERROR: flag " << in_str << " requres an argument. Please refer to the manual for usage instructions." << std::endl;
		} else {
			std::cout << "ERROR: flag " << in_str << " seems to require " + std::to_string(num) + " arguments. No arguments of this type should be implemented yet.." << std::endl;
		}
		exit(EXIT_FAILURE);
	}
}

void parse_arguments(parameters &p, int argc, char *argv[]) {
	char *in_str;
	int i;
	std::set<std::string> option_list {
		"--bgen",
		"--chunk",
		"--range"
	};

	std::set<std::string>::iterator set_it;
	// for --version (& no other args), print splash screen and exit.
	if (argc == 2 && strcmp(argv[1], "--version") == 0) {
		printf("%s\n\n", splash);
		std::exit(EXIT_SUCCESS);
	}

	if (argc == 1) {
		std::cout << "======-----"<< std::endl;
		std::cout << "Matt's BGEN PROG" << std::endl; 
		std::cout << "======-----" << std::endl << std::endl;
		std::exit(EXIT_SUCCESS);
	}

	// read in and check option flags
	for (i = 0; i < argc; i++) {
		in_str = argv[i];	
		if (in_str == "--version" || in_str == "--help") {
			std::cout << "ERROR: flag '" << in_str << "' cannot be used with any other flags." << std::endl;
			exit(EXIT_FAILURE);
		}
	}

	for(i = 0; i < argc; i++){
		if(*argv[i] == '-'){
			in_str = argv[i];	
			set_it = option_list.find(in_str);

			if(set_it == option_list.end()) {
				std::cout << "ERROR: flag '" << in_str <<
					"' not valid. Please refer to the manual for usage instructions." <<
					std::endl;

				exit(EXIT_FAILURE);
			}
	
			// flags with parameters should eat their arguments
			// & also make sure they don't go over argc
			if(strcmp(in_str, "--bgen") == 0) {
				check_counts(in_str, i, 1, argc);
				p.bgen_file = argv[i + 1]; // bgen file
				i += 1;
			}
			
			if(strcmp(in_str, "--range") == 0) {
				static bool check = 0;
				if (check == 1) {
					std::cout << "ERROR: flag '" << in_str << "' can only be provided once." << std::endl;
					exit(EXIT_FAILURE);
				}
				check = 1;
				check_counts(in_str, i, 3, argc);
				p.range = true;
				p.chr = argv[i + 1];
				p.start = atoi(argv[i + 2]);
				p.end = atoi(argv[i + 3]);
				i += 3;
			}
		}
	}
}

#endif
