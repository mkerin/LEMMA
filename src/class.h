// File of classes for use with src/bgen_prog.cpp
#ifndef CLASS_H
#define CLASS_H

#include <iostream>
#include <string>

class parameters {
	public :
		std::string bgen_file, chr, out_file;
		int chunk_size;
		uint32_t start, end;
		bool range, no_maf_lim;
		double min_maf;
	
	// constructors/destructors	
	parameters() {
		bgen_file = "NULL";
		chunk_size = 256;
		range = false;
		no_maf_lim = true;
	}

	~parameters() {
	}
};

#endif
