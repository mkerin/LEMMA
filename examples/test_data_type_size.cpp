#include <iostream>
#include <boost/filesystem.hpp>
#include <string>
using namespace std;

int main() {
   cout << "Size of char : " << sizeof(char) << endl;
   cout << "Size of int : " << sizeof(int) << endl;
   cout << "Size of short int : " << sizeof(short int) << endl;
   cout << "Size of long int : " << sizeof(long int) << endl;
   cout << "Size of float : " << sizeof(float) << endl;
   cout << "Size of double : " << sizeof(double) << endl;
   cout << "Size of wchar_t : " << sizeof(wchar_t) << endl;

	// Also file path manipulation

	std::string filepath = "/well/marchini/kebl4230/software/bgen_prog/tmp.out.gz";
	std::string dir = filepath.substr(0, filepath.rfind("/")+1);
	std::string stem_w_dir = filepath.substr(0, filepath.find("."));
	std::string stem = stem_w_dir.substr(stem_w_dir.rfind("/")+1, stem_w_dir.size());

	std::string ext = filepath.substr(filepath.find("."), filepath.size());
	std::cout << filepath << std::endl;
	std::cout << dir << std::endl;
	std::cout << stem_w_dir << std::endl;
	std::cout << stem << std::endl;
	std::cout << ext << std::endl;

	std::cout << "Removing gzip" << std::endl;
	std::string ext1 = ".out", ext2 = ".out.gz";
	std::cout << ext1.substr(0, ext1.find(".gz")) << std::endl;
	std::cout << ext2.substr(0, ext2.find(".gz")) << std::endl;

	std::string file_prefix = "round1_", file_suffix = "_hyps";
	std::string ofile = dir + file_prefix + stem + file_suffix + ext;
	std::cout << ofile << std::endl;


	// 
	int ii = 3;
	bool main_loop;
	std::string ss = "interim_files/grid_point_" + std::to_string(ii);
	if(!main_loop){
		ss = "r1_" + ss;
	}
		boost::filesystem::path interim_ext(ss), p(filepath), dir_point;

		std::cout << p.parent_path() << std::endl;
		std::cout << p.parent_path() / interim_ext << std::endl;
		dir_point = p.parent_path() / interim_ext;
		boost::filesystem::create_directories(dir_point);

	

	return 0;
}
