#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <iterator>
#include <sys/types.h>
#include <dirent.h>
#include <regex>

// http://www.cplusplus.com/reference/regex/ECMAScript/

typedef std::vector<std::string> stringvec;
 
void read_directory(const std::string& name, stringvec& v)
{
    DIR* dirp = opendir(name.c_str());
    struct dirent * dp;
    while ((dp = readdir(dirp)) != NULL) {
        v.push_back(dp->d_name);
    }
    closedir(dirp);
}

 
int main()
{

        std::string bgen = "data/io_test/t5_bgen_wildcard/chr*.bgen";
        std::string dir, filename;
        std::regex_search(str, m, re1);
        std::cout << "Dir: " << m[0] << '\n';
        
        std::cout << "filename: " << m[0] << '\n';


    std::string wildcard = "chr*.bgen";
    std::regex dot_regex("\\.");
    std::string new_wildcard = std::regex_replace(wildcard, dot_regex, "\\.");
    std::cout << new_wildcard << std::endl;
    std::regex asterisk_regex("\\*");
    std::string new_wildcard2 = std::regex_replace(new_wildcard, asterisk_regex, "[[:digit:]]*");
    std::cout << new_wildcard2.c_str() << std::endl;
    stringvec v;
    read_directory("data/io_test/t5_bgen_wildcard", v);
    std::regex self_regex(new_wildcard2.c_str());
    for (int ii = 0; ii < v.size(); ii++){
        if (std::regex_search(v[ii], self_regex)) {
            std::cout << v[ii] << " contains the phrase " << new_wildcard2 << std::endl;
        }
    }
    std::copy(v.begin(), v.end(),
         std::ostream_iterator<std::string>(std::cout, "\n"));
}
