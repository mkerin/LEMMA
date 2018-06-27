#include <iostream>
#include <string>
#include <vector>

class Class1{
public:
	std::vector<int> vv;

	Class1(int n){
		for(int ii = 0; ii < n; ii++){
			vv.push_back(ii);
		}
	}
};

class Class2 {
public:
	std::vector<int>& vv;

	Class2(Class1& st): vv(st.vv) {
		// vv = st.vv;
	}
	~Class2(){}
};

int main() {
	Class1 obj1(5);
	Class2 obj2(obj1);

	std::cout << "obj1.vv:" << std::endl;
	for(int ii = 0; ii < 5; ii++){
		std::cout << obj1.vv[ii] << std::endl;
	}

	// Changing obj2
	obj2.vv[3] = 100;
	std::cout << "obj2.vv:" << std::endl;
	for(int ii = 0; ii < 5; ii++){
		std::cout << obj2.vv[ii] << std::endl;
	}
	std::cout << "obj1.vv:" << std::endl;
	for(int ii = 0; ii < 5; ii++){
		std::cout << obj1.vv[ii] << std::endl;
	}

	return 0;
}
