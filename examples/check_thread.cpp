#include <thread>
#include <iostream>
#include <vector>

void outside_foo() {
	std::cout << "hello from global function" << std::endl;
}


class Bar {
	int step;
	public:
	Bar(){
		step = 2;
	}

	~Bar(){
	}

	void foo(int& ii) {
		std::cout << "hello from member function" << std::endl;
		ii += step;
	}
	void spawn(int n_thread){
		std::thread t1[n_thread];
		std::vector< int > chunks(n_thread, 0);

		std::cout << "Chunks before calling threads: " << std::endl;
		for (int ch = 0; ch < n_thread; ch++){
			std::cout << chunks[ch] << " ";
		}
		std::cout << std::endl;

		std::cout << "Launching threads" << std::endl;
		for (int ch = 1; ch < n_thread; ch++){
			t1[ch] = std::thread( [this, &chunks, ch] { foo(chunks[ch]); } );
		}

		std::cout << "hello from main thread in spawn function" << std::endl;

		for (int ch = 1; ch < n_thread; ch++){
			t1[ch].join();
		}
		std::cout << "All threads joined." << std::endl;

		std::cout << "Chunks after calling threads: " << std::endl;
		for (int ch = 0; ch < n_thread; ch++){
			std::cout << chunks[ch] << " ";
		}
		std::cout << std::endl;
	}
};

int main() {
	Bar bar;

	bar.spawn(2);
	return 0;
}
