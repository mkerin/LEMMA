#ifndef MY_TIMER_HPP
#define MY_TIMER_HPP

#include <iostream>
#include <string>
#include <boost/timer/timer.hpp>

class MyTimer {
public:
	boost::timer::auto_cpu_timer c_timer; // cumulative time
	boost::timer::auto_cpu_timer l_timer; // lap time

	MyTimer(const std::string& format) : l_timer(""), c_timer(format) {
		c_timer.stop();
		l_timer.stop();
	}

	MyTimer(){
		c_timer.stop();
		l_timer.stop();
	}

	~MyTimer(){
		c_timer.report();
	}

	void resume(){
		c_timer.resume();
		l_timer = boost::timer::auto_cpu_timer("");
	}

	void stop(){
		c_timer.stop();
		l_timer.stop();
	}

	double get_lap_seconds(){
		boost::timer::cpu_times cpu_time(l_timer.elapsed());
		double secs = (cpu_time.user + cpu_time.system) / 1000.0/1000.0/1000.0;
		return secs;
	}

	void report(){
		c_timer.report();
	}
};

#endif
