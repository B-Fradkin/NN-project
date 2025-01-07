#pragma once
#include "Vector.h"
/*
* The outcome of an evaluation.
* Will always contain the output data.
* Might contain the cost.   Fixme: Optional?
*/

class Result
{

public:
	Result();

	Result(const Vector& output) {
		m_output = output;
		m_cost = 0;
	}

	 Result(const Vector& output, double cost) {
		m_output = output;
		m_cost = cost;
	}

	 const Vector& getOutput() {
		return m_output;
	}

	 double getCost() {
		return m_cost;
	}

	
	/* std::string toString() {
		return "Result{" + "output=" + output +", cost=" + cost +'}';
		}
	*/
private:
	Vector m_output;
	double m_cost;
};

