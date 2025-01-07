#pragma once
#include <string>
#include "Vector.h"

class CostFunction
{
public:
	CostFunction();
	~CostFunction();

	virtual std::string getName() =  0;
	virtual double getTotal(const Vector& expected, const Vector& actual) = 0;
	virtual Vector getDerivative(const Vector& expected, const Vector& actual) = 0;
};


/**
 * Cost function: Quadratic, C = ∑(y−exp)^2
 */
class Quadratic :public CostFunction {

public:

	std::string getName()  override {
		return "Quadratic";
	}

	double getTotal(const Vector& expected, const Vector& actual) override {
		Vector diff = actual.sub(expected);
		return diff.dot(diff);
	}

	Vector getDerivative(const Vector& expected, const Vector& actual) override {
		return actual.sub(expected).mul(2);
	}

};


/**
 * Cost function: HalfQuadratic, C = 0.5 ∑(y−exp)^2
 */
class HalfQuadratic:public CostFunction {
		
public:
	std::string getName() override{
		return "HalfQuadratic";
	}

	double getTotal(const Vector &expected, const Vector &actual) override{
		Vector diff = expected.sub(actual);
		return diff.dot(diff) * 0.5;
	}

	Vector getDerivative(const Vector &expected, const Vector &actual) override {
		return actual.sub(expected);
	}
};




