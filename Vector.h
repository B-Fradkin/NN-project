#pragma once
#include <iostream>
#include "Function.h"


class Matrix;


class Vector
{
	static const double EPS;
public:

	Vector() : data(nullptr), length(0){}

	Vector(int size) : length(size) {
		data = new double[length];
	}
	Vector(int size, double initialValue) {
		length = size;
		data = new double[length];
		for (int i = 0; i < length; i++)
			data[i] = initialValue;

	}
	Vector(const std::initializer_list<double> & v)
	{		
		length = v.size();
		data = new double[length];
		int index = 0;
		for (auto & value : v)
			data[index++] = value;		
	}

	template<typename T>
	Vector(const T& v, int size)
	{
		length = size;
		data = new double[length];

		for (int i = 0; i < length; i++)
			data[i] = v[i];
	}

	Vector(const Vector& copy) : length(copy.length) {
		data = new double[length];
		for (int i = 0; i < length; i++)
			data[i] = copy.data[i];
	}

	void zeros(){
		for (int i = 0; i < length; i++)
			data[i] = 0.;
	}

	Vector& operator=(const  Vector& other) {
		if (data != nullptr)
			delete[] data;
		length = other.length;
		data = new double[length];
		for (int i = 0; i < length; i++)
			data[i] = other.data[i];
		return *this;
	}
	 int dimension() const{
		return length;
	}

	~Vector();
	const double* getData() { return data; }
	const double* getData() const{ return data; }

	double& operator[](int index);
	const double& operator[](int index) const;
	Vector add(const Vector& u) const;
	Vector sub(const Vector& u) const;
	Vector sub(double a) const;
	Vector mul(const Matrix& m) const;
	double dot(const Vector& u) const;
	Vector mul(double s)const;
	Vector map(const Function& fn)const;
	Vector elementProduct(const Vector& u)const;
	Matrix outerProduct(const Vector& u) const;
	friend std::ostream& operator<<(std::ostream& os, const Vector& v);
	double max()const;
	double sumElements()const;
	int indexOfLargestElement() const;
	bool equals(const Vector& other, double tol = EPS) const;

private: 
	double *data;
	int length;

};


std::ostream& operator<<(std::ostream& os, const Vector& v);

