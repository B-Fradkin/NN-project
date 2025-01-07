#pragma once
#include <iostream>
#include "Vector.h"

class Matrix
{
public:
	Matrix()  = default;
	Matrix(int rows, int cols) : numRows(rows), numCols(cols)
	{
		data = new double[numRows*numCols];
	}

	Matrix(const Matrix& other) : numRows(other.rows()), numCols(other.cols())
	{
		data = new double[numRows*numCols];
		for (int i = 0; i < numRows*numCols; i++)
			data[i] = other.data[i];
	}
	Matrix& operator=(const Matrix  &other)  {
		if (data != nullptr)
			delete[] data;
		numCols = other.numCols;
		numRows = other.numRows;
		data = new double[numCols*numRows];
		for (int i = 0; i < numRows*numCols; i++)
			data[i] = other.data[i];
		return *this;
	}

	double &operator()(int row, int col) {
		return data[row * numCols + col];
	}
	const double &operator() (int row, int col)const  {
		return data[row * numCols + col];
	}

	Matrix& sub(const Matrix& other);
	Matrix& mul(double s);
	Matrix& add(const Matrix &other);
	Vector multiply(const Vector& v);
	Matrix& map(Function fn);
	int rows()const {return numRows;}
	int cols()const {return numCols;}
	void zeros();

	~Matrix() {
		if (data != nullptr)
			delete[] data;
	};


	friend std::ostream& operator<<(std::ostream& os, const Matrix& m);
	
private:
	int numRows = 0;
	int numCols = 0;
	double *data = nullptr;
};

std::ostream& operator<<(std::ostream& os, const Matrix& m);
