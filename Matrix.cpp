#include "Matrix.h"



std::ostream& operator<<(std::ostream& os, const Matrix& m)
{
	using namespace std;
	for (int i = 0; i < m.numRows; i++) {
		for (int j = 0; j < m.numCols; j++) {
			os << m(i, j) << " ";
		}
		os << endl;
	}
	return os;
}

 Matrix& Matrix::sub(const Matrix& other) {
	//assertCorrectDimension(other);

	for (int y = 0; y < rows(); y++)
		for (int x = 0; x < cols(); x++)
			data[y * numCols + x] -= other(y,x);

	return *this;
}

 Matrix& Matrix::mul(double s) {

	 for (int y = 0; y < rows(); y++)
		 for (int x = 0; x < cols(); x++)
			 data[y * numCols + x] *= s;

	 return *this;
 }
  Vector Matrix::multiply(const Vector& v) {

	 Vector out(numRows);
	 out.zeros();
	 for (int y = 0; y < numRows; y++)
		 for (int x = 0; x < numCols; x++)
			out[y] += data[y * numCols + x] * v[x];
	 return  out;
 }

  Matrix& Matrix::add(const Matrix &other) {
	// assertCorrectDimension(other);

	 for (int y = 0; y < numRows; y++)
		 for (int x = 0; x < numCols; x++)
			 data[y * numCols + x] += other(y, x);

	 return *this;
 }

	Matrix& Matrix::map(Function fn) {
	  for (int y = 0; y < numRows; y++)
		  for (int x = 0; x < numCols; x++)
			  data[y * numCols + x] = fn(data[y * numCols + x]);

	  return *this;
  }

	void Matrix::zeros() {
		for (int y = 0; y < numRows; y++)
			for (int x = 0; x < numCols; x++)
				data[y * numCols + x] = 0.;
	}
