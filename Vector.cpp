#include "Vector.h"
#include "Matrix.h"
#include <iostream>
#include <algorithm>

const double Vector::EPS = 0.00001;

Vector::~Vector()
{
	if(data != nullptr)
	delete[] data;
}

double& Vector::operator[](int index) {

	return data[index];
}
const double& Vector::operator[](int index) const{

	return data[index];
}

Vector Vector::add(const Vector& u) const {
	//assertCorrectDimension(u.dimension());			TODO FIX!!!

	Vector result(length);

	for (int i = 0; i < length; i++)
		result[i] = data[i] + u.data[i];

	return result;
}


// Attention it's a row vector by matrix multiplication !!!
Vector Vector::mul(const Matrix& m) const {
	//assertCorrectDimension(m.rows());

	Vector result(m.cols());
	result.zeros();

	for (int col = 0; col < m.cols(); col++)
		for (int row = 0; row < m.rows(); row++)
			result[col] += m(row, col) * data[row];

	return result;
}

std::ostream& operator<<(std::ostream& os, const Vector& v)
{
	using namespace std;
	for (int i = 0; i < v.length; i++) {
		os << v.data[i] << endl;
	}
	return os;
}

 Vector Vector::sub(const Vector& u) const {
	//assertCorrectDimension(u.dimension());

	 Vector result(length);

	 for (int i = 0; i < length; i++)
		 result[i] = data[i] - u.data[i];

	return result;
}

  double Vector::dot(const  Vector& u) const{
	 //assertCorrectDimension(u.dimension());

	 double sum = 0;
	 for (int i = 0; i < length; i++)
		 sum += data[i] * u.data[i];

	 return sum;
 }

   Vector Vector::mul(double s)const {
	   Vector result(length);

	   for (int i = 0; i < length; i++)
		   result[i] = data[i]*s;

	   return result;
  }

    Vector Vector::map(const Function& fn)const 
	{
		Vector result(length);
		for (int i = 0; i < length; i++)
			result[i] = fn(data[i]);
		return result; 
   }

	 Vector Vector::elementProduct(const Vector& u)const {
		//assertCorrectDimension(u.dimension());
		 Vector result(length);
		 for (int i = 0; i < length; i++)
			 result[i] = data[i] * u.data[i];
		 return result;
	}

	Matrix Vector::outerProduct(const Vector& u) const{
		 Matrix result(u.dimension(), dimension());
		 for (int i = 0; i < length; i++)
			 for (int j = 0; j < u.dimension(); j++)
				 result(j,i) = data[i] * u.data[j];

		 return result;
	 }

	double Vector::max() const{
		const double* result = std::max_element(data, data + length);
		return *result;
	}

	double Vector::sumElements() const
	{
		double sum = 0;
		for (int i = 0; i < length; i++)
			sum += data[i];
		return sum;
	}

	Vector  Vector::sub(double a) const
	{
		Vector result(length);
		for (int i = 0; i < length; i++)
			result[i] = data[i] - a;
		return result;
	}

	int Vector::indexOfLargestElement() const{
		int ixOfLargest = 0;
		for (int i = 0; i < length; i++)
			if (data[i] > data[ixOfLargest]) ixOfLargest = i;
		return ixOfLargest;
	}

bool Vector::equals(const Vector&  o, double tol) const {
	if (length != o.length)
		return false;

	for (int i = 0; i < length; i++) 
		if (data[i] != o.data[i])
			return false;
	
		return true;
	}
