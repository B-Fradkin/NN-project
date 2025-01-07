#include "Initializer.h"



Initializer::Initializer()
{
}


Initializer::~Initializer()
{
}



std::ostream& operator<<(std::ostream& os, const Constant& v)
{
	using namespace std;
	int idx, i, j, k;
	const int  size = v.m_height * v.m_length * v.m_depth;
	for (int idx = 0; idx < size; idx++) {
		v.to3D(idx, i, j, k);
		os << i <<' '<< j << ' ' << k << ' ' << v.m_initWeights[idx] << endl;
	}
	return os;
}
