#pragma once
#include "Vector.h"
#include "Function.h"

class Activation
{

	using pFn_t = Vector(Activation::*)(const Vector& in)const;
	using pdCdI_t = Vector(Activation::*)(const Vector& out, const Vector& dCdO)const;

public:
	Activation();
	~Activation();

	 Activation(const std::string &name) {
		 m_name = name; 
		 Init();
	}

	 Activation(const std::string &name, Function fn, Function dFn) {
		m_name = name;
		m_fn = fn;
		m_dFn = dFn;
		Init();
	}

	 Activation(const std::string &name,  pFn_t pFn, pdCdI_t pdCdI) {
		 m_name = name;
		 m_fn = nullptr;
		 m_dFn = nullptr;
		 m_pFn = pFn;
		 m_pdCdI = pdCdI;
	 }

	 void Init()
	 {
		 m_pFn = &Activation::fn_Impl;
		 m_pdCdI = &Activation::dCdI_Impl;
	 }

	// For most activation function it suffice to map each separate element. 
	// I.e. they depend only on the single component in the vector.
	 Vector fn(const Vector& in) const{
		 return (this->*m_pFn)(in);
	}

	 Vector dFn(const Vector& out) const{
		return out.map(m_dFn);
	}

	// Also when calculating the Error change rate in terms of the input (dCdI)
	// it is just a matter of multiplying, i.e. ∂C/∂I = ∂C/∂O * ∂O/∂I.
	 Vector dCdI(const Vector& out, const Vector& dCdO) const{
		 return (this->*m_pdCdI)(out, dCdO);
	}

	 std::string getName() const{
		return m_name;
	}


private:

	Vector Softmax_fn_Impl(const Vector& in) const;
	Vector Softmax_dCdI_Impl(const Vector& out, const Vector& dCdO) const;

	Vector fn_Impl(const Vector& in) const {
		return in.map(m_fn);
	}

	Vector dCdI_Impl(const Vector& out, const Vector& dCdO) const {
		return dCdO.elementProduct(dFn(out));
	}



	pFn_t m_pFn;
	pdCdI_t m_pdCdI;

	Function m_fn;
	Function m_dFn;
	std::string m_name;

public:
	static Activation Identity;
	static Activation Sigmoid;
	static Activation ReLU;
	static Activation Softmax;
};

