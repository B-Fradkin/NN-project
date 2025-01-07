#include "Activation.h"

//static Activation Identity ;

Activation::Activation()
{
	Init();
}


Activation::~Activation()
{
}

double sigm(double x) { return 1.0 / (1.0 + exp(-x)); };

Activation Activation::Sigmoid = Activation("Sigmoid",
										[](double x) { return 1.0 / (1.0 + exp(-x)); },
										[](double x) { return x * (1.0 - x); });  //FIX ME
//										[](double x) { return sigm(x) * (1.0 - sigm(x)); });  //FIX ME


Activation Activation::Identity = Activation("Identity",
										[](double x) { return x; },
										[](double x) { return 1; });
Activation Activation::ReLU =  Activation("ReLU",
	[](double x) { return x <= 0 ? 0 : x; },               // fn
	[](double x) { return x <= 0 ? 0 : 1;  });



// --------------------------------------------------------------------------
// Softmax needs a little extra love since element output depends on more
// than one component of the vector. Simple element mapping will not suffice.
// --------------------------------------------------------------------------
Activation Activation::Softmax = Activation("Softmax", &Activation::Softmax_fn_Impl, &Activation::Softmax_dCdI_Impl);

Vector Activation::Softmax_fn_Impl(const Vector& in) const{

	double sum = 0;
	double max = in.max();    // Trick: translate the input by largest element to avoid overflow.
	auto translate = [max](double a) { return exp(a - max); };

	sum = in.map(translate).sumElements();

	double finalSum = sum;
	auto softmax = [max, finalSum](double a) { return exp(a - max)/ finalSum; };
	return in.map(softmax);
}

	
Vector Activation::Softmax_dCdI_Impl(const Vector& out, const Vector& dCdO) const{
	double x = out.elementProduct(dCdO).sumElements();
	Vector sub = dCdO.sub(x);
	return out.elementProduct(sub);
}
