#pragma once
#include "Matrix.h"
#include "Vector.h"

class Optimizer
{
public:
	Optimizer();
	~Optimizer();

	virtual void updateWeights(Matrix& weights, Matrix dCdW)  = 0;
	virtual  Vector updateBias(Vector& bias, Vector dCdB) = 0;
	virtual  Optimizer* copy() = 0;

};


/**
 * Updates Weights and biases based on a constant learning rate - i.e. W -= η * dC/dW
 */
 class GradientDescent:public Optimizer {

 public:

	GradientDescent(double learningRate): m_learningRate(learningRate){}


	void updateWeights(Matrix& weights, Matrix dCdW) override {
		weights.sub(dCdW.mul(m_learningRate));
	}

	
	Vector updateBias(Vector& bias, Vector dCdB) override {
		return bias.sub(dCdB.mul(m_learningRate));
	}

	
	Optimizer* copy() override {
		// no need to make copies since this optimizer has
		// no state. Same instance can serve all layers.
		return this;
	}
 private:
	 double m_learningRate;
 };
