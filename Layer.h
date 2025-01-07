#pragma once
#include "Matrix.h"
#include "Vector.h"
#include "Activation.h"
#include "Optimizer.h"
/**
 * A single layer in the network.
 * Contains the weights and biases coming into this layer->
 */
class Layer
{
public:
	Layer();
	~Layer();
	 Layer(int size, Activation activation) : Layer(size, activation, nullptr){}

	 Layer(int size, Activation activation, double initialBias) {
		this->size = size;
		bias = new Vector(size,initialBias);
		deltaBias =  Vector(size);
		deltaBias.zeros();
		this->activation = activation;
	}

	 Layer(int size, Activation activation, Vector* bias) {
		this->size = size;
		this->bias = bias;
		deltaBias =  Vector(size);
		deltaBias.zeros();
		this->activation = activation;
	}

	int getSize() {
		return size;
	}
	const Vector& evaluate(const Vector& i);

	 void setWeights(const Matrix& weights) {
		this->weights = weights;
		deltaWeights = Matrix(weights.rows(), weights.cols());
		deltaWeights.zeros();
	}

	 void setOptimizer(Optimizer* optimizer) {
		this->optimizer = optimizer;
	}

	 void setL2(double l2) {
		this->l2 = l2;
	}

	 Matrix getWeights() {
		return weights;
	}

	  Layer* getPrecedingLayer() const{
		return precedingLayer;
	}

	 void setPrecedingLayer(Layer* precedingLayer) {
		this->precedingLayer = precedingLayer;
	}

	 bool hasPrecedingLayer() {
		return precedingLayer != nullptr;
	}

	 Vector* getBias() {
		return bias;
	}


	const Vector& getOut() const{
		return out;
	}

	 const Activation& getActivation() const{
		 return activation;
	}

	 void addDeltaWeightsAndBiases(const Matrix& dW, const Vector& dB);
	 void updateWeightsAndBias();
private: 
	 int size;
	 Vector out;
	 Activation activation;
	 Optimizer* optimizer;
	 Matrix weights;
	 Vector* bias;
	 double l2 = 0;

	 Layer* precedingLayer = nullptr;

	// Not yet realized changes to the weights and biases ("observed things not yet learned")
	  Matrix deltaWeights;
	  Vector deltaBias;
	  int deltaWeightsAdded = 0;
	  int deltaBiasAdded = 0;
};

