#include "Layer.h"



Layer::Layer()
{
}


Layer::~Layer()
{
}


/**
 * Feed the in-vector, i, through this layer->
 * Stores a copy of the out vector.
 *
 * @param i The input vector
 * @return The out vector o (i.e. the result of o = iW + b)
 */
 const Vector& Layer::evaluate(const Vector& i) {
	if (!hasPrecedingLayer()) {
		out = i;    // No calculation i input layer, just store data
	}
	else {
		out = activation.fn(i.mul(weights).add(*bias));
	}
	return out;
}

 /**
 * Add upcoming changes to the Weights and Biases.
 * This does not mean that the network is updated.
 */
  void Layer::addDeltaWeightsAndBiases(const Matrix& dW, const Vector& dB) {
	 deltaWeights.add(dW);
	 deltaWeightsAdded++;
	 deltaBias = deltaBias.add(dB);
	 deltaBiasAdded++;
 }


  /**
   * Takes an average of all added Weights and Biases and tell the
   * optimizer to apply them to the current weights and biases.
   *
   * Also applies L2 regularization on the weights if used.
   */
  void Layer::updateWeightsAndBias() {

	  auto L2 = this->l2;
	  if (deltaWeightsAdded > 0) {
		  if (l2 > 0)
			  //weights.map(value->value - l2 * value);
			  weights.map([L2](double value) { return value - L2 * value; });

		  Matrix average_dW = deltaWeights.mul(1.0 / deltaWeightsAdded);
		  optimizer->updateWeights(weights, average_dW);
		  //deltaWeights.map(a -> 0);   // Clear
		  deltaWeights.map([](double a){ return 0; } );   // Clear
		  deltaWeightsAdded = 0;
	  }

	  if (deltaBiasAdded > 0) {
		  Vector average_bias = deltaBias.mul(1.0 / deltaBiasAdded);
		  *bias = optimizer->updateBias(*bias, average_bias);
		  //deltaBias = deltaBias.map(a -> 0);  // Clear
		  deltaBias = deltaBias.map([](double a) { return 0; });
		  deltaBiasAdded = 0;
	  }
  }
