#include "NeuralNetwork.h"



NeuralNetwork::NeuralNetwork()
{
}

/**
 * Evaluates an input vector, returning the networks output,
 * without cost or learning anything from it.
 */
Result NeuralNetwork::evaluate(const Vector & input)
{
	return evaluate(input, Vector());
}

/**
 * Evaluates an input vector, returning the networks output.
 * If <code>expected</code> is specified the result will contain
 * a cost and the network will gather some learning from this
 * operation.
 */
Result NeuralNetwork::evaluate(const Vector & input, const Vector & expected)
{
	Vector signal = input;
	for (Layer* layer : m_layers)
		signal = layer->evaluate(signal);

	if (expected.dimension() != 0) {
		learnFrom(expected);
		double cost = m_costFunction->getTotal(expected, signal);
		return Result(signal, cost);
	}

	return Result(signal);
}

/**
 * Will gather some learning based on the <code>expected</code> vector
 * and how that differs to the actual output from the network. This
 * difference (or error) is backpropagated through the net. To make
 * it possible to use mini batches the learning is not immediately
 * realized - i.e. <code>learnFrom</code> does not alter any weights.
 * Use <code>updateFromLearning()</code> to do that.s
 */
void NeuralNetwork::learnFrom(const Vector & expected)
{
	Layer* layer = getLastLayer();

	// The error is initially the derivative of the cost-function.
	Vector dCdO = m_costFunction->getDerivative(expected, layer->getOut());

	// iterate backwards through the layers
	do {
		Vector dCdI = layer->getActivation().dCdI(layer->getOut(), dCdO);
		Matrix dCdW = dCdI.outerProduct(layer->getPrecedingLayer()->getOut());

		// Store the deltas for weights and biases
 		layer->addDeltaWeightsAndBiases(dCdW, dCdI);

		// prepare error propagation and store for next iteration
		dCdO = layer->getWeights().multiply(dCdI);

		layer = layer->getPrecedingLayer();
	} while (layer->hasPrecedingLayer());     // Stop when we are at input layer
}

NeuralNetwork::~NeuralNetwork()
{
	for (auto& layer : m_layers)
		if (layer != nullptr)
			delete layer;
}

/**
 * Let all gathered (but not yet realised) learning "sink in".
 * That is: Update the weights and biases based on the deltas
 * collected during evaluation & training.
 */
void NeuralNetwork::updateFromLearning() {
	for (Layer* l : m_layers)
		if (l->hasPrecedingLayer())         // Skip input layer
			l->updateWeightsAndBias();

}
