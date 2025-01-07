#pragma once
#include "Optimizer.h"
#include "Matrix.h"
#include "Layer.h"
#include "CostFunction.h"
#include "Initializer.h"
#include "Result.h"
#include <vector>




class NeuralNetwork
{
public:

	// --------------------------------------------------------------------

/**
 * Simple builder for a NeuralNetwork
 */
	class Builder {
		friend NeuralNetwork;

	public:
		Builder() {
			init();
		}
		Builder(int networkInputSize)  : Builder() {
			m_networkInputSize = networkInputSize;
			init();
		}

		void  init() {
			m_initializer = new Random(-0.5, 0.5);
			m_costFunction = new Quadratic();
			m_optimizer = new GradientDescent(0.005);
		}

		/**
		 * Create a builder from an existing neural network, hence making
		 * it possible to do a copy of the entire state and modify as needed.
		 */
		Builder(NeuralNetwork* other) {
			m_networkInputSize = other->m_networkInputSize;
			m_costFunction = other->m_costFunction;
			m_optimizer = other->m_optimizer;
			m_l2 = other->m_l2;


			for (auto otherLayer : other->m_layers) {

				m_layers.push_back(
					new Layer(
						otherLayer->getSize(),
						otherLayer->getActivation(),
						otherLayer->getBias()
					)
				);
			}

			//m_initializer = [](weights, layer) {                                        
			//	Layer otherLayer = otherLayers.get(layer + 1);
			//	Matrix otherLayerWeights = otherLayer.getWeights();
			//	weights.fillFrom(otherLayerWeights);
			//};


			//m_initializer = (weights, layer) -> {                                         TODO FIX ME!!
			//	Layer otherLayer = otherLayers.get(layer + 1);
			//	Matrix otherLayerWeights = otherLayer.getWeights();
			//	weights.fillFrom(otherLayerWeights);
			//};

		}

		Builder& initWeights(Initializer* initializer) {
			m_initializer = initializer;
			return *this;
		}

		Builder& setCostFunction(CostFunction* costFunction) {
			m_costFunction = costFunction;
			return *this;
		}

		Builder& setOptimizer(Optimizer* optimizer) {
			m_optimizer = optimizer;
			return *this;
		}

		Builder& l2(double l2) {
			m_l2 = l2;
			return *this;
		}

		Builder& addLayer(Layer* layer) {
			m_layers.push_back(layer);
			return *this;
		}

		NeuralNetwork* create() {
			return new NeuralNetwork(*this);
		}

	private:
		std::vector<Layer*> m_layers;
		int m_networkInputSize;

		// defaults:
		Initializer* m_initializer ;
		CostFunction* m_costFunction;
		Optimizer* m_optimizer;
		double m_l2 = 0;



	};

	// --------END OF BUILDER -----------





	NeuralNetwork();

	/**
	 * Creates a neural network given the configuration set in the builder
	 *
	 * @param nb The config for the neural network
	 */
	private:

		NeuralNetwork(NeuralNetwork::Builder& nb) {

		m_costFunction = nb.m_costFunction;
		m_networkInputSize = nb.m_networkInputSize;
		m_optimizer = nb.m_optimizer;
		m_l2 = nb.m_l2;

		// Adding inputLayer

		Layer* inputLayer =  new Layer(m_networkInputSize, Activation::Identity);
		m_layers.push_back(inputLayer);

		Layer* precedingLayer = inputLayer;
		int currentLayer = 0;
		for (auto layer : nb.m_layers) {
			Matrix w = Matrix(precedingLayer->getSize(), layer->getSize());                         //TODO fix matrix creation.
			nb.m_initializer->initWeights(w, currentLayer);
			layer->setWeights(w);    // Each layer contains the weights between preceding layer and itself
			layer->setOptimizer(m_optimizer->copy());
			layer->setL2(m_l2);
			layer->setPrecedingLayer(precedingLayer);
			m_layers.push_back(layer);
			currentLayer++;
			precedingLayer = layer;
		}
	}

public:
	Layer* getLastLayer() {
		return (m_layers.at(m_layers.size() - 1));
	}
		 
	Result evaluate(const Vector& input);
	Result evaluate(const Vector& input, const Vector& expected) ;
	void learnFrom(const Vector& expected);
	void updateFromLearning();
	~NeuralNetwork();

private:
	 CostFunction* m_costFunction;
	 int m_networkInputSize;
	 double m_l2;
	 Optimizer* m_optimizer;

	std::vector<Layer*> m_layers;
};

