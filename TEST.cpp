#include "Vector.h"
#include "Matrix.h"
#include "NeuralNetwork.h"
#include "Activation.h"
#include <iostream>
using namespace std;

static const double EPS = 0.00001;

bool assertEquals(double expected, double data, double eps = EPS)
{
	bool condition = fabs(data - expected) < eps;
	if (!condition)
		std::cout << "assered failed :actual " << data << " expected " << expected << std::endl;
	return condition; 
}

bool assertEquals(const Vector& expected, const Vector& actual, double eps = EPS)
{

	bool condition = expected.equals(actual, eps);
	if (!condition)
		std::cout << "assered failed :actual\n " << actual << " vs expected: \n" << expected << std::endl;
	return condition;
}

bool assertNotEquals(const Vector& expected, const Vector& actual, double eps = EPS)
{
	bool condition = expected.equals(actual, eps);
	if (condition)
		std::cout << "assered failed :actual\n " << actual << " vs expected: \n" << expected << std::endl;
	return condition;
}


void testFFExampleFromBlog() 
{
	

	 double initWeights[][2][2] = {
			 //{{0.3, 0.7}, {-.4, 0.5}},
			 //{{0.2, -.3}, {0.6, -.1}}
			 { {0.3, 0.2}, {-.4, 0.6}},
			 {{0.7, -.3}, {0.5, -.1}}
	 };	


	 NeuralNetwork* network =
		 NeuralNetwork::Builder(2)
		 .addLayer(new Layer(2, Activation::Sigmoid, new Vector({ 0.25, 0.45 })))
		 .addLayer(new Layer(2, Activation::Sigmoid, new Vector({0.15, 0.35})))
		 .setCostFunction(new Quadratic())
		 .setOptimizer(new GradientDescent(0.1))
		 .initWeights(new Constant(initWeights, 2, 2, 2))
		 .create();

	 Vector out = network->evaluate(Vector({ 2, 3 }), Vector({ 1, 0.2 })).getOutput();  //TODO FIX evaluate

	 const double* data = out.getData();
	 assertEquals(0.712257432295742, data[0], EPS);
	 assertEquals(0.533097573871501, data[1], EPS);

	 network->updateFromLearning();

	 Result result = network->evaluate(Vector({ 2, 3 }), Vector({ 1, 0.2 }));
	 out = result.getOutput();
	 data = out.getData();
	 assertEquals(0.7187729999291985, data[0], EPS);
	 assertEquals(0.5238074518609882, data[1], EPS);
	 return;
};

void testEvaluate() {

	double initWeights[][3][3] = {
			{{0.1, 0.2, 0.3}, {0.3, 0.2, 0.7}, {0.4, 0.3, 0.9}},
			{{0.2, 0.3, 0.5}, {0.3, 0.5, 0.7}, {0.6, 0.4, 0.8}},
			{{0.1, 0.4, 0.8}, {0.3, 0.7, 0.2}, {0.5, 0.2, 0.9}}
	};

	NeuralNetwork* network =
		 NeuralNetwork::Builder(3)
		.addLayer(new Layer(3, Activation::ReLU, 1))
		.addLayer(new Layer(3, Activation::Sigmoid, 1))
		.addLayer(new Layer(3, Activation::Softmax, 1))
		.initWeights(new Constant(initWeights, 3, 3, 3))
		.create();

	Vector out = network->evaluate(Vector({ 0.1, 0.2, 0.7 })).getOutput();

	const double* data = out.getData();
	assertEquals(0.1984468942, data[0], EPS);
	assertEquals(0.2853555304, data[1], EPS);
	assertEquals(0.5161975753, data[2], EPS);
	assertEquals(1, data[0] + data[1] + data[2], EPS);
}

// Based on forward pass here
// https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

void testEvaluateAndLearn() {

	double initWeights[][2][2] = {
			{{0.15, 0.25}, {0.20, 0.30}},
			{{0.40, 0.50}, {0.45, 0.55}},
			//{{0.15, 0.40}, {0.20, 0.45}},
			//{{0.25, 0.50}, {0.30, 0.55}},
	};

	NeuralNetwork* network =
		 NeuralNetwork::Builder(2)
		.addLayer(new Layer(2, Activation::Sigmoid, new Vector({ 0.35, 0.35 })))
		.addLayer(new Layer(2, Activation::Sigmoid, new Vector({ 0.60, 0.60 })))
		.setCostFunction(new HalfQuadratic())
		.setOptimizer(new GradientDescent(0.5))
		.initWeights(new Constant(initWeights, 2, 2, 2))
		.create();


	Vector expected({ 0.01, 0.99 });
	Vector input({ 0.05, 0.1 });

	Result result = network->evaluate(input, expected);

	Vector out = result.getOutput();

	assertEquals(0.29837110, result.getCost(), EPS);
	assertEquals(0.75136507, out.getData()[0], EPS);
	assertEquals(0.77292846, out.getData()[1], EPS);

	network->updateFromLearning();

	result = network->evaluate(input, expected);
	out = result.getOutput();

	assertEquals(0.28047144, result.getCost(), EPS);
	assertEquals(0.72844176, out.getData()[0], EPS);
	assertEquals(0.77837692, out.getData()[1], EPS);

	for (int i = 0; i < 10000; i++) {
		network->updateFromLearning();
		result = network->evaluate(input, expected);
	}

	out = result.getOutput();
	assertEquals(0.0000024485, result.getCost(), EPS);
	assertEquals(0.011587777, out.getData()[0], EPS);
	assertEquals(0.9884586899, out.getData()[1], EPS);
}



   
void testEvaluateAndLearn2() {

	   NeuralNetwork* network =
		    NeuralNetwork::Builder(4)
		   .addLayer(new Layer(6, Activation::Sigmoid, 0.5))
		   .addLayer(new Layer(14, Activation::Sigmoid, 0.5))
		   .setCostFunction(new Quadratic())
		   .setOptimizer(new GradientDescent(1))
//		   .initWeights(new XavierNormal())
		   .initWeights(new Constant("C:\\temp\\XavierNormal.txt"))
		   .create();


	   int trainInputs[][14] = {
			   {1, 1, 1, 0},
			   {1, 1, 0, 0},
			   {0, 1, 1, 0},
			   {1, 0, 1, 0},
			   {1, 0, 0, 0},
			   {0, 1, 0, 0},
			   {0, 0, 1, 0},
			   {1, 1, 1, 1},
			   {1, 1, 0, 1},
			   {0, 1, 1, 1},
			   {1, 0, 1, 1},
			   {1, 0, 0, 1},
			   {0, 1, 0, 1},
			   {0, 0, 1, 1}
	   };

	   int trainOutput[][14] = {
			   {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			   {0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			   {0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			   {0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			   {0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			   {0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
			   {0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
			   {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
			   {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
			   {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
			   {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
			   {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
			   {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
			   {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1}
	   };

	   const size_t length = std::size(trainInputs);
	   int cnt = 0;
	   for (int i = 0; i < 1100; i++) {
		   Vector input(trainInputs[cnt], 4);
		   Vector expected(trainOutput[cnt], 14);
		   network->evaluate(input, expected);
		   network->updateFromLearning();
		   cnt = (cnt + 1) % length;
	   }

	   for (int i = 0; i < length; i++) {
		   Result result = network->evaluate(Vector(trainInputs[i], 4));
		   int ix = result.getOutput().indexOfLargestElement();
		   assertEquals(Vector(trainOutput[i] , 14), Vector(trainOutput[ix] , 14),EPS);
	   }
}

  
	void testMakeACopyOfNetwork() {

	   NeuralNetwork* network1 =
		    NeuralNetwork::Builder(4)
		   .addLayer(new Layer(6, Activation::Sigmoid, 0.5))
		   .addLayer(new Layer(14, Activation::Sigmoid, 0.5))
		   .setCostFunction(new Quadratic())
		   .setOptimizer(new GradientDescent(1))
		   .initWeights(new XavierNormal())
		   .create();

	   NeuralNetwork* network2 =  NeuralNetwork::Builder(network1).create();

	   int trainInputs[][4] = {
			   {1, 1, 1, 0},
			   {1, 1, 0, 0},
			   {0, 1, 1, 0},
			   {1, 0, 1, 0},
	   };

	   int trainOutput[][14] = {
			   {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			   {0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			   {0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			   {0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	   };

	   const size_t length = std::size(trainInputs);
	   int cnt = 0;
	   for (int i = 0; i < 1100; i++) {
		   Vector input(trainInputs[cnt],4);
		   Vector expected(trainOutput[cnt],4);
		   network1->evaluate(input, expected);
		   network1->updateFromLearning();
		   network2->evaluate(input, expected);
		   network2->updateFromLearning();
		   cnt = (cnt + 1) % length;
	   }

	   for (auto trainInput : trainInputs) {
		   Vector input(trainInput,4);
		   Result result1 = network1->evaluate(input);
		   Result result2 = network2->evaluate(input);
		   Vector out1 = result1.getOutput();
		   Vector out2 = result2.getOutput();
		   assertEquals(out1, out2, EPS);
	   }
   }

	 void testBatching() {

		NeuralNetwork* network =
			 NeuralNetwork::Builder(2)
			.addLayer(new Layer(2, Activation::ReLU, 0.5))
			.addLayer(new Layer(2, Activation::Sigmoid, 0.5))
			.initWeights(new XavierNormal())
			.create();

		Vector i1({0.1, 0.7});
		Vector i2({-0.2, 0.3});
		Vector w1({0.2, 0.1});
		Vector w2({1.2, -0.4});

		// First, evaluate without learning
		Result out1a = network->evaluate(i1);
		Result out2a = network->evaluate(i2);

		// Then this should do nothing to the net
		network->updateFromLearning();

		Result out1b = network->evaluate(i1, w1);
		Result out2b = network->evaluate(i2, w2);
		assertEquals(out1a.getOutput(), out1b.getOutput(),EPS);
		assertEquals(out2a.getOutput(), out2b.getOutput(),EPS);

		// Now, evaluate and learn
		out1a = network->evaluate(i1, w1);
		out2a = network->evaluate(i2, w2);

		double cost1BeforeLearning = out1b.getCost();
		double cost2BeforeLearning = out2b.getCost();

		// First verify that we still have not changed any weights
		assertEquals(out1a.getOutput(), out1b.getOutput(),EPS);
		assertEquals(out2a.getOutput(), out2b.getOutput(),EPS);

		// This should however change things ...
		network->updateFromLearning();

		out1b = network->evaluate(i1, w1);
		out2b = network->evaluate(i2, w2);

		assertNotEquals(out1a.getOutput(), out1b.getOutput());
		assertNotEquals(out2a.getOutput(), out2b.getOutput());

		// ... and the cost should be lower
		double cost1AfterLearning = out1b.getCost();
		double cost2AfterLearning = out2b.getCost();

		bool assertTrue = (cost1AfterLearning < cost1BeforeLearning);
		assertTrue = (cost2AfterLearning < cost2BeforeLearning);
	}



int main() {



	//testFFExampleFromBlog();
	//testEvaluate();
	//testEvaluateAndLearn();
	//testEvaluateAndLearn2();
	testBatching();
	return 0;
}
