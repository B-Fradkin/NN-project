cmake_minimum_required(VERSION 3.10)

project(NeuralNetwork)

set(NN_SOURCE_FILES
	Activation.cpp
	CostFunction.cpp
	Initializer.cpp
	Layer.cpp
	Matrix.cpp
	NeuralNetwork.cpp
	Optimizer.cpp
	Result.cpp
	TEST.cpp
	Vector.cpp
	Activation.h
	CostFunction.h
	Function.h
	Initializer.h
	Layer.h
	Matrix.h
	NeuralNetwork.h
	Optimizer.h
	Result.h
	Vector.h)
	
add_executable(NeuralNetwork ${NN_SOURCE_FILES})
