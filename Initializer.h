#pragma once

#include <chrono>
#include <random>
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h> 
#include "Matrix.h"
class Initializer
{
public:
	Initializer();
	~Initializer();

	virtual void initWeights(Matrix& weights, int layer) = 0;
};

class Random:public Initializer {

public:
	Random(double min, double max) {
		m_min = min;
		m_max = max;
	}


	void initWeights(Matrix& weights, int layer) override
	{
		// construct a trivial random generator engine from a time-based seed:
		unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
		std::default_random_engine generator(seed);

		std::uniform_int_distribution<int> distribution(0, 1);

		double delta = m_max - m_min;

		for (int i = 0; i < weights.rows(); i++) {
			for (int j = 0; j < weights.cols(); j++) {
				weights(i, j) = m_min + distribution(generator) * delta;
			}
		}
		//weights.map(value->m_min + getRnd().nextDouble() * delta);
	}

private:
	double m_min;
	double m_max;

};

class XavierNormal:public Initializer {
	
public:
		void initWeights(Matrix& weights, int layer) override {

			// construct a trivial random generator engine from a time-based seed:
			unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
			std::default_random_engine generator(seed);

			std::normal_distribution<> distribution{ 0,1 };

			const double factor = sqrt(2.0 / (weights.cols() + weights.rows()));
			//const double factor = 1;
			auto f = [&](double a) { return distribution(generator) *factor; };

			weights.map(f);
			std::cout << weights << std::endl;
			return;
	}
};

class Constant :public Initializer {

public:
	//Constant(double*** initWeights, int height, int length, int depth){

	//Constant(double initWeights[][2][2], int height, int length, int depth) :
	//	m_height(height), m_length(length), m_depth(depth)
	//{
	//	const int  size = height * length * depth;
	//	m_initWeights = new double[size];

	//	for (int i = 0; i < height; i++) {
	//		for (int j = 0; j < length; j++) {
	//			for (int k = 0; k < depth; k++) {
	//				m_initWeights[to1D(i, j, k)] = initWeights[i][j][k];
	//				std::cout << i << ' ' << j << ' ' << k << ' ' << initWeights[i][j][k] << ' ' << to1D(i, j, k) << std::endl;
	//			}
	//		}
	//	}
	//	std::cout << "-----------------------" << std::endl;
	//}	
	
	//Constant(double initWeights[][2][2], int height, int length, int depth) :
	//	m_height(height), m_length(length), m_depth(depth)
	//{
	//	const int  size = height * length * depth;
	//	m_initWeights = new double[size];

	//	for (int k = 0; k < depth; k++) {
	//		for (int i = 0; i < height; i++) {
	//			for (int j = 0; j < length; j++) {

	//				m_initWeights[to1D(i, j,k)] = initWeights[k][i][j];
	//				std::cout << k << ' ' << i << ' ' << j << ' ' << initWeights[k][i][j] << "   " << to1D(i, j, k) << std::endl;
	//			}
	//		}
	//	}
	//	std::cout << "-----------------------" << std::endl;
	//}

	template<typename  T>
	Constant(T initWeights, int height, int length, int depth) :
		m_height(height), m_length(length), m_depth(depth)
	{
		const int  size = height * length * depth;
		m_initWeights = new double[size];

		for (int k = 0; k < depth; k++) {
			for (int i = 0; i < height; i++) {
				for (int j = 0; j < length; j++) {

					m_initWeights[to1D(i, j, k)] = initWeights[k][i][j];
					std::cout << k << ' ' << i << ' ' << j << ' ' << initWeights[k][i][j] << "   " << to1D(i, j, k) << std::endl;
				}
			}
		}
		std::cout << "-----------------------" << std::endl;
	}

	Constant(const std::string& fileName)
		:m_fileName(fileName)
		,m_isFromFile(true)
		,m_initWeights(nullptr)
		,m_height(0)
		,m_length(0)
		,m_depth(0)
	{}

	void initWeights(Matrix& weights, int layer) override
	{
		using namespace std;
		if (!m_isFromFile)
		{
			for (int i = 0; i < weights.rows(); i++) {
				for (int j = 0; j < weights.cols(); j++) {
					weights(i, j) = m_initWeights[to1D(i, j, layer)];
				}
			}
		}
		else {

			string line;
			int rows, cols;

			ifstream pFile(m_fileName);
			if (!pFile.is_open())
			{
				cout << "Can't open file" << m_fileName << endl;
				return;
			}
			int counter(0);
			while (!pFile.eof())
			{
				getline(pFile, line);
				if (counter == layer) {
					stringstream ss(line);
					ss >> rows >> cols;
					for (int i = 0; i < rows; i++) {
						for (int j = 0; j < cols; j++)
							ss >> weights(i, j);
					}
				}
				counter++;
			}
			pFile.close();
		}
	}


	int to1D(int i, int j, int k) const {
		return (k * m_height * m_length) + (i * m_length) + j;
		//return (k * m_height * m_length) + (j * m_height) + i;
	}

	void  to3D(int idx, int& i, int& j, int& k) const
	{
		k = idx / (m_height * m_length);
		idx -= (k * m_height * m_length);
		j = idx / m_length;
		i = idx % m_length;
		//j = idx / m_height;
		///i = idx % m_height;

	}



	friend std::ostream& operator<<(std::ostream& os, const Constant& v);

private:
	double* m_initWeights;
	int m_height;
	int m_length;
	int m_depth;

	std::string m_fileName;
	bool m_isFromFile = false;

};



std::ostream& operator<<(std::ostream& os, const Constant& v);
