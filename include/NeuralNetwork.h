# ifndef NEURALNETWORK_H
# define NEURALNETWORK_H

# include <iostream>
# include <vector>
# include <algorithm>
# include <random>
# include "../libs/eigen/Eigen/Eigen"

using namespace std;

typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VectorXd; // dynamic vector
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixXd; // dynamic matrix

struct VWofLayer // biases and weights of layer
{
    vector<VectorXd> biases;
    vector<MatrixXd> weights;
};

class NeuralNetwork
{
    public:
    NeuralNetwork(vector<int>); // ex. NeuralNetwork([1,1,1]) for a 3 layer neural network with a single node in each layer
    VectorXd feedForward(VectorXd);
    VWofLayer propagateBackward(VectorXd, VectorXd);
    void BSGD(vector<vector<VectorXd>>, double, int, int); // batch stochastic gradient descent
    void updateWeightsAndBiases(vector<vector<VectorXd>>, double);
    VectorXd squishify(VectorXd); // sigmoid function
    VectorXd squishify_der(VectorXd); // derivative of sigmoid function
    void accuracy(vector<vector<VectorXd>>, int);
    void showWeights();
    void showBiases();

    private:
    vector<int> neurons_by_layer;
    int num_of_layers;
    vector<VectorXd> biases;
    vector<MatrixXd> weights;
};

# endif
