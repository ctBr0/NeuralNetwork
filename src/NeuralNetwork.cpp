# include <iostream>
# include "../include/NeuralNetwork.h"

using namespace std;

NeuralNetwork::NeuralNetwork(vector<int> neurons_by_layer)
{
    this->neurons_by_layer = neurons_by_layer;
    this->num_of_layers = neurons_by_layer.size();
    this->biases = {};
    this->weights = {};

    // intialize the biases vector and weights matrix with random values between -1 and 1
    // no biases in the input layer
    for (int i = 1; i < num_of_layers; i++) // biases
    {
        VectorXd layer_b = VectorXd::Random(neurons_by_layer[i]); // vector of random values between -1 and 1
        this->biases.push_back(layer_b);
        MatrixXd layer_w = MatrixXd::Random(neurons_by_layer[i],neurons_by_layer[i-1]);
        this->weights.push_back(layer_w);
    }
}

VectorXd NeuralNetwork::feedForward(VectorXd input)
{
    VectorXd output = input;
    for (int i = 0; i < biases.size(); i++)
    {
        output = squishify(((weights[i].row(i).dot(output)) + biases[i].array()).matrix());
    }
    return output;
}

void NeuralNetwork::BSGD(vector<vector<VectorXd>> train_data, double learning_rate, int epochs, int batch_size)
{
    int train_size = train_data.size();

    vector<vector<vector<VectorXd>>> batches;
    vector<vector<VectorXd>> batch;
    auto rng = default_random_engine();
    for (int i = 0; i < epochs; i++)
    {
        batches = {};
        shuffle(train_data.begin(), train_data.end(), rng);

        /*
        Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", " << ", ";");

        cout << "After shuffle: " << endl;
        for (int i = 0; i < 3; i++)
        {
            cout << train_data[i][0].format(CommaInitFmt) << endl;
            cout << train_data[i][1].format(CommaInitFmt) << endl;
        }
        */

        for (int j = 0; j < int(train_data.size() / batch_size); j++) // for each batch
        {
            batch = {};
            for (int k = (j * batch_size); k < (j * batch_size) + batch_size; k++)
            {
                batch.push_back(train_data[k]);
            }
            batches.push_back(batch);
        }
        
        for (int j = 0; j < batches.size(); j++)
        {
            this->updateWeightsAndBiases(batches[j], learning_rate);
        }
    }

    cout << "Done!" << endl;
}

void NeuralNetwork::updateWeightsAndBiases(vector<vector<VectorXd>> batch, double learning_rate)
{
    // initialize matrix elements to 0
    vector<VectorXd> grad_biases = {};
    vector<MatrixXd> grad_weights = {}; // gradient for the cost function

     for (int i = 1; i < this->num_of_layers; i++)
    {
        // VectorXd layer(neurons_by_layer[i]);
        // VectorXd layer = VectorXd::Zero(neurons_by_layer[i]);
        // grad_biases.push_back(VectorXd::Zero(neurons_by_layer[i]));
        // grad_biases.push_back(layer.setZero());
        grad_biases.push_back(VectorXd::Zero(neurons_by_layer[i]));
        // MatrixXd layer(neurons_by_layer[i],neurons_by_layer[i-1]);
        // MatrixXd layer = MatrixXd::Zero(this->neurons_by_layer[i],this->neurons_by_layer[i+1]);
        // grad_weights.push_back(layer.setZero());
        grad_weights.push_back(MatrixXd::Zero(neurons_by_layer[i],neurons_by_layer[i-1]));
    }

    VWofLayer gradient = {};

    for (int i = 0; i < batch.size(); i++)
    {
        gradient = this->propagateBackward(batch[i][0], batch[i][1]);
        for (int j = 0; j < grad_biases.size(); j++)
        {
            grad_biases[j] = (grad_biases[j].array() + gradient.biases[j].array()).matrix();
            grad_weights[j] = (grad_weights[j].array() + gradient.weights[j].array()).matrix();
        }
    }

    for (int i = 0; i < this->biases.size(); i++)
    {
        this->biases[i] = (this->biases[i].array() - (learning_rate * grad_biases[i].array() / batch.size())).matrix();
        this->weights[i] = (this->weights[i].array() - (learning_rate * grad_weights[i].array() / batch.size())).matrix();
    }
}

VWofLayer NeuralNetwork::propagateBackward(VectorXd input, VectorXd output)
{
    // initialize matrix elements to 0
    vector<VectorXd> grad_biases = {};
    vector<MatrixXd> grad_weights = {}; // gradient for the cost function

     for (int i = 1; i < this->num_of_layers; i++)
    {
        // VectorXd layer(neurons_by_layer[i]);
        // VectorXd layer = VectorXd::Zero(this->neurons_by_layer[i]);
        // grad_biases.push_back(layer.setZero());
        grad_biases.push_back(VectorXd::Zero(neurons_by_layer[i]));
        // MatrixXd layer(neurons_by_layer[i],neurons_by_layer[i-1]);
        // MatrixXd layer = MatrixXd::Zero(this->neurons_by_layer[i],this->neurons_by_layer[i+1]);
        // grad_weights.push_back(layer.setZero());
        grad_weights.push_back(MatrixXd::Zero(neurons_by_layer[i],neurons_by_layer[i-1]));
    }
    
    // feed forward
    vector<VectorXd> activations;
    VectorXd z_value;
    vector<VectorXd> list_of_z_values;

    VectorXd curr_active = input;
    activations.push_back(input);
    for (int i = 0; i < grad_biases.size(); i++)
    {
        // z_value = squishify(((this->weights[i].row(i).dot(curr_active)) + this->biases[i].array()).matrix());
        // z_value = ((this->weights[i]*curr_active).array() + this->biases[i].array()).matrix();
        // z_value = ((curr_active.asDiagonal()*this->weights[i]).array() + this->biases[i].array()).matrix();
        z_value = ((this->weights[i]*curr_active.matrix()).array() + this->biases[i].array()).matrix();
        list_of_z_values.push_back(z_value);
        curr_active = squishify(z_value);
        activations.push_back(curr_active);
    }

    // propagate backward
    VectorXd delta;
    delta = ((activations.back().array() - output.array()) * squishify_der(list_of_z_values.back()).array()).matrix();
    grad_biases.pop_back();
    grad_biases.push_back(delta);
    // int rows = grad_weights[grad_weights.size() - 1].rows();
    // int cols = grad_weights[grad_weights.size() - 1].cols();
    grad_weights.pop_back();
    // grad_weights.push_back((delta.dot(activations[activations.size() - 2].transpose()) + MatrixXd::Zero(rows, cols).array()).matrix());
    grad_weights.push_back(delta*((activations[activations.size() - 2].transpose()).matrix()));

    for (int i = 2; i < this->num_of_layers; i++)
    {
        z_value = list_of_z_values[list_of_z_values.size() - i];
        // delta = ((((this->weights[this->weights.size() - i + 1]).row(i).transpose()).dot(delta)) * squishify_der(z_value).array()).matrix();
        delta = ((((this->weights[this->weights.size() - i + 1]).transpose()).matrix()*delta).array() * squishify_der(z_value).array()).matrix();
        grad_biases[grad_biases.size() - i] = delta;
        grad_weights[grad_weights.size() - i] = delta*((activations[activations.size() - i - 1]).transpose().matrix());
    }

    VWofLayer result = {grad_biases, grad_weights};
    return result;
}

VectorXd NeuralNetwork::squishify(VectorXd input)
{
    return (1.0 / (1.0 + ((-1.0) * input.array()).exp())).matrix();
}

VectorXd NeuralNetwork::squishify_der(VectorXd input)
{
    return (squishify(input).array() * (1.0 - squishify(input).array())).matrix();
}

void NeuralNetwork::accuracy(vector<vector<VectorXd>> test_data, int num_to_test)
{
    double accuracy;
    int count = 0;
    int actual_index = 0;
    int test_index = 0;
    VectorXd actual_output;
    for (int i = 0; i < num_to_test; i++)
    {
        actual_output = this->feedForward(test_data[i][0]);
        int actual_index = indexOfVectorCoeff(actual_output,actual_output.maxCoeff());
        int test_index = indexOfVectorCoeff(test_data[i][1],test_data[i][1].maxCoeff());
        /*
        for (int i = 0; i < actual_output.cols(); i++)
        {
            if (max_index == actual_output.coeff(i, 1))
            {
                actual_index = i;
            }
            if (test_index == 1)
            {
                test_index = i;
            }
        }
        */

        if (actual_index == test_index)
        {
            count++;
        }
    }
    accuracy = ((double)count / test_data.size()) * 100;
    cout << "Accuracy: " + to_string(accuracy) + "%" << endl;
}

void NeuralNetwork::showWeights()
{
    for (int i = 0; i < this->num_of_layers - 1; i++)
    {
        cout << this->weights[i] << endl;
        cout << endl;
    }
}

void NeuralNetwork::showBiases()
{
    for (int i = 0; i < this->num_of_layers - 1; i++)
    {
        cout << this->biases[i] << endl;
        cout << endl;
    }
}

int NeuralNetwork::indexOfVectorCoeff(VectorXd vector, double coeff)
{
    int index = 0;
    int output;
    bool found = false;
    while (!found || index < vector.size())
    {
        if (coeff == vector(index))
        {
            found = true;
            output = index;
        }
        index++;
    }
    return output;
}