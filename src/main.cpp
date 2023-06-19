# include <iostream>
# include <fstream>
# include <sstream>
# include "../include/NeuralNetwork.h"

using namespace std;

VectorXd numToVectorXd(int);
void head(vector<vector<VectorXd>>, int);

int main(int argc, char** argv)
{

    // Instantiate neural network

    int input, hidden, output, num_of_layers;
    vector<int> neurons_by_layer;
    string train_loc, test_loc;
    string sep = "\n---------------------------------------------------------------------------\n";
    
    cout << sep << "Enter the topology for a neural network... " << sep << endl;

    do
    {
        cout << "Number of layers: (3 is the minimum)" << endl;
        cin >> num_of_layers;
    }
    while(num_of_layers < 3);

    cout << "Neurons in the input layer: (784 neurons for the MNIST dataset)" << endl;
    cin >> input;
    neurons_by_layer.push_back(input);

    for (int i = 1; i < num_of_layers - 1; i++)
    {
        cout << "Neurons in hidden layer " << i << ": "<< endl;
        cin >> hidden;
        neurons_by_layer.push_back(hidden);
    }

    cout << "Neurons in the output layer: (10 neurons for the MNIST dataset)" << endl;
    cin >> output;
    neurons_by_layer.push_back(output);

    NeuralNetwork *network = new NeuralNetwork(neurons_by_layer);

    cout << sep << "Generating network..." << sep << endl;

    // Show intial weights and biases

    // cout << "Initial weights: " << endl;
    // cout << endl;
    // network->showWeights();
    // cout << "Intital biases: " << endl;
    // cout << endl;
    network->showBiases();

    // Obtain locations of input training and testing files

    /*
    cout << sep << "Enter training file location... " << sep << endl;

    cout << "Location: " << endl;
    cin >> train_loc;

    cout << sep << "Enter testing file location... " << sep << endl;

    cout << "Location: " << endl;
    cin >> test_loc;
    */

    train_loc = "../mnist_train_small.csv";
    test_loc = "../mnist_test_small.csv";

    // Parse the csv input files and convert the data into vectors

    // For MNIST dataset:
    // First column of the csv file is the output value which is converted into a vectorxd eg.[0,0,0,0,0,0,1,0,0,0] for an output of 6
    // The next 784 columns are the input pixel values

    vector<vector<VectorXd>> train_data;
    vector<vector<VectorXd>> test_data;

    string line;
    ifstream train_file(train_loc);
    ifstream test_file(test_loc);

    if(train_file.is_open())
    {
        vector<VectorXd> input_output;
        VectorXd input(784); // according to the MNIST dataset
        VectorXd output(10); // according to the MNIST dataset

        train_file.ignore(10000,'\n'); // skip the first line
        while(getline(train_file,line))
        {
            stringstream lineStream(line);
            string cell;

            getline(lineStream,cell,','); // output value (first column of the csv file)
            output = numToVectorXd(stoi(cell));

            int index = 0;
            while(getline(lineStream,cell,',')) // input value (the next 784 columns of the csv file)
            {
                input[index] = stoi(cell);
                index++;
            }

            // combine the input and output values into an array
            input_output = {};
            input_output.push_back(input);
            input_output.push_back(output);
            train_data.push_back(input_output);
        }
        train_file.close();
    }
    else
    {
        cout << "Unable to open training file!" << endl;
        exit(0);
    }

    if(test_file.is_open())
    {
        vector<VectorXd> input_output;
        VectorXd input(784); // according to the MNIST dataset
        VectorXd output(10); // according to the MNIST dataset

        test_file.ignore(10000,'\n'); // skip the first line
        while(getline(test_file,line))
        {
            stringstream lineStream(line);
            string cell;

            getline(lineStream,cell,','); // output value (first column of the csv file)
            output = numToVectorXd(stoi(cell));

            int index = 0;
            while(getline(lineStream,cell,',')) // input value (the next 784 columns of the csv file)
            {
                input[index] = stoi(cell);
                index++;
            }

            // combine the input and output values into an array
            input_output = {};
            input_output.push_back(input);
            input_output.push_back(output);
            test_data.push_back(input_output);
        }
        test_file.close();
    }
    else
    {
        cout << "Unable to open testing file!" << endl;
        exit(0);
    }
    
    // Show vector sizes

    cout << sep << "Number of samples... " << sep << endl;

    cout << "Training samples: " << endl;
    cout << train_data.size() << endl;
    // cout << "First 3 samples of the training dataset: " << endl;
    // head(train_data,3);
    cout << "Testing samples: " << endl;
    cout << test_data.size() << endl;
    // cout << "First 3 samples of the testing dataset: " << endl;
    // head(test_data,3);

    // Train the network

    network->BSGD(train_data, 0.01, 100, 20);

    network->showBiases();

    // Test data

    // call the accuracy method with the testing data

    network->accuracy(test_data,50);

    delete network;

    return 0;
}

VectorXd numToVectorXd(int input)
{
    VectorXd output(10);
    switch(input) // convert numbers into vector
    {
        case 0:
            output << 1,0,0,0,0,0,0,0,0,0;
            break;
        case 1:
            output << 0,1,0,0,0,0,0,0,0,0;
            break;
        case 2:
            output << 0,0,1,0,0,0,0,0,0,0;
            break;
        case 3:
            output << 0,0,0,1,0,0,0,0,0,0;
            break;
        case 4:
            output << 0,0,0,0,1,0,0,0,0,0;
            break;
        case 5:
            output << 0,0,0,0,0,1,0,0,0,0;
            break;
        case 6:
            output << 0,0,0,0,0,0,1,0,0,0;
            break;
        case 7:
            output << 0,0,0,0,0,0,0,1,0,0;
            break;
        case 8:
            output << 0,0,0,0,0,0,0,0,1,0;
            break;
        case 9:
            output << 0,0,0,0,0,0,0,0,0,1;
            break;
    }
    return output;
}

// Prints a specified number of samples from a dataset
void head(vector<vector<VectorXd>> data, int num)
{
    Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", " << ", ";");
    for (int i = 0; i < num; i++)
    {
        cout << data[i][0].format(CommaInitFmt) << endl;
        cout << data[i][1].format(CommaInitFmt) << endl;
    }
}
