# include <iostream>
# include "../include/NeuralNetwork.h"

using namespace std;

int main(int argc, char** argv)
{
    int input, hidden, output;
    cout << "Enter the topology for a neural network: " << endl;

    cout << "Neurons in the input layer: " << endl;
    cin >> input;
    cout << "Neurons in the hidden layer: " << endl;
    cin >> hidden;
    cout << "Neurons in the output layer: " << endl;
    cin >> output;

    return 0;
}