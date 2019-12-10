#include "Dense.h"
#include "Activation.h"
#include "Softmax.h"
#include "Sigmoid.h"
#include "BinaryCrossEntropy.h"
#include <random>
#include <algorithm>
#include <vector>
#include <iostream>
#include <ctime>
#include "TGraph.h"
#include "TCanvas.h"
#include "Layer.h"
using namespace UkrNet;

int main()
{
    auto&& inputDim = 10;
    
    auto&& denseLayer1 = new Dense<double>(inputDim, 5);
    auto&& activationLayer1 = new Activation<double>();
    auto&& denseLayer2 = new Dense<double>(5, 1);
    auto&& sigmoidLayer1 = new Sigmoid<double>();
    auto&& crossEntropyLayer = new BinaryCrossEntropy<double>();
    
    std::vector<double> input(10);
    
    std::generate(input.begin(), input.end(), [n = 0.] () mutable { n++;return 0.1*(n-1); });
//     for(auto&& el : input)
//         std::cout << el << '\n';
//     
    auto&& out1 = denseLayer1->Forward(input);
    auto&& out2 = activationLayer1->Forward(out1);
    auto&& out3 = denseLayer2->Forward(out2);
    auto&& out4 = sigmoidLayer1->Forward(out3);
//     std::cout << " SIZE: " << out4.size() << '\n';
    std::cout << out4[0] << '\n';
    auto&& out5 = crossEntropyLayer->Forward(std::vector<double>{1.}, out4);
    std::cout << "Loss: " << out5 << '\n';
    
    auto&& outGrad5 = crossEntropyLayer->Backward();
//     std::cout << outGrad5[0] << '\n';
    auto&& outGrad4 = sigmoidLayer1->Backward(outGrad5);
//     std::cout << outGrad4[0] << '\n';
    auto&& outGrad3 = denseLayer2->Backward(outGrad4);
//     for(auto&& el : outGrad3)
//         std::cout << el << '\n';
    auto&& outGrad2 = activationLayer1->Backward(outGrad3);
//     for(auto&& el : outGrad2)
//         std::cout << el << '\n';
    auto&& outGrad1 = denseLayer1->Backward(outGrad2);
//     for(auto&& el : outGrad1)
//         std::cout << el << '\n';
    
    denseLayer1->UpdateWeights(1, 1);
    denseLayer2->UpdateWeights(1, 1);
        
    delete denseLayer1;
    delete denseLayer2;
    delete activationLayer1;
    delete sigmoidLayer1;
    delete crossEntropyLayer;
    
    return 0;
}
