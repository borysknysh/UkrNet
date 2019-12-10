#ifndef DENSE_H
#define DENSE_H

#include <vector>
#include <random>
#include <iostream>

namespace UkrNet
{

template<class T>
class Dense
{
public:
    Dense(const int& inputSize, const int& outputSize)
    {
        fInputSize = inputSize;
        fOutputSize = outputSize;
        std::random_device rd;
        std::default_random_engine generator(rd());
        std::normal_distribution<T> distribution(0,std::sqrt(2./(fInputSize+fOutputSize)));
  
        fWeightsMatrix.resize(fOutputSize);
        for(auto& el : fWeightsMatrix)
            el.resize(fInputSize);
        for(auto& el: fWeightsMatrix)
            for(auto& el2 : el)
            {
                el2 = distribution(generator);
            }
        fAccumulatedWeightGradient.resize(fOutputSize);
        for(auto& el : fAccumulatedWeightGradient)
            el.resize(fInputSize);
        fBiasVector.resize(fOutputSize);
        fOutcomingGradient.resize(fInputSize);
        fAccumulatedBiasGradient.resize(fOutputSize);
    }
    
    std::vector<T> Forward(const std::vector<T>& input)
    {
        fInput = input;
        std::vector<T> output;
        for(auto j = 0; j < fOutputSize; j++)
        {
            auto&& res = 0.;
            for(auto i = 0; i < fInputSize; i++)
                res += fWeightsMatrix[j][i] * fInput[i];
            
            output.emplace_back(res+fBiasVector[j]);
        }
//         std::cout << "fWeightsMatrix: " << fWeightsMatrix[0][0] << '\n';
        return std::move(output);
    }
    
    std::vector<T> Backward(const std::vector<T>& incomingGradient)
    {
        fIncomingGradient = incomingGradient;
        std::vector<double> output;
        for(auto i = 0; i < fInputSize; i++)
        {
            auto&& res = 0.;
            for(auto j = 0; j < fOutputSize; j++)
                res += fWeightsMatrix[j][i] * fIncomingGradient[j];
            
            output.emplace_back(res);
        }
        return output;
    }
    
    void UpdateWeights(const int& batchSize, const T& lr)
    {
        for(auto j = 0; j < fOutputSize; j++)
        {
            fAccumulatedBiasGradient[j] += fIncomingGradient[j];
            for(auto i = 0; i < fInputSize; i++)
            {
                    fAccumulatedWeightGradient[j][i] += fInput[i] * fIncomingGradient[j];
            }
        }
        for(auto&& j = 0; j < fOutputSize; j++)
        {
            fBiasVector[j] -= lr * fAccumulatedBiasGradient[j] / batchSize;
            fAccumulatedBiasGradient[j] = 0;
            for(auto&& i = 0; i < fInputSize; i++)
            {
                fWeightsMatrix[j][i] -= lr * fAccumulatedWeightGradient[j][i] / batchSize;
                fAccumulatedWeightGradient[j][i] = 0;
            }
        }
    }
private:
    int fInputSize;
    int fOutputSize;
    std::vector<std::vector<T>> fWeightsMatrix;
    std::vector<T> fBiasVector;
    std::vector<T> fIncomingGradient;
    std::vector<T> fOutcomingGradient;
    std::vector<T> fInput;
    std::vector<T> fOutput;
    std::vector<std::vector<T>> fAccumulatedWeightGradient;
    std::vector<T> fAccumulatedBiasGradient;
    void _resetAccumulatedGradients()
    {
        for(auto&& j = 0; j < fOutputSize; j++)
        {
            fAccumulatedBiasGradient[j] = 0;
            for(auto&& i = 0; i < fInputSize; i++)
                fAccumulatedWeightGradient[j][i] = 0;
        }
    }
};
}

#endif
