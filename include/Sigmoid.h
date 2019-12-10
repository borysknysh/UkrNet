#ifndef SIGMOID_H
#define SIGMOID_H

#include <cmath>
#include <vector>

namespace UkrNet
{
    template<class T>
    class Sigmoid
    {
    public:
        
        std::vector<T> Forward(const std::vector<T>& input)
        {
            fInput = input;
            std::vector<T> output(input.size(),0);
            for(auto&& i = 0; i < (int)output.size(); i++)
            {
                output[i] = 1 / ( 1 + std::exp(-fInput[i]) );
            }
            fOutput = output;
            return std::move(output);
        }
        std::vector<T> Backward(const std::vector<T>& inputGradients)
        {
            auto&& gradientWrtInput = fOutput;
            for(auto&& i = 0; i < (int)gradientWrtInput.size(); i++)
            {
                gradientWrtInput[i] = fOutput[i] * (1 - fOutput[i]) * inputGradients[i];
            }
            
            return std::move(gradientWrtInput);
        }
    private:
        std::vector<T> fInput;
        std::vector<T> fOutput;
    };
}
#endif
