#ifndef SOFTMAX_H
#define SOFTMAX_H

#include <climits>
#include <vector>
#include <cmath>
#include <cfloat>

namespace UkrNet{
    template <class T>
    class Softmax
    {
    public:
        std::vector<T> Forward(const std::vector<T>& input)
        {
            auto&& maxTemp = -FLT_MAX;
            auto&& sum = 0.;
            for(auto el : input)
            {
                if(el > maxTemp)
                    maxTemp = el;
            }
            for(auto el : input)
            {
                sum += std::exp(el-maxTemp);
            }
//             std::cout << sum << '\n';
            std::vector<T> output(input.size(), 0);
            for(auto i = 0; i < int(output.size()); i++)
            {
                output[i] = std::exp(input[i]-maxTemp)/sum;
            }
            
            fOutput = output;
            return std::move(output);
        }
        
        std::vector<T> Backward(const std::vector<T>& inputGrads)
        {
            std::vector<T> outputGrads(inputGrads.size(), 0);
            for(auto j = 0; j < int(inputGrads.size()); j++)
                for(auto i = 0; i < int(inputGrads.size()); i++)
                {
                    auto&& deltaij = (i == j)?1.:0.;
                    outputGrads[j] += inputGrads[i] * (deltaij - inputGrads[i]);
                }

            return std::move(outputGrads);
        }
    private:
        std::vector<T> fOutput;
        
    };
}

#endif
