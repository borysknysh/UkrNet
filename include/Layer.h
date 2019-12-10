#ifndef LAYER_H
#define LAYER_H

#include <vector>

namespace UkrNet
{
    template <class T>
    class Layer
    {
    public:
        virtual std::vector<T> Forward(const std::vector<T>& input) = 0;
        virtual std::vector<T> Backward(const std::vector<T>& inputGrad) = 0;
        virtual ~Layer(){}
    };
}

#endif
