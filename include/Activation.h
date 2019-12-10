#ifndef ACTIVATION_H
#define ACTIVATION_H
// ReLU activation
namespace UkrNet
{
    template <class T>
    class Activation
    {
    public:
        std::vector<T> Forward(const std::vector<T>& input)
        {
            std::vector<T> output(input.size(),0);
            for(auto&& i = 0; i < (int)output.size(); i++)
            {
                output[i] = (input[i]<0.)?0.:input[i];
            }
            fOutput = output;
            return std::move(output);
        }
        std::vector<T> Backward(const std::vector<T>& inputGradients)
        {
            auto&& gradientWrtInput = fOutput;
            for(auto&& i = 0; i < (int)gradientWrtInput.size(); i++)
            {
                auto&& temp = (gradientWrtInput[i] > 0)?1.:0.;
                gradientWrtInput[i] = temp * inputGradients[i];
            }
            
            return std::move(gradientWrtInput);
        }
    private:
        std::vector<T> fOutput;
    };
}
#endif
