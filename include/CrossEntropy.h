#ifndef CROSSENTROPY_H
#define CROSSENTROPY_H

#include <cmath>

namespace UkrNet
{
    template <class T>
    class CrossEntropy
    {
    public:
        T Forward(const std::vector<T>& actualDistr, const std::vector<T>& predictedDistr)
        {
            fActualDistr = actualDistr;
            fPredictedDistr = predictedDistr;
            auto&& res = 0.;
            for(auto&& i = 0; i < int(fActualDistr.size()); i++)
            {
//                 fPredictedDistr[i] += 1e-3;
                res += -fActualDistr[i] * std::log(fPredictedDistr[i]);
//                 std::cout << fActualDistr[i] << "  " << fPredictedDistr[i] << "  "<< res << '\n';
            }
            return res;
        }
        std::vector<T> Backward()
        {
            std::vector<T> res(fActualDistr.size(), 0);
            for(auto&& i = 0; i < int(res.size()); i++)
            {
//                 fPredictedDistr[i] += 1e-3;
                res[i] = -1/(fPredictedDistr[i]) * fActualDistr[i];
            }
            return std::move(res);
        }
    private:
        std::vector<T> fActualDistr;
        std::vector<T> fPredictedDistr;
    };
}

#endif
