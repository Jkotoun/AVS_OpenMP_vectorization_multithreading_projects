/**
 * @file LineMandelCalculator.h
 * @author Josef Kotoun <xkotou06@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over lines
 * @date 29. 10. 2022
 */

#include <BaseMandelCalculator.h>

class LineMandelCalculator : public BaseMandelCalculator
{
public:
    LineMandelCalculator(unsigned matrixBaseSize, unsigned limit);
    ~LineMandelCalculator();
    int *calculateMandelbrot();

private:
    int*points_iterations_to_out;
    float* points_imag;
    float* points_real;
};