/**
 * @file LineMandelCalculator.cc
 * @author Josef Kotoun <xkotou06@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over lines
 * @date 29.10.2022
 */
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <stdlib.h>
#include <iostream>
#include "LineMandelCalculator.h"

LineMandelCalculator::LineMandelCalculator(unsigned matrixBaseSize, unsigned limit) : BaseMandelCalculator(matrixBaseSize, limit, "LineMandelCalculator")
{
	points_iterations_to_out = (int *)(aligned_alloc(64, height * width * sizeof(int)));
	points_imag = (float *)(aligned_alloc(64, (height / 2) * width * sizeof(float)));
	points_real = (float *)(aligned_alloc(64, (height / 2)* width * sizeof(float)));


	for (int i = 0; i < height / 2; i++)
	{
		for (int j = 0; j < width; j++)
		{
			points_real[i*width+j] = x_start + j * dx;
			points_imag[i*width+j] = y_start + i * dy;
			points_iterations_to_out[i*width+j] = limit;
		}
	}
}

LineMandelCalculator::~LineMandelCalculator()
{
	free(points_iterations_to_out);
	free(points_imag);
	free(points_real);

	points_imag = nullptr;
	points_real = nullptr;
	points_iterations_to_out = nullptr;
}



int *LineMandelCalculator::calculateMandelbrot()
{
	int* points_iterations_to_out_cpy = points_iterations_to_out;
	float* points_imag_cpy = points_imag;
	float* points_real_cpy = points_real;

	for (int i = 0; i < (height/2); i++)
	{
		int points_done = 0;
		for(int j = 0;j<limit && points_done < width;j++)
		{

			#pragma omp simd reduction(+:points_done) aligned(points_imag_cpy:64, points_real_cpy:64,points_iterations_to_out_cpy:64) 
			for(int k = 0;k<width;k++)
			{
				float imag = points_imag_cpy[i*width +k]; 
				float real = points_real_cpy[i*width +k];  
				float realimag2 = real*real +  imag*imag;
				//when abs val > 2, set iteration number in array (in next iterations, real and imag are set to zero)
				points_iterations_to_out_cpy[i*width + k] = (realimag2 > 4.0f) ? j : points_iterations_to_out_cpy[i*width + k];
				if((realimag2) > 4.0f)
				{
					points_done += 1;
				}
				//set point real and imaginary part to zero after its abs value is greater than 2 - no need to calc more iterations
				points_imag_cpy[i*width + k] = (real == 0 || (realimag2) > 4.0f) ? 0 : 2.0f * real * imag + y_start + i *dy;
				points_real_cpy[i*width + k] = (imag == 0 || (realimag2) > 4.0f) ? 0 : real*real - imag*imag + x_start + k * dx;
			}
		}
	}
	//symmetry
	for(int i = 0;i<height/2;i++)
	{
		for(int j =0;j<width;j++)
		{
			points_iterations_to_out[(height - i - 1)*width+j] = points_iterations_to_out[i*width+j];
		}
	}
	return points_iterations_to_out;
}



