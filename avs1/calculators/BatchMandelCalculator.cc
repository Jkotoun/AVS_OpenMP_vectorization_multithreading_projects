/**
 * @file BatchMandelCalculator.cc
 * @author Josef Kotoun <xkotou06@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over small batches
 * @date 30.10.2022
 */

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <stdlib.h>
#include <stdexcept>

#include "BatchMandelCalculator.h"

BatchMandelCalculator::BatchMandelCalculator(unsigned matrixBaseSize, unsigned limit) : BaseMandelCalculator(matrixBaseSize, limit, "BatchMandelCalculator")
{
	points_iterations_to_out = (int *)(aligned_alloc(64, height * width * sizeof(int)));
	points_imag = (float *)(aligned_alloc(64, (height / 2) * width * sizeof(float)));
	points_real = (float *)(aligned_alloc(64, (height / 2) * width * sizeof(float)));

	for (int i = 0; i < height / 2; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int index = i * width + j;
			points_real[index] = x_start + j * dx;
			points_imag[index] = y_start + i * dy;
			points_iterations_to_out[index] = limit;
		}
	}
}

BatchMandelCalculator::~BatchMandelCalculator()
{
	free(points_iterations_to_out);
	free(points_imag);
	free(points_real);

	points_imag = nullptr;
	points_real = nullptr;
	points_iterations_to_out = nullptr;
}

int *BatchMandelCalculator::calculateMandelbrot()
{
	const int batch_size = 64;
	int *points_iterations_to_out_cpy = points_iterations_to_out;
	float *points_imag_cpy = points_imag;
	float *points_real_cpy = points_real;

	for (int i = 0; i < (height / 2); i++)
	{
		for (int batch_index = 0; batch_index * batch_size < width; batch_index++)
		{
			int points_done = 0;
			int current_batch_size = batch_size;
			if (batch_index * batch_size + batch_size > width)
			{
				current_batch_size = width - current_batch_size * batch_index;
			}
			for (int j = 0; j < limit && points_done < current_batch_size; j++)
			{
				#pragma omp simd reduction(+ : points_done) aligned(points_imag_cpy : 64, points_real_cpy : 64, points_iterations_to_out_cpy : 64)
				for (int k = batch_size * batch_index; k < batch_size * batch_index + current_batch_size; k++)
				{
					float imag = points_imag_cpy[i * width + k];
					float real = points_real_cpy[i * width + k];
					float real2 = real * real;
					float imag2 = imag * imag;
					//when abs val > 2, set iteration number in array (in next iterations, real and imag are set to zero)

					points_iterations_to_out_cpy[i * width + k] = ((real2 + imag2) > 4.0f) ? j : points_iterations_to_out_cpy[i * width + k];
					if ((real2 + imag2) > 4.0f)
					{
						points_done = points_done + 1;
					}
					//set point real and imaginary part to zero after its abs value is greater than 2 - no need to calc more iterations
					points_imag_cpy[i * width + k] = (real == 0 || (real2 + imag2) > 4.0f) ? 0 : 2.0f * real * imag + y_start + i * dy;
					points_real_cpy[i * width + k] = (imag == 0 || (real2 + imag2) > 4.0f) ? 0 : real2 - imag2 + x_start + k * dx;
				}
			}
		}
	}
	// symmetry
	for (int i = 0; i < height / 2; i++)
	{
		for (int j = 0; j < width; j++)
		{
			points_iterations_to_out[(height - i - 1) * width + j] = points_iterations_to_out[i * width + j];
		}
	}
	return points_iterations_to_out;
}