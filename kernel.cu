#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <string>
#include <omp.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

__constant__ float convolutionKernelStore[256];

__global__ void convolve(unsigned char* source, int width, int height, int paddingX, int paddingY, size_t kOffset, int kWidth, int kHeight, unsigned char* destination)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	float sum = 0.0;
	int   pWidth = kWidth / 2;
	int   pHeight = kHeight / 2;

	if (x >= pWidth + paddingX &&
		y >= pHeight + paddingY &&
		x < (blockDim.x * gridDim.x) - pWidth - paddingX &&
		y < (blockDim.y * gridDim.y) - pHeight - paddingY)
	{
		for (int j = -pHeight; j <= pHeight; j++)
		{
			for (int i = -pWidth; i <= pWidth; i++)
			{
				int ki = (i + pWidth);
				int kj = (j + pHeight);
				float w = convolutionKernelStore[(kj * kWidth) + ki + kOffset];


				sum += w * float(source[((y + j) * width) + (x + i)]);
			}
		}
	}

	destination[(y * width) + x] = (unsigned char)sum;
}

__global__ void pythagoras(unsigned char* a, unsigned char* b, unsigned char* c)
{
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    float af = float(a[idx]);
    float bf = float(b[idx]);

    c[idx] = (unsigned char)sqrtf(af * af + bf * bf);
}

unsigned char* createImageBuffer(unsigned int bytes, unsigned char** devicePtr)
{
    unsigned char* ptr = NULL;
    cudaSetDeviceFlags(cudaDeviceMapHost);
    cudaHostAlloc(&ptr, bytes, cudaHostAllocMapped);
    cudaHostGetDevicePointer(devicePtr, ptr, 0);
    return ptr;
}

int main(int argc, char* argv[])
{
    int num_gpus = 0;   // number of CUDA GPUs
    int filterSel = 1;

    printf("%s Starting...\n\n", argv[0]);

    /////////////////////////////////////////////////////////////////
    // determine the number of CUDA capable GPUs
    //
    cudaGetDeviceCount(&num_gpus);

    if (num_gpus < 1)
    {
        printf("no CUDA capable devices were detected\n");
        return 1;
    }

    /////////////////////////////////////////////////////////////////
    // display CPU and GPU configuration
    //
    printf("number of host CPUs:\t%d\n", omp_get_num_procs());
    printf("number of CUDA devices:\t%d\n", num_gpus);

    for (int i = 0; i < num_gpus; i++)
    {
        cudaDeviceProp dprop;
        cudaGetDeviceProperties(&dprop, i);
        printf("   %d: %s\n", i, dprop.name);
    }

    printf("---------------------------\n");

    /////////////////////////////////////////////////////////////////
    // initialize data
    //
    cv::VideoCapture camera(0);
    cv::Mat          frame;
    if (!camera.isOpened())
        return -1;

    cv::namedWindow("Source");
    cv::namedWindow("GrayScale");
    cv::namedWindow("Filtered");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //filtro gausiano
	const float gaussianKernel5x5[25] =
	{
		2.f / 159.f, 4.f / 159.f, 5.f / 159.f, 4.f / 159.f, 2.f / 159.f,
		4.f / 159.f, 9.f / 159.f, 12.f / 159.f, 9.f / 159.f, 4.f / 159.f,
		5.f / 159.f, 12.f / 159.f, 15.f / 159.f, 12.f / 159.f, 5.f / 159.f,
		4.f / 159.f, 9.f / 159.f, 12.f / 159.f, 9.f / 159.f, 4.f / 159.f,
		2.f / 159.f, 4.f / 159.f, 5.f / 159.f, 4.f / 159.f, 2.f / 159.f,
	};
	cudaMemcpyToSymbol(convolutionKernelStore, gaussianKernel5x5, sizeof(gaussianKernel5x5), 0);
	const size_t gaussianKernel5x5Offset = 0;

	const float sobelGradientX[9] =
	{
		-1.f, 0.f, 1.f,
		-2.f, 0.f, 2.f,
		-1.f, 0.f, 1.f,
	};
	const float sobelGradientY[9] =
	{
		1.f, 2.f, 1.f,
		0.f, 0.f, 0.f,
		-1.f, -2.f, -1.f,
	};
	cudaMemcpyToSymbol(convolutionKernelStore, sobelGradientX, sizeof(sobelGradientX), sizeof(gaussianKernel5x5));
	cudaMemcpyToSymbol(convolutionKernelStore, sobelGradientY, sizeof(sobelGradientY), sizeof(gaussianKernel5x5) + sizeof(sobelGradientX));
	const size_t sobelGradientXOffset = sizeof(gaussianKernel5x5) / sizeof(float);
	const size_t sobelGradientYOffset = sizeof(sobelGradientX) / sizeof(float) + sobelGradientXOffset;

    const float sharpen[9] =
    {
        0.f, -1.f, 0.f,
        -1.f, 5.f, -1.f,
        0.f, -1.f, 0.f,
    };
    cudaMemcpyToSymbol(convolutionKernelStore, sharpen, sizeof(sharpen), sizeof(gaussianKernel5x5) + sizeof(sobelGradientX) + sizeof(sobelGradientY));
    const size_t sharpenoffset = sizeof(sobelGradientY) / sizeof(float) + sobelGradientYOffset /* + sobelGradientXOffset*/;

    // Creando imagenes compartidas CPU/GPU shared images - una para la inicial y otra para el resultado
	camera >> frame;
	unsigned char* grayImage, * filteredImage;
	cv::Mat gray(frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &grayImage));
    cv::Mat filtered(frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &filteredImage));
    
    //Creando dos imagenes temporales
	unsigned char* deviceGradientX, * deviceGradientY;
	cudaMalloc(&deviceGradientX, frame.size().width * frame.size().height);
	cudaMalloc(&deviceGradientY, frame.size().width * frame.size().height);

    //ciclo para capturar imagenes
    while (1)
    {
        // capturar la imagen y almacenar la imagen
        camera >> frame;
        
        cv::cvtColor(frame, gray, 6);
        
        cudaEventRecord(start);
		{
            //  parametros de lanzamiento del kernel
			dim3 cblocks(frame.size().width / 16, frame.size().height / 16);
			dim3 cthreads(16, 16);

			//  parametros del kernel de pitagoras
			dim3 pblocks(frame.size().width * frame.size().height / 256);
			dim3 pthreads(256, 1);

            if (filterSel == 1)
            {
                convolve << <cblocks, cthreads >> > (grayImage, frame.size().width, frame.size().height, 0, 0, gaussianKernel5x5Offset, 5, 5, filteredImage);
            }
            else if (filterSel == 2)
            {
                convolve << <cblocks, cthreads >> > (grayImage, frame.size().width, frame.size().height, 0, 0, sharpenoffset, 3, 3, filteredImage);
            }
            else if (filterSel == 3)
            {
                convolve << <cblocks, cthreads >> > (grayImage, frame.size().width, frame.size().height, 2, 2, sobelGradientXOffset, 3, 3, deviceGradientX);
                convolve << <cblocks, cthreads >> > (grayImage, frame.size().width, frame.size().height, 2, 2, sobelGradientYOffset, 3, 3, deviceGradientY);
                pythagoras << <pblocks, pthreads >> > (deviceGradientX, deviceGradientY, filteredImage);
            }
            cudaThreadSynchronize();
        }
        cudaEventRecord(stop);

        cv::imshow("Source", frame);
        cv::imshow("GrayScale", gray);
        cv::imshow("Filtered", filtered);
 
        if (cv::waitKey(1) == 27)
        {
            break;
        }
        else if (cv::waitKey(1) == 101)
        {
            filterSel = 1;
        }
        else if (cv::waitKey(1) == 115)
        {
            filterSel = 2;
        }
        else if (cv::waitKey(1) == 114)
        {
            filterSel = 3;
        }
    }
}