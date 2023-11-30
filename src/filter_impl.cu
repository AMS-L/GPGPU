#include "filter_impl.h"

#include <iostream>
#include <cassert>
#include <chrono>
#include <thread>
#include <cstdio>
#include "logo.h"

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)

bool first = true;
uint8_t* background = nullptr;


template <typename T>
void check(T err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::fprintf(stderr, "CUDA Runtime Error at: %s: %d\n", file, line);
        std::fprintf(stderr, "%s %s\n", cudaGetErrorString(err), func);
        // We don't exit when we encounter CUDA errors in this example.
        std::exit(EXIT_FAILURE);
    }
}

struct rgb {
    uint8_t r, g, b;
};

struct lab {
    uint8_t l, a, b;
};


__device__ void rgb2xyz_kernel(const rgb& in, double& x, double& y, double& z) {
    auto r = in.r / 255.0;
    auto g = in.g / 255.0;
    auto b = in.b / 255.0;

    if (r > 0.04045) r = __powf((r + 0.055) / 1.055, 2.4);
    else r = r / 12.92;
    if (g > 0.04045) g = __powf((g + 0.055) / 1.055, 2.4);
    else g = g / 12.92;
    if (b > 0.04045) b = __powf((b + 0.055) / 1.055, 2.4);
    else b = b / 12.92;

    r *= 100.0;
    g *= 100.0;
    b *= 100.0;

    // Observer. = 2°, Illuminant = D65
    x = r * 0.4124 + g * 0.3576 + b * 0.1805;
    y = r * 0.2126 + g * 0.7152 + b * 0.0722;
    z = r * 0.0193 + g * 0.1192 + b * 0.9505;
}

__device__ void xyz2lab_kernel(double x, double y, double z, lab& out) {
    x /= 95.047; // Observer= 2°, Illuminant= D65
    y /= 100.000;
    z /= 108.883;

    if (x > 0.008856) x = __powf(x, 1.0 / 3.0);
    else x = (7.787 * x) + (16.0 / 116.0);
    if (y > 0.008856) y = __powf(y, 1.0 / 3.0);
    else y = (7.787 * y) + (16.0 / 116.0);
    if (z > 0.008856) z = __powf(z, 1.0 / 3.0);
    else z = (7.787 * z) + (16.0 / 116.0);

    out.l = (116.0 * y) - 16.0;
    out.a = 500.0 * (x - y);
    out.b = 200.0 * (y - z);
}



__global__ void residual_filter_kernel(uint8_t* dBuffer, uint8_t* background, int width, int height, int stride, int pixel_stride) {
    // Define shared memory for buffer and background pixels
    __shared__ rgb buffer_shared[32][32];
    __shared__ rgb background_shared[32][32];

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < height && j < width) {
        // Get the pixel from the buffer and background
	int idx = i * stride + j * pixel_stride;

        // Load buffer and background pixels into shared memory
	buffer_shared[threadIdx.y][threadIdx.x].r = dBuffer[idx];
	buffer_shared[threadIdx.y][threadIdx.x].g = dBuffer[idx + 1];
	buffer_shared[threadIdx.y][threadIdx.x].b = dBuffer[idx + 2];

	background_shared[threadIdx.y][threadIdx.x].r = background[idx];
	background_shared[threadIdx.y][threadIdx.x].g = background[idx + 1];
	background_shared[threadIdx.y][threadIdx.x].b = background[idx + 2];
    }

    // Synchronize to make sure the pixels are loaded before starting the computation
    __syncthreads();

    if (i < height && j < width) {
        // Convert RGB to Lab using shared memory
	int idx = i * stride + j * pixel_stride;
        lab buffer_lab, background_lab;
    	double x, y, z;
    	rgb2xyz_kernel(buffer_shared[threadIdx.y][threadIdx.x], x, y, z);
    	xyz2lab_kernel(x, y, z, buffer_lab);

    	rgb2xyz_kernel(background_shared[threadIdx.y][threadIdx.x], x, y, z);
    	xyz2lab_kernel(x, y, z, background_lab);

        float deltaL = buffer_lab.l - background_lab.l;
        float deltaA = buffer_lab.a - background_lab.a;
        float deltaB = buffer_lab.b - background_lab.b;

        float distance = __fsqrt_rd(deltaL * deltaL + deltaA * deltaA + deltaB * deltaB);
        // Calculate the lab distance

        if (distance < 10) {
    	    dBuffer[idx] = 0;     // Set red to 0
    	    dBuffer[idx + 1] = 0; // Set green to 0
    	    dBuffer[idx + 2] = 0; // Set blue to 0
        }
    }
}

/*
__global__ void residual_filter_kernel(uint8_t* dBuffer, uint8_t* background, int width, int height, int stride, int pixel_stride) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < height && j < width) {
        // Get the pixel from the buffer and background
	int index = i * stride + j * pixel_stride;

        rgb buffer_pixel;
	buffer_pixel.r = dBuffer[index];
	buffer_pixel.g = dBuffer[index + 1];
	buffer_pixel.b = dBuffer[index + 2];

        rgb background_pixel;
	background_pixel.r = dBuffer[index];
	background_pixel.g = dBuffer[index + 1];
	background_pixel.b = dBuffer[index + 2];

        // Convert RGB to Lab
        lab buffer_lab, background_lab;
    	double x, y, z;
    	rgb2xyz_kernel(buffer_pixel, x, y, z);
    	xyz2lab_kernel(x, y, z, buffer_lab);

    	rgb2xyz_kernel(background_pixel, x, y, z);
    	xyz2lab_kernel(x, y, z, background_lab);

        float deltaL = buffer_lab.l - background_lab.l;
        float deltaA = buffer_lab.a - background_lab.a;
        float deltaB = buffer_lab.b - background_lab.b;

        float distance = sqrt(deltaL * deltaL + deltaA * deltaA + deltaB * deltaB);
        // Calculate the lab distance

        if (distance < 10) {
    	    dBuffer[index] = 0;     // Set red to 0
    	    dBuffer[index + 1] = 0; // Set green to 0
    	    dBuffer[index + 2] = 0; // Set blue to 0
        }
    }
}
*/

__global__ void rgb_to_grayscale_kernel(uint8_t* buffer, uint8_t* grayscale, int width, int height, int stride, int pixel_stride) {
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= width || y >= height)
        return;

    int idx = y * stride + x * pixel_stride;

    uint8_t r = buffer[idx];
    uint8_t g = buffer[idx + 1];
    uint8_t b = buffer[idx + 2];

    grayscale[y * width + x] =  (r + g + b) / 3;
}

__global__ void grayscale_to_rgb_kernel(uint8_t* buffer_gray, uint8_t* buffer_rgb, int width, int height, int stride, int pixel_stride) {
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= width || y >= height)
        return;

    int idx_gray = y * width + x;
    int idx_rgb = y * stride + x * pixel_stride;
    /*buffer_rgb[idx_rgb] = buffer_gray[idx_gray]; // R
    buffer_rgb[idx_rgb + 1] = buffer_gray[idx_gray]; // G
    buffer_rgb[idx_rgb + 2] = buffer_gray[idx_gray]; // B*/
    uint8_t pixel_value = buffer_gray[idx_gray];

    if (pixel_value != 0) {
        buffer_rgb[idx_rgb] = 0.5 * 255;
        buffer_rgb[idx_rgb + 1] = 0;
        buffer_rgb[idx_rgb + 2] = 0;
    }
}

__global__ void erode_kernel(uint8_t* dBuffer, uint8_t* temp, int width, int height, int stride, int pixel_stride) {
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (y < height && x < width) {
        uint8_t min_pixel = 255;
        for (int j = -1; j <= 1; ++j) {
            for (int i = -1; i <= 1; ++i) {
                int nx = x + i;
                int ny = y + j;
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    uint8_t pixel = dBuffer[ny * stride + nx * pixel_stride];
                    min_pixel = min(min_pixel, pixel);
                }
            }
        }
        temp[y * stride + x * pixel_stride] = min_pixel;
    }
}

__global__ void dilate_kernel(uint8_t* buffer_gray, uint8_t* temp, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        uint8_t max_pixel = 0;
        for (int j = -1; j <= 1; ++j) {
            for (int i = -1; i <= 1; ++i) {
                int nx = x + i;
                int ny = y + j;
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    uint8_t pixel = buffer_gray[ny * width + nx];
                    max_pixel = max(max_pixel, pixel);
                }
            }
        }
        temp[y * width + x] = max_pixel;
    }
}

void dilate(uint8_t* buffer_gray, int width, int height) {
    uint8_t* temp;
    cudaMalloc(&temp, width * height * sizeof(uint8_t));

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    dilate_kernel<<<gridSize, blockSize>>>(buffer_gray, temp, width, height);

    cudaMemcpy(buffer_gray, temp, width * height * sizeof(uint8_t), cudaMemcpyDeviceToDevice);

    cudaFree(temp);
}

__global__ void erode_kernel(uint8_t* buffer_gray, uint8_t* temp, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        uint8_t min_pixel = 255;
        for (int j = -1; j <= 1; ++j) {
            for (int i = -1; i <= 1; ++i) {
                int nx = x + i;
                int ny = y + j;
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    uint8_t pixel = buffer_gray[ny * width + nx];
                    min_pixel = min(min_pixel, pixel);
                }
            }
        }
        temp[y * width + x] = min_pixel;
    }
}

void erode(uint8_t* buffer_gray, int width, int height) {
    uint8_t* temp;
    cudaMalloc(&temp, width * height * sizeof(uint8_t));

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    erode_kernel<<<gridSize, blockSize>>>(buffer_gray, temp, width, height);

    cudaMemcpy(buffer_gray, temp, width * height * sizeof(uint8_t), cudaMemcpyDeviceToDevice);

    cudaFree(temp);
}

__global__ void hysteresis_thresholding_kernel(uint8_t* buffer_gray, uint8_t* temp_buffer, int width, int height, int lowThreshold, int highThreshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        uint8_t pixel_value = buffer_gray[idx];

        if (pixel_value >= highThreshold) {
            temp_buffer[idx] = 255;
        } else if (pixel_value < lowThreshold) {
            temp_buffer[idx] = 0;
        } else {
            bool hasStrongNeighbor = false;
            for (int j = -1; j <= 1; ++j) {
                for (int i = -1; i <= 1; ++i) {
                    int nx = x + i;
                    int ny = y + j;
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        int neighbor_idx = ny * width + nx;
                        if (buffer_gray[neighbor_idx] == 255) {
                            hasStrongNeighbor = true;
                            break;
                        }
                    }
                }
                if (hasStrongNeighbor) {
                    break;
                }
            }
            temp_buffer[idx] = hasStrongNeighbor ? 128 : 0;
        }
    }
}

void hysteresis_thresholding(uint8_t* buffer_gray, int width, int height, int lowThreshold, int highThreshold) {
    uint8_t* temp_buffer;
    cudaMalloc(&temp_buffer, width * height * sizeof(uint8_t));

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    hysteresis_thresholding_kernel<<<gridSize, blockSize>>>(buffer_gray, temp_buffer, width, height, lowThreshold, highThreshold);

    cudaMemcpy(buffer_gray, temp_buffer, width * height * sizeof(uint8_t), cudaMemcpyDeviceToDevice);

    cudaFree(temp_buffer);
}

int num_frame = 0;

__global__ void mean_background_kernel(uint8_t* background, uint8_t* buffer, int width, int height,int stride, int pixel_stride, int num_frame) {

    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= width || y >= height)
        return;

    int idx_rgb = y * stride + x * pixel_stride;

    background[idx_rgb] = (background[idx_rgb] * num_frame + buffer[idx_rgb]) / (num_frame + 1);
    background[idx_rgb + 1] = (background[idx_rgb + 1] * num_frame + buffer[idx_rgb + 1]) / (num_frame + 1);
    background[idx_rgb + 2] = (background[idx_rgb + 2] * num_frame + buffer[idx_rgb + 2]) / (num_frame + 1);
    num_frame++;
}


extern "C" {
    void filter_impl(uint8_t* src_buffer, int width, int height, int src_stride, int pixel_stride)
    {

        dim3 blockSize(16,16);
        dim3 gridSize((width + (blockSize.x - 1)) / blockSize.x, (height + (blockSize.y - 1)) / blockSize.y);


        assert(sizeof(rgb) == pixel_stride);
        uint8_t* dBuffer;
        size_t pitch;

        cudaError_t err;

        CHECK_CUDA_ERROR(cudaMallocPitch(&dBuffer, &pitch, width * sizeof(rgb), height));
        CHECK_CUDA_ERROR(cudaMemcpy2D(dBuffer, pitch, src_buffer, src_stride, width * sizeof(rgb), height, cudaMemcpyDefault));


	if (first) {
            CHECK_CUDA_ERROR(cudaMallocPitch(&background, &pitch, width * sizeof(rgb), height));
            CHECK_CUDA_ERROR(cudaMemcpy2D(background, pitch, src_buffer, src_stride, width * sizeof(rgb), height, cudaMemcpyDefault));

	    num_frame = 1;
            first = false;
	}
	else
	    mean_background_kernel<<<gridSize, blockSize>>>(background, dBuffer, width, height, pitch, pixel_stride, num_frame);

	uint8_t* final_buffer;
        CHECK_CUDA_ERROR(cudaMallocPitch(&final_buffer, &pitch, width * sizeof(rgb), height));
        CHECK_CUDA_ERROR(cudaMemcpy2D(final_buffer, pitch, src_buffer, src_stride, width * sizeof(rgb), height, cudaMemcpyDefault));

	residual_filter_kernel<<<gridSize, blockSize>>>(dBuffer, background, width, height, pitch, pixel_stride);
        uint8_t* grayscale;
        CHECK_CUDA_ERROR(cudaMalloc(&grayscale, width * height));

        rgb_to_grayscale_kernel<<<gridSize, blockSize>>>(dBuffer, grayscale, width, height, pitch, pixel_stride);

	erode(grayscale, width, height);
	dilate(grayscale, width, height);
	int lowThreshold = 4;
       	int highThreshold = 30;
	hysteresis_thresholding(grayscale, width, height, lowThreshold, highThreshold);

	grayscale_to_rgb_kernel<<<gridSize, blockSize>>>(grayscale, final_buffer, width, height, pitch, pixel_stride);

        err = cudaMemcpy2D(src_buffer, src_stride, final_buffer, pitch, width * sizeof(rgb), height, cudaMemcpyDefault);
        CHECK_CUDA_ERROR(err);

        cudaFree(dBuffer);
        cudaFree(grayscale);

        err = cudaDeviceSynchronize();
        CHECK_CUDA_ERROR(err);
    }
}
