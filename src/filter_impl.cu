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

    if (r > 0.04045) r = pow((r + 0.055) / 1.055, 2.4);
    else r = r / 12.92;
    if (g > 0.04045) g = pow((g + 0.055) / 1.055, 2.4);
    else g = g / 12.92;
    if (b > 0.04045) b = pow((b + 0.055) / 1.055, 2.4);
    else b = b / 12.92;

    r *= 100.0;
    g *= 100.0;
    b *= 100.0;

    // Observer. = 2°, Illuminant = D65
    x = r * 0.4124 + g * 0.3576 + b * 0.1805;
    y = r * 0.2126 + g * 0.7152 + b * 0.0722;
    // Your lab_distance function code goes here
    z = r * 0.0193 + g * 0.1192 + b * 0.9505;
}

__device__ void xyz2lab_kernel(double x, double y, double z, lab& out) {
    x /= 95.047; // Observer= 2°, Illuminant= D65
    y /= 100.000;
    z /= 108.883;

    if (x > 0.008856) x = pow(x, 1.0 / 3.0);
    else x = (7.787 * x) + (16.0 / 116.0);
    if (y > 0.008856) y = pow(y, 1.0 / 3.0);
    else y = (7.787 * y) + (16.0 / 116.0);
    if (z > 0.008856) z = pow(z, 1.0 / 3.0);
    else z = (7.787 * z) + (16.0 / 116.0);

    out.l = (116.0 * y) - 16.0;
    out.a = 500.0 * (x - y);
    out.b = 200.0 * (y - z);
}

__device__ void rgb2lab_kernel(const rgb& in, lab& out) {
    double x, y, z;
    rgb2xyz_kernel(in, x, y, z);
    xyz2lab_kernel(x, y, z, out);
}


__device__ float lab_distance_kernel(lab image1, lab image2) {
    float deltaL = image1.l - image2.l;
    float deltaA = image1.a - image2.a;
    float deltaB = image1.b - image2.b;

    return sqrt(deltaL * deltaL + deltaA * deltaA + deltaB * deltaB);
}

__device__ void get_rgb(uint8_t* buffer, rgb& pixel, int i, int j, int stride, int pixel_stride) {
    int index = i * stride + j * pixel_stride;
    pixel.r = buffer[index];
    pixel.g = buffer[index + 1];
    pixel.b = buffer[index + 2];
}

__device__ void set_black(uint8_t* dBuffer, int i, int j, int stride, int pixel_stride) {
    int index = i * stride + j * pixel_stride;
    dBuffer[index] = 0;     // Set red to 0
    dBuffer[index + 1] = 0; // Set green to 0
    dBuffer[index + 2] = 0; // Set blue to 0
}

__global__ void residual_filter_kernel(uint8_t* dBuffer, uint8_t* background, int width, int height, int stride, int pixel_stride) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < height && j < width) {
        // Get the pixel from the buffer and background
        rgb buffer_pixel;
	get_rgb(dBuffer, buffer_pixel, i, j, stride, pixel_stride);
        rgb background_pixel;
        get_rgb(background, background_pixel, i, j, stride, pixel_stride);

        // Convert RGB to Lab
        lab buffer_lab, background_lab;
        rgb2lab_kernel(buffer_pixel, buffer_lab);
        rgb2lab_kernel(background_pixel, background_lab);

        // Calculate the lab distance
        float distance = lab_distance_kernel(buffer_lab, background_lab);

        if (distance < 10) {
            set_black(dBuffer, i, j, stride, pixel_stride);
        }
    }
}

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
    buffer_rgb[idx_rgb] = buffer_gray[idx_gray]; // R
    buffer_rgb[idx_rgb + 1] = buffer_gray[idx_gray]; // G
    buffer_rgb[idx_rgb + 2] = buffer_gray[idx_gray]; // B
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

	if (first) {
            err = cudaMallocPitch(&background, &pitch, width * sizeof(rgb), height);
            CHECK_CUDA_ERROR(err);
            err = cudaMemcpy2D(background, pitch, src_buffer, src_stride, width * sizeof(rgb), height, cudaMemcpyDefault);
            CHECK_CUDA_ERROR(err);
            first = false;
	}

        err = cudaMallocPitch(&dBuffer, &pitch, width * sizeof(rgb), height);
        CHECK_CUDA_ERROR(err);
        err = cudaMemcpy2D(dBuffer, pitch, src_buffer, src_stride, width * sizeof(rgb), height, cudaMemcpyDefault);
        CHECK_CUDA_ERROR(err);


	residual_filter_kernel<<<gridSize, blockSize>>>(dBuffer, background, width, height, pitch, pixel_stride);
        uint8_t* grayscale;
        err = cudaMalloc(&grayscale, width * height);
        CHECK_CUDA_ERROR(err);

        rgb_to_grayscale_kernel<<<gridSize, blockSize>>>(dBuffer, grayscale, width, height, pitch, pixel_stride);

	grayscale_to_rgb_kernel<<<gridSize, blockSize>>>(grayscale, dBuffer, width, height, pitch, pixel_stride);

        err = cudaMemcpy2D(src_buffer, src_stride, dBuffer, pitch, width * sizeof(rgb), height, cudaMemcpyDefault);
        CHECK_CUDA_ERROR(err);

        cudaFree(dBuffer);
        cudaFree(grayscale);

        err = cudaDeviceSynchronize();
        CHECK_CUDA_ERROR(err);
    }
}
