#include "filter_impl.h"

#include <iostream>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <chrono>
#include <thread>
#include "logo.h"

struct rgb {
    uint8_t r, g, b;
};

struct lab {
    double l, a, b;
};


// Function to convert RGB to XYZ
static void rgb2xyz(const rgb& in, double& x, double& y, double& z) {
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
    z = r * 0.0193 + g * 0.1192 + b * 0.9505;
}

// Function to convert XYZ to Lab
static void xyz2lab(double x, double y, double z, lab& out) {
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

// Function to convert RGB to Lab
static void rgb2lab(const rgb& in, lab& out) {
    double x, y, z;
    rgb2xyz(in, x, y, z);
    xyz2lab(x, y, z, out);
}

float lab_distance(lab image1, lab image2) {
    float deltaL = image1.l - image2.l;
    float deltaA = image1.a - image2.a;
    float deltaB = image1.b - image2.b;

    return sqrt(deltaL * deltaL + deltaA * deltaA + deltaB * deltaB);
}

void rgb_to_grayscale(uint8_t* buffer, int width, int height, int stride) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            uint8_t* pixel = &buffer[y * stride + x * 3];
            uint8_t r = pixel[0];
            uint8_t g = pixel[1];
            uint8_t b = pixel[2];
            uint8_t gray = static_cast<uint8_t>(0.299 * r + 0.587 * g + 0.114 * b);
            pixel[0] = gray;
            pixel[1] = gray;
            pixel[2] = gray;
        }
    }
}

void erode(uint8_t* buffer, int width, int height, int stride, int pixel_stride) {
    uint8_t* temp = new uint8_t[width * height];

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            uint8_t min_pixel = 255;
            for (int j = -1; j <= 1; ++j) {
                for (int i = -1; i <= 1; ++i) {
                    int nx = x + i;
                    int ny = y + j;
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        uint8_t pixel = buffer[ny * stride + nx * pixel_stride];
                        min_pixel = std::min(min_pixel, pixel);
                    }
                }
            }
            temp[y * width + x] = min_pixel;
        }
    }

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            buffer[y * stride + x * pixel_stride] = temp[y * width + x];
        }
    }

    delete[] temp;
}

void dilate(uint8_t* buffer, int width, int height, int stride, int pixel_stride) {
    uint8_t* temp = new uint8_t[width * height];

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            uint8_t max_pixel = 0;
            for (int j = -1; j <= 1; ++j) {
                for (int i = -1; i <= 1; ++i) {
                    int nx = x + i;
                    int ny = y + j;
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        uint8_t pixel = buffer[ny * stride + nx * pixel_stride];
                        max_pixel = std::max(max_pixel, pixel);
                    }
                }
            }
            temp[y * width + x] = max_pixel;
        }
    }

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            buffer[y * stride + x * pixel_stride] = temp[y * width + x];
        }
    }

    delete[] temp;
}

uint8_t* rgb_to_grayscale(uint8_t* buffer, int width, int height, int stride, int pixel_stride) {

    // MEMORY ALLOCATION. DO NOT FORGET TO FREE
    uint8_t* grayscale = (uint8_t*)malloc(width * height);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * stride + x * pixel_stride;

            uint8_t r = buffer[idx];
            uint8_t g = buffer[idx + 1];
            uint8_t b = buffer[idx + 2];

            grayscale[y * width + x] =  (r + g + b) / 3;
        }
    }

    return grayscale;
}

void grayscale_to_rgb(uint8_t* buffer_gray, uint8_t* buffer_rgb, int width, int height, int stride, int pixel_stride) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx_gray = y * width + x;
            int idx_rgb = y * width * 3 + x * 3;

            buffer_rgb[idx_rgb] = buffer_gray[idx_gray];     // R
            buffer_rgb[idx_rgb + 1] = buffer_gray[idx_gray]; // G
            buffer_rgb[idx_rgb + 2] = buffer_gray[idx_gray]; // B
        }
    }
}



extern "C" {
    void filter_impl(uint8_t* buffer, int width, int height, int stride, int pixel_stride)
    {
	/*
        for (int y = 0; y < height; ++y)
        {
            rgb* lineptr = (rgb*) (buffer + y * stride);
            for (int x = 0; x < width; ++x)
            {
                lineptr[x].r = 0; // Back out red component

                if (x < logo_width && y < logo_height)
                {
                    float alpha = logo_data[y * logo_width + x] / 255.f;
                    lineptr[x].g = uint8_t(alpha * lineptr[x].g + (1-alpha) * 255);
                    lineptr[x].b = uint8_t(alpha * lineptr[x].b + (1-alpha) * 255);

                }
            }
        }
	*/

	uint8_t* buffer_gray = rgb_to_grayscale(buffer, width, height, stride, pixel_stride);
	//erode(buffer_gray, width, height, stride, pixel_stride);
	//dilate(buffer_gray, width, height, stride, pixel_stride);
	grayscale_to_rgb(buffer_gray, buffer, width, height, stride, pixel_stride);
	free(buffer_gray);

	

        // You can fake a long-time process with sleep
        {
            using namespace std::chrono_literals;
            //std::this_thread::sleep_for(100ms);
        }
    }
}
