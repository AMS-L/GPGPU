#include "filter_impl.h"

#include <cstring>
#include <fstream>
#include <vector>
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

bool first = true;

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


void erode(uint8_t* buffer_gray, int width, int height, int stride, int pixel_stride) {
    uint8_t* temp = new uint8_t[width * height];
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            uint8_t min_pixel = 255;
            for (int j = -1; j <= 1; ++j) {
                for (int i = -1; i <= 1; ++i) {
                    int nx = x + i;
                    int ny = y + j;
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        uint8_t pixel = buffer_gray[ny * stride + nx * pixel_stride];
                        min_pixel = std::min(min_pixel, pixel);
                    }
                }
            }
            temp[y * width + x] = min_pixel;
        }
    }

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            //std::cout << buffer_gray[0];
            //buffer_gray[y * stride + x * pixel_stride] = temp[y * width + x];
        }
    }

    delete[] temp;
}

void dilate(uint8_t* buffer_gray, int width, int height, int stride, int pixel_stride) {
    uint8_t* temp = new uint8_t[height * stride];

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            uint8_t max_pixel = 0;
            for (int j = -1; j <= 1; ++j) {
                for (int i = -1; i <= 1; ++i) {
                    int nx = x + i;
                    int ny = y + j;
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        uint8_t pixel = buffer_gray[ny * stride + nx * pixel_stride];
                        max_pixel = std::max(max_pixel, pixel);
                    }
                }
            }
            temp[y * stride + x * pixel_stride] = max_pixel;
        }
    }

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            buffer_gray[y * stride + x * pixel_stride] = temp[y * width + x];
        }
    }
    delete[] temp;
}

/*
void dilate(uint8_t* buffer, int width, int height, int stride, int pixel_stride) {
    uint8_t* temp = new uint8_t[height * stride * pixel_stride];

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            uint8_t max_pixel[3] = {0, 0, 0}; // Assuming RGB format
            for (int j = -1; j <= 1; ++j) {
                for (int i = -1; i <= 1; ++i) {
                    int nx = x + i;
                    int ny = y + j;
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        int idx = (ny * stride) + (nx * pixel_stride);
                        for (int k = 0; k < 3; ++k) { // Assuming RGB format
                            max_pixel[k] = std::max(max_pixel[k], buffer[idx + k]);
                        }
                    }
                }
            }
            int idx_temp = (y * stride) + (x * pixel_stride);
            for (int k = 0; k < 3; ++k) { // Assuming RGB format
                temp[idx_temp + k] = max_pixel[k];
            }
        }
    }

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            buffer[y * stride + x * pixel_stride] = temp[y * width + x];
        }
    }
    delete[] temp;
}
*/
/*
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
*/


void rgb_to_grayscale(uint8_t* buffer, uint8_t* grayscale, int width, int height, int stride, int pixel_stride) {


    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * stride + x * pixel_stride;

            uint8_t r = buffer[idx];
            uint8_t g = buffer[idx + 1];
            uint8_t b = buffer[idx + 2];

            grayscale[y * width + x] =  (r + g + b) / 3;
        }
    }
}

void grayscale_to_rgb(uint8_t* buffer_gray, uint8_t* buffer_rgb, int width, int height, int stride, int pixel_stride) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx_gray = y * width + x;
            int idx_rgb = y * stride + x * pixel_stride;
            buffer_rgb[idx_rgb] = buffer_gray[idx_gray];     // R
            buffer_rgb[idx_rgb + 1] = buffer_gray[idx_gray]; // G
            buffer_rgb[idx_rgb + 2] = buffer_gray[idx_gray]; // B
        }
    }
}

// Function to get the RGB values of a pixel from a buffer
rgb get_rgb(uint8_t* buffer, int i, int j, int stride, int pixel_stride) {
    rgb pixel;
    int index = i * stride + j * pixel_stride;
    pixel.r = buffer[index];
    pixel.g = buffer[index + 1];
    pixel.b = buffer[index + 2];
    return pixel;
}

// Function to set a pixel in a buffer to black
void set_black(uint8_t* buffer, int i, int j, int stride, int pixel_stride) {
    int index = i * stride + j * pixel_stride;
    buffer[index] = 0;     // Set red to 0
    buffer[index + 1] = 0; // Set green to 0
    buffer[index + 2] = 0; // Set blue to 0
}

uint8_t* background;

extern "C" {
    void filter_impl(uint8_t* buffer, int width, int height, int stride, int pixel_stride)
    {
        if (first) {
            background = new uint8_t[height*stride*pixel_stride];
            memcpy(background, buffer, height*stride);
            first = false;
        }

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                // Get the pixel from the buffer and background
                rgb buffer_pixel = get_rgb(buffer, i, j, stride, pixel_stride);
                rgb background_pixel = get_rgb(background, i, j, stride, pixel_stride);

                // Convert RGB to Lab
                lab buffer_lab, background_lab;
                rgb2lab(buffer_pixel, buffer_lab);
                rgb2lab(background_pixel, background_lab);

                // Calculate the lab distance
                float distance = lab_distance(buffer_lab, background_lab);


                // If the distance is zero, blackout the pixel
                // FIX ME : Trouver un calcul (variance) pour estimer le seuil de la distance
                if (distance < 5) {
                    set_black(buffer, i, j, stride, pixel_stride);
                }
            }
        }

        /*for (int y = 0; y < height; ++y)
        {
            rgb* lineptr = (rgb*) (buffer + y * stride);
            for (int x = 0; x < width; ++x)
            {
                if (y * width + x < height*stride*pixel_stride)
                {
                    lineptr[x].r = 0; // Back out red component
                    float alpha = background[y * width + x] / 255.f;
                    lineptr[x].g = uint8_t(alpha * lineptr[x].g + (1-alpha) * 255);
                    lineptr[x].b = uint8_t(alpha * lineptr[x].b + (1-alpha) * 255);
                }

            }
        }*/
        uint8_t* grayscale = new uint8_t[width * height];
        rgb_to_grayscale(buffer, grayscale, width, height, stride, pixel_stride);
        erode(grayscale, width, height, stride, pixel_stride);
        //dilate(grayscale, width, height, stride, pixel_stride);
        grayscale_to_rgb(grayscale, buffer, width, height, stride, pixel_stride);
        delete[] grayscale;

        // You can fake a long-time process with sleep
        {
            using namespace std::chrono_literals;
            //std::this_thread::sleep_for(100ms);
        }
    }
}
