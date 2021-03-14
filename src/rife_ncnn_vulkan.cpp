#include <stdio.h>
#include <pybind11/pybind11.h>

// ncnn
#include "cpu.h"
#include "gpu.h"
#include "platform.h"
#include "benchmark.h"

#include "rife.h"

int add(int i, int j) {
    return i + j;
}

// **** input / output functions to test data conversion

#include <algorithm>
#include <queue>
#include <vector>
#include <clocale>
// image decoder and encoder with stb
#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_PSD
#define STBI_NO_TGA
#define STBI_NO_GIF
#define STBI_NO_HDR
#define STBI_NO_PIC
#define STBI_NO_STDIO
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "filesystem_utils.h"

static int decode_image(const path_t& imagepath, ncnn::Mat& image)
{
    unsigned char* pixeldata = 0;
    int w;
    int h;
    int c;
    
    FILE* fp = fopen(imagepath.c_str(), "rb");
    if (fp)
    {
        // read whole file
        unsigned char* filedata = 0;
        int length = 0;
        {
            fseek(fp, 0, SEEK_END);
            length = ftell(fp);
            rewind(fp);
            filedata = (unsigned char*)malloc(length);
            if (filedata)
            {
                fread(filedata, 1, length, fp);
            }
            fclose(fp);
        }
        if (filedata)
        {
            pixeldata = stbi_load_from_memory(filedata, length, &w, &h, &c, 3);
            c = 3;
            free(filedata);
        }
    }

    if (!pixeldata)
    {
        printf("decode image %s failed\n", imagepath.c_str());
        return -1;
    }

    image = ncnn::Mat(w, h, (void*)pixeldata, (size_t)3, 3);

    return 0;

}

static int encode_image(const path_t& imagepath, const ncnn::Mat& image)
{
    int success = 0;
    path_t ext = get_file_extension(imagepath);
    if (ext == PATHSTR("jpg") || ext == PATHSTR("JPG") || ext == PATHSTR("jpeg") || ext == PATHSTR("JPEG"))
    {
        success = stbi_write_jpg(imagepath.c_str(), image.w, image.h, image.elempack, image.data, 100);
    }
    if (!success)
    {
        printf("encode image %s failed\n", imagepath.c_str());
    }
    return success ? 0 : -1;
}

namespace py = pybind11;

PYBIND11_MODULE(rife_ncnn_vulkan, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: cmake_example

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

    m.def("create_gpu_instance", &ncnn::create_gpu_instance, R"pbdoc(
        create_gpu_instance

        Some other explanation about the add function.
    )pbdoc");

    m.def("destroy_gpu_instance", &ncnn::destroy_gpu_instance, R"pbdoc(
        destroy_gpu_instance

        Some other explanation about the add function.
    )pbdoc");

    // test io functions
    m.def("decode_image", &decode_image);
    m.def("encode_image", &encode_image);

    py::class_<RIFE> rife(m, "RIFE");
    rife.def(py::init<int, bool, bool, int, bool>(), 
                py::arg("gpuid"), 
                py::arg("tta_mode") = false, 
                py::arg("uhd_mode") = false,
                py::arg("num_threads") = 1,
                py::arg("rife_v2") = false
            )
        .def("load", &RIFE::load)
        .def("process", &RIFE::process);
}

int main(int argc, char** argv)
{
    printf("hello\n");
}