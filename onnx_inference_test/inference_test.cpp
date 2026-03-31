#include "npy.hpp"
#include <cpu_provider_factory.h>
#include <onnxruntime_cxx_api.h>
#include <array>
#include <iostream>
#include <vector>
#include <cstdio>
#include <iterator>
#include <format>
#include <chrono>

constexpr int64_t batch_size   = 1;
constexpr int64_t buffer_size  = 96;   
constexpr int64_t input_size   = 2;    
constexpr int64_t hidden_size  = 64;   
constexpr int64_t num_layers   = 2;    

auto run_buffer_inference(
    Ort::Session& session, 
    std::vector<float>& x_data,
    std::vector<float>::iterator input_buffer_begin, 
    std::size_t input_buffer_size, 
    std::vector<float>& hidden_in
) {
    for (size_t i = 0; i < input_buffer_size; ++i)
        x_data[i * 2] = *(input_buffer_begin+i);

    // ── Build input tensors ──────────────────────────────────────────────────
    static const auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::array<int64_t, 3> x_shape      = {batch_size, buffer_size, input_size};
    std::array<int64_t, 3> hidden_shape = {num_layers, batch_size, hidden_size};

    Ort::Value input_x = Ort::Value::CreateTensor<float>(
        memory_info,
        x_data.data(), x_data.size(),
        x_shape.data(), x_shape.size()
    );
    Ort::Value input_hidden = Ort::Value::CreateTensor<float>(
        memory_info,
        hidden_in.data(), hidden_in.size(),
        hidden_shape.data(), hidden_shape.size()
    );

    // ── Run inference ────────────────────────────────────────────────────────
    static const char* input_names[]  = {"x", "hidden_in"};
    static const char* output_names[] = {"output", "hidden_out"};

    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(input_x));
    inputs.push_back(std::move(input_hidden));

    return session.Run(
        Ort::RunOptions{nullptr},
        input_names,  inputs.data(), inputs.size(),
        output_names, std::size(output_names)
    );
}

int main(int argc, char ** argv)
{
    using std::chrono::high_resolution_clock;
    std::string input_filename {"../dataset-8/inputs/input-599.npy"};
    std::string output_filename {"output-599.npy"};
    std::string model_name {"../lowpass_rnn.onnx"};
    if(argc == 1)
    {
        printf("usage : onnx_inference_test <path_to_onnx_model> <path_to_npy_input> <path_to_npy_output>\n");
        return 0;
    }
    if(argc > 1)
    {
        model_name = argv[1];
    }

    if(argc > 2)
    {
        input_filename = argv[2];
    }

    if(argc > 3)
    {
        output_filename = argv[3];
    }
    // Open example input file, retrieve normalized cutoff frequency and remove it from samples
    npy::npy_data py_input = npy::read_npy<float>(input_filename);
    std::vector input_float = py_input.data;
    auto fc_normed = input_float.back();
    input_float.pop_back();
    printf("fc normed = %f\n", fc_normed);

    // load onnx model
    /// session setup
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "lowpass_rnn\n");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    Ort::Session session(env, model_name.c_str(), session_options);
    Ort::AllocatorWithDefaultOptions allocator;

    // prepare buffer of fc_norm
    std::vector<float> x_data(buffer_size * input_size, fc_normed); // all slots = fc_norm

    // inference on per-buffer basis
    const auto buffer_count = input_float.size() / buffer_size;
    const size_t hidden_layer_size = num_layers * batch_size * hidden_size;
    std::vector<float> hidden_in (num_layers * batch_size * hidden_size, 0.f);
    std::vector<float> output_float {};
    printf("starting inference per buffer (%d buffers to run)\n", buffer_count);

    auto avg_execution_time = 0.0;
    for(auto buffer = 0; buffer < buffer_count; ++buffer)
    {
        // --- Inference + copy inference output operation-------------------------------------
        auto t1 = high_resolution_clock::now();
        auto outputs = run_buffer_inference(
            session, 
            x_data,
            input_float.begin()+buffer * buffer_size,
            buffer_size,
            hidden_in
        );
        // keep hidden out to pass for next inference
        float * hidden_out_ptr = outputs[1].GetTensorMutableData<float>();
        std::copy(hidden_out_ptr, hidden_out_ptr + hidden_layer_size,  hidden_in.begin());
        
        auto t2 = high_resolution_clock::now();
        avg_execution_time += std::chrono::duration<double, std::milli>(t2-t1).count();
        // ------------------------------------------------------------------------------------

        // store result
        float * output_buffer = outputs[0].GetTensorMutableData<float>();
        std::copy(output_buffer, output_buffer + buffer_size, std::back_inserter(output_float));
    }

    printf("inference done\n");
    printf(std::format("average inference time = {} ms \n", avg_execution_time/buffer_count).c_str());
    printf("writing to file ...\n");
    // write result to file
    npy::npy_data<float> out;
    out.shape = { output_float.size() };
    std::copy(output_float.begin(), output_float.end(), std::back_inserter(out.data));

    const std::string path_to_out { output_filename };
    npy::write_npy(path_to_out, out);
}