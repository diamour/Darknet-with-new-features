#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "se_layer.h"
#include "avgpool_layer.h"
#include "connected_layer.h"
#include "batchnorm_layer.h"
#include "gemm.h"
#include "blas.h"
#include "im2col.h"
#include "col2im.h"
#include "utils.h"
#include "cuda.h"
}
__global__ void forward_avgpool_layer_kernel_se(int n, int w, int h, int c, float *input, float *output)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;

    int k = id % c;
    id /= c;
    int b = id;

    int i;
    int out_index = (k + c*b);
    output[out_index] = 0;
    for(i = 0; i < w*h; ++i){
        int in_index = i + h*w*(k + b*c);
        output[out_index] += input[in_index];
    }
    output[out_index] /= w*h;
}

__global__ void forward_fusion_se(int n, int w, int h, int c, float *input, float *se_mul, float *output)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;

    int k = id % c;
    id /= c;
    int b = id;

    int i;
    int out_index = (k + c*b);
    for(i = 0; i < w*h; ++i){
        int in_index = i + h*w*(k + b*c);
        output[in_index] = input[in_index]*se_mul[out_index];
    }
}

__global__ void backward_delta_se(int n, int w, int h, int c, float *input, float *se_mul, float *output)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;

    int k = id % c;
    id /= c;
    int b = id;

    int i;
    int out_index = (k + c*b);
    for(i = 0; i < w*h; ++i){
        int in_index = i + h*w*(k + b*c);
        output[in_index] = input[in_index]*se_mul[out_index];
    }
}
// backward_delta_to_connect_se<<<cuda_gridsize(l.c*l.batch), BLOCK>>>(l.c*l.batch, l.w, l.h, l.c, l.delta_gpu,net.input_gpu,connect2.delta_gpu);
__global__ void backward_delta_to_connect_se(int n, int w, int h, int c, float *input_delta, float *l_output, float *output_delta)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;

    int k = id % c;
    id /= c;
    int b = id;

    int i;
    int out_index = (k + c*b);
    float average_out=0;

    for(i = 0; i < w*h; ++i){
        int in_index = i + h*w*(k + b*c);
        average_out = l_output[in_index] ;
    }
    average_out=average_out/w/h;

    float average_delta=0;

    for(i = 0; i < w*h; ++i){
        int in_index = i + h*w*(k + b*c);
        average_delta = input_delta[in_index] ;
    }
    average_delta=average_delta/w/h;

    output_delta[out_index]=average_out*average_delta;
}

void forward_se_layer_gpu(se_layer l, network net)
{
    // printf("forward_se_layer_gpu\n");
    layer ave_pool=l.sub_layers[0];
    // size_t n = ave_pool.c*ave_pool.batch;
    forward_avgpool_layer_kernel_se<<<cuda_gridsize(ave_pool.c*ave_pool.batch), BLOCK>>>(ave_pool.c*ave_pool.batch, ave_pool.w, ave_pool.h, ave_pool.c, net.input_gpu, ave_pool.output_gpu);


    layer connect1=l.sub_layers[1];
    
    fill_gpu(connect1.outputs*connect1.batch, 0, connect1.output_gpu, 1);
    int m = connect1.batch;
    int k = connect1.inputs;
    int n = connect1.outputs;
    float *a = ave_pool.output_gpu;
    float *b = connect1.weights_gpu;
    float *c = connect1.output_gpu;
    gemm_gpu(0,1,m,n,k,1,a,k,b,k,1,c,n);
    if(connect1.batch_normalize){
        forward_batchnorm_layer_gpu2(connect1,ave_pool.output_gpu);
    } else {
        add_bias_gpu(connect1.output_gpu,connect1.biases_gpu, connect1.batch, connect1.outputs, 1);
    }
    activate_array_gpu(connect1.output_gpu, connect1.outputs*connect1.batch,connect1.activation);

    layer connect2=l.sub_layers[2];
    fill_gpu(connect2.outputs*connect2.batch, 0, connect2.output_gpu, 1);
    m = connect2.batch;
    k = connect2.inputs;
    n = connect2.outputs;
    a = connect1.output_gpu;
    b = connect2.weights_gpu;
    c = connect2.output_gpu;
    gemm_gpu(0,1,m,n,k,1,a,k,b,k,1,c,n);
    if(connect2.batch_normalize){
        forward_batchnorm_layer_gpu2(connect2,connect1.output_gpu);
    } else {
        add_bias_gpu(connect2.output_gpu,connect2.biases_gpu, connect2.batch, connect2.outputs, 1);
    }
    activate_array_gpu(connect2.output_gpu, connect2.outputs*connect2.batch,connect2.activation);

    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
    forward_fusion_se<<<cuda_gridsize(l.c*l.batch), BLOCK>>>(l.c*l.batch, l.w, l.h, l.c, net.input_gpu,connect2.output_gpu,l.output_gpu);

#if 1
    cuda_pull_array(connect2.output_gpu, connect2.output, connect2.out_c);
    cuda_pull_array(connect2.weights_gpu, connect2.weights, connect2.out_c*connect2.out_c/16);
    cuda_pull_array(connect1.weights_gpu, connect1.weights, connect1.out_c*connect1.out_c/16);
    char filename[100];
    char filename_output0[100];
    char filename_weights[100];
    sprintf(filename, "sigResult_se%dlayer.txt",connect2.outputs*connect2.batch);
    sprintf(filename_output0, "weights_fc1_se%dlayer.txt",connect1.outputs*connect1.batch);
    sprintf(filename_weights, "weights_fc2_se%dlayer.txt",connect2.outputs*connect2.batch);
    FILE *fp = fopen(filename, "w");
    FILE *fp_output0 = fopen(filename_output0, "w");
    FILE *fp_weights = fopen(filename_weights, "w");
    for(int i = 0; i < connect2.out_c; i++){
      fprintf(fp, "%f\n", connect2.output[i]);
    }
    for(int j = 0; j < connect2.out_c*connect2.out_c/16; j++){
      fprintf(fp_output0, "%f\n", connect1.weights[j]);
      fprintf(fp_weights, "%f\n", connect2.weights[j]);
    }
    fclose(fp);
    fclose(fp_output0);
    fclose(fp_weights);
  #endif  
    // printf("end  forward_se_layer_gpu\n");
}


void backward_se_layer_gpu(se_layer l, network net)
{
    // printf("backward_se_layer_gpu\n");
    layer connect2=l.sub_layers[2];
    backward_delta_se<<<cuda_gridsize(l.c*l.batch), BLOCK>>>(l.c*l.batch, l.w, l.h, l.c, l.delta_gpu,connect2.output_gpu,net.delta_gpu);

    layer connect1=l.sub_layers[1];
    backward_delta_to_connect_se<<<cuda_gridsize(l.c*l.batch), BLOCK>>>(l.c*l.batch, l.w, l.h, l.c, l.delta_gpu,net.input_gpu,connect2.delta_gpu);
    constrain_gpu(connect2.outputs*connect2.batch, 1, connect2.delta_gpu, 1);
    gradient_array_gpu(connect2.output_gpu, connect2.outputs*l.batch, connect2.activation, connect2.delta_gpu);
    if(connect2.batch_normalize){
        backward_batchnorm_layer_gpu2(connect2, connect1.delta_gpu);
    } else {
        backward_bias_gpu(connect2.bias_updates_gpu, connect2.delta_gpu, connect2.batch, connect2.outputs, 1);
    }

    int m = connect2.outputs;
    int k = connect2.batch;
    int n = connect2.inputs;
    float * a = connect2.delta_gpu;
    float * b = connect1.output_gpu;
    float * c = connect2.weight_updates_gpu;
    gemm_gpu(1,0,m,n,k,1,a,m,b,n,1,c,n);

    m = connect2.batch;
    k = connect2.outputs;
    n = connect2.inputs;

    a = connect2.delta_gpu;
    b = connect2.weights_gpu;
    c = connect1.delta_gpu;

    if(c) gemm_gpu(0,0,m,n,k,1,a,k,b,n,1,c,n);

    layer ave_pool=l.sub_layers[0];
    constrain_gpu(connect1.outputs*connect1.batch, 1, connect1.delta_gpu, 1);
    gradient_array_gpu(connect1.output_gpu, connect1.outputs*l.batch, connect1.activation, connect1.delta_gpu);
    if(connect1.batch_normalize){
        backward_batchnorm_layer_gpu2(connect1, ave_pool.delta_gpu);
    } else {
        backward_bias_gpu(connect1.bias_updates_gpu, connect1.delta_gpu, connect1.batch, connect1.outputs, 1);
    }

    m = connect1.outputs;
    k = connect1.batch;
    n = connect1.inputs;
    a = connect1.delta_gpu;
    b = ave_pool.output_gpu;
    c = connect1.weight_updates_gpu;
    gemm_gpu(1,0,m,n,k,1,a,m,b,n,1,c,n);
    // printf("end backward_se_layer_gpu\n");
}


void update_se_layer_gpu(layer l, update_args a)
{
    layer connect2=l.sub_layers[2];
    
    connect2.update_gpu(connect2, a);
    layer connect1=l.sub_layers[1];
    
    connect1.update_gpu(connect1, a);
}

void pull_se_layer(layer l)
{
    layer connect2=l.sub_layers[2];
    pull_connected_layer(connect2);
    layer connect1=l.sub_layers[1];
    pull_connected_layer(connect1);
}

void push_se_layer(layer l)
{
    layer connect2=l.sub_layers[2];
    push_connected_layer(connect2);
    layer connect1=l.sub_layers[1];
    push_connected_layer(connect1);
}




