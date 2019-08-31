#include "se_layer.h"
#include "utils.h"
#include "batchnorm_layer.h"
#include "avgpool_layer.h"
#include "connected_layer.h"
#include "im2col.h"
#include "col2im.h"
#include "blas.h"
#include "gemm.h"
#include <stdio.h>
#include <time.h>


static size_t get_workspace_size(layer l){
    return (size_t)l.c*l.c*l.c*l.c/16/16*sizeof(float);
}


se_layer make_se_layer(int batch, int h, int w, int c, ACTIVATION activation, int batch_normalize,  int adam, myParameters myParamStr)
{
    
    // printf("make_se_layer\n");
    int i;
    se_layer l = {0};
    l.type=SE;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.out_w = h;
    l.out_h = w;
    l.out_c = c;
    l.outputs = h*w*c;
    l.inputs = h*w*c;
    int output_size = l.outputs * batch;

    l.sub_layers_Num=3;
    l.sub_layers=calloc(l.sub_layers_Num, sizeof(layer));

    l.sub_layers[0]=make_avgpool_layer(l.batch,l.h,l.w,l.c);
    // make_connected_layer(int batch, int inputs, int outputs, ACTIVATION activation, int batch_normalize, int adam)
    l.sub_layers[1]=make_connected_layer(l.batch, l.c, l.c/16, LINEAR, batch_normalize, adam);
    l.sub_layers[1].L2_Flg=0;
    l.sub_layers[2]=make_connected_layer(l.batch, l.c/16, l.c, activation, 0, adam);
    l.sub_layers[2].L2_Flg=0;
    l.output =  calloc(output_size, sizeof(float));
    l.delta =   calloc(output_size, sizeof(float));

    l.forward = forward_se_layer;
    l.backward = backward_se_layer;
    l.update = update_se_layer;

#ifdef GPU
    l.forward_gpu = forward_se_layer_gpu;
    l.backward_gpu = backward_se_layer_gpu;
    l.update_gpu = update_se_layer_gpu;

    l.output_gpu  = cuda_make_array(l.output, output_size);
    l.delta_gpu   = cuda_make_array(l.delta, output_size);
#endif
    l.workspace_size = get_workspace_size(l);
    // printf("workspace size:%d\n",l.workspace_size);
    l.activation = activation;
    // if(l.prune){
    // fprintf(stderr, "mix conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d  %5.3f BFLOPs Prune Min:%lf Max:%lf Step:%lf \n", c, l.ksize_h, l.ksize_w, stride, w, h, c, l.out_w, l.out_h, l.out_c, (2.0 * l.n * l.ksize*l.c * l.out_h*l.out_w)/1000000000.,l.prune_threshold_min,l.prune_threshold_max,l.prune_threshold_step);
    // }else{
    //     fprintf(stderr, "mix conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d  %5.3f BFLOPs\n", c, l.ksize_h, l.ksize_w, stride, w, h, c, l.out_w, l.out_h, l.out_c, (2.0 * l.n * l.ksize*l.c * l.out_h*l.out_w)/1000000000.);
    // }
    fprintf(stderr, "     SE LAYER  %4d x%4d x%4d   ->  %4d x%4d x%4d\n",w, h, c, l.out_w, l.out_h, l.out_c);
    return l;
}


void forward_se_layer(se_layer l, network net)
{
    layer ave_pool=l.sub_layers[0];

    for(int b = 0; b < ave_pool.batch; ++b){
        for(int k = 0; k < ave_pool.c; ++k){
            int out_index = k + b*ave_pool.c;
            ave_pool.output[out_index] = 0;
            for(int i = 0; i < ave_pool.h*ave_pool.w; ++i){
                int in_index = i + ave_pool.h*ave_pool.w*(k + b*ave_pool.c);
                ave_pool.output[out_index] += net.input[in_index];
            }
            ave_pool.output[out_index] /= ave_pool.h*ave_pool.w;
        }
    }

    layer connect1=l.sub_layers[1];
    fill_cpu(connect1.outputs*connect1.batch, 0, connect1.output, 1);
    int m = connect1.batch;
    int k = connect1.inputs;
    int n = connect1.outputs;
    float *a = ave_pool.output;
    float *b = connect1.weights;
    float *c = connect1.output;
    gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);
    if(connect1.batch_normalize){
        forward_batchnorm_layer2(connect1,ave_pool.output);
    } else {
        add_bias(connect1.output,connect1.biases, connect1.batch, connect1.outputs, 1);
    }
    activate_array(connect1.output, connect1.outputs*connect1.batch,connect1.activation);

    layer connect2=l.sub_layers[2];
    fill_cpu(connect2.outputs*connect2.batch, 0, connect2.output, 1);
    m = connect2.batch;
    k = connect2.inputs;
    n = connect2.outputs;
    a = connect1.output;
    b = connect2.weights;
    c = connect2.output;
    gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);
    if(connect2.batch_normalize){
        forward_batchnorm_layer2(connect2, connect1.output);
    } else {
        add_bias(connect2.output,connect2.biases, connect2.batch, connect2.outputs, 1);
    }
    activate_array(connect2.output, connect2.outputs*connect2.batch,connect2.activation);

    //output
    fill_cpu(l.outputs*l.batch, 0, l.output, 1);
    
    for(int b = 0; b < l.batch; ++b){
        for(int k = 0; k < l.c; ++k){
            int out_index = k + b*l.c;
            for(int i = 0; i < l.h*l.w; ++i){
                int in_index = i + l.h*l.w*(k + b*l.c);
                l.output[in_index] += net.input[in_index]*connect2.output[out_index];
            }
        }
    }
}

void backward_se_layer(se_layer l, network net)
{

}

void update_se_layer(se_layer l, update_args a)
{
  
}

void resize_se_layer(se_layer *l, int w, int h)
{
    l->h = h;
    l->w = w;
    // l->c = c;
    l->out_w = h;
    l->out_h = w;
    // l->out_c = c;
    l->outputs = h*w*l->out_c;
    l->inputs = h*w*l->out_c;
    int output_size = l->outputs * l->batch;
    l->output = realloc(l->output, l->batch*l->outputs*sizeof(float));
    l->delta  = realloc(l->delta,  l->batch*l->outputs*sizeof(float));
    // layer connect2=l->sub_layers[2];
    // layer connect1=l->sub_layers[1];
    // layer ave_pool=l->sub_layers[0];

    #ifdef GPU
        cuda_free(l->delta_gpu);
        cuda_free(l->output_gpu);

        l->delta_gpu =  cuda_make_array(l->delta,  l->batch*l->outputs);
        l->output_gpu = cuda_make_array(l->output, l->batch*l->outputs);

    #endif
    l->workspace_size = get_workspace_size(*l);
}

