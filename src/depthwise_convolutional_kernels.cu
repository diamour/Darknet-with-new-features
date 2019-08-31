#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "depthwise_convolutional_layer.h"
// #include "convolutional_layer.h"
#include "batchnorm_layer.h"
#include "gemm.h"
#include "blas.h"
#include "im2col.h"
#include "col2im.h"
#include "utils.h"
#include "cuda.h"
}

/**************************prune network weights*************************/

void forward_depthwise_convolutional_layer_gpu(depthwise_convolutional_layer l, network net)
{
    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
    // printf("forward_depthwise_convolutional_layer_gpu\n");
#ifdef CUDNN
    float *net_input_base_pointer=net.input_gpu;
    float *layer_weights_base_pointer=l.weights_gpu;
    float *layer_output_base_pointer=l.output_gpu;

    for(int b = 0; b < l.batch; ++b){
        for (int c=0;c<l.c;c++)
        {
            int l_id=c+b*l.c;
            layer sub_l = l.sub_layers[l_id];

            if(sub_l.delta_gpu){
                fill_gpu(sub_l.outputs, 0, sub_l.delta_gpu, 1);
            }

            net.input_gpu=net_input_base_pointer+l_id*l.h*l.w;
            sub_l.weights_gpu=layer_weights_base_pointer+l_id*l.ksize;
            sub_l.output_gpu=layer_output_base_pointer+l_id*l.out_h*l.out_w;
            // sub_l.x_gpu=l.x_gpu+l_id*l.out_h*l.out_w;
            // sub_l.x_norm_gpu=l.x_norm_gpu+l_id*l.out_h*l.out_w;
            float one = 1;
            cudnnConvolutionForward(cudnn_handle(),
                        &one,
                        sub_l.srcTensorDesc,
                        net.input_gpu,
                        sub_l.weightDesc,
                        sub_l.weights_gpu,
                        sub_l.convDesc,
                        sub_l.fw_algo,
                        net.workspace,
                        sub_l.workspace_size,
                        &one,
                        sub_l.dstTensorDesc,
                        sub_l.output_gpu);
            // if (l.batch_normalize) {
            //     forward_batchnorm_layer_gpu(sub_l, net);
            // } else {
            //     add_bias_gpu(sub_l.output_gpu, sub_l.biases_gpu, sub_l.batch, sub_l.n, sub_l.out_w*sub_l.out_h);
            // }
        }
    }
    net.input_gpu=net_input_base_pointer;
    l.weights_gpu=layer_weights_base_pointer;
    l.output_gpu=layer_output_base_pointer;
// #endif
#else
    int k = l.ksize;
    int n = l.out_w*l.out_h;
    for(int b = 0; b < l.batch; ++b){
        for (int c=0;c<l.c;c++)
        {
            float *aoffset = l.weights_gpu+c*l.ksize;
            float *boffset = net.workspace;
            float *coffset = l.output_gpu+c*l.out_h*l.out_w+b*l.n*l.out_h*l.out_w;
            float *input_offset = net.input_gpu + c*l.h*l.w+ b*l.c*l.h*l.w;
            // printf("Check point b:%d c:%d\n",b,c);
            if(l.rectFlg){
                // printf("**************************************gpu ONLY**********************************************\n");
                rect_im2col_gpu(input_offset, 1, l.h, l.w, l.ksize_h,l.ksize_w, l.stride_h,l.stride_w, l.pad_h,l.pad_w, boffset);
            }else{
                // printf("**************************************im2col_gpu**********************************************\n");
                im2col_gpu(input_offset, l.c, l.h, l.w, l.size, l.stride, l.pad, boffset);
            }
                
            gemm_gpu(0, 0, 1, n, k, 1, aoffset, k, boffset, n, 1, coffset, n);
        }
    }
    // if (l.batch_normalize) {
    //     forward_batchnorm_layer_gpu(l, net);
    // } else {
    //     add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
    // }
#endif
    // printf("Check point3\n");
    // printf("Check point4\n");
    // printf("end forward_depthwise_convolutional_layer_gpu\n");
    if (l.batch_normalize) {
        forward_batchnorm_layer_gpu(l, net);
    } else {
        add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
    }
    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);

}


void backward_depthwise_convolutional_layer_gpu(depthwise_convolutional_layer l, network net)
{
    gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
    // printf("backward_depthwise_convolutional_layer_gpu chepoint:1\n");
    if(l.batch_normalize){
        backward_batchnorm_layer_gpu(l, net);
    } else {
        backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.n, l.out_w*l.out_h);
    }
    // float *original_input = net.input_gpu;

#ifdef CUDNN
    float *net_input_base_pointer=net.input_gpu;
    float *net_delta_base_pointer=net.delta_gpu;
    float *layer_weight_updates_base_pointer=l.weight_updates_gpu;
    float *layer_weights_base_pointer=l.weights_gpu;
    float *layer_delta_base_pointer=l.delta_gpu;

    for(int b = 0; b < l.batch; ++b){
        for (int c=0;c<l.c;c++)
        {
            int l_id=c+b*l.c;
            layer sub_l = l.sub_layers[l_id];
            net.input_gpu=net_input_base_pointer+l_id*l.h*l.w;
            net.delta_gpu=net_delta_base_pointer+l_id*l.h*l.w;
            sub_l.weights_gpu=layer_weights_base_pointer+l_id*l.ksize;
            sub_l.weight_updates_gpu=layer_weight_updates_base_pointer+l_id*l.ksize;
            sub_l.delta_gpu=layer_delta_base_pointer+l_id*l.out_h*l.out_w;

            float one = 1;
            cudnnConvolutionBackwardFilter(cudnn_handle(),
                    &one,
                    sub_l.srcTensorDesc,
                    net.input_gpu,
                    sub_l.ddstTensorDesc,
                    sub_l.delta_gpu,
                    sub_l.convDesc,
                    sub_l.bf_algo,
                    net.workspace,
                    sub_l.workspace_size,
                    &one,
                    sub_l.dweightDesc,
                    sub_l.weight_updates_gpu);

            if(net.delta_gpu){
                cudnnConvolutionBackwardData(cudnn_handle(),
                        &one,
                        sub_l.weightDesc,
                        sub_l.weights_gpu,
                        sub_l.ddstTensorDesc,
                        sub_l.delta_gpu,
                        sub_l.convDesc,
                        sub_l.bd_algo,
                        net.workspace,
                        sub_l.workspace_size,
                        &one,
                        sub_l.dsrcTensorDesc,
                        net.delta_gpu);
            }
        }
    }
    net.input_gpu=net_input_base_pointer;
    net.delta_gpu=net_delta_base_pointer;
    l.weights_gpu=layer_weights_base_pointer;
    l.weight_updates_gpu=layer_weight_updates_base_pointer;
    l.delta_gpu=layer_delta_base_pointer;
// #endif
#else
    int n = l.ksize;
    int k = l.out_w*l.out_h;
    for (int b = 0; b < l.batch; ++b) {
        for (int c = 0; c<l.c; c++)
        {
            float *aoffset = l.delta_gpu + c*l.out_h*l.out_w + b*l.n*l.out_h*l.out_w;
            float *boffset = net.workspace;
            float *coffset = l.weight_updates_gpu + c*l.ksize;

            float *im = net.input_gpu + c*l.h*l.w + b*l.c*l.h*l.w;

            if(l.rectFlg){
                // printf("**************************************gpu ONLY**********************************************\n");
                rect_im2col_gpu(im, 1, l.h, l.w, l.ksize_h,l.ksize_w, l.stride_h,l.stride_w, l.pad_h,l.pad_w, boffset);
            }else{
                im2col_gpu(im, 1, l.h, l.w, l.size, l.stride, l.pad, boffset);
            }

            gemm_gpu(0, 1, 1, n, k, 1, aoffset, k, boffset, k, 1, coffset, n);

            if (net.delta_gpu) {
                aoffset = l.weights_gpu+ c*l.ksize;
                boffset = l.delta_gpu+ c*l.out_h*l.out_w + b*l.n*l.out_h*l.out_w;
                coffset = net.workspace;

                gemm_gpu(1, 0, n, k, 1, 1, aoffset, n, boffset, k, 0, coffset, k);

                if(l.rectFlg){
                // printf("**************************************gpu ONLY**********************************************\n");
                    rect_col2im_gpu(net.workspace, 1, l.h, l.w, l.ksize_h,l.ksize_w, l.stride_h, l.stride_w, l.pad_h,l.pad_w, net.delta_gpu + c*l.h*l.w + b*l.n*l.h*l.w);
                }else{
                    col2im_gpu(net.workspace, 1, l.h, l.w, l.size, l.stride, l.pad, net.delta_gpu + c*l.h*l.w + b*l.n*l.h*l.w);
                }
            }
        }
    }
#endif
}


void update_depthwise_convolutional_layer_gpu(layer l, update_args a)
{
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;
    if(a.adam){
        adam_update_gpu(l.weights_gpu, l.weight_updates_gpu, l.m_gpu, l.v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.nweights, batch, a.t);
        adam_update_gpu(l.biases_gpu, l.bias_updates_gpu, l.bias_m_gpu, l.bias_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.n, batch, a.t);
        if(l.scales_gpu){
            adam_update_gpu(l.scales_gpu, l.scale_updates_gpu, l.scale_m_gpu, l.scale_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.n, batch, a.t);
        }
    }else{
        axpy_gpu(l.nweights, -decay*batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
        axpy_gpu(l.nweights, learning_rate/batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
        scal_gpu(l.nweights, momentum, l.weight_updates_gpu, 1);

        axpy_gpu(l.n, learning_rate/batch, l.bias_updates_gpu, 1, l.biases_gpu, 1);
        scal_gpu(l.n, momentum, l.bias_updates_gpu, 1);

        if(l.scales_gpu){
            axpy_gpu(l.n, learning_rate/batch, l.scale_updates_gpu, 1, l.scales_gpu, 1);
            scal_gpu(l.n, momentum, l.scale_updates_gpu, 1);
        }
    }
    // cuda_pull_array(l.weights_gpu, l.weights, l.nweights);
    // printf("*******************************************************\n");
    // for(int i=6;i<7;i++){
    //     for(int j=0;j<9;j++){
    //         printf("%f ",l.weights[i*9+j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");
}

void pull_depthwise_convolutional_layer(layer l)
{
    cuda_pull_array(l.weights_gpu, l.weights, l.nweights);
    cuda_pull_array(l.biases_gpu, l.biases, l.n);
    cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
    cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.n);
    if (l.batch_normalize){
        cuda_pull_array(l.scales_gpu, l.scales, l.n);
        cuda_pull_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
        cuda_pull_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
    }
}

void push_depthwise_convolutional_layer(layer l)
{
    cuda_push_array(l.weights_gpu, l.weights, l.nweights);
    cuda_push_array(l.biases_gpu, l.biases, l.n);
    cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
    cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.n);
    if (l.batch_normalize){
        cuda_push_array(l.scales_gpu, l.scales, l.n);
        cuda_push_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
        cuda_push_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
    }
}




