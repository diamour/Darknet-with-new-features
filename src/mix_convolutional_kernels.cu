#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "mix_convolutional_layer.h"
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

void forward_mix_convolutional_layer_gpu(mix_convolutional_layer l, network net)
{
    // printf("forward_mix_convolutional_layer_gpu\n");
    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
    int a_add=0;
    // int b_add=0;
    int c_add=0;
    int m_add=0;
    int m;
    int k;
    int n;
    for(int i = 0; i < l.batch; ++i){
        a_add=0;
        for(int j=0;j<l.sub_layers_Num;j++){
            layer sub_l=l.sub_layers[j];

            m = sub_l.n;
            k = sub_l.size*sub_l.size*sub_l.c;
            n = l.out_w*l.out_h;
            // printf("m:%d k:%d n:%d l.nweights:%d\n",m,k,n,l.nweights);
            // printf("a_add:%d l.nweights:%d\n",a_add,sub_l.nweights);
            float *a = l.weights_gpu + a_add;
            float *b = net.workspace;
            float *c = l.output_gpu +c_add;
            float *im =  net.input_gpu + m_add;

            if(sub_l.rectFlg){
                // printf("**************************************CPU ONLY**********************************************\n");
                rect_im2col_gpu(im, sub_l.c, sub_l.h, sub_l.w, sub_l.ksize_h,sub_l.ksize_w,sub_l.stride_h,sub_l.stride_w, sub_l.pad_h,sub_l.pad_w, b);
            }else{
                im2col_gpu(im, sub_l.c, sub_l.h, sub_l.w, sub_l.size, sub_l.stride, sub_l.pad, b);
            }

            // printf("check point2\n");
            gemm_gpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
            // printf("check point4\n");
            a_add+=sub_l.nweights;
            c_add+=n*m;
            m_add+=sub_l.c*sub_l.h*sub_l.w;
        }
    }
    if (l.batch_normalize) {
        forward_batchnorm_layer_gpu(l, net);
    } else {
        add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
    }
    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
    // printf("end forward_mix_convolutional_layer_gpu\n");
}


void backward_mix_convolutional_layer_gpu(mix_convolutional_layer l, network net)
{
    // printf("backward_mix_convolutional_layer_gpu chepoint:1\n");
    gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
    if(l.batch_normalize){
        backward_batchnorm_layer_gpu(l, net);
    } else {
        backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.n, l.out_w*l.out_h);
    }

    int a_add=0;
    // int b_add=0;
    int c_add=0;
    int m_add=0;


    for(int i = 0; i < l.batch; ++i){
        c_add=0;
        for(int j=0;j<l.sub_layers_Num;j++){
            layer sub_l=l.sub_layers[j];
            int m = sub_l.n;
            int n = sub_l.ksize*sub_l.c;
            int k = sub_l.out_w*sub_l.out_h;


            float *a = l.delta_gpu + a_add;
            float *b = net.workspace;
            float *c = l.weight_updates_gpu + c_add;

            float *im  = net.input_gpu+m_add;
            float *imd = net.delta_gpu+m_add;

            if(l.rectFlg){
                // rect_im2col_gpu(im, sub_l.c, l.h, l.w, l.ksize_h,l.ksize_w,  l.stride_h,l.stride_w, l.pad_h,l.pad_w, b);
            }else{
                im2col_gpu(im, sub_l.c, sub_l.h, sub_l.w, sub_l.size, sub_l.stride, sub_l.pad, b);
            }
            // im2col_gpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
            gemm_gpu(0,1,m,n,k,1,a,k,b,k,1,c,n);

            if (net.delta_gpu) {
                a = l.weights_gpu + c_add;
                b = l.delta_gpu + a_add;
                c = net.workspace;
                if (sub_l.size == 1) {
                    c = imd;
                }

                gemm_gpu(1,0,n,k,m,1,a,n,b,k,0,c,k);

                if (sub_l.size != 1) {
                    if(l.rectFlg){
                        // rect_col2im_gpu(net.workspace, l.c/l.groups, l.h, l.w, l.ksize_h,l.ksize_w,  l.stride_h,l.stride_w, l.pad_h,l.pad_w, imd);
                    }else{
                        col2im_gpu(net.workspace, sub_l.c, sub_l.h, sub_l.w, sub_l.size, sub_l.stride, sub_l.pad, imd);
                    }
                }
                a_add+=m*k;
                c_add+=sub_l.nweights;
                m_add+=sub_l.c*sub_l.h*sub_l.w;
            }
        }
    }

}


void update_mix_convolutional_layer_gpu(layer l, update_args a)
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

void pull_mix_convolutional_layer(layer l)
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

void push_mix_convolutional_layer(layer l)
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




