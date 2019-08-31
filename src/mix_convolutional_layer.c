#include "mix_convolutional_layer.h"
#include "utils.h"
#include "batchnorm_layer.h"
#include "im2col.h"
#include "col2im.h"
#include "blas.h"
#include "gemm.h"
#include <stdio.h>
#include <time.h>

#ifdef AI2
#include "xnor_layer.h"
#endif


int mix_convolutional_out_height(mix_convolutional_layer l)
{
    return (l.h + 2*l.pad_h - l.ksize_h) / l.stride_h + 1;
}

int mix_convolutional_out_width(mix_convolutional_layer l)
{
    return (l.w + 2*l.pad_w - l.ksize_w) / l.stride_w + 1;
}

static size_t get_workspace_size(layer l){
#ifdef CUDNN
    if(gpu_index >= 0){
        size_t most = 0;
        size_t s = 0;
        cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle(),
                l.srcTensorDesc,
                l.weightDesc,
                l.convDesc,
                l.dstTensorDesc,
                l.fw_algo,
                &s);
        if (s > most) most = s;
        cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle(),
                l.srcTensorDesc,
                l.ddstTensorDesc,
                l.convDesc,
                l.dweightDesc,
                l.bf_algo,
                &s);
        if (s > most) most = s;
        cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle(),
                l.weightDesc,
                l.ddstTensorDesc,
                l.convDesc,
                l.dsrcTensorDesc,
                l.bd_algo,
                &s);
        if (s > most) most = s;
        return most;
    }
#endif
    return (size_t)l.out_h*l.out_w*l.nweights/l.n*sizeof(float);
}

#ifdef GPU
#ifdef CUDNN
void cudnn_mix_convolutional_setup(layer *l)
{
    for(int b=0;b<l->batch;b++){
        for(int c=0;c<l->c;c++){
            int l_id=c+b*l->c;   
            layer sub_l = l->sub_layers[l_id];
            cudnn_2d_mix_convolutional_setup(&sub_l);
        }
    }
}
void cudnn_2d_mix_convolutional_setup(layer *l)
{
    cudnnSetTensor4dDescriptor(l->dsrcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, l->h, l->w); 
    cudnnSetTensor4dDescriptor(l->ddstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, l->out_h, l->out_w); 

    cudnnSetTensor4dDescriptor(l->srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, l->h, l->w); 
    cudnnSetTensor4dDescriptor(l->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, l->out_h, l->out_w); 
    cudnnSetTensor4dDescriptor(l->normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, 1); 

    cudnnSetFilter4dDescriptor(l->dweightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1 ,1, l->ksize_h, l->ksize_w); 
    cudnnSetFilter4dDescriptor(l->weightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1, 1, l->ksize_h, l->ksize_w); 
    #if CUDNN_MAJOR >= 6
    cudnnSetConvolution2dDescriptor(l->convDesc, l->pad_h, l->pad_w, l->stride_h, l->stride_w, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    #else
    cudnnSetConvolution2dDescriptor(l->convDesc, l->pad_h, l->pad_w, l->stride_h, l->stride_w, 1, 1, CUDNN_CROSS_CORRELATION);
    #endif

    #if CUDNN_MAJOR >= 7
    cudnnSetConvolutionGroupCount(l->convDesc, 1);
    #else
    if(l->groups > 1){
        error("CUDNN < 7 doesn't support groups, please upgrade!");
    }
    #endif

    cudnnGetConvolutionForwardAlgorithm(cudnn_handle(),
            l->srcTensorDesc,
            l->weightDesc,
            l->convDesc,
            l->dstTensorDesc,
            CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
            2000000000,
            &l->fw_algo);
    cudnnGetConvolutionBackwardDataAlgorithm(cudnn_handle(),
            l->weightDesc,
            l->ddstTensorDesc,
            l->convDesc,
            l->dsrcTensorDesc,
            CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
            2000000000,
            &l->bd_algo);
    cudnnGetConvolutionBackwardFilterAlgorithm(cudnn_handle(),
            l->srcTensorDesc,
            l->ddstTensorDesc,
            l->convDesc,
            l->dweightDesc,
            CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
            2000000000,
            &l->bf_algo);
}

#endif
#endif


layer make_sub_mix_convolutional_layer(int h, int w, int size, int stride, int padding, int c, int n)
{
    // printf("make_sub_mix_convolutional_layer\n");
    layer l = {0};
    l.type = CONVOLUTIONAL;

    l.h = h;
    l.w = w;
    l.c = c;
    l.n = n;

    l.batch = 1;
    l.stride = stride;
    l.pad = padding;
    l.pad_h = l.pad;
    l.pad_w = l.pad;
    l.size=size;
    l.nweights=l.size*l.size*c*n;
    l.ksize_h =size;
    l.ksize_w =size;
    l.ksize= size*size;
    l.stride_h=l.stride;
    l.stride_w=l.stride;
    int out_w = mix_convolutional_out_width(l);
    int out_h = mix_convolutional_out_height(l);
    l.out_h = out_h;
    l.out_w = out_w;
    l.out_c = n;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = l.w * l.h * l.c;
    printf("make_mix_convolutional_layer size:%d ksize:%d ksize_h:%d ksize_w:%d pad_h:%d pad_w:%d pad:%d stride:%d stride_h:%d stride_w:%d\n",l.size,l.ksize,l.ksize_h,l.ksize_w,l.pad_h,l.pad_w,l.pad,l.stride,l.stride_h,l.stride_w);
    printf("mix_sub_conv out_w:%d out_h:%d l.n:%d l.c:%d l.out_c:%d\n",l.out_w,l.out_h,l.n,l.c,l.out_c);
    return l;
}


mix_convolutional_layer make_mix_convolutional_layer(int batch, int h, int w, int c, int n,int mixNum, int *sizeList, int stride, int padding, ACTIVATION activation, int batch_normalize,  int adam, myParameters myParamStr)
{
    
    // printf("make_mix_convolutional_layer\n");

    int i;
    mix_convolutional_layer l = {0};

    // int mixNum=3;
    int *mixKernalSize=sizeList;
    int weightsSizeTotal=0;
    l.sub_layers_Num=mixNum;
    l.sub_layers=calloc(l.sub_layers_Num, sizeof(layer));
    // printf("c:%d n:%d\n",c/mixNum,mixKernalSize[0]);
    // printf("make sub layers\n");
    for(int mixID=0;mixID<mixNum-1;mixID++){
        //layer make_sub_mix_convolutional_layer(int h, int w, int size, int stride, int padding, int c, int n, int norm)
        l.sub_layers[mixID]=make_sub_mix_convolutional_layer(h,w,mixKernalSize[mixID],stride,mixKernalSize[mixID]/2, c/mixNum, n/mixNum);
        weightsSizeTotal+=l.sub_layers[mixID].nweights;
    }
    int c_rest=c-c*(mixNum-1)/mixNum;
    int n_rest=n-n*(mixNum-1)/mixNum;
    l.sub_layers[mixNum-1]=make_sub_mix_convolutional_layer(h,w,mixKernalSize[mixNum-1],stride,mixKernalSize[mixNum-1]/2, c_rest, n_rest);
    weightsSizeTotal+=l.sub_layers[mixNum-1].nweights;

    l.type = MIX_CONVOLUTIONAL;
    l.h = h;
    l.w = w;
    l.c = c;
    //the out put channel equals to the input channel
    l.n = n;
    int size=l.sub_layers[0].size;
    l.batch = batch;
    l.stride = stride;
    l.pad = size/2;
    l.pad_h = l.pad;
    l.pad_w = l.pad;

    l.rectFlg=myParamStr.rectFlg;
    if(myParamStr.rectFlg){
        l.size=size;
        l.ksize_h =myParamStr.ksize_h;
        l.ksize_w =myParamStr.ksize_w;
        l.rectFlg =myParamStr.rectFlg;
        l.stride_h=myParamStr.stride_h;
        l.stride_w=myParamStr.stride_w;
        l.ksize=l.ksize_h*l.ksize_w;
        l.pad_h=l.ksize_h/2;
        l.pad_w=l.ksize_w/2;
        // printf("**********************RectConv********************\n");
        // printf("make_mix_convolutional_layer size:%d ksize:%d ksize_h:%d ksize_w:%d pad_h:%d pad_w:%d pad:%d stride:%d stride_h:%d stride_w:%d\n",l.size,l.ksize,l.ksize_h,l.ksize_w,l.pad_h,l.pad_w,l.pad,l.stride,l.stride_h,l.stride_w);
    }else{
        l.size = size;
        l.ksize_h =size;
        l.ksize_w =size;
        l.ksize= size*size;
        l.stride_h=l.stride;
        l.stride_w=l.stride;
        l.pad_h=l.ksize_h/2;
        l.pad_w=l.ksize_w/2;
    }
    
    // printf("make_mix_convolutional_layer size:%d ksize:%d ksize_h:%d ksize_w:%d pad_h:%d pad_w:%d pad:%d stride:%d stride_h:%d stride_w:%d\n",l.size,l.ksize,l.ksize_h,l.ksize_w,l.pad_h,l.pad_w,l.pad,l.stride,l.stride_h,l.stride_w);
    
    l.batch_normalize = batch_normalize;
    l.nweights = weightsSizeTotal;

    l.weights = calloc(l.nweights, sizeof(float));
    l.weight_updates = calloc(l.nweights, sizeof(float));

    l.biases = calloc(l.n, sizeof(float));
    l.bias_updates = calloc(l.n, sizeof(float));


    l.nbiases = l.n;
    // printf("l.nweights:%d\n",l.nweights);
    l.prune=myParamStr.prune;
    l.prune_threshold=calloc(1, sizeof(double));
    l.prune_threshold_min=myParamStr.prune_threshold_min;
    l.prune_threshold_max=myParamStr.prune_threshold_max;
    l.prune_threshold_step=myParamStr.prune_threshold_step;
    
    float scale = sqrt(2./(l.ksize*c));
    for(i = 0; i < l.nweights; ++i) l.weights[i] = scale*rand_normal();
    int out_w = mix_convolutional_out_width(l);
    int out_h = mix_convolutional_out_height(l);
    l.out_h = out_h;
    l.out_w = out_w;
    l.out_c = l.n;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = l.w * l.h * l.c;
//  printf("mix_conv out_w:%d out_h:%d:\n",l.out_w,l.out_h);
    l.output = calloc(l.batch*l.outputs, sizeof(float));
    l.delta  = calloc(l.batch*l.outputs, sizeof(float));

    l.forward = forward_mix_convolutional_layer;
    l.backward = backward_mix_convolutional_layer;
    l.update = update_mix_convolutional_layer;

    if(batch_normalize){
        l.scales = calloc(n, sizeof(float));
        l.scale_updates = calloc(n, sizeof(float));
        for(i = 0; i < n; ++i){
            l.scales[i] = 1;
        }

        l.mean = calloc(n, sizeof(float));
        l.variance = calloc(n, sizeof(float));

        l.mean_delta = calloc(n, sizeof(float));
        l.variance_delta = calloc(n, sizeof(float));

        l.rolling_mean = calloc(n, sizeof(float));
        l.rolling_variance = calloc(n, sizeof(float));
        l.x = calloc(l.batch*l.outputs, sizeof(float));
        l.x_norm = calloc(l.batch*l.outputs, sizeof(float));
    }
    if(adam){
        l.m = calloc(l.nweights, sizeof(float));
        l.v = calloc(l.nweights, sizeof(float));
        l.bias_m = calloc(n, sizeof(float));
        l.scale_m = calloc(n, sizeof(float));
        l.bias_v = calloc(n, sizeof(float));
        l.scale_v = calloc(n, sizeof(float));
    }

#ifdef GPU
    l.forward_gpu = forward_mix_convolutional_layer_gpu;
    l.backward_gpu = backward_mix_convolutional_layer_gpu;
    l.update_gpu = update_mix_convolutional_layer_gpu;

    if(gpu_index >= 0){
        if (adam) {
            l.m_gpu = cuda_make_array(l.m, l.nweights);
            l.v_gpu = cuda_make_array(l.v, l.nweights);
            l.bias_m_gpu = cuda_make_array(l.bias_m, n);
            l.bias_v_gpu = cuda_make_array(l.bias_v, n);
            l.scale_m_gpu = cuda_make_array(l.scale_m, n);
            l.scale_v_gpu = cuda_make_array(l.scale_v, n);
        }

        l.weights_gpu = cuda_make_array(l.weights, l.nweights);
        l.weight_updates_gpu = cuda_make_array(l.weight_updates, l.nweights);

        l.biases_gpu = cuda_make_array(l.biases, n);
        l.bias_updates_gpu = cuda_make_array(l.bias_updates, n);

        l.delta_gpu = cuda_make_array(l.delta, l.batch*out_h*out_w*n);
        l.output_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);

        if(batch_normalize){
            l.mean_gpu = cuda_make_array(l.mean, n);
            l.variance_gpu = cuda_make_array(l.variance, n);

            l.rolling_mean_gpu = cuda_make_array(l.mean, n);
            l.rolling_variance_gpu = cuda_make_array(l.variance, n);

            l.mean_delta_gpu = cuda_make_array(l.mean, n);
            l.variance_delta_gpu = cuda_make_array(l.variance, n);

            l.scales_gpu = cuda_make_array(l.scales, n);
            l.scale_updates_gpu = cuda_make_array(l.scale_updates, n);

            l.x_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);
            l.x_norm_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);
        }
#ifdef CUDNN
        // l.sub_layers=calloc(c*batch, sizeof(layer));
        // printf("make sub layers\n");
        // // make_2d_convolutional_layer(int h, int w, int size, int stride, int padding, myParameters myParamStr)
        // for(int i=0;i<c*batch;i++){
        //     l.sub_layers[i]=make_2d_convolutional_layer(h,w,size,stride,padding,batch_normalize,myParamStr);
        // }
#endif
    }
#endif
    l.workspace_size = get_workspace_size(l);
    // printf("workspace size:%d\n",l.workspace_size);
    l.activation = activation;
    if(l.prune){
        fprintf(stderr, "mix conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d  %5.3f BFLOPs Prune Min:%lf Max:%lf Step:%lf \n", c, l.ksize_h, l.ksize_w, stride, w, h, c, l.out_w, l.out_h, l.out_c, (2.0 * l.n * l.ksize*l.c * l.out_h*l.out_w)/1000000000.,l.prune_threshold_min,l.prune_threshold_max,l.prune_threshold_step);
    }else{
        fprintf(stderr, "mix conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d  %5.3f BFLOPs\n", c, l.ksize_h, l.ksize_w, stride, w, h, c, l.out_w, l.out_h, l.out_c, (2.0 * l.n * l.ksize*l.c * l.out_h*l.out_w)/1000000000.);
    }

    return l;
}


void forward_mix_convolutional_layer(mix_convolutional_layer l, network net)
{
    // printf("forward_mix_convolutional_layer check point 0\n");
    fill_cpu(l.outputs*l.batch, 0, l.output, 1);

    int a_add=0;
    int b_add=0;
    int c_add=0;
    int m_add=0;
    int m;
    int k;
    int n;
    for(int i = 0; i < l.batch; ++i){
        for(int j=0;j<l.sub_layers_Num;j++){
            layer sub_l=l.sub_layers[j];

            m = sub_l.n;
            k = sub_l.ksize*sub_l.c;
            n = sub_l.out_w*sub_l.out_h;
            printf("m:%d k:%d n:%d l.nweights:%d\n",m,k,n,l.nweights);

            float *a = l.weights + a_add;
            float *b = net.workspace;
            float *c = l.output +c_add;
            float *im =  net.input + m_add;
            if (sub_l.size == 1) {
                b = im;
            } else {
                if(l.rectFlg){
                    // printf("**************************************CPU ONLY**********************************************\n");
                    rect_im2col_cpu(im, l.c, l.h, l.w, l.ksize_h,l.ksize_w, l.stride_h,l.stride_w, l.pad_h,l.pad_w, b);
                }else{
                    im2col_cpu(im, sub_l.c, sub_l.h, sub_l.w, sub_l.size, sub_l.stride, sub_l.pad, b);
                }
                
            }
            gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
            a_add+=sub_l.nweights;
            c_add+=n*m;
            m_add+=sub_l.c*sub_l.h*sub_l.w;
        }
    }
    if (l.batch_normalize) {
        forward_batchnorm_layer(l, net);
    } else {
        add_bias(l.output, l.biases, l.batch, l.n, l.out_w*l.out_h);
    }
    activate_array(l.output, l.outputs*l.batch, l.activation);
}

void backward_mix_convolutional_layer(mix_convolutional_layer l, network net)
{
    // printf("backward_mix_convolutional_layer_gpu chepoint:1\n");
    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);
    if(l.batch_normalize){
        backward_batchnorm_layer(l, net);
    } else {
        backward_bias(l.bias_updates, l.delta, l.batch, l.n, l.out_w*l.out_h);
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


            float *a = l.delta + a_add;
            float *b = net.workspace;
            float *c = l.weight_updates + c_add;

            float *im  = net.input+m_add;
            float *imd = net.delta+m_add;

            if(l.rectFlg){
                // rect_im2col_gpu(im, sub_l.c, l.h, l.w, l.ksize_h,l.ksize_w,  l.stride_h,l.stride_w, l.pad_h,l.pad_w, b);
            }else{
                im2col_cpu(im, sub_l.c, sub_l.h, sub_l.w, sub_l.size, sub_l.stride, sub_l.pad, b);
            }
            // im2col_gpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
            gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);

            if (net.delta) {
                a = l.weights + c_add;
                b = l.delta + a_add;
                c = net.workspace;
                if (sub_l.size == 1) {
                    c = imd;
                }

                gemm(1,0,n,k,m,1,a,n,b,k,0,c,k);

                if (sub_l.size != 1) {
                    if(l.rectFlg){
                        // rect_col2im_gpu(net.workspace, l.c/l.groups, l.h, l.w, l.ksize_h,l.ksize_w,  l.stride_h,l.stride_w, l.pad_h,l.pad_w, imd);
                    }else{
                        col2im_cpu(net.workspace, sub_l.c, sub_l.h, sub_l.w, sub_l.size, sub_l.stride, sub_l.pad, imd);
                    }
                }
                a_add+=m*k;
                c_add+=sub_l.nweights;
                m_add+=sub_l.c*sub_l.h*sub_l.w;
            }
        }
    }
}

void update_mix_convolutional_layer(mix_convolutional_layer l, update_args a)
{
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;

    // int size = l.ksize*l.c;

    axpy_cpu(l.n, learning_rate/batch, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.n, momentum, l.bias_updates, 1);

    if(l.scales){
        axpy_cpu(l.n, learning_rate/batch, l.scale_updates, 1, l.scales, 1);
        scal_cpu(l.n, momentum, l.scale_updates, 1);
    }

    axpy_cpu(l.nweights, -decay*batch, l.weights, 1, l.weight_updates, 1);
    axpy_cpu(l.nweights, learning_rate/batch, l.weight_updates, 1, l.weights, 1);
    scal_cpu(l.nweights, momentum, l.weight_updates, 1);
}

void denormalize_mix_convolutional_layer(mix_convolutional_layer l)
{
    int i, j;
    for(i = 0; i < l.n; ++i){
        float scale = l.scales[i]/sqrt(l.rolling_variance[i] + .00001);
        for(j = 0; j < l.ksize; ++j){
            l.weights[i*l.size*l.size + j] *= scale;
        }
        l.biases[i] -= l.rolling_mean[i] * scale;
        l.scales[i] = 1;
        l.rolling_mean[i] = 0;
        l.rolling_variance[i] = 1;
    }
}


void resize_mix_convolutional_layer(mix_convolutional_layer *l, int w, int h)
{
    
    // printf("mix_conv out_w:%d out_h:%d l.n:%d l.c:%d l.out_c:%d\n",l->out_w,l->out_h,l->n,l->c,l->out_c);
    for(int mixID=0;mixID<l->sub_layers_Num;mixID++){
        layer sub_l=l->sub_layers[mixID];
        sub_l.out_w = mix_convolutional_out_width(sub_l);
        sub_l.out_h = mix_convolutional_out_height(sub_l);
        sub_l.w = w;
        sub_l.h = h;
        sub_l.outputs = sub_l.out_h * sub_l.out_w * sub_l.n;
        sub_l.inputs = sub_l.w * sub_l.h * sub_l.c;
        // printf("mix_sub_conv out_w:%d out_h:%d l.n:%d l.c:%d l.out_c:%d\n",sub_l.out_w,sub_l.out_h,sub_l.n,sub_l.c,sub_l.out_c);
    }
    
    l->w = w;
    l->h = h;

    int out_w = l->sub_layers[0].out_w;
    int out_h = l->sub_layers[0].out_h;

    l->out_w = out_w;
    l->out_h = out_h;

    l->outputs = l->out_h * l->out_w * l->out_c;
    l->inputs = l->w * l->h * l->c;

    l->output = realloc(l->output, l->batch*l->outputs*sizeof(float));
    l->delta  = realloc(l->delta,  l->batch*l->outputs*sizeof(float));
    if(l->batch_normalize){
        l->x = realloc(l->x, l->batch*l->outputs*sizeof(float));
        l->x_norm  = realloc(l->x_norm, l->batch*l->outputs*sizeof(float));
    }

#ifdef GPU
    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);

    l->delta_gpu =  cuda_make_array(l->delta,  l->batch*l->outputs);
    l->output_gpu = cuda_make_array(l->output, l->batch*l->outputs);

    if(l->batch_normalize){
        cuda_free(l->x_gpu);
        cuda_free(l->x_norm_gpu);

        l->x_gpu = cuda_make_array(l->output, l->batch*l->outputs);
        l->x_norm_gpu = cuda_make_array(l->output, l->batch*l->outputs);
    }
    #ifdef CUDNN
        // for(int i=0;i<l->c*l->batch;i++){
        //     l->sub_layers[i].w = w;
        //     l->sub_layers[i].h = h;

        //     l->sub_layers[i].out_w = out_w;
        //     l->sub_layers[i].out_h = out_h;

        //     l->sub_layers[i].outputs = l->out_h * l->out_w;
        //     l->sub_layers[i].inputs = l->w * l->h;
            
        //     l->sub_layers[i].output = realloc(l->sub_layers[i].output, l->sub_layers[i].outputs*sizeof(float));
        //     l->sub_layers[i].delta  = realloc(l->sub_layers[i].delta,  l->sub_layers[i].outputs*sizeof(float));

        //     cuda_free(l->sub_layers[i].delta_gpu);
        //     cuda_free(l->sub_layers[i].output_gpu);


        //     if(l->batch_normalize){
        //         l->sub_layers[i].x = realloc(l->sub_layers[i].x, l->sub_layers[i].outputs*sizeof(float));
        //         l->sub_layers[i].x_norm  = realloc(l->sub_layers[i].x_norm, l->sub_layers[i].outputs*sizeof(float));
        //     }


        //     if(l->batch_normalize){
        //         cuda_free(l->sub_layers[i].x_gpu);
        //         cuda_free(l->sub_layers[i].x_norm_gpu);

        //         l->sub_layers[i].x_gpu = cuda_make_array(l->sub_layers[i].output, l->sub_layers[i].outputs);
        //         l->sub_layers[i].x_norm_gpu = cuda_make_array(l->sub_layers[i].output, l->sub_layers[i].outputs);
        //     }
        //     l->sub_layers[i].delta_gpu =  cuda_make_array(l->sub_layers[i].delta, l->sub_layers[i].outputs);
        //     l->sub_layers[i].output_gpu = cuda_make_array(l->sub_layers[i].output, l->sub_layers[i].outputs);
        // }
        cudnn_mix_convolutional_setup(l);
    #endif
    #endif
    l->workspace_size = get_workspace_size(*l);
    // printf("resize***********\n");
}


void add_bias_mix(float *output, float *biases, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] += biases[i];
            }
        }
    }
}

void scale_bias_mix(float *output, float *scales, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] *= scales[i];
            }
        }
    }
}

void backward_bias_mix(float *bias_updates, float *delta, int batch, int n, int size)
{
    int i,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            bias_updates[i] += sum_array(delta+size*(i+b*n), size);
        }
    }
}
