#include "depthwise_convolutional_layer.h"
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


int depthwise_convolutional_out_height(depthwise_convolutional_layer l)
{
    return (l.h + 2*l.pad_h - l.ksize_h) / l.stride_h + 1;
}

int depthwise_convolutional_out_width(depthwise_convolutional_layer l)
{
    return (l.w + 2*l.pad_w - l.ksize_w) / l.stride_w + 1;
}

static size_t get_workspace_size(layer l){
#ifdef CUDNN
    if(gpu_index >= 0){
        int size_total=0;
        for(int i=0;i<l.batch*l.c;i++){
            size_total+=l.sub_layers[i].workspace_size;
        }
        return (size_t)size_total;
    }
#endif
    return (size_t)l.out_h*l.out_w*l.ksize*l.c*sizeof(float);
}

static size_t get_2d_workspace_size(layer l){
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
    return (size_t)l.out_h*l.out_w*l.ksize*l.c*sizeof(float);
}

#ifdef GPU
#ifdef CUDNN
void cudnn_depthwise_convolutional_setup(layer *l)
{
    for(int b=0;b<l->batch;b++){
        for(int c=0;c<l->c;c++){
            int l_id=c+b*l->c;   
            layer sub_l = l->sub_layers[l_id];
            cudnn_2d_convolutional_setup(&sub_l);
        }
    }
}
void cudnn_2d_convolutional_setup(layer *l)
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


layer make_2d_convolutional_layer(int h, int w, int size, int stride, int padding, int norm, myParameters myParamStr)
{
    // printf("make_depthwise_convolutional_layer\n");
    int i;
    int c=1;
    layer l = {0};
    l.type = DEPTHWISE_CONVOLUTIONAL;

    l.h = h;
    l.w = w;
    l.c = 1;
    l.n = 1;

    l.batch = 1;
    l.stride = stride;
    l.pad = padding;
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
    }else{
        l.size = size;
        l.ksize_h =size;
        l.ksize_w =size;
        l.ksize= size*size;
        l.stride_h=l.stride;
        l.stride_w=l.stride;
    }
    
    // printf("make_2d_convolutional_layer size:%d ksize:%d ksize_h:%d ksize_w:%d pad_h:%d pad_w:%d pad:%d stride:%d stride_h:%d stride_w:%d\n",l.size,l.ksize,l.ksize_h,l.ksize_w,l.pad_h,l.pad_w,l.pad,l.stride,l.stride_h,l.stride_w);

    l.weights = calloc(c*l.ksize, sizeof(float));
    l.weight_updates = calloc(c*l.ksize, sizeof(float));

    l.biases = calloc(l.n, sizeof(float));
    l.bias_updates = calloc(l.n, sizeof(float));

    l.nweights = c*l.ksize;
    l.nbiases = l.n;

    l.prune=myParamStr.prune;
    l.prune_threshold=calloc(1, sizeof(double));
    l.prune_threshold_min=myParamStr.prune_threshold_min;
    l.prune_threshold_max=myParamStr.prune_threshold_max;
    l.prune_threshold_step=myParamStr.prune_threshold_step;
    float scale = sqrt(2./(l.ksize*c));

    for(i = 0; i < l.nweights; ++i) l.weights[i] = scale*rand_normal();
    int out_w = depthwise_convolutional_out_width(l);
    int out_h = depthwise_convolutional_out_height(l);
    l.out_h = out_h;
    l.out_w = out_w;
    l.out_c = l.n;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = l.w * l.h * l.c;
    if(norm){
        l.scales = calloc(c, sizeof(float));
        l.scale_updates = calloc(c, sizeof(float));
        for(i = 0; i < c; ++i){
            l.scales[i] = 1;
        }

        l.mean = calloc(c, sizeof(float));
        l.variance = calloc(c, sizeof(float));

        l.mean_delta = calloc(c, sizeof(float));
        l.variance_delta = calloc(c, sizeof(float));

        l.rolling_mean = calloc(c, sizeof(float));
        l.rolling_variance = calloc(c, sizeof(float));
        l.x = calloc(l.batch*l.outputs, sizeof(float));
        l.x_norm = calloc(l.batch*l.outputs, sizeof(float));
    }
    l.output = calloc(l.batch*l.outputs, sizeof(float));
    l.delta  = calloc(l.batch*l.outputs, sizeof(float));

#ifdef GPU
    if(gpu_index >= 0){
        l.weights_gpu = cuda_make_array(l.weights, l.nweights);
        l.weight_updates_gpu = cuda_make_array(l.weight_updates, l.nweights);

        l.biases_gpu = cuda_make_array(l.biases, c);
        l.bias_updates_gpu = cuda_make_array(l.bias_updates, c);

        l.delta_gpu = cuda_make_array(l.delta, out_h*out_w*c);
        l.output_gpu = cuda_make_array(l.output, out_h*out_w*c);

        if(norm){
            l.mean_gpu = cuda_make_array(l.mean, c);
            l.variance_gpu = cuda_make_array(l.variance, c);

            l.rolling_mean_gpu = cuda_make_array(l.mean, c);
            l.rolling_variance_gpu = cuda_make_array(l.variance, c);

            l.mean_delta_gpu = cuda_make_array(l.mean, c);
            l.variance_delta_gpu = cuda_make_array(l.variance, c);

            l.scales_gpu = cuda_make_array(l.scales, c);
            l.scale_updates_gpu = cuda_make_array(l.scale_updates, c);

            l.x_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*c);
            l.x_norm_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*c);
        }
#ifdef CUDNN
        cudnnCreateTensorDescriptor(&l.normTensorDesc);
        cudnnCreateTensorDescriptor(&l.srcTensorDesc);
        cudnnCreateTensorDescriptor(&l.dstTensorDesc);
        cudnnCreateFilterDescriptor(&l.weightDesc);
        cudnnCreateTensorDescriptor(&l.dsrcTensorDesc);
        cudnnCreateTensorDescriptor(&l.ddstTensorDesc);
        cudnnCreateFilterDescriptor(&l.dweightDesc);
        cudnnCreateConvolutionDescriptor(&l.convDesc);
        cudnn_2d_convolutional_setup(&l);
#endif
    }
#endif
    l.workspace_size = get_2d_workspace_size(l);

    return l;
}


depthwise_convolutional_layer make_depthwise_convolutional_layer(int batch, int h, int w, int c, int size, int stride, int padding, ACTIVATION activation, int batch_normalize,  int adam, myParameters myParamStr)
{
    
    printf("make_depthwise_convolutional_layer\n");

    int i;
    depthwise_convolutional_layer l = {0};
    l.type = DEPTHWISE_CONVOLUTIONAL;

    l.h = h;
    l.w = w;
    l.c = c;
    //the out put channel equals to the input channel
    l.n = c;

    l.batch = batch;
    l.stride = stride;
    l.pad = padding;
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
        // printf("make_depthwise_convolutional_layer size:%d ksize:%d ksize_h:%d ksize_w:%d pad_h:%d pad_w:%d pad:%d stride:%d stride_h:%d stride_w:%d\n",l.size,l.ksize,l.ksize_h,l.ksize_w,l.pad_h,l.pad_w,l.pad,l.stride,l.stride_h,l.stride_w);
    }else{
        l.size = size;
        l.ksize_h =size;
        l.ksize_w =size;
        l.ksize= size*size;
        l.stride_h=l.stride;
        l.stride_w=l.stride;
    }
    
    // printf("make_depthwise_convolutional_layer size:%d ksize:%d ksize_h:%d ksize_w:%d pad_h:%d pad_w:%d pad:%d stride:%d stride_h:%d stride_w:%d\n",l.size,l.ksize,l.ksize_h,l.ksize_w,l.pad_h,l.pad_w,l.pad,l.stride,l.stride_h,l.stride_w);
    
    l.batch_normalize = batch_normalize;

    l.weights = calloc(c*l.ksize, sizeof(float));
    l.weight_updates = calloc(c*l.ksize, sizeof(float));

    l.biases = calloc(l.n, sizeof(float));
    l.bias_updates = calloc(l.n, sizeof(float));

    l.nweights = c*l.ksize;
    l.nbiases = l.n;
    // printf("l.nweights:%d\n",l.nweights);
    l.prune=myParamStr.prune;
    l.prune_threshold=calloc(1, sizeof(double));
    l.prune_threshold_min=myParamStr.prune_threshold_min;
    l.prune_threshold_max=myParamStr.prune_threshold_max;
    l.prune_threshold_step=myParamStr.prune_threshold_step;
    
    // float scale = 1./sqrt(l.ksize*c);
    float scale = sqrt(2./(l.ksize*c));
    //printf("convscale %f\n", scale);
    //scale = .02;
    //for(i = 0; i < c*n*l.ksize; ++i) l.weights[i] = scale*rand_uniform(-1, 1);
    for(i = 0; i < l.nweights; ++i) l.weights[i] = scale*rand_normal();
    int out_w = depthwise_convolutional_out_width(l);
    int out_h = depthwise_convolutional_out_height(l);
    l.out_h = out_h;
    l.out_w = out_w;
    l.out_c = l.n;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = l.w * l.h * l.c;

    l.output = calloc(l.batch*l.outputs, sizeof(float));
    l.delta  = calloc(l.batch*l.outputs, sizeof(float));

    l.forward = forward_depthwise_convolutional_layer;
    l.backward = backward_depthwise_convolutional_layer;
    l.update = update_depthwise_convolutional_layer;


    if(batch_normalize){
        l.scales = calloc(c, sizeof(float));
        l.scale_updates = calloc(c, sizeof(float));
        for(i = 0; i < c; ++i){
            l.scales[i] = 1;
        }

        l.mean = calloc(c, sizeof(float));
        l.variance = calloc(c, sizeof(float));

        l.mean_delta = calloc(c, sizeof(float));
        l.variance_delta = calloc(c, sizeof(float));

        l.rolling_mean = calloc(c, sizeof(float));
        l.rolling_variance = calloc(c, sizeof(float));
        l.x = calloc(l.batch*l.outputs, sizeof(float));
        l.x_norm = calloc(l.batch*l.outputs, sizeof(float));
    }
    if(adam){
        l.m = calloc(l.nweights, sizeof(float));
        l.v = calloc(l.nweights, sizeof(float));
        l.bias_m = calloc(c, sizeof(float));
        l.scale_m = calloc(c, sizeof(float));
        l.bias_v = calloc(c, sizeof(float));
        l.scale_v = calloc(c, sizeof(float));
    }

#ifdef GPU
    l.forward_gpu = forward_depthwise_convolutional_layer_gpu;
    l.backward_gpu = backward_depthwise_convolutional_layer_gpu;
    l.update_gpu = update_depthwise_convolutional_layer_gpu;

    if(gpu_index >= 0){
        if (adam) {
            l.m_gpu = cuda_make_array(l.m, l.nweights);
            l.v_gpu = cuda_make_array(l.v, l.nweights);
            l.bias_m_gpu = cuda_make_array(l.bias_m, c);
            l.bias_v_gpu = cuda_make_array(l.bias_v, c);
            l.scale_m_gpu = cuda_make_array(l.scale_m, c);
            l.scale_v_gpu = cuda_make_array(l.scale_v, c);
        }

        l.weights_gpu = cuda_make_array(l.weights, l.nweights);
        l.weight_updates_gpu = cuda_make_array(l.weight_updates, l.nweights);

        l.biases_gpu = cuda_make_array(l.biases, c);
        l.bias_updates_gpu = cuda_make_array(l.bias_updates, c);

        l.delta_gpu = cuda_make_array(l.delta, l.batch*out_h*out_w*c);
        l.output_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*c);


        if(batch_normalize){
            l.mean_gpu = cuda_make_array(l.mean, c);
            l.variance_gpu = cuda_make_array(l.variance, c);

            l.rolling_mean_gpu = cuda_make_array(l.mean, c);
            l.rolling_variance_gpu = cuda_make_array(l.variance, c);

            l.mean_delta_gpu = cuda_make_array(l.mean, c);
            l.variance_delta_gpu = cuda_make_array(l.variance, c);

            l.scales_gpu = cuda_make_array(l.scales, c);
            l.scale_updates_gpu = cuda_make_array(l.scale_updates, c);

            l.x_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*c);
            l.x_norm_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*c);
        }
#ifdef CUDNN
        l.sub_layers=calloc(c*batch, sizeof(layer));
        printf("make sub layers\n");
        // make_2d_convolutional_layer(int h, int w, int size, int stride, int padding, myParameters myParamStr)
        for(int i=0;i<c*batch;i++){
            l.sub_layers[i]=make_2d_convolutional_layer(h,w,size,stride,padding,batch_normalize,myParamStr);
        }
#endif
    }
#endif
    l.workspace_size = get_workspace_size(l);
    // printf("workspace size:%d\n",l.workspace_size);
    l.activation = activation;
    if(l.prune){
        fprintf(stderr, "depthwise conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d  %5.3f BFLOPs Prune Min:%lf Max:%lf Step:%lf \n", c, l.ksize_h, l.ksize_w, stride, w, h, c, l.out_w, l.out_h, l.out_c, (2.0 * l.n * l.ksize*l.c * l.out_h*l.out_w)/1000000000.,l.prune_threshold_min,l.prune_threshold_max,l.prune_threshold_step);
    }else{
        fprintf(stderr, "depthwise conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d  %5.3f BFLOPs\n", c, l.ksize_h, l.ksize_w, stride, w, h, c, l.out_w, l.out_h, l.out_c, (2.0 * l.n * l.ksize*l.c * l.out_h*l.out_w)/1000000000.);
    }

    return l;
}


void forward_depthwise_convolutional_layer(depthwise_convolutional_layer l, network net)
{
    fill_cpu(l.outputs*l.batch, 0, l.output, 1);
    int k = l.ksize;
    int n = l.out_w*l.out_h;
    for(int b = 0; b < l.batch; ++b){
        for (int c=0;c<l.c;c++)
        {
            float *aoffset = l.weights+c*l.ksize;
            float *boffset = net.workspace;
            float *coffset = l.output+c*l.out_h*l.out_w+b*l.n*l.out_h*l.out_w;
            float *input_offset = net.input + c*l.h*l.w+ b*l.c*l.h*l.w;

            if(l.rectFlg){
                rect_im2col_cpu(input_offset, 1, l.h, l.w, l.ksize_h,l.ksize_w, l.stride_h,l.stride_w, l.pad_h,l.pad_w, boffset);
            }else{
                im2col_cpu(input_offset, l.c, l.h, l.w, l.size, l.stride, l.pad, boffset);
            }
                
            gemm(0, 0, 1, n, k, 1, aoffset, k, boffset, n, 1, coffset, n);
        }
    }
    if (l.batch_normalize) {
        forward_batchnorm_layer(l, net);
    } else {
        add_bias(l.output, l.biases, l.batch, l.n, l.out_w*l.out_h);
    }
    activate_array(l.output, l.outputs*l.batch, l.activation);
}

void backward_depthwise_convolutional_layer(depthwise_convolutional_layer l, network net)
{
    // int i;
    int m = l.n;
    int n = l.ksize;
    int k = l.out_w*l.out_h;
    gradient_array(l.output, m*k*l.batch, l.activation, l.delta);

    if(l.batch_normalize){
        backward_batchnorm_layer(l, net);
    } else {
        backward_bias(l.bias_updates, l.delta, l.batch, l.n, k);
    }


	for (int b = 0; b < l.batch; ++b) {
		for (int c = 0; c<l.c; c++)
		{
			float *aoffset = l.delta + c*l.out_h*l.out_w + b*l.n*l.out_h*l.out_w;
			float *boffset = net.workspace;
			float *coffset = l.weight_updates + c*l.ksize;

			float *im = net.input + c*l.h*l.w + b*l.c*l.h*l.w;

            if(l.size == 1){
                boffset = im;
            } else {
                if(l.rectFlg){
                    // printf("**************************************CPU ONLY**********************************************\n");
                    rect_im2col_cpu(im, 1, l.h, l.w, l.ksize_h,l.ksize_w, l.stride_h,l.stride_w, l.pad_h,l.pad_w, boffset);
                }else{
                    im2col_cpu(im, 1, l.h, l.w, l.size, l.stride, l.pad, boffset);
                }
            }

            gemm(0, 1, 1, n, k, 1, aoffset, k, boffset, k, 1, coffset, n);

            if (net.delta) {
				aoffset = l.weights+ c*l.ksize;
				boffset = l.delta + c*l.out_h*l.out_w + b*l.n*l.out_h*l.out_w;
				coffset = net.workspace;

                gemm(1, 0, n, k, 1, 1, aoffset, n, boffset, k, 0, coffset, k);

                if(l.rectFlg){
                // printf("**************************************CPU ONLY**********************************************\n");
                    rect_col2im_cpu(net.workspace, 1, l.h, l.w, l.ksize_h,l.ksize_w, l.stride_h, l.stride_w, l.pad_h,l.pad_w, net.delta + c*l.h*l.w + b*l.n*l.h*l.w);
                }else{
                    col2im_cpu(net.workspace, 1, l.h, l.w, l.size, l.stride, l.pad, net.delta + c*l.h*l.w + b*l.n*l.h*l.w);
                }
            }
        }
    }
}

void update_depthwise_convolutional_layer(depthwise_convolutional_layer l, update_args a)
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

void denormalize_depthwise_convolutional_layer(depthwise_convolutional_layer l)
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


void resize_depthwise_convolutional_layer(depthwise_convolutional_layer *l, int w, int h)
{
    l->w = w;
    l->h = h;
    int out_w = depthwise_convolutional_out_width(*l);
    int out_h = depthwise_convolutional_out_height(*l);

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
    // printf("");
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
        cudnn_depthwise_convolutional_setup(l);
    #endif
    #endif
    l->workspace_size = get_workspace_size(*l);

}


void add_bias_depthwise(float *output, float *biases, int batch, int n, int size)
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

void scale_bias_depthwise(float *output, float *scales, int batch, int n, int size)
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

void backward_bias_depthwise(float *bias_updates, float *delta, int batch, int n, int size)
{
    int i,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            bias_updates[i] += sum_array(delta+size*(i+b*n), size);
        }
    }
}
