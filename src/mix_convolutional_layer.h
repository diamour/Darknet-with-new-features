#ifndef MIX_CONVOLUTIONAL_LAYER_H
#define MIX_CONVOLUTIONAL_LAYER_H

#include "cuda.h"
#include "image.h"
#include "activations.h"
#include "layer.h"
#include "network.h"

typedef layer mix_convolutional_layer;

#ifdef GPU
void forward_mix_convolutional_layer_gpu(mix_convolutional_layer layer, network net);
void backward_mix_convolutional_layer_gpu(mix_convolutional_layer layer, network net);
void update_mix_convolutional_layer_gpu(mix_convolutional_layer layer, update_args a);

void push_mix_convolutional_layer(mix_convolutional_layer layer);
void pull_mix_convolutional_layer(mix_convolutional_layer layer);

void add_bias_gpu(float *output, float *biases, int batch, int n, int size);
void backward_bias_gpu(float *bias_updates, float *delta, int batch, int n, int size);
void adam_update_gpu(float *w, float *d, float *m, float *v, float B1, float B2, float eps, float decay, float rate, int n, int batch, int t);
#ifdef CUDNN
void cudnn_mix_convolutional_setup(layer *l);
#endif
#endif
// layer make_2d_convolutional_layer(int h, int w, int size, int stride, int padding, myParameters myParamStr);
mix_convolutional_layer make_mix_convolutional_layer(int batch, int h, int w, int c, int n,int mixNum ,int *sizeList, int stride, int padding, ACTIVATION activation, int batch_normalize,  int adam, myParameters myParamStr);
// mix_convolutional_layer make_mix_convolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, int padding, ACTIVATION activation, int batch_normalize,  int adam, myParameters myParamStr);
void resize_mix_convolutional_layer(mix_convolutional_layer *layer, int w, int h);
void forward_mix_convolutional_layer(const mix_convolutional_layer layer, network net);
void update_mix_convolutional_layer(mix_convolutional_layer layer, update_args a);


void denormalize_mix_convolutional_layer(mix_convolutional_layer l);
void backward_mix_convolutional_layer(mix_convolutional_layer layer, network net);

void add_bias(float *output, float *biases, int batch, int n, int size);
void backward_bias(float *bias_updates, float *delta, int batch, int n, int size);

int mix_convolutional_out_height(mix_convolutional_layer layer);
int mix_convolutional_out_width(mix_convolutional_layer layer);

#endif

