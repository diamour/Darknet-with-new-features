#ifndef SE_LAYER_H
#define SE_LAYER_H

#include "cuda.h"
#include "image.h"
#include "activations.h"
#include "layer.h"
#include "network.h"

typedef layer se_layer;

#ifdef GPU
void forward_se_layer_gpu(se_layer layer, network net);
void backward_se_layer_gpu(se_layer layer, network net);
void update_se_layer_gpu(se_layer layer, update_args a);

void push_se_layer(se_layer layer);
void pull_se_layer(se_layer layer);

#endif
// layer make_2d_convolutional_layer(int h, int w, int size, int stride, int padding, myParameters myParamStr);
se_layer make_se_layer(int batch, int h, int w, int c, ACTIVATION activation, int batch_normalize,  int adam, myParameters myParamStr);
// se_layer make_se_layer(int batch, int h, int w, int c, int n, int size, int stride, int padding, ACTIVATION activation, int batch_normalize,  int adam, myParameters myParamStr);
void resize_se_layer(se_layer *layer, int w, int h);
void forward_se_layer(const se_layer layer, network net);
void backward_se_layer(se_layer layer, network net);
void update_se_layer(se_layer layer, update_args a);
#endif

