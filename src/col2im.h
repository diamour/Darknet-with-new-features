#ifndef COL2IM_H
#define COL2IM_H

void col2im_cpu(float* data_col,
        int channels, int height, int width,
        int ksize, int stride, int pad, float* data_im);
void rect_col2im_cpu(float* data_col,
         int channels,  int height,  int width,
         int ksize_h,int ksize_w, int stride_h,int stride_w, int pad_h, int pad_w, float* data_im);
#ifdef GPU
void col2im_gpu(float *data_col,
        int channels, int height, int width,
        int ksize, int stride, int pad, float *data_im);
void rect_col2im_gpu(float *data_col,
    int channels, int height, int width,
    int ksize_h,int ksize_w, int stride_h,int stride_w, int pad_h,int pad_w, float *data_im);
#endif
#endif
