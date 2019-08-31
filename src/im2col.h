#ifndef IM2COL_H
#define IM2COL_H

void im2col_cpu(float* data_im,
        int channels, int height, int width,
        int ksize, int stride, int pad, float* data_col);
        
void rect_im2col_cpu(float* data_im,
     int channels,  int height,  int width,
     int ksize_h,int ksize_w, int stride_h, int stride_w,int pad_h, int pad_w, float* data_col);
#ifdef GPU

void im2col_gpu(float *im,
         int channels, int height, int width,
         int ksize, int stride, int pad,float *data_col);

void rect_im2col_gpu(float *im,
        int channels, int height, int width,
        int ksize_h,int ksize_w, int stride_h,int stride_w, int pad_h,int pad_w, float *data_col);
#endif
#endif
