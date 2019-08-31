#include <stdio.h>
#include <math.h>
void col2im_add_pixel(float *im, int height, int width, int channels,
                        int row, int col, int channel, int pad_h,int pad_w, float val)
{
    row -= pad_h;
    col -= pad_w;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return;
    im[col + width*(row + height*channel)] += val;
}
//This one might be too, can't remember.
void col2im_cpu(float* data_col,
         int channels,  int height,  int width,
         int ksize, int stride, int pad, float* data_im) 
{
    int c,h,w;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                double val = data_col[col_index];
                col2im_add_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad,pad, val);
            }
        }
    }
}

void rect_col2im_cpu(float* data_col,
         int channels,  int height,  int width,
         int ksize_h,int ksize_w, int stride_h,int stride_w, int pad_h, int pad_w, float* data_im) 
{
    int c,h,w;
    int height_col = (height + 2*pad_h - ksize_h) / stride_h + 1;
    int width_col = (width + 2*pad_w - ksize_w) / stride_w + 1;

    int channels_col = channels * ksize_h * ksize_w;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize_w;
        int h_offset = (c / ksize_h) % ksize_h;
        int c_im = c / ksize_h / ksize_w;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride_h;
                int im_col = w_offset + w * stride_w;
                int col_index = (c * height_col + h) * width_col + w;
                double val = data_col[col_index];
                col2im_add_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad_h,pad_w, val);
            }
        }
    }
}
