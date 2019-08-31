#include "darknet.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "network.h"
// #include "n.h"
#include "depthwise_convolutional_layer.h"
#include "mix_convolutional_layer.h"
static int coco_ids[] = {1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90};


int DT_classifier(float center_x,float center_y, float obj_w,float obj_h,float img_w,float img_h){
    if(center_y<=867.25){
        if(center_x<=386.25){
            if(center_x<224.75){
                if(center_y<=797.75){
                    if(img_w <=589.5){
                        return  1;
                    }else{
                        if(center_x <=188.25){
                            return  1;
                        }
                        else{
                            return  0;
                        }
                    }
                }
                else{
                    return  0;
                }
            }else{
                if(center_y <= 835.25){
                    if(obj_w <= 16.5){
                        if(center_y <=734.75){
                            return  1;
                        }else{
                            if(center_y <=761.75){
                                return  0;
                            }else{
                                return  1;
                            }
                        }
                    }else{
                        return  1;
                    }
                }else{
                    if(center_y <= 837.25){
                        return  0;
                    }else{
                        if(center_y <= 839.5){
                            if(obj_w <= 37.0){
                                return  1;
                            }else{
                                return  0;
                            }
                        }else{
                            return  1;
                        }
                    }
                }
            }
        }else{
            if(obj_h <= 59.5){
                return  1;
            }else{
                return  0;
            }
        }
    }else{
        if(obj_h <= 73.5){
            if(center_x <= 325.75){
                if(center_y <= 933.25){
                    if(center_y < 930.25){
                        return  1;
                    }else{
                        return  0;
                    }
                }else{
                    return  1;
                }
            }else{
                if(obj_h <= 54.5){
                    return  1;
                }else{
                    return  0;
                }
            }
        }else{
           if(center_y <= 1035.75){
                if(obj_w <= 47.5){
                    if(img_w <= 572.5){
                        if(center_y <= 928.75){
                            if(img_h <= 1380.0){
                              if(center_y <= 928.25){
                                    if(obj_h <= 87.0){
                                        return  0;
                                    }else{
                                        if(obj_h <= 88.5){
                                            if(center_x <= 278.0){
                                                return  1;
                                            }else{
                                                return  0;
                                            }
                                        }else{
                                            return  0;
                                        }
                                    }
                              }else{
                                    if(center_x <= 272.25){
                                        return  1;
                                    }else{
                                        return  0;
                                    }
                              }
                            }else{
                                return  1;
                            }
                        }else{
                            if(obj_h <= 75.5){
                                if(img_h <= 1340.0){
                                    return  0;
                                }else{
                                    return  1;
                                }
                            }else{
                                if(center_x <= 166.25){
                                    if(center_x <= 165.75){
                                        return  0;
                                    }else{
                                        return  1;
                                    }
                                }else{
                                    if(obj_h <= 86.5){
                                        if(obj_h <= 85.5){
                                            return  0;
                                        }else{
                                            if(center_x <= 310.75){
                                                if(center_x <= 223.75){
                                                    return  0;
                                                }else{
                                                    return  1;
                                                }
                                            }else{
                                                return  0;
                                            }
                                        }
                                    }else{
                                        return  0;
                                    }
                                }
                            }
                        }
                    }
                    else{
                        return  1;
                    }
                }else{
                    if(center_y <= 1010.0){
                        return  1;
                    }else{
                        return  0;
                    }
                }
           }else{
                return  1;
           }
        }
    }
}


void train_detector(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear)
{
    list *options = read_data_cfg(datacfg);
    char *train_images = option_find_str(options, "train", "data/train.list");
    char *backup_directory = option_find_str(options, "backup", "/backup/");

    FILE *myfp = 0;
    myfp = fopen("loss.text", "w");
    
    srand(time(0));
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    float avg_loss = -1;
    // printf("ngpus:%d\n",ngpus);
    network **nets = calloc(ngpus, sizeof(network));

    srand(time(0));
    int seed = rand();
    int i;
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        // printf("ngpus:%d\n",gpus[i]);
        cuda_set_device(gpus[i]);
#endif
        nets[i] = load_network(cfgfile, weightfile, clear);
        nets[i]->learning_rate *= ngpus;
    }
    srand(time(0));
    network *net = nets[0];

    int imgs = net->batch * net->subdivisions * ngpus;
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    data train, buffer;

    layer l = net->layers[net->n - 1];

    int classes = l.classes;
    float jitter = l.jitter;

    list *plist = get_paths(train_images);
    //int N = plist->size;
    char **paths = (char **)list_to_array(plist);

    load_args args = get_base_args(net);
    args.coords = l.coords;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.classes = classes;
    args.jitter = jitter;
    args.num_boxes = l.max_boxes;
    args.d = &buffer;
    args.type = DETECTION_DATA;
    //args.type = INSTANCE_DATA;
    args.threads = 64;

    pthread_t load_thread = load_data(args);
    double time;
    int count = 0;
    //while(i*imgs < N*120){
    while(get_current_batch(net) < net->max_batches){
        if(l.random && count++%10 == 0){
            printf("Resizing\n");
            int dim = 416;
            // int dim = (rand() % 10 + 10) * 32;
            // if (get_current_batch(net)+200 > net->max_batches) dim = 608;
            // int dim = (rand() % 4 + 16) * 32;
            printf("%d\n", dim);
            args.w = dim;
            args.h = dim;

            pthread_join(load_thread, 0);
            train = buffer;
            free_data(train);
            load_thread = load_data(args);

            #pragma omp parallel for
            for(i = 0; i < ngpus; ++i){
                resize_network(nets[i], dim, dim);
            }
            printf("Resizing finish\n");
            net = nets[0];
        }
        time=what_time_is_it_now();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);

        /*
           int k;
           for(k = 0; k < l.max_boxes; ++k){
           box b = float_to_box(train.y.vals[10] + 1 + k*5);
           if(!b.x) break;
           printf("loaded: %f %f %f %f\n", b.x, b.y, b.w, b.h);
           }
         */
        /*
           int zz;
           for(zz = 0; zz < train.X.cols; ++zz){
           image im = float_to_image(net->w, net->h, 3, train.X.vals[zz]);
           int k;
           for(k = 0; k < l.max_boxes; ++k){
           box b = float_to_box(train.y.vals[zz] + k*5, 1);
           printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);
           draw_bbox(im, b, 1, 1,0,0);
           }
           show_image(im, "truth11");
           cvWaitKey(0);
           save_image(im, "truth11");
           }
         */

        printf("Loaded: %lf seconds\n", what_time_is_it_now()-time);

        time=what_time_is_it_now();
        float loss = 0;
#ifdef GPU
        if(ngpus == 1){
            loss = train_network(net, train);
        } else {
            loss = train_networks(nets, ngpus, train, 4);
        }
#else
        loss = train_network(net, train);
#endif
        if (avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        i = get_current_batch(net);
        
        fprintf(myfp, "%d loss:%f\n",count,loss);
        
        printf("%ld: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), loss, avg_loss, get_current_rate(net), what_time_is_it_now()-time, i*imgs);
        if(i%100==0){
#ifdef GPU
            if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
            char buff[256];
            sprintf(buff, "%s/%s.backup", backup_directory, base);
            save_weights(net, buff);
        }
        if(i%5000==0 || (i < 5000 && i%500 == 0)){
#ifdef GPU
            if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            save_weights(net, buff);
        }
        free_data(train);
    }
#ifdef GPU
    if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
    fclose(myfp);
}


void test_detector(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh, char *outfile, int fullscreen)
{
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);

    image **alphabet = load_alphabet();
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);
    double time;
    char buff[256];
    char *input = buff;
    float nms=.45;
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        } else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image im = load_image_color(input,0,0);
        image sized = letterbox_image(im, net->w, net->h);
        //image sized = resize_image(im, net->w, net->h);
        //image sized2 = resize_max(im, net->w);
        //image sized = crop_image(sized2, -((net->w - sized2.w)/2), -((net->h - sized2.h)/2), net->w, net->h);
        //resize_network(net, sized.w, sized.h);
        layer l = net->layers[net->n-1];


        float *X = sized.data;
        time=what_time_is_it_now();
        float *_out=network_predict(net, X);
        // int _i,_j;
        // for(_i=0;_i<26*26;_i++){
        //     for(_j=0;_j<5;_j++){
        //         printf("%f ",_out[_i+_j*26*26]);
        //         printf("%f ",_out[_i+(_j+1)*26*26]);
        //         printf("%f ",_out[_i+(_j+2)*26*26]);
        //         printf("%f ",_out[_i+(_j+3)*26*26]);
        //         printf("%f ",_out[_i+(_j+4)*26*26]);
        //         printf("%f ",_out[_i+(_j+5)*26*26]);
        //         printf("%f ",_out[_i+(_j+6)*26*26]);
        //         printf("%f ",_out[_i+(_j+7)*26*26]);
        //         printf("\n");
        //     }
            
        // }
        printf("%s: Predicted in %f seconds.\n", input, what_time_is_it_now()-time);
        
        int nboxes = 0;
        detection *dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes);
        //printf("%d\n", nboxes);
        //if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
        draw_detections(im, dets, nboxes, thresh, names, alphabet, l.classes);
        free_detections(dets, nboxes);
        if(outfile){
            save_image(im, outfile);
        }
        else{
            save_image(im, "predictions");
#ifdef OPENCV
            make_window("predictions", im.w, im.h, 0);
            show_image(im, "predictions", 0);
#endif
        }

        free_image(im);
        free_image(sized);
        if (filename) break;
    }
    
    char *base = basecfg(cfgfile);
    char *backup_directory = option_find_str(options, "backup", "/backup/");
    char _buff[256];
    sprintf(_buff, "%s/%s_final_prune.weights", backup_directory, base);
    // save_weights(net, _buff);
    
}

static int get_coco_image_id(char *filename)
{
    char *p = strrchr(filename, '/');
    char *c = strrchr(filename, '_');
    if(c) p = c;
    return atoi(p+1);
}

static void print_cocos(FILE *fp, char *image_path, detection *dets, int num_boxes, int classes, int w, int h)
{
    int i, j;
    int image_id = get_coco_image_id(image_path);
    for(i = 0; i < num_boxes; ++i){
        float xmin = dets[i].bbox.x - dets[i].bbox.w/2.;
        float xmax = dets[i].bbox.x + dets[i].bbox.w/2.;
        float ymin = dets[i].bbox.y - dets[i].bbox.h/2.;
        float ymax = dets[i].bbox.y + dets[i].bbox.h/2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        float bx = xmin;
        float by = ymin;
        float bw = xmax - xmin;
        float bh = ymax - ymin;

        for(j = 0; j < classes; ++j){
            if (dets[i].prob[j]) fprintf(fp, "{\"image_id\":%d, \"category_id\":%d, \"bbox\":[%f, %f, %f, %f], \"score\":%f},\n", image_id, coco_ids[j], bx, by, bw, bh, dets[i].prob[j]);
        }
    }
}

void print_detector_detections(FILE **fps, char *id, detection *dets, int total, int classes, int w, int h)
{
    int i, j;
    for(i = 0; i < total; ++i){
        float xmin = dets[i].bbox.x - dets[i].bbox.w/2. + 1;
        float xmax = dets[i].bbox.x + dets[i].bbox.w/2. + 1;
        float ymin = dets[i].bbox.y - dets[i].bbox.h/2. + 1;
        float ymax = dets[i].bbox.y + dets[i].bbox.h/2. + 1;

        if (xmin < 1) xmin = 1;
        if (ymin < 1) ymin = 1;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for(j = 0; j < classes; ++j){
            if (dets[i].prob[j]) fprintf(fps[j], "%s %f %f %f %f %f\n", id, dets[i].prob[j],
                    xmin, ymin, xmax, ymax);
        }
    }
}

float my_print_detector_detections(FILE **fps, char *id, detection *dets, int total, int classes, int w, int h)
{
    int i, j;
    float mean_det=0;
    for(i = 0; i < total; ++i){
        float xmin = dets[i].bbox.x - dets[i].bbox.w/2. + 1;
        float xmax = dets[i].bbox.x + dets[i].bbox.w/2. + 1;
        float ymin = dets[i].bbox.y - dets[i].bbox.h/2. + 1;
        float ymax = dets[i].bbox.y + dets[i].bbox.h/2. + 1;

        if (xmin < 1) xmin = 1;
        if (ymin < 1) ymin = 1;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for(j = 0; j < classes; ++j){
            mean_det+=dets[i].prob[j];
            if (dets[i].prob[j]) fprintf(fps[j], "%s %f %f %f %f %f\n", id, dets[i].prob[j],
                    xmin, ymin, xmax, ymax);
        }
    }
    printf("total:%d classes:%d mean_det:%f\n",total,classes,mean_det);
    return mean_det/classes;
}

void print_imagenet_detections(FILE *fp, int id, detection *dets, int total, int classes, int w, int h)
{
    int i, j;
    for(i = 0; i < total; ++i){
        float xmin = dets[i].bbox.x - dets[i].bbox.w/2.;
        float xmax = dets[i].bbox.x + dets[i].bbox.w/2.;
        float ymin = dets[i].bbox.y - dets[i].bbox.h/2.;
        float ymax = dets[i].bbox.y + dets[i].bbox.h/2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for(j = 0; j < classes; ++j){
            int class = j;
            if (dets[i].prob[class]) fprintf(fp, "%d %d %f %f %f %f %f\n", id, j+1, dets[i].prob[class],
                    xmin, ymin, xmax, ymax);
        }
    }
}

void validate_detector_flip(char *datacfg, char *cfgfile, char *weightfile, char *outfile)
{
    int j;
    list *options = read_data_cfg(datacfg);
    char *valid_images = option_find_str(options, "valid", "data/val.list");
    char *name_list = option_find_str(options, "names", "data/names.list");
    char *prefix = option_find_str(options, "results", "results");
    char **names = get_labels(name_list);
    char *mapf = option_find_str(options, "map", 0);
    int *map = 0;
    if (mapf) map = read_map(mapf);

    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 2);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    srand(time(0));

    list *plist = get_paths(valid_images);
    char **paths = (char **)list_to_array(plist);

    layer l = net->layers[net->n-1];
    int classes = l.classes;

    char buff[1024];
    char *type = option_find_str(options, "eval", "voc");
    FILE *fp = 0;
    FILE **fps = 0;
    int coco = 0;
    int imagenet = 0;
    if(0==strcmp(type, "coco")){
        if(!outfile) outfile = "coco_results";
        snprintf(buff, 1024, "%s/%s.json", prefix, outfile);
        fp = fopen(buff, "w");
        fprintf(fp, "[\n");
        coco = 1;
    } else if(0==strcmp(type, "imagenet")){
        if(!outfile) outfile = "imagenet-detection";
        snprintf(buff, 1024, "%s/%s.txt", prefix, outfile);
        fp = fopen(buff, "w");
        imagenet = 1;
        classes = 200;
    } else {
        if(!outfile) outfile = "comp4_det_test_";
        fps = calloc(classes, sizeof(FILE *));
        for(j = 0; j < classes; ++j){
            snprintf(buff, 1024, "%s/%s%s.txt", prefix, outfile, names[j]);
            fps[j] = fopen(buff, "w");
        }
    }

    int m = plist->size;
    int i=0;
    int t;

    float thresh = .005;
    float nms = .45;

    int nthreads = 4;
    image *val = calloc(nthreads, sizeof(image));
    image *val_resized = calloc(nthreads, sizeof(image));
    image *buf = calloc(nthreads, sizeof(image));
    image *buf_resized = calloc(nthreads, sizeof(image));
    pthread_t *thr = calloc(nthreads, sizeof(pthread_t));

    image input = make_image(net->w, net->h, net->c*2);

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    //args.type = IMAGE_DATA;
    args.type = LETTERBOX_DATA;

    for(t = 0; t < nthreads; ++t){
        args.path = paths[i+t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    double start = what_time_is_it_now();
    for(i = nthreads; i < m+nthreads; i += nthreads){
        fprintf(stderr, "%d\n", i);
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }
        for(t = 0; t < nthreads && i+t < m; ++t){
            args.path = paths[i+t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            char *path = paths[i+t-nthreads];
            char *id = basecfg(path);
            copy_cpu(net->w*net->h*net->c, val_resized[t].data, 1, input.data, 1);
            flip_image(val_resized[t]);
            copy_cpu(net->w*net->h*net->c, val_resized[t].data, 1, input.data + net->w*net->h*net->c, 1);

            network_predict(net, input.data);
            int w = val[t].w;
            int h = val[t].h;
            int num = 0;
            detection *dets = get_network_boxes(net, w, h, thresh, .5, map, 0, &num);
            if (nms) do_nms_sort(dets, num, classes, nms);
            if (coco){
                print_cocos(fp, path, dets, num, classes, w, h);
            } else if (imagenet){
                print_imagenet_detections(fp, i+t-nthreads+1, dets, num, classes, w, h);
            } else {
                print_detector_detections(fps, id, dets, num, classes, w, h);
            }
            free_detections(dets, num);
            free(id);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }
    for(j = 0; j < classes; ++j){
        if(fps) fclose(fps[j]);
    }
    if(coco){
        fseek(fp, -2, SEEK_CUR); 
        fprintf(fp, "\n]\n");
        fclose(fp);
    }
    fprintf(stderr, "Total Detection Time: %f Seconds\n", what_time_is_it_now() - start);
}


void validate_detector(char *datacfg, char *cfgfile, char *weightfile, char *outfile)
{
    int j;
    list *options = read_data_cfg(datacfg);
    char *valid_images = option_find_str(options, "valid", "data/train.list");
    char *name_list = option_find_str(options, "names", "data/names.list");
    char *prefix = option_find_str(options, "results", "results");
    char **names = get_labels(name_list);
    char *mapf = option_find_str(options, "map", 0);
    int *map = 0;
    if (mapf) map = read_map(mapf);

    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    srand(time(0));

    list *plist = get_paths(valid_images);
    char **paths = (char **)list_to_array(plist);

    layer l = net->layers[net->n-1];
    int classes = l.classes;

    char buff[1024];
    char *type = option_find_str(options, "eval", "voc");
    FILE *fp = 0;
    FILE **fps = 0;
    int coco = 0;
    int imagenet = 0;
    if(0==strcmp(type, "coco")){
        if(!outfile) outfile = "coco_results";
        snprintf(buff, 1024, "%s/%s.json", prefix, outfile);
        fp = fopen(buff, "w");
        fprintf(fp, "[\n");
        coco = 1;
    } else if(0==strcmp(type, "imagenet")){
        if(!outfile) outfile = "imagenet-detection";
        snprintf(buff, 1024, "%s/%s.txt", prefix, outfile);
        fp = fopen(buff, "w");
        imagenet = 1;
        classes = 200;
    } else {
        if(!outfile) outfile = "";
        fps = calloc(classes, sizeof(FILE *));
        for(j = 0; j < classes; ++j){
            snprintf(buff, 1024, "%s/%s%s.txt", prefix, outfile, names[j]);
            fps[j] = fopen(buff, "w");
        }
    }


    int m = plist->size;
    int i=0;
    int t;

    float thresh = .005;
    float nms = .45;

    int nthreads = 4;
    image *val = calloc(nthreads, sizeof(image));
    image *val_resized = calloc(nthreads, sizeof(image));
    image *buf = calloc(nthreads, sizeof(image));
    image *buf_resized = calloc(nthreads, sizeof(image));
    pthread_t *thr = calloc(nthreads, sizeof(pthread_t));

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    //args.type = IMAGE_DATA;
    args.type = LETTERBOX_DATA;

    for(t = 0; t < nthreads; ++t){
        args.path = paths[i+t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    double start = what_time_is_it_now();
    for(i = nthreads; i < m+nthreads; i += nthreads){
        fprintf(stderr, "%d\n", i);
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }
        for(t = 0; t < nthreads && i+t < m; ++t){
            args.path = paths[i+t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            char *path = paths[i+t-nthreads];
            char *id = basecfg(path);
            float *X = val_resized[t].data;
            network_predict(net, X);
            int w = val[t].w;
            int h = val[t].h;
            int nboxes = 0;
            detection *dets = get_network_boxes(net, w, h, thresh, .5, map, 0, &nboxes);
            if (nms) do_nms_sort(dets, nboxes, classes, nms);
            if (coco){
                print_cocos(fp, path, dets, nboxes, classes, w, h);
            } else if (imagenet){
                print_imagenet_detections(fp, i+t-nthreads+1, dets, nboxes, classes, w, h);
            } else {
                print_detector_detections(fps, id, dets, nboxes, classes, w, h);
            }
            free_detections(dets, nboxes);
            free(id);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }
    for(j = 0; j < classes; ++j){
        if(fps) fclose(fps[j]);
    }
    if(coco){
        fseek(fp, -2, SEEK_CUR); 
        fprintf(fp, "\n]\n");
        fclose(fp);
    }
    fprintf(stderr, "Total Detection Time: %f Seconds\n", what_time_is_it_now() - start);
}




void my_validate_detector(char *datacfg, char *cfgfile, char *weightfile, char *outfile)
{
    int j;
    list *options = read_data_cfg(datacfg);
    char *valid_images = option_find_str(options, "valid", "data/train.list");
    char *name_list = option_find_str(options, "names", "data/names.list");
    char *prefix = option_find_str(options, "results", "results");
    char **names = get_labels(name_list);
    char *mapf = option_find_str(options, "map", 0);
    int *map = 0;
    if (mapf) map = read_map(mapf);

    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    srand(time(0));

    list *plist = get_paths(valid_images);
    char **paths = (char **)list_to_array(plist);

    layer l = net->layers[net->n-1];
    int classes = l.classes;

    char buff[1024];
    char *type = option_find_str(options, "eval", "voc");
    FILE *fp = 0;
    FILE **fps = 0;
    int coco = 0;
    int imagenet = 0;
    if(0==strcmp(type, "coco")){
        if(!outfile) outfile = "coco_results";
        snprintf(buff, 1024, "%s/%s.json", prefix, outfile);
        fp = fopen(buff, "w");
        fprintf(fp, "[\n");
        coco = 1;
    } else if(0==strcmp(type, "imagenet")){
        if(!outfile) outfile = "imagenet-detection";
        snprintf(buff, 1024, "%s/%s.txt", prefix, outfile);
        fp = fopen(buff, "w");
        imagenet = 1;
        classes = 200;
    } else {
        if(!outfile) outfile = "comp4_det_test_";
        fps = calloc(classes, sizeof(FILE *));
        for(j = 0; j < classes; ++j){
            snprintf(buff, 1024, "%s/%s%s.txt", prefix, outfile, names[j]);
            fps[j] = fopen(buff, "w");
        }
    }


    int m = plist->size;
    int i=0;
    int t;
    float mean_det=0;

    float thresh = .50;
    float nms = .45;

    int nthreads = 4;
    image *val = calloc(nthreads, sizeof(image));
    image *val_resized = calloc(nthreads, sizeof(image));
    image *buf = calloc(nthreads, sizeof(image));
    image *buf_resized = calloc(nthreads, sizeof(image));
    pthread_t *thr = calloc(nthreads, sizeof(pthread_t));

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    //args.type = IMAGE_DATA;
    args.type = LETTERBOX_DATA;

    for(t = 0; t < nthreads; ++t){
        args.path = paths[i+t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    double start = what_time_is_it_now();
    for(i = nthreads; i < m+nthreads; i += nthreads){
        fprintf(stderr, "%d\n", i);
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }
        for(t = 0; t < nthreads && i+t < m; ++t){
            args.path = paths[i+t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            char *path = paths[i+t-nthreads];
            char *id = basecfg(path);
            float *X = val_resized[t].data;
            network_predict(net, X);
            int w = val[t].w;
            int h = val[t].h;
            int nboxes = 0;
            detection *dets = get_network_boxes(net, w, h, thresh, .5, map, 0, &nboxes);
            if (nms) do_nms_sort(dets, nboxes, classes, nms);
            //printf("map:%d\n",map);
            if (coco){
                print_cocos(fp, path, dets, nboxes, classes, w, h);
            } else if (imagenet){
                print_imagenet_detections(fp, i+t-nthreads+1, dets, nboxes, classes, w, h);
            } else {
                 mean_det+=my_print_detector_detections(fps, id, dets, nboxes, classes, w, h);
            }
            free_detections(dets, nboxes);
            free(id);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }
    for(j = 0; j < classes; ++j){
        if(fps) fclose(fps[j]);
    }
    if(coco){
        fseek(fp, -2, SEEK_CUR); 
        fprintf(fp, "\n]\n");
        fclose(fp);
    }
    mean_det/= m;
    printf("m:%d",m);
    fprintf(stderr, "Total Detection Time: %f Seconds with mean det %f\n", what_time_is_it_now() - start,mean_det);
}

void validate_detector_recall(char *datacfg, char *cfgfile, char *weightfile, float thresh, float hier_thresh)
{
     network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    srand(time(0));

    list *plist = get_paths("data/NOK/2007_val.txt");
    char **paths = (char **)list_to_array(plist);

    layer l = net->layers[net->n-1];

    int j, k;

    int m = plist->size;
    int i=0;

    // thresh = .5;
    float iou_thresh = .1;
    float nms = .45;

    int total = 0;
    int correct = 0;
    int proposals = 0;
    float avg_iou = 0;

    for(i = 0; i < m; ++i){
        char *path = paths[i];
        image orig = load_image_color(path, 0, 0);
        image sized = resize_image(orig, net->w, net->h);
        char *id = basecfg(path);
        network_predict(net, sized.data);
        int nboxes = 0;
        detection *dets = get_network_boxes(net, sized.w, sized.h, thresh, .5, 0, 1, &nboxes);
        if (nms) do_nms_obj(dets, nboxes, 1, nms);

        char labelpath[4096];
        find_replace(path, "images", "labels", labelpath);
        find_replace(labelpath, "JPEGImages", "labels", labelpath);
        find_replace(labelpath, ".jpg", ".txt", labelpath);
        find_replace(labelpath, ".JPEG", ".txt", labelpath);

        int num_labels = 0;
        box_label *truth = read_boxes(labelpath, &num_labels);
        for(k = 0; k < nboxes; ++k){
            if(dets[k].objectness > thresh){
                ++proposals;
            }
        }
        for (j = 0; j < num_labels; ++j) {
            ++total;
            box t = {truth[j].x, truth[j].y, truth[j].w, truth[j].h};
            // printf("%f:\n",truth[j].x);
            float best_iou = 0;
            for(k = 0; k < nboxes; ++k){
                float iou = box_iou(dets[k].bbox, t);
                if(dets[k].objectness > thresh && iou > best_iou){
                    best_iou = iou;
                }
            }
            avg_iou += best_iou;
            if(best_iou > iou_thresh){
                ++correct;
            }
        }

        fprintf(stderr, "%5d %5d %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tRecall:%.2f%%\n", i, correct, total, (float)proposals/(i+1), avg_iou*100/total, 100.*correct/total);
        free(id);
        free_image(orig);
        free_image(sized);
    }
}


void validate_detector_f1(char *datacfg, char *cfgfile, char *weightfile, float thresh, float hier_thresh)
{
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    srand(time(0));

    list *plist = get_paths("data/NOK/2007_val.txt");
    char **paths = (char **)list_to_array(plist);

    layer l = net->layers[net->n-1];

    int j, k;

    int m = plist->size;
    int i=0;

    // thresh = .5;
    float iou_thresh = .1;
    float nms = .45;

    int TP_FN = 0;
    int TP_FP=0;
    int TP = 0;
    int proposals = 0;
    float avg_iou = 0;

    for(i = 0; i < m; ++i){
        char *path = paths[i];
        image orig = load_image_color(path, 0, 0);
        image sized = resize_image(orig, net->w, net->h);
        char *id = basecfg(path);
        network_predict(net, sized.data);
        int nboxes = 0;
        detection *dets = get_network_boxes(net, sized.w, sized.h, thresh, .5, 0, 1, &nboxes);
        if (nms) do_nms_obj(dets, nboxes, 1, nms);

        char labelpath[4096];
        find_replace(path, "images", "labels", labelpath);
        find_replace(labelpath, "JPEGImages", "labels", labelpath);
        find_replace(labelpath, ".jpg", ".txt", labelpath);
        find_replace(labelpath, ".JPEG", ".txt", labelpath);

        int num_labels = 0;
        box_label *truth = read_boxes(labelpath, &num_labels);
        for(k = 0; k < nboxes; ++k){
            if(dets[k].objectness > thresh){
                ++proposals;
            }
        }
        // printf("number of boxes:%d\n",nboxes);
        for(k = 0; k < nboxes; ++k){
            TP_FP++;
        }
        for (j = 0; j < num_labels; ++j) {
            ++TP_FN;
            box t = {truth[j].x, truth[j].y, truth[j].w, truth[j].h};
            // printf("%f:\n",truth[j].x);
            float best_iou = 0;
            for(k = 0; k < nboxes; ++k){
                float iou = box_iou(dets[k].bbox, t);
                if(dets[k].objectness > thresh && iou > best_iou){
                    best_iou = iou;
                }
            }
            avg_iou += best_iou;
            if(best_iou > iou_thresh){
                ++TP;
            }
        }

        fprintf(stderr, "%5d %5d %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tRecall:%.2f%%\n", i, TP, TP_FN, (float)proposals/(i+1), avg_iou*100/TP_FN, 100.*TP/TP_FN);
        fprintf(stderr, "%5d %5d %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tPrecision:%.2f%%\n", i, TP, TP_FP, (float)proposals/(i+1), avg_iou*100/TP_FN, 100.*TP/TP_FP);
        fprintf(stderr, "%5d %5d %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tF1:%.2f%%\n\n", i, TP, (TP_FN+TP_FP)/2, (float)proposals/(i+1), avg_iou*100/TP_FN, 100.*2*TP/(TP_FP+TP_FN));
        free(id);
        free_image(orig);
        free_image(sized);
    }
}


void validate_detector_f1_each(char *datacfg, char *cfgfile, char *weightfile, float thresh, float hier_thresh)
{
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    srand(time(0));
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    list *plist = get_paths("data/NOK/2007_val.txt");
    char **paths = (char **)list_to_array(plist);
    char **names = get_labels(name_list);
    layer l = net->layers[net->n-1];

    int j, k;

    int m = plist->size;
    int i=0;

    // thresh = .5;
    float iou_thresh = .1;
    float nms = .45;

    int TP_FN = 0;
    int TP_FP=0;
    int TP = 0;
    int proposals = 0;

    int TP_FN_each[100] = {0};
    int TP_FP_each[100] = {0};
    float TP_each[100] = {0};
    int TP_each_test[100] = {0};
    float avg_iou = 0;

    for(i = 0; i < m; ++i){
        char *path = paths[i];
        image orig = load_image_color(path, 0, 0);
        image sized = resize_image(orig, net->w, net->h);
        char *id = basecfg(path);
        network_predict(net, sized.data);
        int nboxes = 0;
        detection *dets = get_network_boxes(net, sized.w, sized.h, thresh, .5, 0, 1, &nboxes);
        if (nms) do_nms_obj(dets, nboxes, 1, nms);
        

		for (int i = 0; i < nboxes; i++) {

            int best_class = -1;
            float img_w=orig.w;
            float img_h=orig.h;
            float center_x=(dets[i].bbox.x+dets[i].bbox.w/2)*img_w;
            float center_y=(dets[i].bbox.y+dets[i].bbox.h/2)*img_h;
            float obj_w=dets[i].bbox.w*img_w;
            float obj_h=dets[i].bbox.h*img_h;
            int DT_result=DT_classifier(center_x, center_y,obj_w,obj_h,img_w,img_h);
            // printf("x:%f y:%f w:%f h:%f\n",dets[i].bbox.x,dets[i].bbox.y,dets[i].bbox.w,dets[i].bbox.h);
            printf("center_x:%f center_y:%f obj_w:%f obj_h:%f img_w:%f img_h:%f\n",center_x,center_y,obj_w,obj_h,img_w,img_h);
            for(j = 0; j < l.classes; ++j){
                if (dets[i].prob[j] > thresh){
                    if (best_class < 0) {
                        best_class = j;
                    }
                }
            }

            if (best_class==0&&DT_result==1) {
                dets[i].prob[0]=0;
                dets[i].prob[1]=1;
            }
            if (best_class==2&&DT_result==1) {
                dets[i].prob[2]=0;
                dets[i].prob[1]=1;
            }
            if(best_class>-1){
                char labelstr[4096] = {0};
                strcat(labelstr, names[best_class]);
                fprintf(stderr,"best class:%s DT_result:%d\n",labelstr,DT_result);
            }
            // 
        }


        char labelpath[4096];
        find_replace(path, "images", "labels", labelpath);
        find_replace(labelpath, "JPEGImages", "labels", labelpath);
        find_replace(labelpath, ".jpg", ".txt", labelpath);
        find_replace(labelpath, ".JPEG", ".txt", labelpath);

        int num_labels = 0;
        box_label *truth = read_boxes(labelpath, &num_labels);
        for(k = 0; k < nboxes; ++k){
            if(dets[k].objectness > thresh){
                ++proposals;
            }
        }
        // printf("number of boxes:%d\n",nboxes);
        for(k = 0; k < nboxes; ++k){
            TP_FP++;
        }
		for (int i = 0; i < nboxes; i++) {
            int best_class = -1;
            for(j = 0; j < l.classes; ++j){
                if (dets[i].prob[j] > thresh){
                    if (best_class < 0) {
                        best_class = j;
                    }
                }
            }
            if (best_class > -1) {
                TP_FP_each[best_class]++;
            }
        }
        for (j = 0; j < num_labels; ++j) {
            ++TP_FN;
            box t = {truth[j].x, truth[j].y, truth[j].w, truth[j].h};
            TP_FN_each[truth[j].id]++;
            int t_class=truth[j].id;

            // TP_each[t_class]++;
            // printf("truth[j].id:%d\n",t_class);
            float best_iou = 0;
            for(k = 0; k < nboxes; ++k){
                float iou = box_iou(dets[k].bbox, t);
                int best_class = -1;
                if(dets[k].objectness > thresh && iou > best_iou){
                    best_iou = iou;
                }
            }

            int check=0;
            for(k = 0; k < nboxes; ++k){
                float iou = box_iou(dets[k].bbox, t);
                int best_class = -1;
                for(int c = 0; c < l.classes; ++c){
                    if (dets[k].prob[c] > thresh){
                        if (best_class < 0) {
                            best_class = c;
                        }
                    }
                }
                if(best_class==t_class && iou > iou_thresh){
                    check=1;
                }
            }
            if(check){
                TP_each[t_class]++;
            }

            avg_iou += best_iou;
            if(best_iou > iou_thresh){
                ++TP;
            }
        }
        for(int index = 0; index < l.classes; ++index){
            char labelstr[4096] = {0};
            strcat(labelstr, names[index]);
            // printf("check point2\n");
            // printf("l.classes:%d index:%d TP_each:%d\n",l.classes,index,TP_each_test[index]);
            if(TP_FN_each[index]>0&&TP_FP_each[index]>0){
                // printf("Recall:%d\n", TP_each[index]);
                fprintf(stderr, "label:%s Recall:%.2f%%\n",labelstr, 100.*TP_each[index]/TP_FN_each[index]);
                fprintf(stderr, "label:%s Precision:%.2f%%\n",labelstr, 100.*TP_each[index]/TP_FP_each[index]);
                fprintf(stderr, "label:%s F1:%.2f%%\n\n",labelstr, 100.*2*TP_each[index]/(TP_FP_each[index]+TP_FN_each[index]));
            }

        }
        fprintf(stderr, "%5d %5d %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tRecall:%.2f%%\n", i, TP, TP_FN, (float)proposals/(i+1), avg_iou*100/TP_FN, 100.*TP/TP_FN);
        fprintf(stderr, "%5d %5d %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tPrecision:%.2f%%\n", i, TP, TP_FP, (float)proposals/(i+1), avg_iou*100/TP_FN, 100.*TP/TP_FP);
        fprintf(stderr, "%5d %5d %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tF1:%.2f%%\n\n", i, TP, (TP_FN+TP_FP)/2, (float)proposals/(i+1), avg_iou*100/TP_FN, 100.*2*TP/(TP_FP+TP_FN));
        free(id);
        free_image(orig);
        free_image(sized);
    }
}


void detect_nok(char *datacfg, char *cfgfile, char *weightfile, float thresh, float hier_thresh)
{
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    srand(time(0));

    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);
    image **alphabet = load_alphabet();

    layer l = net->layers[net->n-1];
    int j;
    int i=0;

    // float thresh = .001;
    float nms = .5;
    // make_window("predictions", orig.w, orig.h, 0);
    for(i = 0; i < 25000; ++i){
        char *path= malloc(50);
        sprintf(path, "/home/gong/nok_test/testData/%d.jpg", i);
        
        image origRec = load_image_color(path, 0, 0);
        image orig = load_image_color(path, 0, 0);
        image sized = resize_image(orig, net->w, net->h);
        network_predict(net, sized.data);
        int nboxes = 0;
        detection *dets = get_network_boxes(net, sized.w, sized.h, thresh, hier_thresh, 0, 1, &nboxes);
        if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
        


		for (int i = 0; i < nboxes; i++) {

            int best_class = -1;
            float img_w=orig.w;
            float img_h=orig.h;
            float center_x=(dets[i].bbox.x+dets[i].bbox.w/2)*img_w;
            float center_y=(dets[i].bbox.y+dets[i].bbox.h/2)*img_h;
            float obj_w=dets[i].bbox.w*img_w;
            float obj_h=dets[i].bbox.h*img_h;
            int DT_result=DT_classifier(center_x, center_y,obj_w,obj_h,img_w,img_h);
            // printf("x:%f y:%f w:%f h:%f\n",dets[i].bbox.x,dets[i].bbox.y,dets[i].bbox.w,dets[i].bbox.h);
            
            for(j = 0; j < l.classes; ++j){
                if (dets[i].prob[j] > thresh){
                    if (best_class < 0) {
                        best_class = j;
                    }
                }
            }

            if (best_class==0&&DT_result==0) {
                dets[i].prob[0]=0.;
                dets[i].prob[1]=1.0;
            }
            if (best_class==2&&DT_result==0) {
                dets[i].prob[2]=0.;
                dets[i].prob[1]=1.0;
            }
            if(best_class==1){
                char labelstr[4096] = {0};
                strcat(labelstr, names[best_class]);
                fprintf(stderr,"best class:%s DT_result:%d\n",labelstr,DT_result);
                printf("center_x:%f center_y:%f obj_w:%f obj_h:%f img_w:%f img_h:%f\n",center_x,center_y,obj_w,obj_h,img_w,img_h);
            }
            // 
        }

        draw_detections(origRec, dets, nboxes, thresh, names, alphabet, l.classes);
		int defectFlag = 0;
		int cfineFlag = 0;
        int fineFlag=0;
        // int cdefectFlag=0;
        int numerrFlag=0;
		for (int i = 0; i < nboxes; i++) {
            char labelstr[4096] = {0};
            int best_class = -1;
            for(j = 0; j < l.classes; ++j){
                if (dets[i].prob[j] > thresh){
                    if (best_class < 0) {
                        strcat(labelstr, names[j]);
                        best_class = j;
                    } else {
                        strcat(labelstr, ", ");
                        strcat(labelstr, names[j]);
                    }
                    printf("%s: %.0f%%\n", names[j], dets[i].prob[j]*100);
                }
            }
			if (best_class == 0) {
				printf("\nmy result defect: class:%d %s: %.0f%%", best_class, labelstr, dets[i].prob[best_class] * 100);
				defectFlag = 1;
				//break;
			}
            if (best_class == 1) {
				printf("\nmy result defect: class:%d %s: %.0f%%", best_class, labelstr, dets[i].prob[best_class] * 100);
				fineFlag = 1;
				//break;
			}
			if (best_class == 2) {
				printf("\nmy result numErr: class:%d %s: %.0f%%", best_class, labelstr, dets[i].prob[best_class] * 100);
				numerrFlag = 1;
			}
			if (best_class == 3) {
				cfineFlag = 1;
			}
		}

        if(cfineFlag == 0 && numerrFlag==0 && defectFlag == 0 && fineFlag == 0)
        {
            char *pathSave= malloc(50);
            printf("There is no injector!");
            sprintf(pathSave, "/home/gong/nok_test/resultOut/blank/%d", i);
            save_image(origRec, pathSave);
            free(pathSave);
        }else{
            if (cfineFlag == 0) {
                printf("There is no seal ring!");

            }
            if (defectFlag == 0 && numerrFlag==0 && cfineFlag == 1) {
                char *pathSave= malloc(50);
                sprintf(pathSave, "/home/gong/nok_test/resultOut/fine/%d", i);
                save_image(origRec, pathSave);
                free(pathSave);
            }
            if(defectFlag == 1){
                char *pathSave= malloc(50);
                sprintf(pathSave, "/home/gong/nok_test/resultOut/defect/%d", i);
                save_image(origRec, pathSave);
                free(pathSave);
                char *pathSave2= malloc(50);
                sprintf(pathSave2, "/home/gong/nok_test/resultOut/defectNolabel/%d", i);
                save_image(orig, pathSave2);
                free(pathSave2);
            }
            if(cfineFlag == 0){
                char *pathSave= malloc(50);
                sprintf(pathSave, "/home/gong/nok_test/resultOut/noRing/%d", i);
                save_image(origRec, pathSave);
                free(pathSave);
                char *pathSave2= malloc(50);
                sprintf(pathSave2, "/home/gong/nok_test/resultOut/defectNolabel/%d", i);
                save_image(orig, pathSave2);
                free(pathSave2);
            }
            if(numerrFlag == 1){
                char *pathSave= malloc(50);
                sprintf(pathSave, "/home/gong/nok_test/resultOut/numerr/%d", i);
                save_image(origRec, pathSave);
                free(pathSave);
                char *pathSave2= malloc(50);
                sprintf(pathSave2, "/home/gong/nok_test/resultOut/defectNolabel/%d", i);
                save_image(orig, pathSave2);
                free(pathSave2);
            }
        }


        #ifdef OPENCV
            //将图片保存
            if (defectFlag == 0 && numerrFlag==0 && cfineFlag == 1) {
                image origRecsized = resize_image(origRec, orig.w/2, orig.h/2);
                show_image(origRecsized, "predictions", 1);
                free_image(origRecsized);
            }else{
                image origRecsized = resize_image(origRec, orig.w/2, orig.h/2);
                show_image(origRecsized, "predictions", 1);
                free_image(origRecsized);
            }
        #endif
        free(path);
        free_detections(dets, nboxes);
        free_image(orig);
        free_image(origRec);
        free_image(sized);
    }
}


void code(void){
   
   float data[] = {1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1};

    float delta_data[] = {1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2};

	float truth[] = { 0,0,1 };
	float delta[75] = {0 };


	int num_layer = 1;
	network *net = make_network(num_layer);
	net->h=5;
	net->w=5;
	net->c=3;
	net->batch = 1;
	net->input = data;
    net->inputs=75;
	net->truth = truth;
	net->train = 1;
    myParameters myParamStr;
    myParamStr.prune = 0;
    myParamStr.prune_threshold_min = 0;
    myParamStr.prune_threshold_max = 0;
    myParamStr.prune_step_num = 0;
    myParamStr.prune_threshold_step=(myParamStr.prune_threshold_max-myParamStr.prune_threshold_min)/myParamStr.prune_step_num;
    myParamStr.rectFlg=0;
    myParamStr.ksize_h=0;
    myParamStr.ksize_w=0;
    
    //make_depthwise_convolutional_layer(int batch, int h, int w, int c, int size, int stride, int padding, ACTIVATION activation, int batch_normalize,  int adam, myParameters myParamStr)
	// depthwise_convolutional_layer depthwise_conv1 = make_depthwise_convolutional_layer(net->batch, net->h, net->w, net->c, 3, 1, 1, LINEAR, 0, 0,myParamStr);
	//  make_mix_convolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, int padding, ACTIVATION activation, int batch_normalize,  int adam, myParameters myParamStr)
    int sizeList[3]={3,5,7};
    mix_convolutional_layer depthwise_conv1 = make_mix_convolutional_layer(net->batch, net->h, net->w, net->c,3, 3,sizeList, 1, 1, LINEAR, 0, 0, myParamStr);
    printf("FINISH MAKE LAYER\n");
    net->layers[0] = depthwise_conv1;
    net->workspace = calloc(1, 100000);
    
    net->index = 0;
	layer l = net->layers[0];
    float conv_data[]={1,1,1,
                        1,1,1,
                        1,1,1,
                        1,1,1,1,1,
                        1,1,1,1,1,
                        1,1,1,1,1,
                        1,1,1,1,1,
                        1,1,1,1,1,
                        1,1,1,1,1,1,1,
                        1,1,1,1,1,1,1,
                        1,1,1,1,1,1,1,
                        1,1,1,1,1,1,1,
                        1,1,1,1,1,1,1,
                        1,1,1,1,1,1,1,
                        1,1,1,1,1,1,1};
    l.weights = conv_data;
#ifdef GPU
    
    cuda_set_device(0);
    net->input_gpu=cuda_make_array(net->input, net->inputs*net->batch);
    net->workspace = cuda_make_array(0, (100000-1)/sizeof(float)+1);
    printf("CHECK 0\n");
    cuda_push_array(net->input_gpu, net->input, net->batch*net->inputs);
    cuda_push_array(l.weights_gpu, l.weights, l.nweights);

    printf("CHECK 1\n");

    l.forward_gpu(l, *net);
    // l.forward_gpu(l, *net);
    printf("CHECK 3\n");
    cuda_pull_array(l.output_gpu, l.output, 75);
#else
    printf("l.outputs:%d\n",l.outputs);
    if (l.delta) {
		fill_cpu(l.outputs * l.batch, 0, l.delta, 1);
	}
    l.forward(l, *net);
    printf("forward\n");
    l.forward(l, *net);
    l.backward(l, *net);
#endif
    printf("FINISH FORWARD LAYER\n");
    for(int i=0;i<15;i++){
        for(int j=0;j<5;j++){
            printf("%f ",l.output[i*5+j]);
        }
        printf("\n");
    }
}

/*
void censor_detector(char *datacfg, char *cfgfile, char *weightfile, int cam_index, const char *filename, int class, float thresh, int skip)
{
#ifdef OPENCV
    char *base = basecfg(cfgfile);
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);

    srand(2222222);
    CvCapture * cap;

    int w = 1280;
    int h = 720;

    if(filename){
        cap = cvCaptureFromFile(filename);
    }else{
        cap = cvCaptureFromCAM(cam_index);
    }

    if(w){
        cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_WIDTH, w);
    }
    if(h){
        cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_HEIGHT, h);
    }

    if(!cap) error("Couldn't connect to webcam.\n");
    cvNamedWindow(base, CV_WINDOW_NORMAL); 
    cvResizeWindow(base, 512, 512);
    float fps = 0;
    int i;
    float nms = .45;

    while(1){
        image in = get_image_from_stream(cap);
        //image in_s = resize_image(in, net->w, net->h);
        image in_s = letterbox_image(in, net->w, net->h);
        layer l = net->layers[net->n-1];

        float *X = in_s.data;
        network_predict(net, X);
        int nboxes = 0;
        detection *dets = get_network_boxes(net, in.w, in.h, thresh, 0, 0, 0, &nboxes);
        //if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        if (nms) do_nms_sort(dets, nboxes, l.classes, nms);

        for(i = 0; i < nboxes; ++i){
            if(dets[i].prob[class] > thresh){
                box b = dets[i].bbox;
                int left  = b.x-b.w/2.;
                int top   = b.y-b.h/2.;
                censor_image(in, left, top, b.w, b.h);
            }
        }
        show_image(in, base);
        cvWaitKey(10);
        free_detections(dets, nboxes);


        free_image(in_s);
        free_image(in);


        float curr = 0;
        fps = .9*fps + .1*curr;
        for(i = 0; i < skip; ++i){
            image in = get_image_from_stream(cap);
            free_image(in);
        }
    }
    #endif
}

void extract_detector(char *datacfg, char *cfgfile, char *weightfile, int cam_index, const char *filename, int class, float thresh, int skip)
{
#ifdef OPENCV
    char *base = basecfg(cfgfile);
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);

    srand(2222222);
    CvCapture * cap;

    int w = 1280;
    int h = 720;

    if(filename){
        cap = cvCaptureFromFile(filename);
    }else{
        cap = cvCaptureFromCAM(cam_index);
    }

    if(w){
        cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_WIDTH, w);
    }
    if(h){
        cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_HEIGHT, h);
    }

    if(!cap) error("Couldn't connect to webcam.\n");
    cvNamedWindow(base, CV_WINDOW_NORMAL); 
    cvResizeWindow(base, 512, 512);
    float fps = 0;
    int i;
    int count = 0;
    float nms = .45;

    while(1){
        image in = get_image_from_stream(cap);
        //image in_s = resize_image(in, net->w, net->h);
        image in_s = letterbox_image(in, net->w, net->h);
        layer l = net->layers[net->n-1];

        show_image(in, base);

        int nboxes = 0;
        float *X = in_s.data;
        network_predict(net, X);
        detection *dets = get_network_boxes(net, in.w, in.h, thresh, 0, 0, 1, &nboxes);
        //if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        if (nms) do_nms_sort(dets, nboxes, l.classes, nms);

        for(i = 0; i < nboxes; ++i){
            if(dets[i].prob[class] > thresh){
                box b = dets[i].bbox;
                int size = b.w*in.w > b.h*in.h ? b.w*in.w : b.h*in.h;
                int dx  = b.x*in.w-size/2.;
                int dy  = b.y*in.h-size/2.;
                image bim = crop_image(in, dx, dy, size, size);
                char buff[2048];
                sprintf(buff, "results/extract/%07d", count);
                ++count;
                save_image(bim, buff);
                free_image(bim);
            }
        }
        free_detections(dets, nboxes);


        free_image(in_s);
        free_image(in);


        float curr = 0;
        fps = .9*fps + .1*curr;
        for(i = 0; i < skip; ++i){
            image in = get_image_from_stream(cap);
            free_image(in);
        }
    }
    #endif
}
*/

/*
void network_detect(network *net, image im, float thresh, float hier_thresh, float nms, detection *dets)
{
    network_predict_image(net, im);
    layer l = net->layers[net->n-1];
    int nboxes = num_boxes(net);
    fill_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 0, dets);
    if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
}
*/

void run_detector(int argc, char **argv)
{
    char *prefix = find_char_arg(argc, argv, "-prefix", 0);
    float thresh = find_float_arg(argc, argv, "-thresh", .5);
    float hier_thresh = find_float_arg(argc, argv, "-hier", .5);
    int cam_index = find_int_arg(argc, argv, "-c", 0);
    int frame_skip = find_int_arg(argc, argv, "-s", 0);
    int avg = find_int_arg(argc, argv, "-avg", 3);
    if(argc < 3){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }
    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    char *outfile = find_char_arg(argc, argv, "-out", 0);
    int *gpus = 0;
    int gpu = 0;
    int ngpus = 0;
    if(gpu_list){
        printf("%s\n", gpu_list);
        int len = strlen(gpu_list);
        ngpus = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (gpu_list[i] == ',') ++ngpus;
        }
        gpus = calloc(ngpus, sizeof(int));
        for(i = 0; i < ngpus; ++i){
            gpus[i] = atoi(gpu_list);
            gpu_list = strchr(gpu_list, ',')+1;
        }
    } else {
        gpu = gpu_index;
        gpus = &gpu;
        ngpus = 1;
    }

    int clear = find_arg(argc, argv, "-clear");
    int fullscreen = find_arg(argc, argv, "-fullscreen");
    int width = find_int_arg(argc, argv, "-w", 0);
    int height = find_int_arg(argc, argv, "-h", 0);
    int fps = find_int_arg(argc, argv, "-fps", 0);
    //int class = find_int_arg(argc, argv, "-class", 0);

    char *datacfg = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;
    char *filename = (argc > 6) ? argv[6]: 0;
    if(0==strcmp(argv[2], "test")) test_detector(datacfg, cfg, weights, filename, thresh, hier_thresh, outfile, fullscreen);
    else if(0==strcmp(argv[2], "train")) train_detector(datacfg, cfg, weights, gpus, ngpus, clear);
    else if(0==strcmp(argv[2], "valid")) validate_detector(datacfg, cfg, weights, outfile);
    else if(0==strcmp(argv[2], "valid2")) validate_detector_flip(datacfg, cfg, weights, outfile);
    else if(0==strcmp(argv[2], "myvalid")) my_validate_detector(datacfg, cfg, weights, outfile);
    else if(0==strcmp(argv[2], "recall")) validate_detector_recall(datacfg, cfg, weights, thresh, hier_thresh);
    else if(0==strcmp(argv[2], "detect_nok")) detect_nok(datacfg, cfg, weights, thresh, hier_thresh);
    else if(0==strcmp(argv[2], "f1")) validate_detector_f1(datacfg, cfg, weights, thresh, hier_thresh);
    else if(0==strcmp(argv[2], "f1_each")) validate_detector_f1_each(datacfg, cfg, weights, thresh, hier_thresh);
    else if(0==strcmp(argv[2], "code")) code();
    else if(0==strcmp(argv[2], "demo")) {
        list *options = read_data_cfg(datacfg);
        int classes = option_find_int(options, "classes", 20);
        char *name_list = option_find_str(options, "names", "data/names.list");
        char **names = get_labels(name_list);
        demo(cfg, weights, thresh, cam_index, filename, names, classes, frame_skip, prefix, avg, hier_thresh, width, height, fps, fullscreen);
    }
    //else if(0==strcmp(argv[2], "extract")) extract_detector(datacfg, cfg, weights, cam_index, filename, class, thresh, frame_skip);
    //else if(0==strcmp(argv[2], "censor")) censor_detector(datacfg, cfg, weights, cam_index, filename, class, thresh, frame_skip);
}

