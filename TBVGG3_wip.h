/*
--------------------------------------------------
    James William Fletcher (james@voxdsp.com)
        JANUARY 2021 - TBVGG3
--------------------------------------------------
    Tiny Binary VGG3
    https://github.com/tfcnn

    This is an adaption inspired by the VGG series of networks.

    This VGG network is designed for binary classification and is
    only three layers deep. It uses Global Average Pooling rather
    than a final fully connected layer, additionally the final
    result is again just an average of the GAP. Essentially making
    this network an FCN version of the VGG network.

    The VGG network was originally created by the Visual Geometry Group
    of Oxford University in the United Kingdom. It was first proposed
    by Karen Simonyan and Andrew Zisserman, the original paper is
    available here; https://arxiv.org/abs/1409.1556

        TBVGG3
        :: ReLU + 0 Padding
        28x28 x32
        > maxpool
        14x14 x64
        > maxpool
        7x7 x128
        > GAP + Average

    I like to call the gradient the error, I am one of those.

    Configuration;
        No batching of the forward passes before backproping the error,
        my argument is that this is a network designed to be light-weight
        and to be used for real-time training.

        Nesterov optimisation of gradient descent, although, I think
        random step size SGD may better fit the reverse/forward
        training model.

        XAVIER GLOROT normal distribution weight initialisation.
        I read some places online that uniform GLOROT works
        better in CNN's, this is something I really need to
        benchmark for myself at some point. Since the original
        VGG paper references GLOROT with normal distribution,
        this is what I chose initially.

    Preferences;
        You can see that I do not make an active effort to avoid
        branching, when I consider the trade off, such as with the
        TBVGG3_CheckPadded() check, I think to myself do I memcpy()
        to a new buffer with padding or include the padding in the
        original buffer or use branches to check if entering a padded
        coordinate, I chose the latter. I would rather a few extra
        branches than to bloat memory in some scenarios, although
        you can also see in TBVGG3_2x2MaxPool() that I choose a
        negligibly higher use of memory to avoid ALU divisions.

    Network size:
        (3x28x28)+(32x3x10)+(64x32x10)+(128x64x10)+(32x3x10)+(64x32x10)+
        (128x64x10)+(32x28x28)+(32x14x14)+(64x14x14)+(64x7x7)+(128x7x7)+
        (32x28x28)+(64x14x14)+(128x7x7) = 306288 floats
        306288*4 = 1225152 bytes = 1.168395996 megabytes
*/

#ifndef TBVGG3_H
#define TBVGG3_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <sys/file.h>

#define uint unsigned int
#define LEARNING_RATE 0.03
#define NAG_MOMENTUM  0.1

/*
--------------------------------------
    structures
--------------------------------------
*/

// network struct
struct
{
    //inputs
    float input[3][28][28];

    //filters:num, d,  w+b
    float l1f[32 ][3 ][10];
    float l2f[64 ][32][10];
    float l3f[128][64][10];

    // filter momentum's
    float l1fm[32 ][3 ][10];
    float l2fm[64 ][32][10];
    float l3fm[128][64][10];

    // outputs
    //       d,  y,  x
    float o1[32][28][28];
        float p1[32][14][14]; // pooled
    float o2[64][14][14];
        float p2[64][7][7];   // pooled
    float o3[128][7][7];

    // error gradients
    //       y,  x
    float e1[32][28][28];
    float e2[64][14][14];
    float e3[128][7][7];
}
typedef TBVGG3_Network;

enum 
{
    LEARN_MAX = 1,
    LEARN_MIN = 0,
    NO_LEARN  = -1
}
typedef TBVGG3_LEARNTYPE;

/*
--------------------------------------
    functions
--------------------------------------
*/

float TBVGG3_Process(TBVGG3_Network* net, const float* inputs, const TBVGG3_LEARNTYPE learn);
void  TBVGG3_Reset(TBVGG3_Network* net);
int   TBVGG3_SaveNetwork(TBVGG3_Network* net, const char* file);
int   TBVGG3_LoadNetwork(TBVGG3_Network* net, const char* file);

/*
--------------------------------------
    the code ...
--------------------------------------
*/

static inline float TBVGG3_RELU(const float x)
{
    if(x < 0){return 0;}
    return x;
}

static inline float TBVGG3_RELU_D(const float x)
{
    if(x > 0){return 1;}
    return 0;
}

static inline float TBVGG3_SIGMOID(const float x)
{
    return 1 / (1 + exp(-x));
}

static inline float TBVGG3_SIGMOID_D(const float x)
{
    return x * (1 - x);
}

static inline float TBVGG3_SGD(TBVGG3_Network* net, const float input, const float error)
{
    return LEARNING_RATE * error * input;
}

static inline float TBVGG3_NAG(TBVGG3_Network* net, const float input, const float error, float* momentum)
{
    const float v = NAG_MOMENTUM * momentum[0] + ( LEARNING_RATE * error * input );
    const float n = v + NAG_MOMENTUM * (v - momentum[0]);
    momentum[0] = v;
    return n;
}

float TBVGG3_NormalRandom() // Box Muller
{
    static const float rmax = (float)RAND_MAX;
    double u = ( (((float)rand())+1e-7) / rmax) * 2 - 1;
    double v = ( (((float)rand())+1e-7) / rmax) * 2 - 1;
    double r = u * u + v * v;
    while(r == 0 || r > 1)
    {
        u = ( (((float)rand())+1e-7) / rmax) * 2 - 1;
        v = ( (((float)rand())+1e-7) / rmax) * 2 - 1;
        r = u * u + v * v;
    }
    return u * sqrt(-2 * log(r) / r);
}

void TBVGG3_Reset(TBVGG3_Network* net)
{
    if(net == NULL){return;}

    // seed random
    srand(time(0));

    // XAVIER GLOROT NORMAL
    // Weight Init

    //l1f
    float d = sqrt(2.0 / 2384);
    for(uint i = 0; i < 32; i++)
        for(uint j = 0; j < 3; j++)
            for(uint k = 0; k < 10; k++)
                net->l1f[i][j][k] = k == 9 ? 0 : TBVGG3_NormalRandom() * d; // 9 = bias

    //l2f
    d = sqrt(2.0 / 96);
    for(uint i = 0; i < 64; i++)
        for(uint j = 0; j < 32; j++)
            for(uint k = 0; k < 10; k++)
                net->l2f[i][j][k] = k == 9 ? 0 : TBVGG3_NormalRandom() * d;

    //l3f
    d = sqrt(2.0 / 129);
    for(uint i = 0; i < 128; i++)
        for(uint j = 0; j < 64; j++)
            for(uint k = 0; k < 10; k++)
                net->l3f[i][j][k] = k == 9 ? 0 : TBVGG3_NormalRandom() * d;
    
    // zero momentum's
    memset(net->l1fm, 0, sizeof(net->l1fm));
    memset(net->l2fm, 0, sizeof(net->l2fm));
    memset(net->l3fm, 0, sizeof(net->l3fm));

    // zero buffers
    memset(net->input, 0, sizeof(net->input));

    memset(net->p1, 0, sizeof(net->p1));
    memset(net->p2, 0, sizeof(net->p2));

    memset(net->o1, 0, sizeof(net->o1));
    memset(net->o2, 0, sizeof(net->o2));
    memset(net->o3, 0, sizeof(net->o3));

    memset(net->e1, 0, sizeof(net->e1));
    memset(net->e2, 0, sizeof(net->e2));
    memset(net->e3, 0, sizeof(net->e3));
}

int TBVGG3_SaveNetwork(TBVGG3_Network* net, const char* file)
{
    if(net == NULL){return -1;}

    FILE* f = fopen(file, "wb");
    if(f == NULL)
        return -1;

    if(fwrite(net, 1, sizeof(TBVGG3_Network), f) != sizeof(TBVGG3_Network))
    {
        fclose(f);
        return -1;
    }

    fclose(f);
    return 0;
}

int TBVGG3_LoadNetwork(TBVGG3_Network* net, const char* file)
{
    if(net == NULL){return -1;}

    FILE* f = fopen(file, "rb");
    if(f == NULL)
        return -1;

    if(fread(net, 1, sizeof(TBVGG3_Network), f) != sizeof(TBVGG3_Network))
    {
        fclose(f);
        return -1;
    }

    fclose(f);
    return 0;
}

void TBVGG3_2x2MaxPool(const float*** input, float*** output, const uint depth, const uint input_width, const uint input_height)
{
    // output tracking, more memory for less alu division ops
    uint oi = 0, oj = 0;

    // for every depth
    for(uint d = 0; d < depth; d++)
    {
        // for every 2x2 chunk of input
        for(uint i = 0; i < input_height; i += 2, oi++)
        {
            for(uint j = 0; j < input_width; j += 2, oj++)
            {
                // // get 2x2 chunk from input
                // const float f[] = {input[i][j], input[i+1][j], input[i][j+1], input[i+1][j+1]};

                // // iterate 2x2 chunk for max val
                // float max = 0;
                // for(uint k = 0; k < 4; k++)
                //     if(f[k] > max)
                //         max = f[k];
                
                // get max val
                float max = 0;
                if(input[d][i][j] > max)
                    max = input[d][i][j];
                if(input[d][i][j+1] > max)
                    max = input[d][i][j+1];
                if(input[d][i+1][j] > max)
                    max = input[d][i+1][j];
                if(input[d][i+1][j+1] > max)
                    max = input[d][i+1][j+1];

                // output max val
                output[d][oi][oj] = max;
            }
        }
    }
}

static inline uint TBVGG3_CheckPadded(const uint x, const uint y, const uint w, const uint h)
{
    if(x < 0 || y < 0 || x > w || y > h)
        return 1;
    return 0;
}

float TBVGG3_3x3Conv(const float*** input, const float** filter, const uint depth, const uint width, const uint height, const uint x, const uint y)
{
    // input depth needs to be same as filter depth
    // This will return a single float output. Call this x*y times per filter.
    // It's zero padded so if TBVGG3_CheckPadded() returns 1 it's a no operation
    float ro = 0;
    for(uint i = 0; i < depth; i++)
    {
        uint nx = 0, ny = 0;

        // lower row
        nx = x-1, ny = y-1;
        if(TBVGG3_CheckPadded(nx, ny, width, height) == 0)
            ro += input[i][ny][nx] * filter[i][0];

        nx = x,   ny = y-1;
        if(TBVGG3_CheckPadded(nx, ny, width, height) == 0)
            ro += input[i][ny][nx] * filter[i][1];

        nx = x+1, ny = y-1;
        if(TBVGG3_CheckPadded(nx, ny, width, height) == 0)
            ro += input[i][ny][nx] * filter[i][2];

        // middle row
        nx = x-1, ny = y;
        if(TBVGG3_CheckPadded(nx, ny, width, height) == 0)
            ro += input[i][ny][nx] * filter[i][3];

        nx = x,   ny = y;
        if(TBVGG3_CheckPadded(nx, ny, width, height) == 0)
            ro += input[i][ny][nx] * filter[i][4];

        nx = x+1, ny = y;
        if(TBVGG3_CheckPadded(nx, ny, width, height) == 0)
            ro += input[i][ny][nx] * filter[i][5];

        // top row
        nx = x-1, ny = y+1;
        if(TBVGG3_CheckPadded(nx, ny, width, height) == 0)
            ro += input[i][ny][nx] * filter[i][6];

        nx = x,   ny = y+1;
        if(TBVGG3_CheckPadded(nx, ny, width, height) == 0)
            ro += input[i][ny][nx] * filter[i][7];

        nx = x+1, ny = y+1;
        if(TBVGG3_CheckPadded(nx, ny, width, height) == 0)
            ro += input[i][ny][nx] * filter[i][8];
    }

    // bias
    ro += filter[i][9];

    // return output
    return ro;
}

float TBVGG3_Process(TBVGG3_Network* net, const float* inputs, const TBVGG3_LEARNTYPE learn)
{
    if(net == NULL){return -1;}

}

#endif
