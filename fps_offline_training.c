/*
--------------------------------------------------
    James William Fletcher (james@voxdsp.com)
--------------------------------------------------
    This trains a TBVGG3 network using data produced by;
    fps_dataset_logger.c

    9408 bytes per sample, 2352 floats.
    order R,G,B.

    I don't reseve 30% of the data for testing when
    computing the RMSE, I also do not shuffle the data
    once loaded but these are not detrimental.

    Make sure TARGET_SAMPLES is tuned to the right amount
    of samples collected for both targets and non-targets,
    you will have to check by dividing the file sizes in
    bytes by 9408.

    Compile:
    clang fps_offline_training.c -Ofast -lX11 -lm -o aim
*/

#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>

#include "TBVGG3_SGD.h"

#define uint unsigned int
#define SCAN_AREA 28
#define TARGET_SAMPLES 300

const uint r0 = SCAN_AREA;  // dimensions of sample image square
const uint r2 = r0*r0;      // total pixels in square
const uint r2i = r2*3;      // total inputs to neural net pixels*channels
const uint rd2 = r0/2;      // total pixels divided by two

float target_samples[TARGET_SAMPLES][3][28][28] = {0};
float nontarget_samples[TARGET_SAMPLES][3][28][28] = {0};
float read_buff[r2i] = {0};

TBVGG3_Network net;

/***************************************************
   ~~ Utilities
*/
unsigned int crc32c(const unsigned char *message, const size_t len) // modified from "Hacker's Delight"
{
    unsigned int byte, crc, mask;
    static unsigned int table[256];

    if(table[1] == 0)
    {
        for(byte = 0; byte <= 255; byte++)
        {
            crc = byte;
            for(int j = 7; j >= 0; j--)
            {
                mask = -(crc & 1);
                crc = (crc >> 1) ^ (0xEDB88320 & mask);
            }
            table[byte] = crc;
        }
    }

    crc = 0xFFFFFFFF;
    for(size_t i = 0; i < len; i++)
        crc = (crc >> 8) ^ table[(crc ^ message[i]) & 0xFF];
    return ~crc;
}

/***************************************************
   ~~ Program Entry Point
*/
int main(int argc, char *argv[])
{
    // init
    printf("James William Fletcher (james@voxdsp.com)\n\n");
    TBVGG3_Reset(&net);

    // load existing weights?
    if(argc == 2 && strcmp(argv[1], "new") == 0)
    {
        printf("!! Started with random weights.\n\n");
    }
    else
    {
        if(TBVGG3_LoadNetwork(&net, "weights.dat") == 0)
        {
            printf(">> Loaded weights.\n\n");
        }
        else
        {
            printf("!! Started with random weights.\n\n");
        }
    }

    // load target samples
    printf("Loading target samples:\n");
    FILE* f = fopen("t.dat", "rb");
    if(f != NULL)
    {
        for(uint i = 0; i < TARGET_SAMPLES; i++)
        {
            while(fread(&read_buff, 1, sizeof(read_buff), f) != sizeof(read_buff))
                sleep(1);
            
            printf("(%i)CRC32: %X\n", i, crc32c((const unsigned char*)&read_buff, sizeof(read_buff)));

            for(uint j = 0; j < r2i; j += 3)
            {
                const uint rj = (j/3);
                uint y = rj/SCAN_AREA;
                if(y > 27){y = 27;}
                const uint x = rj-(y*SCAN_AREA);
                target_samples[i][0][y][x] = read_buff[i];
                target_samples[i][1][y][x] = read_buff[i+1];
                target_samples[i][2][y][x] = read_buff[i+2];
                //printf("%ux%u: %.2f %.2f %.2f\n", x, y, target_samples[i][0][y][x], target_samples[i][1][y][x], target_samples[i][2][y][x]);
            }
        }
        fclose(f);
    }

    // load non-target samples
    printf("\nLoading non-target samples:\n");
    f = fopen("nt.dat", "rb");
    if(f != NULL)
    {
        for(uint i = 0; i < TARGET_SAMPLES; i++)
        {
            while(fread(&read_buff, 1, sizeof(read_buff), f) != sizeof(read_buff))
                sleep(1);
            
            printf("(%i)CRC32: %X\n", i, crc32c((const unsigned char*)&read_buff, sizeof(read_buff)));

            for(uint j = 0; j < r2i; j += 3)
            {
                const uint rj = (j/3);
                uint y = rj/SCAN_AREA;
                if(y > 27){y = 27;}
                const uint x = rj-(y*SCAN_AREA);
                nontarget_samples[i][0][y][x] = read_buff[i];
                nontarget_samples[i][1][y][x] = read_buff[i+1];
                nontarget_samples[i][2][y][x] = read_buff[i+2];
                //printf("%ux%u: %.2f %.2f %.2f\n", x, y, nontarget_samples[i][0][y][x], nontarget_samples[i][1][y][x], nontarget_samples[i][2][y][x]);
            }
        }
        fclose(f);
    }

    // train
    printf("\nTraining...\n");
    for(uint i = 0; i < TARGET_SAMPLES; i++)
    {
        TBVGG3_Process(&net, target_samples[i], LEARN_MAX);
        TBVGG3_Process(&net, nontarget_samples[i], LEARN_MIN);
    }

    // test rmse
    float squaremean = 0;
    for(uint i = 0; i < TARGET_SAMPLES; i++)
    {
        const float r = 1 - TBVGG3_Process(&net, target_samples[i], NO_LEARN);
        squaremean += r*r;
    }
    squaremean /= TARGET_SAMPLES;
    const float rmse = sqrt(squaremean);
    printf("RMSE: %f\n", rmse);

    // save it !
    TBVGG3_SaveNetwork(&net, "weights.dat");
    TBVGG3_Dump(&net, "dump");

    // done
    return 0;
}
