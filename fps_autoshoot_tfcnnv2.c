/*
--------------------------------------------------
    James William Fletcher (james@voxdsp.com)
--------------------------------------------------
    This is an attempt to create a real-time
    trainable autoshoot bot for FPS games
    using TFCNNv2.h.

    Compile:
    clang fps_autoshoot_tfcnnv2.c -Ofast -lX11 -lm -o aim
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
#include <signal.h>

#include "TFCNNv2.h"

#define uint unsigned int
#define SCAN_AREA 30
#define ACTIVATION_SENITIVITY 0.90
#define FIRE_RATE_LIMIT_MS 3000

const uint r0 = SCAN_AREA;  // dimensions of sample image square
const uint r2 = r0*r0;      // total pixels in square
const uint r2i = r2*3;      // total inputs to neural net pixels*channels
const uint rd2 = r0/2;      // total pixels divided by two
uint x=0, y=0;

float input[r2i] = {0};
    double r[r2] = {0};
    double g[r2] = {0};
    double b[r2] = {0};

Display *d;
int si;
Window twin;
GC gc = 0;

network net;


/***************************************************
   ~~ Utils
*/
//https://www.cl.cam.ac.uk/~mgk25/ucs/keysymdef.h
int key_is_pressed(KeySym ks)
{
    Display *dpy = XOpenDisplay(":0");
    char keys_return[32];
    XQueryKeymap(dpy, keys_return);
    KeyCode kc2 = XKeysymToKeycode(dpy, ks);
    int isPressed = !!(keys_return[kc2 >> 3] & (1 << (kc2 & 7)));
    XCloseDisplay(dpy);
    return isPressed;
}

void speakS(const char* text)
{
    char s[256];
    sprintf(s, "/usr/bin/espeak \"%s\"", text);
    if(system(s) <= 0)
        sleep(1);
}

void speakI(const int i)
{
    char s[256];
    sprintf(s, "/usr/bin/espeak %i", i);
    if(system(s) <= 0)
        sleep(1);
}

void speakF(const double f)
{
    char s[256];
    sprintf(s, "/usr/bin/espeak %.1f", f);
    if(system(s) <= 0)
        sleep(1);
}

Window getWindow() // gets child window mouse is over
{
    Display *d = XOpenDisplay((char *) NULL);
    if(d == NULL)
        return 0;
    int si = XDefaultScreen(d);
    XEvent event;
    memset(&event, 0x00, sizeof(event));
    XQueryPointer(d, RootWindow(d, si), &event.xbutton.root, &event.xbutton.window, &event.xbutton.x_root, &event.xbutton.y_root, &event.xbutton.x, &event.xbutton.y, &event.xbutton.state);
    event.xbutton.subwindow = event.xbutton.window;
    while(event.xbutton.subwindow)
    {
        event.xbutton.window = event.xbutton.subwindow;
        XQueryPointer(d, event.xbutton.window, &event.xbutton.root, &event.xbutton.subwindow, &event.xbutton.x_root, &event.xbutton.y_root, &event.xbutton.x, &event.xbutton.y, &event.xbutton.state);
    }
    const Window ret = event.xbutton.window;
    XCloseDisplay(d);
    return ret;
}

void processScanArea(Window w)
{
    // get image block
    XImage *img = XGetImage(d, w, x-rd2, y-rd2, r0, r0, AllPlanes, XYPixmap);
    if(img == NULL)
        return;

    // colour map
    const Colormap map = XDefaultColormap(d, si);

    // extract colour information
    double rh = 0, rl = 99999999999999, rm = 0;
    double gh = 0, gl = 99999999999999, gm = 0;
    double bh = 0, bl = 99999999999999, bm = 0;
    int i = 0;
    for(int y = 0; y < r0; y++)
    {
        for(int x = 0; x < r0; x++)
        {
            XColor c;
            c.pixel = XGetPixel(img, x, y);
            XQueryColor(d, map, &c);

            r[i] = (double)c.red;
            g[i] = (double)c.green;
            b[i] = (double)c.blue;

            if(r[i] > rh){rh = r[i];}
            if(r[i] < rl){rl = r[i];}
            rm += r[i];

            if(g[i] > gh){gh = g[i];}
            if(g[i] < gl){gl = g[i];}
            gm += g[i];

            if(b[i] > bh){bh = b[i];}
            if(b[i] < bl){bl = b[i];}
            bm += b[i];

            i++;
        }
    }

    // free image block
    XFree(img);


    /////////////////
    // 0-1 normalised

    // for(uint i = 0, i2 = 0; i < r2i; i += 3, i2++)
    // {
    //     input[i]   = r[i2] / 65535.0;
    //     input[i+1] = g[i2] / 65535.0;
    //     input[i+2] = b[i2] / 65535.0;
    // }


    /////////////////
    // mean normalised

    rm /= r2;
    gm /= r2;
    bm /= r2;

    const double rmd = rh-rl;
    const double gmd = gh-gl;
    const double bmd = bh-bl;

    for(uint i = 0, i2 = 0; i < r2i; i += 3, i2++)
    {
        input[i]   = ((r[i2]-rm)+1e-7) / (rmd+1e-7);
        input[i+1] = ((g[i2]-gm)+1e-7) / (gmd+1e-7);
        input[i+2] = ((b[i2]-bm)+1e-7) / (bmd+1e-7);
    }


    /////////////////
    // -1 to 1 normalised

    // const double rmd = rh-rl;
    // const double gmd = gh-gl;
    // const double bmd = bh-bl;

    // for(uint i = 0, i2 = 0; i < r2i; i += 3, i2++)
    // {
    //     input[i]   = ((r[i2]-rl)+1e-7) / (rmd+1e-7);
    //     input[i+1] = ((g[i2]-gl)+1e-7) / (gmd+1e-7);
    //     input[i+2] = ((b[i2]-bl)+1e-7) / (bmd+1e-7);
    // }
}

void sigint_handler(int sig_num) 
{
    static int m_qe = 0;
    
    if(m_qe == 0)
    {
        printf("\nPlease Wait while the network state is saved...\n\n");
        m_qe = 1;

        saveNetwork(&net, "weights.dat");
        exportLayers(&net, "export.txt");
        exit(0);
    }
}

/***************************************************
   ~~ Program Entry Point
*/
int main(int argc, char *argv[])
{
    printf("James William Fletcher (james@voxdsp.com)\n\n");
    printf("Hotkeys:\n");
    printf("L-CTRL + L-ALT = Toggle BOT ON/OFF\n");
    printf("R-CTRL + R-ALT = Toggle HOTKEYS ON/OFF\n");
    printf("T = Toggle auto-shoot\n");
    printf("P = Toggle crosshair\n");
    printf("C = Output input array from reticule area.\n");
    printf("M = Reset Network.\n");
    printf("G = Get activation for reticule area.\n");
    printf("Q = Train on reticule area.\n");
    printf("E = Un-Train on reticule area.\n");
    printf("\n\n");

    //

    signal(SIGINT, sigint_handler);

    //

    createNetwork(&net, WEIGHT_INIT_NORMAL_GLOROT, r2i, 3, 128, 1);
    setOptimiser(&net, OPTIM_NESTEROV);
    setActivator(&net, ELLIOT);
    setLearningRate(&net, 0.3);
    setGain(&net, 0.3);

    setUnitDropout(&net, 0.3);
    setMomentum(&net, 0.1);
    // setTargetMin(&net, 0.1);
    // setTargetMax(&net, 0.9);

    //

    if(argc == 2 && strcmp(argv[1], "new") == 0)
    {
        printf("!! Starting with no training data.\n\n");
    }
    else
    {
        if(loadNetwork(&net, "weights.dat") == 0)
        {
            printf(">> Loaded network weights.\n\n");
        }
        else
        {
            printf("!! Starting with no training data.\n\n");
        }
    }

    //
    
    XEvent event;
    memset(&event, 0x00, sizeof(event));
    
    uint enable = 0;
    uint autofire = 0;
    uint crosshair = 0;
    uint hotkeys = 1;

    //
    
    while(1)
    {
        // loop every 10 ms (1,000 microsecond = 1 millisecond)
        usleep(1000);

        // bot toggle
        if(key_is_pressed(XK_Control_L) && key_is_pressed(XK_Alt_L))
        {
            if(enable == 0)
            {
                // open display 0
                d = XOpenDisplay((char *) NULL);
                if(d == NULL)
                    continue;

                // get default screen
                si = XDefaultScreen(d);

                // get graphics context
                gc = DefaultGC(d, si);

                // get window
                twin = getWindow();

                // get center window point (x & y)
                XWindowAttributes attr;
                XGetWindowAttributes(d, twin, &attr);
                x = attr.width/2;
                y = attr.height/2;

                // set mouse event
                memset(&event, 0x00, sizeof(event));
                event.type = ButtonPress;
                event.xbutton.button = Button1;
                event.xbutton.same_screen = True;
                event.xbutton.subwindow = twin;
                event.xbutton.window = twin;

                enable = 1;
                usleep(300000);
                printf("BOT: ON [%ix%i]\n", x, y);
                speakS("on");
            }
            else
            {
                enable = 0;
                usleep(300000);
                XCloseDisplay(d);
                printf("BOT: OFF\n");
                speakS("off");
            }
        }
        
        // toggle bot on/off
        if(enable == 1 && getWindow() == twin)
        {
            // input toggle
            if(key_is_pressed(XK_Control_R) && key_is_pressed(XK_Alt_R))
            {
                if(hotkeys == 0)
                {
                    hotkeys = 1;
                    usleep(300000);
                    printf("HOTKEYS: ON [%ix%i]\n", x, y);
                    speakS("hk on");
                }
                else
                {
                    hotkeys = 0;
                    usleep(300000);
                    printf("HOTKEYS: OFF\n");
                    speakS("hk off");
                }
            }

            if(hotkeys == 1)
            {
                // detect when pressed
                if(key_is_pressed(XK_T))
                {
                    if(autofire == 0)
                    {
                        autofire = 1;
                        usleep(300000);
                        printf("AUTO-FIRE: ON\n");
                        speakS("af on");
                    }
                    else
                    {
                        autofire = 0;
                        usleep(300000);
                        printf("AUTO-FIRE: OFF\n");
                        speakS("af off");
                    }
                }
                
                // crosshair toggle
                if(key_is_pressed(XK_P))
                {
                    if(crosshair == 0)
                    {
                        crosshair = 1;
                        usleep(300000);
                        printf("CROSSHAIR: ON\n");
                        speakS("cx on");
                    }
                    else
                    {
                        crosshair = 0;
                        usleep(300000);
                        printf("CROSSHAIR: OFF\n");
                        speakS("cx off");
                    }
                }

                // print input data
                if(key_is_pressed(XK_C))
                {
                    processScanArea(twin);

                    // per channel
                    printf("R: ");
                    for(uint i = 0; i < r2; i++)
                        printf("%.2f ", r[i]);
                    printf("\n\n");

                    printf("G: ");
                    for(uint i = 0; i < r2; i++)
                        printf("%.2f ", g[i]);
                    printf("\n\n");

                    printf("B: ");
                    for(uint i = 0; i < r2; i++)
                        printf("%.2f ", b[i]);
                    printf("\n\n");

                    // mean normalised 1d input array
                    printf("I: ");
                    // for(uint i = 0; i < r2i; i++)
                    //     printf("%.2f ", input[i]);
                    for(uint i = 0; i < r2i; i += 3)
                        printf("%.2f %.2f %.2f :: ", input[i], input[i+1], input[i+2]);
                    printf("\n\n");
                }

                // reset network
                if(key_is_pressed(XK_M))
                {
                    resetNetwork(&net);
                    printf("!!! NETWORK RESET !!!\n");
                    usleep(300000);
                    speakS("reset");
                }

            }
            
            if(autofire == 1) // left mouse trigger on activation
            {
                processScanArea(twin);
                if(processNetwork(&net, &input[0], NO_LEARN) > ACTIVATION_SENITIVITY)
                {
                    // fire mouse down
                    event.type = ButtonPress;
                    event.xbutton.state = 0;
                    XSendEvent(d, PointerWindow, True, 0xfff, &event);
                    XFlush(d);
                    
                    // wait 100ms (or ban for suspected cheating)
                    usleep(100000);
                    
                    // release mouse down
                    event.type = ButtonRelease;
                    event.xbutton.state = 0x100;
                    XSendEvent(d, PointerWindow, True, 0xfff, &event);
                    XFlush(d);

                    // fire limit
                    usleep(FIRE_RATE_LIMIT_MS * 1000);
                }
            }
            else if(hotkeys == 1 && key_is_pressed(XK_G)) // print activation when pressed
            {
                processScanArea(twin);
                const float ret = processNetwork(&net, &input[0], NO_LEARN);
                printf("A: %f\n", ret);
                if(ret > ACTIVATION_SENITIVITY)
                {
                    XSetForeground(d, gc, 65280);
                    XDrawRectangle(d, event.xbutton.window, gc, x-rd2-1, y-rd2-1, r0+2, r0+2);
                    XSetForeground(d, gc, 0);
                    XDrawRectangle(d, event.xbutton.window, gc, x-rd2-2, y-rd2-2, r0+4, r0+4);
                    XFlush(d);
                }
                else
                {
                    XSetForeground(d, gc, 16711680);
                    XDrawRectangle(d, event.xbutton.window, gc, x-rd2-1, y-rd2-1, r0+2, r0+2);
                    XSetForeground(d, gc, 0);
                    XDrawRectangle(d, event.xbutton.window, gc, x-rd2-2, y-rd2-2, r0+4, r0+4);
                    XFlush(d);
                }                
            }
            else
            {
                if(hotkeys == 1 && key_is_pressed(XK_Q)) // train when pressed
                {
                    processScanArea(twin);
                    processNetwork(&net, &input[0], LEARN_MAX);

                    // draw sample outline
                    XSetForeground(d, gc, 65280);
                    XDrawRectangle(d, event.xbutton.window, gc, x-rd2-1, y-rd2-1, r0+2, r0+2);
                    XSetForeground(d, gc, 0);
                    XDrawRectangle(d, event.xbutton.window, gc, x-rd2-2, y-rd2-2, r0+4, r0+4);
                    XFlush(d);
                }
                else if(hotkeys == 1 && key_is_pressed(XK_E)) // untrain when pressed
                {
                    processScanArea(twin);
                    processNetwork(&net, &input[0], LEARN_MIN);

                    // draw sample outline
                    XSetForeground(d, gc, 16711680);
                    XDrawRectangle(d, event.xbutton.window, gc, x-rd2-1, y-rd2-1, r0+2, r0+2);
                    XSetForeground(d, gc, 0);
                    XDrawRectangle(d, event.xbutton.window, gc, x-rd2-2, y-rd2-2, r0+4, r0+4);
                    XFlush(d);
                }
            }

            if(crosshair == 1)
            {
                XSetForeground(d, gc, 0);
                XDrawPoint(d, event.xbutton.window, gc, x, y);
                XSetForeground(d, gc, 65280);
                XDrawRectangle(d, event.xbutton.window, gc, x-1, y-1, 2, 2);
                XSetForeground(d, gc, 0);
                XDrawRectangle(d, event.xbutton.window, gc, x-2, y-2, 4, 4);
                XFlush(d);
            }

        ///
        }
    }

    // done, never gets here in regular execution flow
    return 0;
}
