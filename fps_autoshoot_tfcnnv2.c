/*
--------------------------------------------------
    James William Fletcher (james@voxdsp.com)
--------------------------------------------------
    This is an attempt to create a real-time
    trainable autoshoot bot for FPS games
    using TFCNNv2.h.

    Compile:
    clang aimbot.c -Ofast -lX11 -lm -o aim
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

// void playTone()
// {
//     if(system("/usr/bin/aplay --quiet /usr/share/sounds/a.wav") <= 0)
//         sleep(1);
// }

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

int isFocus(const Window w)
{
    // open display 0
    Display *d = XOpenDisplay((char *) NULL);
    if(d == NULL)
        return 0;

    // get default screen
    int si = XDefaultScreen(d);

    // mouse event
    XEvent event;
    memset(&event, 0x00, sizeof(event));

    // find target window
    XQueryPointer(d, RootWindow(d, si), &event.xbutton.root, &event.xbutton.window, &event.xbutton.x_root, &event.xbutton.y_root, &event.xbutton.x, &event.xbutton.y, &event.xbutton.state);
    event.xbutton.subwindow = event.xbutton.window;
    while(event.xbutton.subwindow)
    {
        event.xbutton.window = event.xbutton.subwindow;
        XQueryPointer(d, event.xbutton.window, &event.xbutton.root, &event.xbutton.subwindow, &event.xbutton.x_root, &event.xbutton.y_root, &event.xbutton.x, &event.xbutton.y, &event.xbutton.state);
    }

    // not the right window?
    if(event.xbutton.window != w)
    {
        XCloseDisplay(d);
        return 0;
    }

    // done
    XCloseDisplay(d);
    return 1;
}

Window getWindow()
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

network net;
void sigint_handler(int sig_num) 
{
    static int m_qe = 0;
    
    if(m_qe == 0)
    {
        printf("\nPlease Wait while the network state is saved...\n\n");
        m_qe = 1;

        saveNetwork(&net, "weights.dat");
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
    printf("L-CTRL + L-ALT = Toggle BOT & HOTKEYS ON/OFF\n");
    printf("T = Toggle auto-shoot\n");
    printf("C = Output input array from reticule area.\n");
    printf("G = Get activation for reticule area.\n");
    printf("Q = Train on reticule area.\n");
    printf("E = Un-Train on reticule area.\n");
    printf("\n");
    printf("Please input reticule scan area: \n");

    // uint rsa = 30;
    // char in[32] = {0};
    // fgets(in, 32, stdin);
    // if(in[0] != 0x0A)
    // {
    //     rsa = atoi(in);
    // }
    // printf("\n%u has been set.\n\n", rsa);

    //

    signal(SIGINT, sigint_handler);

    //

    const uint r0 = 30;    // dimensions of sample image square
    const uint r2 = r0*r0;  // total pixels in square
    const uint r2i = r2*3;  // total inputs to neural net pixels*channels
    const uint rd2 = r0/2;  // total pixels divided by two

    //

    Display *d;
    int si;
    Window twin;

    //

    createNetwork(&net, WEIGHT_INIT_UNIFORM_GLOROT, r2i, 3, 512, 1);
    setOptimiser(&net, OPTIM_NESTEROV);
    setActivator(&net, ELLIOT);
    setLearningRate(&net, 0.01);

    setUnitDropout(&net, 0.3);
    setMomentum(&net, 0.1);
    // setTargetMin(&net, 0.1);
    // setTargetMax(&net, 0.9);

    //
    if(argc == 2 && strcmp(argv[1], "new") == 0)
    {
        printf("Starting with no training data.\n\n");
    }
    else
    {
        if(loadNetwork(&net, "weights.dat") == 0)
        {
            printf("Loaded network weights.\n\n");
        }
        else
        {
            printf("Starting with no training data.\n\n");
        }
    }

    //
    
    XEvent event;
    memset(&event, 0x00, sizeof(event));
    
    uint x=0, y=0;
    uint enable = 0;
    uint autofire = 0;

    //
    
    while(1)
    {
        // loop every 10 ms (1,000 microsecond = 1 millisecond)
        usleep(10000);

        // input toggle
        if(key_is_pressed(XK_Control_L) && key_is_pressed(XK_Alt_L))
        {
            if(enable == 0)
            {
                twin = getWindow();
                x = 0, y = 0;
                enable = 1;
                printf("\a\n");
                usleep(300000);
                printf("AUTO-SHOOT: ON\n");
                speakS("on");
            }
            else
            {
                enable = 0;
                printf("AUTO-SHOOT: OFF\n");
                speakS("off");
            }
        }
        
        // toggle bot on/off
        if(enable == 1 && isFocus(twin) == 1)
        {
            // open display 0
            d = XOpenDisplay((char *) NULL);
            if(d == NULL)
                continue;

            // get default screen
            si = XDefaultScreen(d);
            
            // reset mouse event
            memset(&event, 0x00, sizeof(event));

            // ready to press down mouse 1
            event.type = ButtonPress;
            event.xbutton.button = Button1;
            event.xbutton.same_screen = True;
            
            // find target window
            XQueryPointer(d, RootWindow(d, si), &event.xbutton.root, &event.xbutton.window, &event.xbutton.x_root, &event.xbutton.y_root, &event.xbutton.x, &event.xbutton.y, &event.xbutton.state);
            event.xbutton.subwindow = event.xbutton.window;
            while(event.xbutton.subwindow)
            {
                event.xbutton.window = event.xbutton.subwindow;
                XQueryPointer(d, event.xbutton.window, &event.xbutton.root, &event.xbutton.subwindow, &event.xbutton.x_root, &event.xbutton.y_root, &event.xbutton.x, &event.xbutton.y, &event.xbutton.state);
            }

            // get center window point (x & y)
            if(x == 0 && y == 0)
            {
                XWindowAttributes attr;
                XGetWindowAttributes(d, event.xbutton.window, &attr);
                x = attr.width/2;
                y = attr.height/2;
                printf("CSET: %ix%i\n", x, y);
            }

            // get image block
            XImage *img = XGetImage(d, event.xbutton.window, x-rd2, y-rd2, r0, r0, AllPlanes, XYPixmap);
            if(img == NULL)
            {
                XCloseDisplay(d);
                continue;
            }

            // colour map
            const Colormap map = XDefaultColormap(d, si);

            // extract colour information
            double rh = 0, rl = 99999999999999, rm = 0;
            double gh = 0, gl = 99999999999999, gm = 0;
            double bh = 0, bl = 99999999999999, bm = 0;
            double r[r2] = {0};
            double g[r2] = {0};
            double b[r2] = {0};
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

            // calculate mean normalised input buffer
            float input[r2i] = {0};

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


            // detect when pressed
            if(key_is_pressed(XK_T))
            {
                if(autofire == 0)
                {
                    autofire = 1;
                    printf("AUTO-FIRE: ON\n");
                    speakS("af on");
                }
                else
                {
                    autofire = 0;
                    printf("AUTO-FIRE: OFF\n");
                    speakS("af off");
                }
            }

            // print input data
            if(key_is_pressed(XK_C))
            {
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
            
            if(autofire == 1) // left mouse trigger on activation
            {
                if(processNetwork(&net, &input[0], NO_LEARN) > 0.7)
                {
                    // fire mouse down
                    XSendEvent(d, PointerWindow, True, 0xfff, &event);
                    XFlush(d);
                    
                    // wait 100ms (or ban for suspected cheating)
                    usleep(100000);
                    
                    // release mouse down
                    event.type = ButtonRelease;
                    event.xbutton.state = 0x100;
                    XSendEvent(d, PointerWindow, True, 0xfff, &event);
                    XFlush(d);
                }
            }
            else if(key_is_pressed(XK_G)) // print activation when pressed
            {
                printf("A: %f\n", processNetwork(&net, &input[0], NO_LEARN));
            }
            else
            {
                if(key_is_pressed(XK_Q)) // train when pressed
                {
                    processNetwork(&net, &input[0], LEARN_MAX);
                }
                else if(key_is_pressed(XK_E)) // untrain when pressed
                {
                    processNetwork(&net, &input[0], LEARN_MIN);
                }
            }

            //Close the display
            XCloseDisplay(d);
        }
    }

    // done, never gets here in regular execution flow
    return 0;
}
