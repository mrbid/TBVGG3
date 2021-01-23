/*
--------------------------------------------------
    James William Fletcher (james@voxdsp.com)
--------------------------------------------------
    This allows users to create 28x28x3 sample
    datasets of the reticule area while playing
    in game.

    All data is saved in r,g,b order.

    2352 bytes per sample in _rgb_bytes.dat

    9408 bytes (2352 floats at 4 byte each)
        per sample in all other files.

    It's important you have espeak installed
    because it is regulating the program from
    taking too many snapshots at once.
    Otherwise replace the function body of
    speakSS() with usleep(300000);

    Compile:
    clang fps_dataset_logger.c -Ofast -lX11 -lm -o aim
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

#define uint unsigned int
#define SCAN_AREA 28

const uint r0 = SCAN_AREA;  // dimensions of sample image square
const uint r2 = r0*r0;      // total pixels in square
const uint r2i = r2*3;      // total inputs to neural net pixels*channels
const uint rd2 = r0/2;      // total pixels divided by two
uint x=0, y=0;

char inputb[r2i] = {0};
float input[r2i] = {0};
    double r[r2] = {0};
    double g[r2] = {0};
    double b[r2] = {0};

Display *d;
int si;
Window twin;
GC gc = 0;

/***************************************************
   ~~ Utils
*/
//https://www.cl.cam.ac.uk/~mgk25/ucs/keysymdef.h
//https://stackoverflow.com/questions/18281412/check-keypress-in-c-on-linux/52801588
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

void speakSS(const char* text)
{
    char s[256];
    sprintf(s, "/usr/bin/espeak -s 360 \"%s\"", text);
    if(system(s) <= 0)
        sleep(1);
}

Window getWindow() // gets child window mouse is over
{
    Display *d = XOpenDisplay((char*) NULL);
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

void saveSample(Window w, const char* name)
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

            // if(r[i] == 0 && b[i] == 0 && g[i] > 65500)
            //     printf("crosshair detected %ix%i\n", x, y);

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
    // regular 0-255 byte per colour channel
    for(uint i = 0, i2 = 0; i < r2i; i += 3, i2++)
    {
        inputb[i]   = (char)((r[i2] / 65535.0) * 255);
        inputb[i+1] = (char)((g[i2] / 65535.0) * 255);
        inputb[i+2] = (char)((b[i2] / 65535.0) * 255);
    }

    char file[256];
    sprintf(file, "%s_rgb_bytes.dat", name);
    FILE* f = fopen(file, "ab");
    {
        if(fwrite(inputb, 1, sizeof(inputb), f) != sizeof(inputb))
        {
            printf("CORRUPTED: %s\n", file);
            speakS("Corrupted write to file exiting.");
            exit(0); // this is rare so on corrruption we exit so that user can manually bytecount and truncate the appended error bytes for recovery
        }
        fclose(f);
    }

    /////////////////
    // 0-1 normalised
    for(uint i = 0, i2 = 0; i < r2i; i += 3, i2++)
    {
        input[i]   = r[i2] / 65535.0;
        input[i+1] = g[i2] / 65535.0;
        input[i+2] = b[i2] / 65535.0;
    }

    sprintf(file, "%s_normalised_floats.dat", name);
    f = fopen(file, "ab");
    {
        if(fwrite(input, 1, sizeof(input), f) != sizeof(input))
        {
            printf("CORRUPTED: %s\n", file);
            speakS("Corrupted write to file exiting.");
            exit(0); // this is rare so on corrruption we exit so that user can manually bytecount and truncate the appended error bytes for recovery
        }
        fclose(f);
    }

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

    sprintf(file, "%s_meancentered_floats.dat", name);
    f = fopen(file, "ab");
    {
        if(fwrite(input, 1, sizeof(input), f) != sizeof(input))
        {
            printf("CORRUPTED: %s\n", file);
            speakS("Corrupted write to file exiting.");
            exit(0); // this is rare so on corrruption we exit so that user can manually bytecount and truncate the appended error bytes for recovery
        }
        fclose(f);
    }

    /////////////////
    // -1 to 1 normalised
    for(uint i = 0, i2 = 0; i < r2i; i += 3, i2++)
    {
        input[i]   = ((r[i2]-rl)+1e-7) / (rmd+1e-7);
        input[i+1] = ((g[i2]-gl)+1e-7) / (gmd+1e-7);
        input[i+2] = ((b[i2]-bl)+1e-7) / (bmd+1e-7);
    }

    sprintf(file, "%s_zerocentered_floats.dat", name);
    f = fopen(file, "ab");
    {
        if(fwrite(input, 1, sizeof(input), f) != sizeof(input))
            printf("CORRUPTED: %s\n", file);
        fclose(f);
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
    printf("P = Toggle crosshair\n");
    printf("V = Show sample frame area\n");
    printf("Q = Sample enemy to dataset.\n");
    printf("E = Sample non-enemy to dataset.\n");
    printf("\n\n");

    //
    
    XEvent event;
    memset(&event, 0x00, sizeof(event));
    
    uint enable = 0;
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
                d = XOpenDisplay((char*) NULL);
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
                        printf("Don't have this enabled while taking samples, the crosshair will be burned into your training data.\n");
                    }
                }
            }
            
            if(hotkeys == 1 && key_is_pressed(XK_V))
            {
                // draw sample outline
                XSetForeground(d, gc, 0);
                XDrawRectangle(d, event.xbutton.window, gc, x-rd2-1, y-rd2-1, r0+2, r0+2);
                XSetForeground(d, gc, 0);
                XDrawRectangle(d, event.xbutton.window, gc, x-rd2-2, y-rd2-2, r0+4, r0+4);
                XFlush(d);
            }
            else if(hotkeys == 1 && key_is_pressed(XK_Q)) // train when pressed
            {
                saveSample(twin, "target");

                // draw sample outline
                XSetForeground(d, gc, 65280);
                XDrawRectangle(d, event.xbutton.window, gc, x-rd2-1, y-rd2-1, r0+2, r0+2);
                XSetForeground(d, gc, 0);
                XDrawRectangle(d, event.xbutton.window, gc, x-rd2-2, y-rd2-2, r0+4, r0+4);
                XFlush(d);
                
                speakSS("v");
            }
            else if(hotkeys == 1 && key_is_pressed(XK_E)) // untrain when pressed
            {
                saveSample(twin, "nontarget");

                // draw sample outline
                XSetForeground(d, gc, 16711680);
                XDrawRectangle(d, event.xbutton.window, gc, x-rd2-1, y-rd2-1, r0+2, r0+2);
                XSetForeground(d, gc, 0);
                XDrawRectangle(d, event.xbutton.window, gc, x-rd2-2, y-rd2-2, r0+4, r0+4);
                XFlush(d);
                
                speakSS("t");
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
