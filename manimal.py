#! /usr/bin/env python3
from aicspylibczi import CziFile
import argparse
import math
import numpy as np
import os
import pathlib
from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageTk
import queue
import sys
import threading
import tkinter as tk
from tkinter import ttk
import tkinter.font as tkfont

def add2D(a, b):
    return (a[0] + b[0], a[1] + b[1])

def diff2D(a, b):
    return (a[0] - b[0], a[1] - b[1])

def findNoGreaterThan(x, xs):
    """
    Find first item in xs no greater than x, assuming xs is in
    descending order
    """
    for i in range(len(xs)):
        if xs[i] <= x:
            return i
    return len(xs)

def strip_zen_nonsense(h):
    """
    Zen adds a lot of weird stuff to a CSV when it saves it out.
    Here we remove it.
    """
    h = h.strip('"')
    start = h.find("::")
    start = 0 if start < 0 else start + 2
    end = h.find("!!")
    if end < 0:
        end = len(h)
    return h[start:end]

class CsvLoader:
    def __init__(self, fh):
        rows = fh.read().splitlines()
        if len(rows) == 0:
            self.headers = None
            self.rows = []
            return
        self.headers = [strip_zen_nonsense(h) for h in rows[0].split(',')]
        self.rows = [r for r in rows[1:] if r.strip(',"')]

    def headersAre(self, cshs):
        """
        Returns True iff cshs (a comma-separated list of headers) matches
        the actual headers.
        """
        if not self.headers:
            return False
        hs = cshs.split(',')
        return hs == self.headers

    def rowCount(self):
        return len(self.rows)

    def generateRows(self):
        for r in self.rows:
            yield [c.strip('"') for c in r.split(',')]


class ImageFile:
    def __init__(self, path):
        pass
    def readImage(self, queue, radius : int, scale, cx : int, cy : int, brightness: float):
        raise NotImplementedError('readImage')
    def surfaceZ(self, x, y):
        return 0
    def getBbox(self):
        raise NotImplementedError('readImage')
    def getPixelSize(self):
        return None
    def close(self):
        pass


class Bbox:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

class PillowImageFile(ImageFile):
    def  __init__(self, path):
        super().__init__(path)
        self.im = Image.open(path)
        self.bbox = Bbox(0, 0, self.im.width, self.im.height)
        self.radius = max(self.im.width, self.im.height)
    def readImage(self, queue, radius : int, scale, cx : int, cy : int, brightness: float):
        queue.put((self.im, (0, 0, self.im.width, self.im.height), 1, self.radius, 1))
    def getBbox(self):
        return self.bbox
    def close(self):
        self.im.close()
        super().close()

class CziImageFile(ImageFile):
    def __init__(self, path):
        super().__init__(path)
        czi = CziFile(path)
        self.czi = czi
        self.bbox = czi.get_mosaic_bounding_box()
        sx = czi.meta.find('Metadata/Scaling/Items/Distance[@Id="X"]/Value')
        sy = czi.meta.find('Metadata/Scaling/Items/Distance[@Id="Y"]/Value')
        if sx == None or sy == None:
            raise Exception("No pixel size in metadata")
        xscale = float(sx.text)
        yscale = float(sy.text)
        if xscale != yscale:
            raise Exception("Cannot handle non-square pixels yet, sorry!")
        self.pixelSize = xscale
        focus_actions = czi.meta.findall(
            'Metadata/Information/TimelineTracks/TimelineTrack/TimelineElements/TimelineElement/EventInformation/FocusAction'
        )
        if focus_actions:
            def focus_action_success(e):
                r = e.find('Result')
                return r != None and r.text == 'Success'
            self.surface = [
                [ float(a.find(tag).text) for tag in ['X', 'Y', 'ResultPosition'] ]
                for a in focus_actions
                if focus_action_success(a)
            ]
        else:
            overall_z_element = (
                czi.meta.find('Metadata/Experiment/ExperimentBlocks/AcquisitionBlock/SubDimensionSetups/RegionsSetup/SampleHolder/TileRegions/TileRegion/Z')
                or czi.meta.find('Metadata/HardwareSetting/ParameterCollection[@Id="MTBFocus"]/Position')
            )
            self.surface = [float(overall_z_element.text)]

    def readImage(self, queue, radius : int, scale, cx : int, cy : int, brightness: float):
        """
        Enqueues an image read from the CZI file and other information:
        (np.ndarray, rect, scale, radius)
        where rect is (left, top, width, height) in image co-ordinates
        parameters:
            czi: CZI image file
            radius: how big we want the image to be, in source pixels
            scale: reciprocal of scaling required, so 2 gives a half-sized image
        """
        scale = max(1.0, scale)
        bbox = self.bbox
        cx = math.floor(max(cx, bbox.x + radius))
        cx = min(cx, bbox.x + bbox.w - radius)
        cy = math.floor(max(cy, bbox.y + radius))
        cy = min(cy, bbox.y + bbox.h - radius)
        left = max(cx - radius, bbox.x)
        top = max(cy - radius, bbox.y)
        right = min(cx + radius, bbox.x + bbox.w)
        bottom = min(cy + radius, bbox.y + bbox.h)
        width = right - left
        height = bottom - top
        rect = (left, top, width, height)
        data = self.czi.read_mosaic(rect, 1 / scale, C=0)
        rescaled = np.minimum(
            np.multiply(data[0], brightness/256.0),
            255.0
        )
        img = np.asarray(rescaled.astype(np.uint8))
        pil = Image.fromarray(img)
        queue.put((pil, rect, scale, radius, brightness))

    def surfaceZ(self, x, y):
        """
        Returns an approximate z position from the x, y positions
        provided (each in micrometers)
        """
        # For now, we'll just find the closest known surface point.
        # An alternative might be to find the closest known points
        # in the NW, NE, SE, SW quadrants (which will define a
        # quadrilateral that definitely includes the point), then
        # arbitrarily divide it by the line between the SW and NE
        # points and choose the triangle the point is actually in,
        # then interpolate between those three points.
        x0 = self.surface[0][0] - x
        y0 = self.surface[0][1] - y
        dist2 = x0 * x0 + y0 * y0
        best = self.surface[0][2]
        for i in range(1,len(self.surface)):
            p = self.surface[i]
            xi = p[0] - x
            yi = p[1] - y
            d2 = xi * xi + yi * yi
            if d2 < dist2:
                dist2 = d2
                best = p[2]
        return best

    def getBbox(self):
        return self.bbox

    def getPixelSize(self):
        return self.pixelSize


fileTypes = {
    '.czi': CziImageFile,
    '.jpg': PillowImageFile,
    '.jpeg': PillowImageFile,
    '.JPG': PillowImageFile
}


def loadImage(path):
    (base, ext) = os.path.splitext(path)
    if ext in fileTypes:
        return fileTypes[ext](path)
    raise Exception('Did not understand {0} file type'.format(ext))


class Screen:
    def __init__(self, canvas):
        self.panX = 0
        self.panY = 0
        self.scale = 1.0
        self.canvas = canvas

    def setScale(self, scale):
        self.scale = scale

    def getScale(self):
        return self.scale

    def setTranslation(self, x, y):
        self.panX = x
        self.panY = y

    def width(self):
        return self.canvas.winfo_width()

    def height(self):
        return self.canvas.winfo_height()

    def toWorld(self, x, y):
        x -= self.width() / 2
        y -= self.height() / 2
        return (x * self.scale + self.panX, y * self.scale + self.panY)

    def fromWorld(self, x, y):
        return (
            (x - self.panX) / self.scale + self.width() / 2,
            (y - self.panY) / self.scale + self.height() / 2
        )

    def dragStart(self, x, y):
        self.dragStartPoint = self.toWorld(x, y)
        self.panX0 = self.panX
        self.panY0 = self.panY

    def dragMove(self, x, y):
        self.panX = self.panX0
        self.panY = self.panY0
        w = self.toWorld(x, y)
        d = diff2D(w, self.dragStartPoint)
        self.panX -= d[0]
        self.panY -= d[1]

    def zoomChange(self, scale, x, y):
        # let p0 be the initial pan point and s0 the original zoom, with m the
        # clicked point relative to the centre of the canvas (in screen pixels).
        # Then we want p0 + m/s0 = p1 + m/s1
        # => p1 = p0 + (1/s0 - 1/s1)m
        mx = x - self.width() / 2
        my = y - self.height() / 2
        ds = self.scale - scale
        self.scale = scale
        self.panX += ds * mx
        self.panY += ds * my

class ZoomableImage:
    def __init__(self, cziPath, brightness=2.0, flip=False):
        self.czi = loadImage(pathlib.Path(cziPath))
        self.queue = queue.SimpleQueue()
        self.pil = None
        self.pilPosition = None
        self.pilRadius = 0
        # the brightness of the image we last received
        self.brightness = brightness
        self.desiredBrightness = brightness
        self.pin = None # pin position in world co-ordinates
        # scale factor: cached image pixels * scale -> world co-ordinates
        self.scale = 1.0
        # radians to rotate anticlockwise to transform into world co-ordinates
        self.angle = 0.0
        # centre of this image transformed into world co-ordinates
        self.tx = 0
        self.ty = 0
        self.flip = flip and -1 or 1
        self.waitingForRead = False

    def pixelSize(self):
        """
        Returns pixelSize in micrometers, or 1 if pixel size is not known.
        """
        s = self.czi.getPixelSize()
        if s:
            return s * 1e6
        return 1

    def setAngle(self, angle):
        self.angle = angle

    def setTranslation(self, x, y):
        self.tx = x
        self.ty = y

    def setBrightness(self, brightness):
        self.desiredBrightness = math.pow(10.0, float(brightness) / 10.0)

    def centre(self):
        """
        returns the centre of the whole image in image co-ordinates.
        """
        bbox = self.czi.getBbox()
        return (
            bbox.x + bbox.w // 2,
            bbox.y + bbox.h // 2
        )

    def toWorld(self, x, y):
        s = math.sin(self.angle)
        c = math.cos(self.angle)
        x *= self.flip
        return (
            c * x + s * y + self.tx,
            c * y - s * x + self.ty
        )

    def fromWorldLinear(self, x, y):
        """ Performs just the rotation and flip without the translation """
        s = math.sin(self.angle)
        c = math.cos(self.angle)
        return (
            self.flip * (c * x - s * y),
            c * y + s * x
        )

    def fromWorld(self, x, y):
        return self.fromWorldLinear(x - self.tx, y - self.ty)
    
    def getMicrometerMatrix(self):
        """
        Returns the mapping TO this image (in micrometers) FROM world
        co-ordinates (in micrometers).
        """
        c0 = self.fromWorldLinear(1, 0)
        c1 = self.fromWorldLinear(0, 1)
        c2 = self.fromWorld(0, 0)
        scale = self.pixelSize()
        return [
            [c0[0], c1[0], c2[0] * scale],
            [c0[1], c1[1], c2[1] * scale]
        ]

    def toScreen(self, screen: Screen, x, y):
        """
        convert from image co-ordinates to screen co-ordinates
        """
        w = self.toWorld(x,y)
        return screen.fromWorld(*w)

    def fromScreen(self, screen: Screen, x, y):
        """
        convert from screen co-ordinates to image co-ordinates
        """
        w = screen.toWorld(x, y)
        return self.fromWorld(*w)

    def update(self):
        """
        Read cache updates from the queue and apply
        """
        count = 0
        try:
            while True:
                block = count == 0 and self.pil == None
                (i, p, z, r, b) = self.queue.get(block=block)
                count += 1
        except queue.Empty:
            if count == 0:
                return False
        self.waitingForRead = False
        self.pil = i
        self.pilPosition = p
        self.scale = z # scaling of image compared to cache
        self.pilRadius = r
        self.brightness = b
        return True

    def updateCacheIfNecessary(self, requestQueue, screen: Screen):
        """
        Puts a new read request on the queue if we want a better
        cached image.
        """
        if self.waitingForRead:
            # We shouldn't put multiple reads on the queue
            return
        cw = screen.width()
        ch = screen.height()
        radius = max(1, math.floor(math.hypot(cw, ch) * screen.scale / 2))
        cacheRadius = radius * 3
        (x,y) = self.fromWorld(screen.panX, screen.panY)
        # clamp to within the bounding box of the image - radius
        # (we are assuming here that bbox.w and bbox.h are at least 2*cacheRadius, which
        # will be true for the images we care about)
        bbox = self.czi.getBbox()
        x = max(bbox.x + cacheRadius, min(bbox.x + bbox.w - cacheRadius, x))
        y = max(bbox.y + cacheRadius, min(bbox.y + bbox.h - cacheRadius, y))
        # if we are not within the old radius, request a new cache update
        magnification = self.scale / screen.scale
        if (self.pilPosition == None
                or magnification < 0.45
                or (1.9 < magnification and 1.0 < self.scale)
                or x - radius < self.pilPosition[0]
                or y - radius < self.pilPosition[1]
                or self.pilPosition[0] + self.pilPosition[2] < x + radius
                or self.pilPosition[1] + self.pilPosition[3] < y + radius
                or self.brightness != self.desiredBrightness):
            requestQueue.put(ImageReadRequest(
                self,
                cacheRadius,
                screen.scale,
                x,
                y,
                self.desiredBrightness
            ))
            self.waitingForRead = True

    def pinScreenPosition(self, screen: Screen):
        if self.pin == None:
            return None
        return screen.fromWorld(*self.pin)

    def setPinIfUnset(self, screen: Screen, x, y):
        if self.pin != None:
            return False
        self.pin = screen.toWorld(x, y)
        return True

    def unsetPin(self):
        self.pin = None

    def angleToPin(self, screen: Screen, x, y):
        """
        returns the angle the screen position (x,y) makes with the pin
        (relative to rightwards going anticlockwise). Returns None if
        there is no pin.
        """
        if not self.pin:
            return None
        s = screen.toWorld(x, y)
        return math.atan2(self.pin[1] - s[1], s[0] - self.pin[0])

    def dragStart(self, screen: Screen, x, y, draggingPin=False):
        self.dragStartPoint = screen.toWorld(x, y)
        self.tx0 = self.tx
        self.ty0 = self.ty
        self.angle0 = self.angle
        self.pinAngle = self.angleToPin(screen, x, y)
        if draggingPin and self.pin:
            self.pinDragStart = self.pin
        else:
            self.pinDragStart = None

    def dragMove(self, screen: Screen, x, y):
        w = screen.toWorld(x, y)
        d = diff2D(w, self.dragStartPoint)
        if self.pinDragStart:
            self.pin = add2D(self.pinDragStart, d)
            return
        angle = self.angleToPin(screen, x, y)
        if angle == None:
            self.tx = self.tx0 + d[0]
            self.ty = self.ty0 + d[1]
        else:
            # We pre-multiply because the pin position P is in world co-ordinates:
            # (R'|P-R'P)(R|T)
            # = (R'R|R'T + P - R'P)
            # = (R'R|R'(T - P) + P)
            angle -= self.pinAngle
            s = math.sin(angle)
            c = math.cos(angle)
            x = self.tx0 - self.pin[0]
            y = self.ty0 - self.pin[1]
            self.tx = c * x + s * y + self.pin[0]
            self.ty = c * y - s * x + self.pin[1]
            self.angle = self.angle0 + angle

    def getCropped(self, screen: Screen):
        (leftPil, topPil, widthPil, heightPil) = self.pilPosition
        px = leftPil + widthPil / 2
        py = topPil + heightPil / 2
        # Screen centre is at inv(Zs)(Pv - T - RFPs) in
        # flipped rotated cached image co-ordinates
        # which is the centre of the cached portion mapped to world
        # co-ordinates, subtracted from the screen position and scaled
        # to cached image co-ordinates
        (wx, wy) = self.toWorld(px, py)
        cx = (screen.panX - wx) / self.scale
        cy = (screen.panY - wy) / self.scale
        pil = self.pil
        # create rotated flipped cached portion
        if self.flip < 0:
            pil = pil.transpose(Image.FLIP_LEFT_RIGHT)
        pil = pil.rotate(self.angle * 180 / math.pi)
        # set  leftPil, topPil, widthPil and heightPil to adjust for the new
        # image's dimensions
        widthPil = pil.width * self.scale
        heightPil = pil.height * self.scale
        leftPil = math.floor(px - widthPil / 2 + 0.5)
        topPil = math.floor(py - heightPil / 2 + 0.5)
        # position on the image (flipped rotated cached co-ordinates)
        magnification = self.scale / screen.scale
        zwidth = screen.width() / magnification
        zheight = screen.height() / magnification
        zleft = math.floor(cx - zwidth / 2 + 0.5)
        ztop = math.floor(cy - zheight / 2 + 0.5)
        zwidth = math.floor(zwidth)
        zheight = math.floor(zheight)
        # position of cropped image on screen
        offsetX = 0
        offsetY = 0
        phw = math.floor(pil.width / 2 + 0.5)
        phh = math.floor(pil.height / 2 + 0.5)
        leftCrop = zleft + phw
        topCrop = ztop + phh
        rightCrop = zleft + zwidth + phw
        bottomCrop = ztop + zheight + phh
        if leftCrop < 0:
            offsetX = -leftCrop
            leftCrop = 0
        if topCrop < 0:
            offsetY = -topCrop
            topCrop = 0
        if pil.width < rightCrop:
            rightCrop = pil.width
        if pil.height < bottomCrop:
            bottomCrop = pil.height
        widthCrop = rightCrop - leftCrop
        heightCrop = bottomCrop - topCrop
        if widthCrop <= 0 or heightCrop <= 0:
            return None
        # width and height in screen pixels
        canvasX = math.floor(offsetX * magnification)
        canvasY = math.floor(offsetY * magnification)
        canvasW = math.ceil(widthCrop * magnification)
        canvasH = math.ceil(heightCrop * magnification)
        image = pil.resize(
            (canvasW, canvasH),
            resample=Image.NEAREST,
            box=(leftCrop, topCrop, rightCrop, bottomCrop)
        )
        if self.brightness != self.desiredBrightness:
            brightnessFactor = self.desiredBrightness / self.brightness
            image = ImageEnhance.Brightness(image).enhance(brightnessFactor)
        return {
            'image': image,
            'x': canvasX,
            'y': canvasY
        }

    def transformImage(self, screen: Screen):
        im = self.getCropped(screen)
        width = screen.width()
        height = screen.height()
        if (im != None
            and im['x'] == 0 and ['y'] == 0
            and im['image'].width == width and im['image'].height == height
        ):
            return im['image']
        bg = Image.new("RGB", (width, height), (0,0,0))
        if im != None:
            bg.paste(im['image'], (im['x'], im['y']))
        return bg

class ImageReadRequest:
    def __init__(self, image: ZoomableImage, radius: int, scale, cx: int, cy:int, brightness: float):
        """
        Prepare a request to read a portion of a CZI image to cache.

        Parameters:
            zim: The image to read
            radius: The number of pixels to read in each direction from the (cx,cy)
            scale: The reciprocal of the scale factor required (so 2 gives a half-sized image)
            (cx, cy): The centre of the image to be read in image co-ordinates
            brightness: How much to mulitply the image values by
        """
        self.queue = image.queue
        self.czi = image.czi
        self.radius = radius
        self.scale = scale
        self.cx = cx
        self.cy = cy
        self.brightness = brightness


def readImages(inputQueue):
    """
    Continuously read ImageReadRequests from the inputQueue, outputting
    the results onto the queues of the ZoomableImages passed in.
    """
    while True:
        i = inputQueue.get()
        i.czi.readImage(i.queue, i.radius, i.scale, i.cx, i.cy, i.brightness)


class ScaleBar:
    def __init__(self, canvas):
        self.bottom = 50
        self.right = 50
        self.height = 12
        self.canvas = canvas
        self.sections = []
        for i in range(0,10):
            col = '#FFFFFF' if i % 2 == 0 else '#808080'
            self.sections.append(canvas.create_rectangle(
                (0,0,1,1), fill=col, state='hidden', outline=''
            ))
        self.font = tkfont.Font(family='Sans', size=9)
        self.micrometers = '\xb5m'
        if self.font.measure('\xb5') == 0:
            self.micrometers = 'um'
        self.text = canvas.create_text(
            (0,0), fill='#000000', font=self.font, state='hidden'
        )
    def show(self, maxLength, maxMicrons):
        magnitude = math.floor(math.log10(maxMicrons))
        microns = math.pow(10.0, magnitude)
        ticks = 10
        if microns * 5.0 <= maxMicrons:
            microns *= 5.0
            ticks = 5
        elif microns * 2.0 <= maxMicrons:
            microns *= 2.0
            ticks = 2
        pixels = microns * maxLength / maxMicrons
        x = self.canvas.winfo_width() - self.right
        y = self.canvas.winfo_height() - self.bottom
        x0 = x - pixels
        y0 = y - self.height
        xprev = math.floor(x0 + 0.5)
        for t in range(0, ticks):
            xe = math.floor(x0 + pixels * (t+1) / ticks)
            self.canvas.coords(self.sections[t], (xprev, y0, xe, y))
            self.canvas.itemconfigure(self.sections[t], state='normal')
            xprev = xe
        for t in range(ticks, len(self.sections)):
            self.canvas.itemconfigure(self.sections[t], state='hidden')
        units = self.micrometers
        v = math.floor(microns)
        if microns < 1:
            v = math.floor(microns*1000)
            units = 'nm'
        elif 1000 <= microns:
            v = math.floor(microns / 1000)
            units = 'mm'
        self.canvas.coords(self.text, (x - pixels / 2, y - self.height / 2))
        self.canvas.itemconfigure(
            self.text, text='{0} {1}'.format(v, units), state='normal'
        )

class SaveDialog(tk.Toplevel):
    def __init__(self, parent, saveFn, quitFn):
        super().__init__(parent)
        self.saveFn = saveFn
        self.quitFn = quitFn
        self.title('Save changes?')
        text = tk.Label(self, text='Do you want to save your changes?')
        text.grid(row=0, column=0, columnspan=3)
        save = tk.Button(self, text='Save', command=self.save)
        save.grid(row=1, column=0)
        dont = tk.Button(self, text="Don't save", command=self.dont)
        dont.grid(row=1, column=1)
        cancel = tk.Button(self, text='Cancel', command=self.cancel)
        cancel.grid(row=1, column=2)
        self.grid()
    def destroy(self):
        super().destroy()
    def save(self, event=None):
        self.saveFn()
        self.quitFn()
        self.destroy()
    def dont(self, event=None):
        self.quitFn()
        self.destroy()
    def cancel(self, event=None):
        self.destroy()


class ManimalApplication(tk.Frame):
    def __init__(self, master, fixed=None, sliding=None, matrix_file=None, poi_file=None, fixed_brightness=2.0, flip='auto'):
        super().__init__(master)
        saveColumn = 0
        fixedBrightnessColumn = 1
        slidingBrightnessColumn = 3
        pinButtonColumn = 5
        leftClickColumn = 6
        rightClickColumn = 8
        closeColumn = 11
        columnCount = 12
        self.changed = False
        self.grid(row=0, column=0, sticky='nsew')
        self.saveButton = tk.Button(self, text='Save', command = self.save)
        self.saveButton.grid(column=saveColumn, row=1, sticky='s')
        master.bind('<Return>', lambda e: self.saveAndQuit())
        master.bind('s', lambda e: self.save())
        self.closeButton = tk.Button(self, text='Close', command = self.maybeQuit)
        self.closeButton.grid(column=closeColumn, row=1, sticky='s')
        master.bind('<Escape>', lambda e: self.maybeQuit())
        self.canvas = tk.Canvas(self)
        self.canvas.grid(column=0, columnspan=columnCount, row=0, sticky='nsew')
        self.screen = Screen(self.canvas)
        self.mode = tk.IntVar(value=0)
        self.mouseFunctionMove = 'move'
        self.mouseFunctionAddPoi = 'add POI'
        self.mouseFunctionAddRegPoint = 'add Reg. point'
        self.mouseFunctionSlide = 'slide'
        self.mouseDownFunction = {
            self.mouseFunctionMove: self.noop,
            self.mouseFunctionAddPoi: self.dragStartAddPoi,
            self.mouseFunctionAddRegPoint: self.dragStartAddRegPoint,
            self.mouseFunctionSlide: self.dragStartSlide
        }
        self.mouseDragFunction = {
            self.mouseFunctionMove: self.dragPan,
            self.mouseFunctionAddPoi: self.noop,
            self.mouseFunctionAddRegPoint: self.noop,
            self.mouseFunctionSlide: self.dragSlide
        }
        functions = [self.mouseFunctionMove]
        if poi_file != None:
            functions += [self.mouseFunctionAddPoi, self.mouseFunctionAddRegPoint]
        self.poi_file = poi_file
        self.fixed = None
        if fixed:
            self.fixed = ZoomableImage(
                pathlib.Path(fixed),
                brightness=fixed_brightness
            )
            centreFixed = self.fixed.centre()
            self.screen.setTranslation(centreFixed[0], centreFixed[1])
        self.matrix_file = matrix_file
        if flip == 'auto':
            flip = bool(fixed)
        elif flip == 'yes':
            flip = True
        else:
            flip = False
        xflip = -1 if flip else 1
        self.axlePin = None
        self.regPins = set([])
        self.poiPins = set([])
        self.podPins = set([])
        self.pinCoords = {}
        self.pinNames = {}
        self.draggingPin = None
        self.unknownPois = []
        self.pinButton = None
        self.sliding = None
        if sliding:
            functions.append(
                self.mouseFunctionSlide
            )
            self.pinButton = tk.Button(
                self,
                text='Pin',
                relief='raised',
                command=self.toggleAxlePin
            )
            self.pinButton.grid(column=pinButtonColumn, row=1, sticky='s')
            self.grid_columnconfigure(pinButtonColumn, weight=10)
            master.bind('p', lambda e: self.toggleAxlePin())
            self.sliding = ZoomableImage(
                pathlib.Path(sliding), flip=flip
            )
            centreSliding = self.sliding.centre()
            if fixed:
                self.sliding.setTranslation(
                    centreFixed[0] - xflip * centreSliding[0],
                    centreFixed[1] - centreSliding[1]
                )
            else:
                self.screen.setTranslation(centreSliding[0], centreSliding[1])
            text = 'brightness (sliding)' if fixed else 'brightness'
            label = tk.Label(self, text=text)
            label.grid(column=slidingBrightnessColumn, row=1, sticky='s')
            self.slidingBrightnessSlider = tk.Scale(
                self,
                length=100, from_=-6, to=24, resolution=1,
                orient='horizontal',
                showvalue=False,
                command=self.setSlidingBrightness
            )
            self.slidingBrightnessSlider.set(10.0 * math.log10(fixed_brightness))
            self.slidingBrightnessSlider.grid(column=slidingBrightnessColumn + 1, row=1, sticky='s')
        else:
            self.grid_columnconfigure(pinButtonColumn, weight=0)
        self.grid_columnconfigure(slidingBrightnessColumn, weight=0)
        self.grid_columnconfigure(slidingBrightnessColumn + 1, weight=0)
        if fixed:
            text = 'brightness (fixed)' if sliding else 'brightness'
            label = tk.Label(self, text=text)
            label.grid(column=fixedBrightnessColumn, row=1, sticky='s')
            self.fixedBrightnessSlider = tk.Scale(
                self,
                length=100, from_=-6, to=24, resolution=1,
                orient='horizontal',
                showvalue=False,
                command=self.setFixedBrightness
            )
            self.fixedBrightnessSlider.set(10.0 * math.log10(fixed_brightness))
            self.fixedBrightnessSlider.grid(column=fixedBrightnessColumn + 1, row=1, sticky='s')
        self.grid_columnconfigure(fixedBrightnessColumn, weight=0)
        self.grid_columnconfigure(fixedBrightnessColumn + 1, weight=0)
        tk.Label(self, text='left mouse button:').grid(
            column=leftClickColumn, row=1, sticky='se'
        )
        self.leftClickFunctionSelector = ttk.Combobox(
            self, values=functions, state='readonly'
        )
        self.leftClickFunctionSelector.grid(column=leftClickColumn+1, row=1, sticky='sw')
        self.leftClickFunctionSelector.set(self.mouseFunctionMove)
        self.grid_columnconfigure(leftClickColumn, weight=0)
        self.grid_columnconfigure(leftClickColumn+1, weight=10)
        tk.Label(self, text='right mouse button:').grid(
            column=rightClickColumn, row=1, sticky='se'
        )
        self.rightClickFunctionSelector = ttk.Combobox(
            self, values=functions, state='readonly'
        )
        self.rightClickFunctionSelector.grid(column=rightClickColumn+1, row=1, sticky='sw')
        self.rightClickFunctionSelector.set(self.mouseFunctionMove)
        self.grid_columnconfigure(rightClickColumn, weight=0)
        self.grid_columnconfigure(rightClickColumn+1, weight=10)
        # zoom 100/percent
        self.zoomLevels = [20.0, 10.0, 4.0, 2.0, 1.0, 0.5, 0.25]
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.canvas.bind("<Button-1>", lambda e: self.dragStart(
            e.x, e.y, self.leftClickFunctionSelector.get()
        ))
        self.canvas.bind("<ButtonRelease-1>", lambda e: self.dragEnd(
            e.x, e.y, self.leftClickFunctionSelector.get()
        ))
        self.canvas.bind("<B1-Motion>", lambda e: self.dragMove(
            e.x, e.y, self.leftClickFunctionSelector.get()
        ))
        self.canvas.bind("<Button-3>", lambda e: self.dragStart(
            e.x, e.y, self.rightClickFunctionSelector.get()
        ))
        self.canvas.bind("<ButtonRelease-3>", lambda e: self.dragEnd(
            e.x, e.y, self.rightClickFunctionSelector.get()
        ))
        self.canvas.bind("<B3-Motion>", lambda e: self.dragMove(
            e.x, e.y, self.rightClickFunctionSelector.get()
        ))
        self.canvas.bind("<MouseWheel>", lambda e: self.zoomChange(e.delta, e.x, e.y))
        self.canvas.bind("<Button-4>", lambda e: self.zoomChange(-120, e.x, e.y))
        self.canvas.bind("<Button-5>", lambda e: self.zoomChange(120, e.x, e.y))
        self.canvas.bind("<Motion>", self.motion)
        self.requestQueue = queue.SimpleQueue()
        self.readThread = threading.Thread(
            target=readImages, args=(self.requestQueue,)
        )
        self.readThread.daemon = True
        self.readThread.start()
        self.createCanvas()
        self.scaleBar = ScaleBar(self.canvas)
        self.update_idletasks()
        self.updateCache()
        self.updateCanvas()
        self.loadPoisIfAvailable(self.poi_file)
        self.axlePin = self.createPin('green', outline='white', hidden=True)
        self.tick()

    def updateCache(self):
        if self.fixed:
            self.fixed.updateCacheIfNecessary(self.requestQueue, self.screen)
            self.fixed.update()
        if self.sliding:
            self.sliding.updateCacheIfNecessary(self.requestQueue, self.screen)
            self.sliding.update()

    def tick(self):
        if self.fixed:
            self.fixed.update()
        if self.sliding:
            self.sliding.update()
        self.updateCanvas()
        self.after(1000, self.tick)

    def setFixedBrightness(self, brightness):
        self.fixed.setBrightness(brightness)
        self.updateCanvas()

    def setSlidingBrightness(self, brightness):
        self.sliding.setBrightness(brightness)
        self.updateCanvas()

    def removeAxlePin(self):
        if self.pinButton == None:
            return
        c = self.pinButton.config
        c(relief='raised')
        self.canvas.itemconfigure(self.axlePin, state='hidden')
        self.sliding.unsetPin()

    def toggleAxlePin(self):
        if self.pinButton == None:
            return
        c = self.pinButton.config
        if c('relief')[-1] == 'raised':
            c(relief='sunken')
            self.canvas.itemconfigure(self.axlePin, state='normal')
        else:
            self.removeAxlePin()

    def isAxlePinIn(self):
        return self.pinButton and self.pinButton.config('relief')[-1] == 'sunken'

    def motion(self, e):
        m = self.mode.get()
        if self.sliding and self.sliding.pin == None and self.isAxlePinIn():
            self.updateCanvas(pin=(e.x, e.y))
        elif m != 0:
            self.updateCanvas()

    def savePois(self, **kwargs):
        print('type,x,y,z,name', **kwargs)
        for (t,x,y,z,name) in self.unknownPois:
            print('{0},{1},{2},{3},{4}'.format(t,x,y,z,name), **kwargs)
        image = self.fixed or self.sliding
        scale = image.pixelSize()
        for (k, ids) in [ ('r', self.regPins), ('i', self.podPins), ('i', self.poiPins) ]:
            for id in ids:
                (x,y) = self.pinCoords[id]
                sx = x * scale
                sy = y * scale
                z = image.czi.surfaceZ(sx, sy)
                name = id in self.pinNames and self.pinNames[id] or ''
                print("{0},{1},{2},{3},{4}".format(k, sx, sy, z, name), **kwargs)

    def saveMatrix(self, **kwargs):
        print('x,y,t', **kwargs)
        for row in self.sliding.getMicrometerMatrix():
            print(','.join(map(str, row)), **kwargs)

    def save(self):
        if self.poi_file:
            with open(self.poi_file, 'w') as fh:
                self.savePois(file=fh)
        if self.sliding:
            if self.matrix_file:
                with open(self.matrix_file, 'w') as fh:
                    self.saveMatrix(file=fh)
            else:
                self.saveMatrix()
        self.changed = False

    def saveAndQuit(self):
        self.save()
        super().quit()

    def maybeQuit(self):
        if self.changed:
            self.wait_window(SaveDialog(self, self.save, self.quit))
        self.quit()

    def pixelSize(self):
        pixelSize = None
        if self.fixed:
            pixelSize = self.fixed.pixelSize()
        if not pixelSize and self.sliding:
            pixelSize = self.sliding.pixelSize()
        return pixelSize

    def loadPois(self, fh):
        csv = CsvLoader(fh)
        if not csv.headersAre('type,x,y,z,name'):
            raise Exception('Could not understand csv file')
        self.unknownPois = []
        scale = self.pixelSize()
        for (t,x,y,z,name) in csv.generateRows():
            pin = None
            if t == 'i':
                if len(name) == 0:
                    pin = self.createPoiPin()
                else:
                    pin = self.createPodPin()
                    self.pinNames[pin] = name
            elif t == 'r':
                pin = self.createRegPin()
            else:
                self.unknownPois.append((t,x,y,z,name))
            if pin:
                self.pinCoords[pin] = (float(x) / scale,float(y) / scale)

    def loadPoisIfAvailable(self, fn):
        if fn and os.path.exists(fn):
            with open(fn, 'r', encoding='utf-8-sig') as fh:
                self.loadPois(fh)
                return True
        return False

    def isAxlePin(self, x, y):
        ids = self.canvas.find_overlapping(x, y, x+1, y+1)
        return self.axlePin in ids

    def pinAt(self, x, y, pins):
        ids = set(self.canvas.find_overlapping(x, y, x+1, y+1))
        ps = ids & pins
        if self.draggingPin in ps:
            ps.remove(self.draggingPin)
        if not ps:
            return None
        return ps.pop()

    def regPinAt(self, x, y):
        return self.pinAt(x, y, self.regPins)

    def poiPinAt(self, x, y):
        return self.pinAt(x, y, self.poiPins)

    def deleteDraggingPin(self):
        p = self.draggingPin
        if p == None:
            return
        self.draggingPin = None
        if p in self.regPins:
            self.regPins.remove(p)
            self.changed = True
        elif p in self.poiPins:
            self.poiPins.remove(p)
            self.changed = True
        elif p == self.axlePin:
            self.removeAxlePin()

    def zoomChange(self, delta, x, y):
        scaleIndex = findNoGreaterThan(self.screen.scale, self.zoomLevels)
        if delta < 0:
            if scaleIndex == 0:
                return
            self.screen.zoomChange(self.zoomLevels[scaleIndex - 1], x, y)
        else:
            if len(self.zoomLevels) <= scaleIndex + 1:
                return
            self.screen.zoomChange(self.zoomLevels[scaleIndex + 1], x, y)
        self.updateCanvas()

    def noop(self, x, y):
        pass

    def dragStartAddPoi(self, x, y):
        p = self.createPoiPin()
        self.pinCoords[p] = self.screen.toWorld(x, y)
        self.updateCanvas()
        self.changed = True

    def dragStartAddRegPoint(self, x, y):
        p = self.createRegPin()
        self.pinCoords[p] = self.screen.toWorld(x, y)
        self.updateCanvas()
        self.changed = True

    def dragStartSlide(self, x, y):
        self.sliding.dragStart(self.screen, x, y)

    def dragSlide(self, x, y):
        self.sliding.dragMove(self.screen, x, y)
        self.updateCanvas()
        self.changed = True

    def dragPan(self, x, y):
        self.screen.dragMove(x, y)
        self.updateCanvas()

    def dragStart(self, x, y, funcName):
        self.screen.dragStart(x, y)
        p = self.regPinAt(x, y)
        if not p:
            p = self.poiPinAt(x, y)
        if not p and self.isAxlePin(x, y):
            p = self.axlePin
            self.sliding.dragStart(self.screen, x, y, draggingPin=True)
        if p:
            # clicked a pin; make it the dragging pin
            self.deleteDraggingPin()
            self.draggingPin = p
        elif self.draggingPin == None:
            # Not clicked a pin, so do whatever is selected for this button
            return self.mouseDownFunction[funcName](x, y)
        if not self.draggingPin:
            return
        ps = self.canvas.coords(self.draggingPin)
        px = ps[0]
        py = ps[1]
        for i in range(2, len(ps) // 2):
            py0 = ps[i*2+1]
            if py < py0:
                px = ps[i*2]
                py = py0
        self.draggingPinOffsetX = px - x
        self.draggingPinOffsetY = py - y
        self.updateCanvas()

    def dragEnd(self, x, y, funcName):
        if self.draggingPin:
            if x < 0 or y < 0 or self.screen.width() < x or self.screen.height() < y:
                self.deleteDraggingPin()
            elif self.draggingPin == self.axlePin:
                self.sliding.setPinIfUnset(self.screen, x, y)
            else:
                self.pinCoords[self.draggingPin] = self.screen.toWorld(
                    self.draggingPinOffsetX + x,
                    self.draggingPinOffsetY + y
                )
                self.changed = True
            self.draggingPin = None
            self.updateCanvas()
    
    def dragMove(self, x, y, funcName):
        if self.draggingPin:
            if self.draggingPin == self.axlePin:
                self.sliding.dragMove(self.screen, x, y)
            else:
                self.pinCoords[self.draggingPin] = self.screen.toWorld(
                    self.draggingPinOffsetX + x,
                    self.draggingPinOffsetY + y
                )
                self.changed = True
            self.updateCanvas()
        else:
            self.mouseDragFunction[funcName](x, y)
    
    def createCanvas(self):
        self.canvasImage = self.canvas.create_image(0, 0, anchor="nw")

    def createPin(self, color, outline='black', hidden=False):
        state = 'hidden' if hidden else 'normal'
        return self.canvas.create_polygon((0,0,0,1,1,1),
            fill=color, joinstyle='miter', outline=outline, state=state)

    def createRegPin(self):
        """
        Create registration point pin
        """
        pin = self.createPin('yellow')
        self.regPins.add(pin)
        return pin

    def createPoiPin(self):
        """
        Create point of interest pin
        """
        pin = self.createPin('white')
        self.poiPins.add(pin)
        return pin

    def createPodPin(self):
        """
        Create point of interest (with z-stack already captured) pin
        """
        pin = self.createPin('grey24', outline='grey64')
        self.podPins.add(pin)
        return pin

    def showScaleBar(self):
        pixelSize = self.pixelSize()
        if not pixelSize:
            return
        pixelSize *= self.screen.getScale()
        scaleBarWidth = 300
        maxMicrons = pixelSize * scaleBarWidth
        self.scaleBar.show(scaleBarWidth, maxMicrons)

    def updateCanvas(self, pin=None):
        self.updateCache()
        image = None
        if self.fixed:
            image = self.fixed.transformImage(self.screen)
        if self.sliding:
            slider = self.sliding.transformImage(self.screen)
            if image:
                image = Image.blend(image, slider, 0.5)
            else:
                image = slider
        for pinId in self.poiPins | self.regPins | self.podPins:
            (x,y) = self.screen.fromWorld(*self.pinCoords[pinId])
            if pinId == self.draggingPin:
                self.canvas.coords(pinId, (x,y,x+5,y-15,x+15,y-5))
            else:
                self.canvas.coords(pinId, (x,y,x+5,y-15,x-5,y-15))
        if pin != None:
            (x,y) = pin
            self.canvas.coords(self.axlePin, (x,y,x+5,y-15,x+15,y-5))
        elif self.sliding:
            p = self.sliding.pinScreenPosition(self.screen)
            if p:
                (x,y) = p
                if self.axlePin == self.draggingPin:
                    self.canvas.coords(self.axlePin, (x,y,x+5,y-15,x+15,y-5))
                else:
                    self.canvas.coords(self.axlePin, (x,y,x-5,y-15,x+5,y-15))
        self.image = ImageTk.PhotoImage(image)
        self.showScaleBar()
        self.canvas.itemconfigure(self.canvasImage, image=self.image)

parser = argparse.ArgumentParser(
    description= 'MANual IMage ALigner: '
    + 'a GUI tool for manually aligning images with each other, '
    + 'finding points of interest, '
    + 'or aligning an image with existing points of interest.'
)
parser.add_argument(
    '-p',
    '--poi',
    help='input/output CSV file for points of interest',
    required=False,
    dest='poi_file',
    metavar='POI-CSV'
)
parser.add_argument(
    '-a',
    '--align',
    help='sliding image, if alignment functionality is wanted (.czi file)',
    dest='sliding',
    required=False,
)
parser.add_argument(
    '-m',
    '--matrix',
    help='matrix output CSV file (if not standard out), only if --align is used',
    required=False,
    dest='matrix_file',
    metavar='MATRIX-CSV'
)
parser.add_argument(
    '-f',
    '--fixed',
    help='fixed image (.czi file) if desired',
    required=False,
    dest='fixed'
)
parser.add_argument(
    '-b',
    '--brightness',
    help='initial brightness of the fixed image',
    dest='fixed_brightness',
    required=False,
    type=float,
    metavar='BRIGHTNESS'
)
parser.add_argument(
    '--flip',
    help='whether to flip the sliding image. Default (auto) means yes if --fixed is used, no otherwise',
    choices=['no', 'yes', 'auto'],
    default='auto'
)
options = parser.parse_args()

kw = {
}
for attr in ['poi_file', 'sliding', 'matrix_file', 'fixed_brightness', 'fixed', 'flip']:
    v = getattr(options, attr, None)
    if v != None:
        kw[attr] = v

if 'fixed' not in kw and 'sliding' not in kw:
    parser.error('Either fixed or sliding (or both) images must be specified')

if 'poi_file' not in kw and 'sliding' not in kw:
    parser.error('Either a poi file or a sliding image (or both) must be specified')

root = tk.Tk()
root.grid_columnconfigure(0, weight=1)
root.grid_rowconfigure(0, weight=1)
app = ManimalApplication(
    root, **kw
)
app.master.title("Manimal")
app.mainloop()
