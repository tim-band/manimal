#! /usr/bin/env python3
from aicspylibczi import CziFile
import argparse
import glob
import itertools
import math
import numpy as np
import os
import pathlib
from PIL import Image, ImageEnhance, ImageTk
import queue
import re
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

    def headersStartWith(self, cshs):
        """
        Returns True iff cshs (a comma-separated list of headers) matches
        the actual headers in those starting positions (more unspecified
        headers than in cshs is OK, but fewer is not)
        """
        if not self.headers:
            return False
        hs = cshs.split(',')
        if len(self.headers) < len(hs):
            return False
        return hs == self.headers[0:len(hs)]

    def rowCount(self):
        return len(self.rows)

    def generateRows(self):
        for r in self.rows:
            yield [c.strip('"') for c in r.split(',')]

    def getHeaders(self):
        return self.headers


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

class SurfaceEstimator:
    # Possibilities for orders of x and y to use
    # for estimation. So, for each `o` in `orders`
    # the independent variables to the linear
    # regression are:
    # 1, y, y^2 ... y^o[0]
    # x, xy, xy^2 ... xy^o[1]
    # x^2, x^2y, x^2y^2 ... x^2y^o[2]
    # ...
    # x^(len(o)-1) ... x^(len(o)-1)y^o[len(o) - 1]
    # We are using the ones we assume Zeiss uses
    # in its tile experiment support points
    # functionality.
    orders = [
        [5, 5, 5, 4, 3],  # biquartic but orders < 7 only
        [4, 4, 4, 3],  # bicubic but orders < 6 only
        [3, 3, 3],  # biquadratic
        [2, 2],  # bilinear
        [1],  # horizontal
    ]
    def choose_order(self, point_count):
        for o in self.orders:
            if sum(o) <= point_count:
                return o
        raise Exception('Internal Error: no surface to estimate')
    def generate_row(self, xy):
        x = xy[0]
        y = xy[1]
        xn = 1
        row = []
        for y_order in self.order:
            xnyn = xn
            for n in range(y_order):
                row.append(xnyn)
                xnyn *= y
            xn *= x
        return row
    def __init__(self, coords):
        point_count = len(coords)
        self.order = self.choose_order(point_count)
        ins = np.array(list(map(self.generate_row, coords)))
        outs = np.array(list(map(lambda xyz: xyz[2], coords)))
        m1 = np.linalg.inv(ins.T @ ins)
        self.coefficients = m1 @ ins.T @ outs
    def getZ(self, x, y):
        vs = np.array(self.generate_row([x, y]))
        return np.dot(self.coefficients, vs)

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
        surface_points = czi.meta.findall(
            'Metadata/Experiment/ExperimentBlocks/'
            + 'AcquisitionBlock[@IsActivated="true"]/SubDimensionSetups/'
            + 'RegionsSetup[@IsActivated="true"]/SampleHolder/'
            + 'TileRegions/TileRegion/SupportPoints/'
            + 'SupportPoint'
        )
        focus_actions = czi.meta.findall(
            'Metadata/Information/TimelineTracks/TimelineTrack/TimelineElements/TimelineElement/EventInformation/FocusAction'
        )
        surface = []
        if surface_points:
            surface += [
                [ float(a.find(tag).text) for tag in ['X', 'Y', 'Z'] ]
                for a in surface_points
            ]
        if focus_actions:
            def focus_action_success(e):
                r = e.find('Result')
                return r != None and r.text == 'Success'
            surface += [
                [ float(a.find(tag).text) for tag in ['X', 'Y', 'ResultPosition'] ]
                for a in focus_actions
                if focus_action_success(a)
            ]
        if len(surface) == 0:
            overall_z_element = (
                czi.meta.find('Metadata/Experiment/ExperimentBlocks/AcquisitionBlock/SubDimensionSetups/RegionsSetup/SampleHolder/TileRegions/TileRegion/Z')
                or czi.meta.find('Metadata/HardwareSetting/ParameterCollection[@Id="MTBFocus"]/Position')
            )
            surface = [[0, 0, float(overall_z_element.text)]]
        self.surface = SurfaceEstimator(surface)
        self.imageBits = int(czi.meta.find('Metadata/Information/Image/ComponentBitCount').text)

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
        pil_type = np.uint8
        data = self.czi.read_mosaic(region=rect, scale_factor=1/scale, C=0)
        pil_bits = 8 * pil_type(0).nbytes
        rescaled = np.floor(np.minimum(
            np.multiply(data[0], brightness * 2**(pil_bits - self.imageBits)),
            2**pil_bits - 1
        ))
        img = np.asarray(rescaled.astype(pil_type))
        pil = Image.fromarray(img)
        queue.put((pil, rect, scale, radius, brightness))

    def surfaceZ(self, x, y):
        """
        Returns an approximate z position from the x, y positions
        provided (each in micrometers)
        """
        return self.surface.getZ(x, y)

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
        self.halfMagWidth = None
        self.halfMagHeight = None

    def setMagnifiedScreenSize(self, width, height):
        """
        Sets the size of the dotted box that will appear when
        dragging POI pins around, in world pixels. This can be
        used to show how big the magnified images will be.
        """
        self.halfMagWidth = width / 2
        self.halfMagHeight = height / 2

    def setScale(self, scale):
        """
        Sets `scale` as the world unit size of a pixel.
        The "world" scale is usually pixels on the fixed image.
        """
        self.scale = scale

    def getScale(self):
        return self.scale

    def setTranslation(self, x, y):
        """
        Sets (x,y) (in world co-ordinates) to be the
        centre of the screen
        """
        self.panX = x
        self.panY = y

    def width(self):
        """ Returns the screen width in screen pixels """
        return self.canvas.winfo_width()

    def height(self):
        """ Returns the screen height in screen pixels """
        return self.canvas.winfo_height()

    def widthWorld(self):
        return self.width() * self.scale

    def heightWorld(self):
        return self.height() * self.scale

    def centreWorld(self):
        """
        Returns the real-world co-ordinates this screen is centred on
        """
        return (self.panX, self.panY)

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

    def setMoveTarget(self, dx, dy):
        """
        Sets the number of screenfulls we want to move by. The actual
        movement happens when calling moveTowardsTarget.
        """
        dx = int(dx * self.width() * self.scale + 0.5)
        dy = int(dy * self.height() * self.scale + 0.5)
        self.panDX = dx
        self.panDY = dy
        self.panStartX = self.panX
        self.panStartY = self.panY

    def moveTowardsTarget(self, interpolation):
        self.panX = self.panStartX + self.panDX * interpolation
        self.panY = self.panStartY + self.panDY * interpolation

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

    def createPin(self, x=0, y=0, tag=None):
        id = self.canvas.create_polygon(
            (0,0), joinstyle='miter',
            state='normal', tags=[tag] if tag else []
        )
        self.setPinShape(id, x, y, tag)
        self.setPinColours(id, tag)
        return id

    def deletePin(self, id):
        if id is not None:
            self.canvas.delete(id)

    def hidePin(self, id):
        if id is not None:
            self.canvas.itemconfigure(id, state='hidden')

    def showPin(self, id):
        if id is not None:
            self.canvas.itemconfigure(id, state='normal')

    def setPinColours(self, id, tag):
        if tag == 'reg':
            self.canvas.itemconfigure(id, fill='yellow', outline='black')
        elif tag == 'pod':
            self.canvas.itemconfigure(id, fill='grey24', outline='grey64')
        elif tag == 'axle':
            self.canvas.itemconfigure(id, fill='red', outline='white')
        else:
            self.canvas.itemconfigure(id, fill='white', outline='black')

    def movePinTo(self, id, x, y, dragging):
        tags = self.canvas.gettags(id)
        tag = tags[0] if len(tags) != 0 else None
        self.setPinShape(id, x, y, tag, dragging)

    def setPinShape(self, id, x, y, tag, dragging=False):
        (x, y) = self.fromWorld(x, y)
        if dragging:
            if tag == 'poi':
                if self.halfMagWidth is not None:
                    w = self.halfMagWidth / self.scale
                    h = self.halfMagHeight / self.scale
                    self.canvas.itemconfigure(id, dash='.', fill='', outline='white')
                    self.canvas.coords(id, (
                        x-w, y-h, x+w, y-h, x+w, y+h, x-w, y+h
                    ))
                    return
            self.canvas.coords(id, (x, y, x+5, y-15, x+15, y-5))
        else:
            if tag == 'poi':
                self.canvas.itemconfigure(id, dash='', fill='white', outline='black')
            self.canvas.coords(id, (x, y, x+5, y-15, x-5, y-15))

    def overlappingIds(self, x, y):
        """
        Returns the IDs of the screen objects at screen co-ordinates (x,y)
        """
        return self.canvas.find_overlapping(x, y, x+1, y+1)

    def screenId(self):
        return self.canvas.winfo_id()


class OverviewScreen(Screen):
    def __init__(self, **kwargs):
        self.realX = None
        self.canvasX = None
        self.centreX = None
        super().__init__(**kwargs)
    def setPinShape(self, id, x, y, tag, dragging=False):
        (x, y) = self.fromWorld(x, y)
        if tag == 'reg':
            self.canvas.coords(id, (x-1, y-1, x+2, y-1, x+2, y+2, x-1, y+2))
        else:
            self.canvas.coords(id, (x, y, x+1, y, x, y-1))
    def setPinColours(self, id, tag):
        if tag == 'reg':
            self.canvas.itemconfigure(id, outline='yellow')
        elif tag == 'pod':
            self.canvas.itemconfigure(id, outline='grey24')
        elif tag == 'axle':
            self.canvas.itemconfigure(id, outline='red')
        else:
            self.canvas.itemconfigure(id, outline='white')

class ZoomableImage:
    def __init__(self, cziPath, brightness=2.0, flip=False):
        self.load(cziPath)
        self.queue = queue.SimpleQueue()
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

    def load(self, cziPath):
        self.czi = loadImage(pathlib.Path(cziPath))
        self.pil = None
        self.pilPosition = None
        self.pilRadius = 0

    def pixelSize(self):
        """
        Returns pixelSize in micrometers, or 1 if pixel size is not known.
        """
        s = self.czi.getPixelSize()
        if s:
            return s * 1e6
        return 1

    def xFlip(self):
        """
        Returns -1 if this image is flipped, 1 otherwise.
        """
        return self.flip

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

    def left(self):
        return self.czi.getBbox().x

    def top(self):
        return self.czi.getBbox().y

    def width(self):
        return self.czi.getBbox().w

    def height(self):
        return self.czi.getBbox().h

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

    def setMicrometerMatrix(self, x0, y0, tx, x1, y1, ty):
        """
        Sets the mapping to this image (in micrometers) from world
        co-ordinates.
        """
        # We need to decompose this into Translate + Rotate * Flip * ImageVector
        # But this is the inverse of that!
        # Flip(s) = I if s = 1 or (-1, 0 \ 0, 1) if s = -1
        # Rotate(a) = (cos(a), sin(a) \ -sin(a), cos(a))
        # inv(Flip(s)*Rotate(a)) = inv(Rotate(a))*inv(Flip(s)) = Rotate(-a)*Flip(s)
        # so we have: v = t + FRw <=> w = inv(FR)(v-t) = inv(R)F(v-t) = -inv(R)Ft + inv(R)Fv
        # So first we determine which F we have, and work out R
        cross = x0 * y1 - x1 * y0
        if cross < 0:
            self.flip = -1
            x0 = -x0
            y0 = -y0
            tx = -tx
        else:
            self.flip = 1
        self.angle = -math.atan2(y0, x0)
        scale = self.pixelSize()
        self.tx = (-x0 * tx + y0 * ty) / scale
        self.ty = (-x0 * ty - y0 * tx) / scale

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
            pil = pil.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
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
            resample=Image.Resampling.NEAREST,
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

class ConfirmationDialog(tk.Toplevel):
    def __init__(self, parent, title, label, *args):
        """
        Confirmation dialog with parent, title, description and
        the remaining arguments are dicts describing the buttons
        the dialog needs, which have the following keys:
        'fn' (optional) is a function to be called when the button
        is pressed, 'text' is the text the button should have, and
        'keys' (optional) is a list of key bindings the button
        should have. Each button dismisses the dialog, so if 'fn'
        is not provided then the corresponding button is just a
        cancel button.
        """
        super().__init__(parent)
        self.title(title)
        text = tk.Label(self, text=label)
        text.grid(row=0, column=0, columnspan=3)
        column = 0
        def getCommand(f):
            def com(e=None):
                if f:
                    f()
                self.destroy()
            return com
        for action in args:
            fn = action.get('fn')
            btn = tk.Button(
                self,
                text=action['text'],
                command=getCommand(fn)
            )
            btn.grid(row=1, column=column)
            column += 1
            if 'keys' in action:
                for k in action['keys']:
                    self.bind(k, fn)
        self.grid()
        self.focus_set()
    def destroy(self):
        super().destroy()

class SaveDialog(ConfirmationDialog):
    def __init__(self, parent, saveFn, quitFn):
        def saveAndQuit():
            saveFn()
            quitFn()
        super().__init__(
            parent,
            'Save changes?',
            'Do you want to save your changes?',
            { 'text': 'Save', 'fn': saveAndQuit, 'keys': ['<Return>', 's'] },
            { 'text': "Don't save", 'fn': quitFn, 'keys': ['d']},
            { 'text': 'Cancel', 'keys': ['<Escape>']}
        )

class ResetDialog(ConfirmationDialog):
    def __init__(self, parent, resetFn):
        super().__init__(
            parent,
            'Reset alignment?',
            'Do you want to reset the alignment to straight and centre-aligned?',
            { 'text': 'Reset', 'fn': resetFn, 'keys': ['<Return>'] },
            { 'text': 'Cancel', 'keys': ['<Escape>']}
        )

class Pin:
    CHARACTER = 'i'
    TAG = 'pin'
    def __init__(self, x=0, y=0, z=None, hidden=False, extra_fields=[]):
        """
        (x, y) is the position in pixels.
        z is the original z text, or None.
        """
        self.x = x
        self.y = y
        self.z = z
        self.state = 'hidden' if hidden else 'normal'
        self.ids = {}
        self.extra_fields = extra_fields
    def setPosition(self, x, y):
        self.x = x
        self.y = y
        self.z = None
        self.extra_fields = []
    def screenCoords(self, screen):
        return screen.fromWorld(self.x, self.y)
    def getId(self, screen):
        screenId = screen.screenId()
        if (screenId not in self.ids):
            return None
        return self.ids[screenId]
    def render(self, screen, isDragging=False):
        screenId = screen.screenId()
        if (screenId not in self.ids):
            self.ids[screenId] = screen.createPin(
                self.x, self.y, self.TAG
            )
            return
        screen.movePinTo(
            id=self.ids[screenId],
            x=self.x, y=self.y,
            dragging=isDragging
        )
    def getName(self):
        return ""
    def getExtras(self):
        return self.extra_fields
    def join(self, *args):
        return ','.join(map(str, args))
    def renderCsvLine(self, image, **kwargs):
        scale = image.pixelSize()
        sx = self.x * scale
        sy = self.y * scale
        z = self.z
        if type(z) is str:
            z = z.strip()
        if z is None or z == '':
            z = image.czi.surfaceZ(sx, sy)
        header_string = self.join(
            self.CHARACTER,
            sx, sy, z,
            self.getName(),
            *self.getExtras()
        )
        print(header_string, **kwargs)
    def hide(self, *screens):
        for screen in screens:
            screen.hidePin(self.getId(screen))
    def show(self, *screens):
        for screen in screens:
            screen.showPin(self.getId(screen))
    def removeFrom(self, *screens):
        for screen in screens:
            screen.deletePin(self.getId(screen))

class PoiPin(Pin):
    """ Point of interest still not captured as a Z-stack """
    TAG = 'poi'
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class NamedPin(Pin):
    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)
        self.name = name
    def getName(self):
        return self.name

class PodPin(NamedPin):
    """ Point of interest that has been captured """
    TAG = 'pod'

class RegPin(NamedPin):
    """ Registration point """
    TAG = 'reg'
    CHARACTER = 'r'

class AxlePin(Pin):
    TAG = 'axle'
    """ Rotation axle """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def renderCsvLine(self, image, **kwargs):
        pass

def getOverlappingElement(es, screen, x, y):
    ps = frozenset(screen.overlappingIds(x,y))
    for e in es:
        if e.getId(screen) in ps:
            return e
    return None

class PinSet:
    def __init__(self):
        self.pois = set([])
        self.pods = set([])
        self.regs = set([])
        self.axle = None
        self.unknownPois = []
        self.removed = []
        self._changed = False
        self.csvHeaders = ['type', 'x', 'y', 'z', 'name']
    def changed(self):
        return self._changed
    def setChanged(self):
        self._changed = True
    def addPoi(self, p):
        self.pois.add(p)
        self._changed = True
    def createPoiPin(self, **kwargs):
        self.addPoi(PoiPin(**kwargs))
    def addPod(self, p):
        self.pods.add(p)
        self._changed = True
    def createPodPin(self, name, **kwargs):
        self.addPod(PodPin(name, **kwargs))
    def addReg(self, p):
        self.regs.add(p)
        self._changed = True
    def createRegPin(self, **kwargs):
        self.addReg(RegPin(**kwargs))
    def setAxle(self, p):
        self.removeAxle()
        self.axle = p
    def setAxlePosition(self, x, y):
        if self.axle is None:
            self.axle = AxlePin()
        self.axle.setPosition(x, y)
    def removeAxle(self):
        if self.axle is not None:
            self.axle.destroy()
            self.axle = None
    def remove(self, p):
        for ps in [self.pois, self.pods, self.regs]:
            if p in ps:
                ps.remove(p)
                p.destroy()
                return
        if p == self.axle:
            self.axle = None
        p.destroy()
        self._changed = True
    def hideAxle(self, *screens):
        if self.axle:
            self.axle.hide(*screens)
    def showAxle(self, *screens):
        if self.axle:
            self.axle.show(*screens)
    def poiCount(self):
        return len(self.pois)
    def podCount(self):
        return len(self.pods)
    def regCount(self):
        return len(self.regs)
    def load(self, image, fh):
        csv = CsvLoader(fh)
        if not csv.headersStartWith('type,x,y,z,name'):
            raise Exception('Could not understand csv file')
        self.csvHeaders = csv.getHeaders()
        self.pois = set([])
        self.pods = set([])
        self.regs = set([])
        self.unknownPois = []
        scale = image.pixelSize()
        for (t,tx,ty,z,name,*extras) in csv.generateRows():
            x = float(tx) / scale
            y = float(ty) / scale
            pin = None
            if t == 'i':
                if len(name) == 0:
                    pin = self.createPoiPin(x=x, y=y, z=z, extra_fields=extras)
                else:
                    pin = self.createPodPin(name, x=x, y=y, z=z, extra_fields=extras)
            elif t == 'r':
                pin = self.createRegPin(x=x, y=y, z=z, name=name, extra_fields=extras)
            else:
                self.unknownPois.append((t,tx,ty,z,name,*extras))
        self._changed = False
    def save(self, image, **kwargs):
        print(','.join(self.csvHeaders), **kwargs)
        for fields in self.unknownPois:
            print(','.join(fields), **kwargs)
        for pins in [ self.regs, self.pods, self.pois ]:
            for pin in pins:
                pin.renderCsvLine(image, **kwargs)
        self._changed = False
    def allPins(self):
        maybe_axle = [] if self.axle is None else [self.axle]
        return itertools.chain(self.pois, self.pods, self.regs, maybe_axle)
    def renderOn(self, screen, draggingPin=None):
        for p in self.allPins():
            p.render(screen, p == draggingPin)
    def isAxlePin(self, screen, x, y):
        if self.axle is None:
            return False
        return self.axle.getId(screen) in screen.overlappingIds(x, y)
    def getPoiPin(self, screen, x, y):
        return getOverlappingElement(self.pois, screen, x, y)
    def getPodPin(self, screen, x, y):
        return getOverlappingElement(self.pods, screen, x, y)
    def getRegPin(self, screen, x, y):
        return getOverlappingElement(self.regs, screen, x, y)
    def deletePin(self, pin, screens):
        if pin in self.pois:
            self.pois.remove(pin)
        elif pin in self.regs:
            self.regs.remove(pin)
        elif pin == self.axle:
            self.axle = None
        elif pin in self.pods:
            self.pods.remove(pin)
        pin.removeFrom(*screens)
        self._changed = True

class ManimalApplication(tk.Frame):
    def __init__(
        self,
        master,
        fixed=None,
        sliding=None,
        matrix_file=None,
        poi_file=None,
        fixed_brightness=2.0,
        flip='auto',
        mag_size=None
    ):
        super().__init__(master)
        self.grid(row=0, column=0, sticky='nsew')
        self.grid_columnconfigure(0, weight=1)
        self.controls1 = tk.Canvas(self)
        self.controls1.grid(column=0, row=1, sticky='nsew')
        self.controls2 = tk.Canvas(self)
        self.controls2.grid(column=0, row=2, sticky='nsew')
        self.counter = tk.Label(self.controls1)
        self.counter.pack(side='right')
        self.saveButton = tk.Button(self.controls2, text='Save', command = self.save)
        self.saveButton.pack(side='right')
        master.bind('<Control-s>', lambda e: self.save())
        self.closeButton = tk.Button(self.controls2, text='Close', command = self.maybeQuit)
        self.closeButton.pack(side='right')
        master.bind('<Escape>', lambda e: self.maybeQuit())
        if sliding:
            self.resetButton = tk.Button(self.controls2, text='Reset', command = self.confirmResetAlignment)
            self.resetButton.pack(side='right')
        self.canvas = tk.Canvas(self)
        self.canvas.grid(column=0, row=0, sticky='nsew')
        master.bind('<Left>', self.moveLeft)
        master.bind('a', self.moveLeft)
        master.bind('<Right>', self.moveRight)
        master.bind('d', self.moveRight)
        master.bind('<Up>', self.moveUp)
        master.bind('w', self.moveUp)
        master.bind('<Down>', self.moveDown)
        master.bind('s', self.moveDown)
        self.screen = Screen(self.canvas)
        self.overview = None
        self.screens = [self.screen]
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
        self.overviewSliderId = None
        if fixed:
            fixeds = glob.glob(os.path.expanduser(fixed))
            fixeds.sort()
            if len(fixeds) == 0:
                print('No such file:', fixed)
                exit(2)
            self.fixed = ZoomableImage(
                pathlib.Path(fixeds[0]),
                brightness=fixed_brightness
            )
            if len(fixeds) > 1:
                self.fixeds = {
                    os.path.splitext(os.path.basename(f))[0] : f
                    for f in fixeds
                }
                ks = list(self.fixeds.keys())
                self.fixedImageLabel = ttk.Label(self.controls1, text='image:')
                self.fixedImageLabel.pack(side='left')
                self.fixedImageSelector = ttk.Combobox(
                    self.controls1, values=ks, state='readonly'
                )
                self.fixedImageSelector.pack(side='left')
                self.fixedImageSelector.set(ks[0])
                self.fixedImageSelector.bind('<<ComboboxSelected>>', self.setFixedImage)
            centreFixed = self.fixed.centre()
            self.screen.setTranslation(centreFixed[0], centreFixed[1])
            self.overviewCanvas = tk.Canvas(self, bg='#001')
            self.overviewCanvas.bind("<Button-1>", self.goOverview)
            self.overviewCanvas.bind("<B1-Motion>", self.goOverview)
            self.overviewCanvas.grid(
                column=0, row=0, sticky='nsew'
            )
            self.overviewCanvas.create_polygon(
                (0, 0), tags=['view'], fill='#222', outline='#888'
            )
            self.overview = OverviewScreen(canvas=self.overviewCanvas)
            self.overviewWidth = 150
            self.overviewHeight = 150
            self.canvas.create_window(
                0, 0, anchor='nw',
                width=self.overviewWidth,
                height=self.overviewHeight,
                window=self.overviewCanvas
            )
            self.screens.append(self.overview)
            self.overview.setTranslation(centreFixed[0], centreFixed[1])
            self.overview.setScale(1.05 * max(
                self.fixed.width() / self.overviewWidth,
                self.fixed.height() / self.overviewHeight
            ))
            pix = self.fixed.pixelSize()
            if mag_size is not None:
                m = re.match(r"^(\d+)\s*(?:[x,]\s*(\d+))?$", mag_size)
                if m:
                    w = int(m.group(1))
                    h = int(m.group(2))
                    if h is None:
                        h = w
                    self.screen.setMagnifiedScreenSize(w / pix, h / pix)
            if sliding:
                self.overviewSliderId = self.overviewCanvas.create_polygon(
                    (0,0), fill='#445'
                )
                self.overviewCanvas.tag_raise('view', self.overviewSliderId)
        if flip == 'auto':
            flip = bool(fixed)
        elif flip == 'yes':
            flip = True
        else:
            flip = False
        self.pins = PinSet()
        self.pinCoords = {}
        self.pinNames = {}
        self.draggingPin = None
        self.pinButton = None
        self.sliding = None
        if sliding:
            functions.append(
                self.mouseFunctionSlide
            )
            self.pinButton = tk.Button(
                self.controls1,
                text='Pin',
                relief='raised',
                command=self.toggleAxlePin
            )
            self.pinButton.pack(side='left')
            master.bind('p', lambda e: self.toggleAxlePin())
            self.sliding = ZoomableImage(
                pathlib.Path(sliding), flip=flip
            )
            self.resetAlignment()
            text = 'brightness (sliding):' if fixed else 'brightness:'
            label = tk.Label(self.controls1, text=text)
            label.pack(side='left')
            self.slidingBrightnessSlider = tk.Scale(
                self.controls1,
                length=100, from_=-6, to=24, resolution=1,
                orient='horizontal',
                showvalue=False,
                command=self.setSlidingBrightness
            )
            self.slidingBrightnessSlider.set(10.0 * math.log10(fixed_brightness))
            self.slidingBrightnessSlider.pack(side='left')
        if fixed:
            text = 'brightness (fixed):' if sliding else 'brightness:'
            label = tk.Label(self.controls1, text=text)
            label.pack(side='left')
            self.fixedBrightnessSlider = tk.Scale(
                self.controls1,
                length=100, from_=-6, to=24, resolution=1,
                orient='horizontal',
                showvalue=False,
                command=self.setFixedBrightness
            )
            self.fixedBrightnessSlider.set(10.0 * math.log10(fixed_brightness))
            self.fixedBrightnessSlider.pack(side='left')
        self.loadMatrix(matrix_file)
        tk.Label(self.controls2, text='left mouse button:').pack(side='left')
        self.leftClickFunctionSelector = ttk.Combobox(
            self.controls2, values=functions, state='readonly'
        )
        self.leftClickFunctionSelector.pack(side='left')
        self.leftClickFunctionSelector.set(self.mouseFunctionMove)
        tk.Label(self.controls2, text='right mouse button:').pack(side='left')
        self.rightClickFunctionSelector = ttk.Combobox(
            self.controls2, values=functions, state='readonly'
        )
        self.rightClickFunctionSelector.pack(side='left')
        self.rightClickFunctionSelector.set(self.mouseFunctionMove)
        # zoom 100/percent
        self.zoomLevels = [20.0, 10.0, 4.0, 2.0, 1.0, 0.5, 0.25]
        self.screen.setScale(self.zoomLevels[0])
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
        self.updateCounter()
        self.loadPoisIfAvailable(self.poi_file)
        self.changed = False
        self.tick()

    def resetAlignment(self):
        centreSliding = self.sliding.centre()
        if self.fixed:
            centreFixed = self.fixed.centre()
            xflip = self.sliding.xFlip()
            self.sliding.setTranslation(
                    centreFixed[0] - xflip * centreSliding[0],
                    centreFixed[1] - centreSliding[1]
                )
        else:
            self.screen.setTranslation(centreSliding[0], centreSliding[1])
        self.sliding.setAngle(0)
        self.removeAxlePin()
        self.changed = True

    def confirmResetAlignment(self):
        self.wait_window(
            ResetDialog(self, self.resetAlignment)
        )

    def updateCache(self):
        if self.fixed:
            self.fixed.updateCacheIfNecessary(self.requestQueue, self.screen)
            self.fixed.update()
        if self.sliding:
            self.sliding.updateCacheIfNecessary(self.requestQueue, self.screen)
            self.sliding.update()

    def updateCounter(self):
        self.counter.configure(
            text='POIs: {0} (+{1} captured), Regs: {2}'.format(
                self.pins.poiCount(), self.pins.podCount(), self.pins.regCount()
            )
        )

    def tick(self):
        if self.fixed:
            self.fixed.update()
        if self.sliding:
            self.sliding.update()
        self.updateCanvas()
        self.updateCounter()
        self.after(1000, self.tick)

    def setFixedImage(self, event):
        cziPath = self.fixeds[event.widget.get()]
        self.update()
        self.fixed.load(cziPath)
        self.updateCache()

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
        self.pins.hideAxle(*self.screens)
        self.sliding.unsetPin()

    def toggleAxlePin(self):
        if self.pinButton == None:
            return
        c = self.pinButton.config
        if c('relief')[-1] == 'raised':
            c(relief='sunken')
            self.pins.showAxle(*self.screens)
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
        image = self.fixed or self.sliding
        self.pins.save(image=image, **kwargs)

    def saveMatrix(self, **kwargs):
        print('x,y,t', **kwargs)
        for row in self.sliding.getMicrometerMatrix():
            print(','.join(map(str, row)), **kwargs)

    def loadMatrix(self, matrix_file):
        if not self.sliding:
            self.matrix_file = None
            return
        self.matrix_file = matrix_file
        if not self.matrix_file:
            return
        try:
            with open(self.matrix_file, 'r') as fh:
                csv = CsvLoader(fh)
                if not csv.headersStartWith('x,y,t'):
                    print('Matrix file does not seem to be a matrix file.')
                    print('(should be a CSV with headers x, y, t)')
                    return
                rows = csv.generateRows()
                try:
                    row0 = next(rows)
                    row1 = next(rows)
                except StopIteration:
                    print('Matrix file does not have two rows')
                    return
                self.sliding.setMicrometerMatrix(
                    float(row0[0]),
                    float(row0[1]),
                    float(row0[2]),
                    float(row1[0]),
                    float(row1[1]),
                    float(row1[2])
                )
        except FileNotFoundError:
            # Matrix will be output, but it does not exist yet.
            # This is fine.
            return

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
        if self.changed or self.pins.changed():
            self.wait_window(SaveDialog(self, self.save, self.quit))
        else:
            self.quit()

    def pixelSize(self):
        pixelSize = None
        if self.fixed:
            pixelSize = self.fixed.pixelSize()
        if not pixelSize and self.sliding:
            pixelSize = self.sliding.pixelSize()
        return pixelSize

    def loadPois(self, fh):
        image = self.fixed if self.fixed else self.sliding
        self.pins.load(image=image, fh=fh)

    def loadPoisIfAvailable(self, fn):
        if fn and os.path.exists(fn):
            with open(fn, 'r', encoding='utf-8-sig') as fh:
                self.loadPois(fh)
                return True
        return False

    def isAxlePin(self, x, y):
        return self.pins.isAxlePin(self.screen, x, y)

    def regPinAt(self, x, y):
        return self.pins.getRegPin(self.screen, x, y)

    def poiPinAt(self, x, y):
        return self.pins.getPoiPin(self.screen, x, y)

    def deleteDraggingPin(self):
        p = self.draggingPin
        if p == None:
            return
        self.draggingPin = None
        self.pins.deletePin(p, self.screens)

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
        (wx, wy) = self.screen.toWorld(x, y)
        self.pins.createPoiPin(x=wx, y=wy)
        self.updateCanvas()
        self.updateCounter()
        self.canvas.focus_set()

    def dragStartAddRegPoint(self, x, y):
        (wx, wy) = self.screen.toWorld(x, y)
        self.pins.createRegPin(x=wx, y=wy)
        self.updateCanvas()
        self.updateCounter()
        self.canvas.focus_set()

    def dragStartSlide(self, x, y):
        self.sliding.dragStart(self.screen, x, y)
        self.canvas.focus_set()

    def dragSlide(self, x, y):
        self.sliding.dragMove(self.screen, x, y)
        self.updateCanvas()
        self.changed = True

    def dragPan(self, x, y):
        self.screen.dragMove(x, y)
        self.updateCanvas()

    def dragStart(self, x, y, funcName):
        self.canvas.focus_set()
        self.screen.dragStart(x, y)
        p = self.pins.getRegPin(self.screen, x, y)
        if not p:
            p = self.pins.getPoiPin(self.screen, x, y)
        if not p and self.pins.isAxlePin(self.screen, x, y):
            p = self.pins.axle
            self.sliding.dragStart(self.screen, x, y, draggingPin=True)
        if p:
            # clicked a pin; make it the dragging pin
            self.deleteDraggingPin()
            self.draggingPin = p
        elif self.draggingPin is None:
            # Not clicked a pin, so do whatever is selected for this button
            return self.mouseDownFunction[funcName](x, y)
        if not self.draggingPin:
            return
        (px, py) = self.draggingPin.screenCoords(self.screen)
        self.draggingPinOffsetX = px - x
        self.draggingPinOffsetY = py - y
        self.updateCanvas()

    def dragEnd(self, x, y, funcName):
        if self.draggingPin:
            if x < 0 or y < 0 or self.screen.width() < x or self.screen.height() < y:
                self.deleteDraggingPin()
            elif self.draggingPin == self.pins.axle:
                self.sliding.setPinIfUnset(self.screen, x, y)
            else:
                (wx, wy) = self.screen.toWorld(
                    self.draggingPinOffsetX + x,
                    self.draggingPinOffsetY + y
                )
                self.draggingPin.setPosition(wx, wy)
                self.pins.setChanged()
            self.draggingPin = None
            self.updateCanvas()
            self.updateCounter()
    
    def dragMove(self, x, y, funcName):
        if self.draggingPin:
            if self.draggingPin == self.pins.axle:
                self.sliding.dragMove(self.screen, x, y)
            else:
                (wx, wy) = self.screen.toWorld(
                    self.draggingPinOffsetX + x,
                    self.draggingPinOffsetY + y
                )
                self.draggingPin.setPosition(wx, wy)
            self.updateCanvas()
        else:
            self.mouseDragFunction[funcName](x, y)

    def startMove(self):
        self.moveInterpolation = 0
        self.doMove()
    def doMove(self):
        if self.moveInterpolation < 4:
            self.after(50, self.doMove)
        self.moveInterpolation += 1
        self.screen.moveTowardsTarget(self.moveInterpolation / 4)
        self.updateCanvas()
    def moveLeft(self, e):
        self.screen.setMoveTarget(-0.5, 0)
        self.startMove()
    def moveRight(self, e):
        self.screen.setMoveTarget(0.5, 0)
        self.startMove()
    def moveUp(self, e):
        self.screen.setMoveTarget(0, -0.5)
        self.startMove()
    def moveDown(self, e):
        self.screen.setMoveTarget(0, 0.5)
        self.startMove()

    def createCanvas(self):
        self.canvasImage = self.canvas.create_image(0, 0, anchor="nw")

    def showScaleBar(self):
        pixelSize = self.pixelSize()
        if not pixelSize:
            return
        pixelSize *= self.screen.getScale()
        scaleBarWidth = 300
        maxMicrons = pixelSize * scaleBarWidth
        self.scaleBar.show(scaleBarWidth, maxMicrons)

    def goOverview(self, e):
        p = self.overview.toWorld(e.x, e.y)
        self.screen.setTranslation(*p)
        self.updateCanvas()

    def updateOverviewCanvas(self):
        if self.overviewSliderId is not None:
            l = self.sliding.left()
            r = l + self.sliding.width()
            t = self.sliding.top()
            b = t + self.sliding.height()
            coords = [
                self.sliding.toScreen(self.overview, *c)
                for c in [(l, t), (r, t), (r, b), (l, b)]
            ]
            self.overviewCanvas.coords(
                self.overviewSliderId,
                list(itertools.chain(*coords))
            )
        (x, y) = self.screen.centreWorld()
        w2 = self.screen.widthWorld() / 2
        h2 = self.screen.heightWorld() / 2
        (lo, to) = self.overview.fromWorld(x - w2, y - h2)
        (ro, bo) = self.overview.fromWorld(x + w2, y + h2)
        self.overviewCanvas.coords('view', [
            lo, to, ro, to, ro, bo, lo, bo
        ])

    def updateCanvas(self, pin=None):
        self.updateCache()
        self.updateOverviewCanvas()
        image = None
        if self.fixed:
            image = self.fixed.transformImage(self.screen)
        if self.sliding:
            slider = self.sliding.transformImage(self.screen)
            if image:
                image = Image.blend(image, slider, 0.5)
            else:
                image = slider
        draggingPin = self.draggingPin
        if pin != None:
            (x,y) = pin
            (wx, wy) = self.screen.toWorld(x, y)
            self.pins.setAxlePosition(wx, wy)
            draggingPin = self.pins.axle
        elif self.sliding:
            p = self.sliding.pinScreenPosition(self.screen)
            if p:
                (x,y) = p
                (wx, wy) = self.screen.toWorld(
                    self.draggingPinOffsetX + x,
                    self.draggingPinOffsetY + y
                )
                self.pins.setAxlePosition(wx, wy)
        for screen in self.screens:
            self.pins.renderOn(screen, draggingPin)
        self.image = ImageTk.PhotoImage(image)
        self.showScaleBar()
        self.canvas.itemconfigure(self.canvasImage, image=self.image)

parser = argparse.ArgumentParser(
    description= 'MANual IMage ALigner: '
    + 'a GUI tool for manually aligning images with each other, '
    + 'finding points of interest, '
    + 'or aligning an image with existing points of interest. '
    + 'Manimal Copyright (C) 2022 Tim Band. '
    + 'This program is free software, and comes with '
    + 'ABSOLUTELY NO WARRANTY. You may distribute it under '
    + 'the terms of the Gnu Public License version 3.'
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
    help=(
        'Fixed image (.czi file) if desired. Use an asterisk to load'
        + ' multiple files which the user can switch between (on Mac'
        + ' or Linux asterisks need to be backslash-escaped or put'
        + ' in quotes)'
    ),
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
parser.add_argument(
    '--magsize',
    help='size of the images to be captured in micrometers e.g. 250x200',
    dest='mag_size',
    required=False
)
options = parser.parse_args()

kw = {
}
for attr in ['poi_file', 'sliding', 'matrix_file', 'fixed_brightness', 'fixed', 'flip', 'mag_size']:
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
root.protocol('WM_DELETE_WINDOW', app.maybeQuit)
app.mainloop()
