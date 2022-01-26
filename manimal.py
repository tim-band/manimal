#! /usr/bin/env python3
from aicspylibczi import CziFile
import math
import numpy as np
import pathlib
from PIL import Image, ImageTk
import queue
import sys
import threading
import tkinter as tk

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

def readImage(queue, czi: CziFile, radius : int, scale, cx : int, cy : int):
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
    bbox = czi.get_mosaic_bounding_box()
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
    data = czi.read_mosaic(rect, 1 / scale, C=0)
    brightness = 2.0
    rescaled = np.minimum(np.multiply(data[0], brightness/256.0), 256.0)
    img = np.asarray(rescaled.astype(np.uint8))
    pil = Image.fromarray(img)
    queue.put((pil, rect, scale, radius))

class Screen:
    def __init__(self, canvas):
        self.panX = 0
        self.panY = 0
        self.scale = 1.0
        self.canvas = canvas

    def setScale(self, scale):
        self.scale = scale

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
    def __init__(self, cziPath, flip=False):
        self.czi = CziFile(pathlib.Path(cziPath))
        self.queue = queue.SimpleQueue()
        self.bbox = self.czi.get_mosaic_bounding_box()
        self.pil = None
        self.pilPosition = None
        self.pilRadius = 0
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

    def setAngle(self, angle):
        self.angle = angle

    def setTranslation(self, x, y):
        self.tx = x
        self.ty = y

    def centre(self):
        """
        returns the centre of the whole image in image co-ordinates.
        """
        return (
            self.flip * (self.bbox.x + self.bbox.w // 2),
            self.bbox.y + self.bbox.h // 2
        )

    def toWorld(self, x, y):
        s = math.sin(self.angle)
        c = math.cos(self.angle)
        return (
            c * x + s * y + self.tx,
            c * y - s * x + self.ty
        )

    def fromWorld(self, x, y):
        x -= self.tx
        y -= self.ty
        s = math.sin(self.angle)
        c = math.cos(self.angle)
        return (
            c * x - s * y,
            c * y + s * x
        )

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
                (i, p, z, r) = self.queue.get(block=block)
                count += 1
        except queue.Empty:
            if count == 0:
                return False
        self.waitingForRead = False
        self.pil = i
        self.pilPosition = p
        self.scale = z # scaling of image compared to cache
        self.pilRadius = r
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
        radius = math.floor(math.hypot(cw, ch) * screen.scale / 2)
        cacheRadius = radius * 3
        (x,y) = self.fromWorld(screen.panX, screen.panY)
        # clamp to within the bounding box of the image - radius
        # (we are assuming here that bbox.w and bbox.h are at least 2*cacheRadius, which
        # will be true for the images we care about)
        x = max(self.bbox.x + cacheRadius, min(self.bbox.x + self.bbox.w - cacheRadius, x))
        y = max(self.bbox.y + cacheRadius, min(self.bbox.y + self.bbox.h - cacheRadius, y))
        # if we are not within the old radius, request a new cache update
        magnification = self.scale / screen.scale
        if (self.pilPosition == None
                or magnification < 0.45
                or (1.9 < magnification and 1.0 < self.scale)
                or x - radius < self.pilPosition[0]
                or y - radius < self.pilPosition[1]
                or self.pilPosition[0] + self.pilPosition[2] < x + radius
                or self.pilPosition[1] + self.pilPosition[3] < y + radius):
            requestQueue.put(ImageReadRequest(
                self,
                cacheRadius,
                screen.scale,
                x,
                y
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
        return {
            'image': pil.resize(
                (canvasW, canvasH),
                resample=Image.NEAREST,
                box=(leftCrop, topCrop, rightCrop, bottomCrop)
            ),
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
    def __init__(self, image: ZoomableImage, radius: int, scale, cx: int, cy:int):
        """
        Prepare a request to read a portion of a CZI image to cache.

        Parameters:
            zim: The image to read
            radius: The number of pixels to read in each direction from the (cx,cy)
            scale: The reciprocal of the scale factor required (so 2 gives a half-sized image)
            (cx, cy): The centre of the image to be read in image co-ordinates
        """
        self.queue = image.queue
        self.czi = image.czi
        self.radius = radius
        self.scale = scale
        self.cx = cx
        self.cy = cy

def readImages(inputQueue):
    """
    Continuously read ImageReadRequests from the inputQueue, outputting
    the results onto the queues of the ZoomableImages passed in.
    """
    while True:
        i = inputQueue.get()
        readImage(i.queue, i.czi, i.radius, i.scale, i.cx, i.cy)

class ManimalApplication(tk.Frame):
    def __init__(self, master, fixed, sliding):
        super().__init__(master)
        self.grid(row=0, column=0, sticky='nsew')
        self.okButton = tk.Button(self, text='OK', command = self.saveAndQuit)
        master.bind('<Return>', lambda e: self.saveAndQuit())
        self.cancelButton = tk.Button(self, text='cancel', command = self.quit)
        master.bind('<Escape>', lambda e: self.quit())
        self.pinButton = tk.Button(self, text='Pin', relief='raised', command = self.togglePin)
        master.bind('p', lambda e: self.togglePin())
        self.fixed = ZoomableImage(pathlib.Path(fixed))
        flip = True
        xflip = flip and -1 or 1
        self.sliding = ZoomableImage(pathlib.Path(sliding), flip=flip)
        self.canvas = tk.Canvas(self)
        self.screen = Screen(self.canvas)
        # centres of the images
        centreFixed = self.fixed.centre()
        centreSliding = self.sliding.centre()
        # rotation (in radians) then translation (in image pixels) of the sliding image
        # Order of operations for sliding layer: flip -> rotate -> translate
        self.sliding.setTranslation(
            centreFixed[0] - xflip * centreSliding[0],
            centreFixed[1] - centreSliding[1]
        )
        # zoom 100/percent
        self.zoomLevels = [20.0, 10.0, 4.0, 2.0, 1.0, 0.5, 0.25]
        self.screen.setTranslation(centreFixed[0], centreFixed[1])
        self.okButton.grid(column=2, row=1, sticky="s")
        self.cancelButton.grid(column=1, row=1, sticky="s")
        self.pinButton.grid(column=0, row=1, sticky="s")
        self.canvas.grid(column=0, columnspan=12, row=0, sticky="nsew")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_rowconfigure(2, weight=0)
        self.pinId = None
        self.canvas.bind("<Button-1>", lambda e: self.dragStart(e.x, e.y))
        self.canvas.bind("<B1-Motion>", lambda e: self.dragMove(e.x, e.y))
        self.canvas.bind("<Button-3>", lambda e: self.dragStart(e.x, e.y))
        self.canvas.bind("<B3-Motion>", lambda e: self.rightDragMove(e.x, e.y))
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
        self.update_idletasks()
        self.fixed.updateCacheIfNecessary(self.requestQueue, self.screen)
        self.sliding.updateCacheIfNecessary(self.requestQueue, self.screen)
        self.createCanvas()
        self.updateCanvas()
        self.tick()

    def tick(self):
        if self.fixed.update() or self.sliding.update():
            self.updateCanvas()
        self.after(1000, self.tick)

    def togglePin(self):
        c = self.pinButton.config
        if c('relief')[-1] == 'raised':
            c(relief='sunken')
            self.canvas.itemconfigure(self.pinId, state='normal')
        else:
            c(relief='raised')
            self.canvas.itemconfigure(self.pinId, state='hidden')
            self.sliding.unsetPin()

    def isPinIn(self):
        return self.pinButton.config('relief')[-1] == 'sunken'

    def motion(self, e):
        if self.sliding.pin != None or not self.isPinIn():
            return
        self.updateCanvas(pin=(e.x, e.y))

    def saveAndQuit(self):
        print("Save!")
        #... save
        super().quit()

    def isPin(self, x, y):
        ids = self.canvas.find_overlapping(x, y, x+1, y+1)
        return self.pinId in ids

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

    def dragStart(self, x, y):
        self.screen.dragStart(x, y)
        self.sliding.dragStart(self.screen, x, y, draggingPin=self.isPin(x, y))
        if self.isPinIn():
            if self.sliding.setPinIfUnset(self.screen, x, y):
                self.updateCanvas()

    def rightDragMove(self, x, y):
        self.screen.dragMove(x, y)
        self.updateCanvas()

    def dragMove(self, x, y):
        self.sliding.dragMove(self.screen, x, y)
        self.updateCanvas()

    def createCanvas(self):
        self.canvasImage = self.canvas.create_image(0, 0, anchor="nw")
        self.pinId = self.canvas.create_polygon((0,0,0,1,1,1,1,0),
            fill='white', joinstyle='miter', outline='black', state='hidden')

    def updateCanvas(self, pin=None):
        # fixed image: -pan -> zoom -> crop
        # = -pan -> crop (antizoomed canvas frame) -> zoom
        # = crop (antizoomed canvas frame + pan) -> zoom
        # sliding image: rotate -> slide -> -pan -> zoom -> crop
        # = rotate + slide -> -pan -> crop (antizoomed canvas frame) -> zoom
        # = rotate + slide -> crop (antizoom canvas frame + pan) -> zoom
        self.fixed.updateCacheIfNecessary(self.requestQueue, self.screen)
        self.fixed.update()
        self.sliding.updateCacheIfNecessary(self.requestQueue, self.screen)
        self.sliding.update()
        image = self.fixed.transformImage(self.screen)
        slider = self.sliding.transformImage(self.screen)
        image = Image.blend(image, slider, 0.5)
        if pin != None:
            (x,y) = pin
            self.canvas.coords(self.pinId, (x,y,x+5,y-15,x+15,y-5))
        else:
            p = self.sliding.pinScreenPosition(self.screen)
            if p:
                (x,y) = p
                self.canvas.coords(self.pinId, (x,y,x-5,y-15,x+5,y-15))
        self.image = ImageTk.PhotoImage(image)
        self.canvas.itemconfigure(self.canvasImage, image=self.image)

if len(sys.argv) != 3:
    raise Exception("Need two inputs: ./manimal <fixed-image.czi> <sliding-image.czi>")

root = tk.Tk()
root.grid_columnconfigure(0, weight=1)
root.grid_rowconfigure(0, weight=1)
app = ManimalApplication(root, sys.argv[1], sys.argv[2])
app.master.title("Manimal")
app.mainloop()
