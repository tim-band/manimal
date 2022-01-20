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

def findNoLessThan(x, xs):
    for i in range(len(xs)):
        if x <= xs[i]:
            return i
    return len(xs)

def readImage(queue, czi: CziFile, radius : int, zoom, cx : int, cy : int):
    """
    parameters:
        czi: CZI image file
        radius: how big we want the image to be, in source pixels
        zoom: how many screen pixels should cover 100 source pixels
        should probably be kept to powers of two
    returns (np.ndarray, (left, top, width, height))
    """
    zoom = min(100, zoom)
    scale = zoom / 100
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
    data = czi.read_mosaic(rect, scale, C=0)
    brightness = 2.0
    rescaled = np.minimum(np.multiply(data[0], brightness/256.0), 256.0)
    img = np.asarray(rescaled.astype(np.uint8))
    pil = Image.fromarray(img)
    queue.put((pil, rect, zoom))

class ZoomableImage:
    def __init__(self, cziPath):
        self.czi = CziFile(pathlib.Path(cziPath))
        self.queue = queue.SimpleQueue()
        self.bbox = self.czi.get_mosaic_bounding_box()
        self.pil = None
        self.pilPosition = None
        self.zoom = 100

    def centre(self):
        return (self.bbox.x + self.bbox.w // 2, self.bbox.y + self.bbox.h // 2)

    def update(self):
        count = 0
        try:
            while True:
                block = count == 0 and self.pil == None
                (i, p, z) = self.queue.get(block=block)
                count += 1
        except queue.Empty:
            if count == 0:
                return False
        self.pil = i
        self.pilPosition = p
        self.zoom = z
        return True

class ImageReadRequest:
    def __init__(self, image: ZoomableImage, radius: int, zoom, cx: int, cy:int):
        self.queue = image.queue
        self.czi = image.czi
        self.radius = radius
        self.zoom = zoom
        self.cx = cx
        self.cy = cy

def readImages(inputQueue):
    while True:
        i = inputQueue.get()
        readImage(i.queue, i.czi, i.radius, i.zoom, i.cx, i.cy)

class TransformableImage:
    def __init__(self, zim: ZoomableImage, cx: int, cy: int, zoom):
        self.zim = zim
        self.cx = cx
        self.cy = cy
        self.zoom = zoom

    def getCropped(self, width, height):
        # position on the fixed background
        zwidth = (width * 100) / self.zoom
        zheight = (height * 100) / self.zoom
        zleft = math.floor(self.cx - zwidth / 2 + 0.5)
        ztop = math.floor(self.cy - zheight / 2 + 0.5)
        zwidth = math.floor(zwidth)
        zheight = math.floor(zheight)
        # position of cropped image on screen
        offsetX = 0
        offsetY = 0
        # crops in background pixels
        bbox = self.zim.bbox
        cropped = False
        leftPil = self.zim.pilPosition[0]
        leftCrop = zleft - leftPil
        if leftCrop < 0:
            offsetX = -leftCrop
            leftCrop = 0
            cropped = bbox.x < leftPil
        topPil = self.zim.pilPosition[1]
        topCrop = ztop - topPil
        if topCrop < 0:
            offsetY = -topCrop
            topCrop = 0
            cropped = cropped or bbox.y < topPil
        rightCrop = zleft + zwidth - leftPil
        widthPil = self.zim.pilPosition[2]
        if widthPil < rightCrop:
            rightCrop = widthPil
            cropped = cropped or leftPil + widthPil < bbox.x + bbox.w
        bottomCrop = ztop + zheight - topPil
        heightPil = self.zim.pilPosition[3]
        if heightPil < bottomCrop:
            bottomCrop = heightPil
            cropped = cropped or topPil + heightPil < bbox.y + bbox.h
        widthCrop = rightCrop - leftCrop
        heightCrop = bottomCrop - topCrop
        if widthCrop <= 0 or heightCrop <= 0:
            return None
        # width and height in screen pixels
        canvasX = math.floor(offsetX * self.zoom / 100)
        canvasY = math.floor(offsetY * self.zoom / 100)
        canvasW = math.ceil(widthCrop * self.zoom / 100)
        canvasH = math.ceil(heightCrop * self.zoom / 100)
        # crops in extracted image pixels
        topIm = max(0, math.floor(topCrop * self.zim.zoom / 100))
        bottomIm = min(self.zim.pil.height, math.floor(bottomCrop * self.zim.zoom / 100))
        leftIm = max(0, math.floor(leftCrop * self.zim.zoom / 100))
        rightIm = min(self.zim.pil.width, math.floor(rightCrop * self.zim.zoom / 100))
        self.image = self.zim.pil.resize(
            (canvasW, canvasH),
            resample=Image.NEAREST,
            box=(leftIm, topIm, rightIm, bottomIm)
        )
        cropped = cropped
        self.x = canvasX
        self.y = canvasY
        magnification = self.zoom / self.zim.zoom
        self.outdated = (magnification <= 0.5
            or (2 <= magnification and self.zim.zoom < 100)
            or cropped)
        return self.image

    def get(self, width, height):
        im = self.getCropped(width, height)
        if (im != None
            and self.x == 0 and self.y == 0
            and im.width == width and im.height == height
        ):
            return im
        bg = Image.new("RGB", (width, height), (0,0,0))
        if im != None:
            bg.paste(im, (self.x, self.y))
        return bg

class ManimalApplication(tk.Frame):
    def __init__(self, master, fixed, sliding):
        super().__init__(master)
        self.grid(row=0, column=0, sticky="nsew")
        self.okButton = tk.Button(self, text="OK", command = self.saveAndQuit)
        master.bind('<Return>', lambda e: self.saveAndQuit())
        self.cancelButton = tk.Button(self, text="cancel", command = self.quit)
        master.bind('<Escape>', lambda e: self.quit())
        self.fixed = ZoomableImage(pathlib.Path(fixed))
        self.sliding = ZoomableImage(pathlib.Path(sliding))
        self.canvas = tk.Canvas(self)
        # centres of the images
        centreFixed = self.fixed.centre()
        centreSliding = self.sliding.centre()
        # rotation (in radians) then translation (in image pixels) of the sliding image
        self.rotation = 0.0
        self.slideX = centreSliding[0] - centreFixed[0]
        self.slideY = centreSliding[1] - centreFixed[1]
        # zoom percent
        self.zoomLevels = [5, 10, 25, 50, 100, 200, 400]
        self.zoom = 100
        # where the centre of the view is on the fixed image
        self.panX = centreFixed[0]
        self.panY = centreFixed[1]
        # None if we are rotating and sliding in "free" mode, an (x,y)
        # co-ordinate pair about which we are rotating if we are in
        # "pinned" mode.
        self.pin = None
        self.okButton.grid(column=0, row=1, sticky="s")
        self.cancelButton.grid(column=1, row=1, sticky="s")
        self.canvas.grid(column=0, columnspan=2, row=0, sticky="nsew")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.canvasPin = None
        self.canvas.bind("<Button-1>", lambda e: self.dragStart(e.x, e.y))
        self.canvas.bind("<B1-Motion>", lambda e: self.dragMove(e.x, e.y))
        self.canvas.bind("<Button-3>", lambda e: self.dragStart(e.x, e.y))
        self.canvas.bind("<B3-Motion>", lambda e: self.rightDragMove(e.x, e.y))
        self.canvas.bind("<MouseWheel>", lambda e: self.zoomChange(e.delta, e.x, e.y))
        self.canvas.bind("<Button-4>", lambda e: self.zoomChange(-120, e.x, e.y))
        self.canvas.bind("<Button-5>", lambda e: self.zoomChange(120, e.x, e.y))
        self.requestQueue = queue.SimpleQueue()
        self.readThread = threading.Thread(
            target=readImages, args=(self.requestQueue,)
        )
        self.readThread.daemon = True
        self.readThread.start()
        self.update_idletasks()
        self.readCurrentLocality()
        self.createCanvas()
        self.updateCanvas()
        self.tick()

    def tick(self):
        if self.fixed.update() or self.sliding.update():
            self.updateCanvas()
        self.after(1000, self.tick)


    def readCurrentLocality(self):
        if not self.requestQueue.empty():
            return
        radius = self.canvas.winfo_width() + self.canvas.winfo_height()
        self.requestQueue.put(ImageReadRequest(
            self.fixed,
            math.floor(radius * 100 / self.zoom),
            self.zoom,
            self.panX,
            self.panY
        ))
        self.requestQueue.put(ImageReadRequest(
            self.sliding,
            math.floor(radius * 100 / self.zoom),
            self.zoom,
            self.panX + self.slideX,
            self.panY + self.slideY
        ))

    def saveAndQuit(self):
        print("Save!")
        #... save
        super().quit()

    def isPin(self, x, y):
        ids = self.canvas.find_overlapping(x, y, x+1, y+1)
        return self.canvasPin in ids

    def zoomChange(self, delta, x, y):
        # let p0 be the initial pan point and s0 the original zoom, with m the
        # clicked point relative to the centre of the canvas (in screen pixels).
        # Then we want p0 + m/s0 = p1 + m/s1
        # => p1 = p0 + (1/s0 - 1/s1)m
        mx = x - self.canvas.winfo_width() / 2
        my = y - self.canvas.winfo_height() / 2
        z0 = 100 / self.zoom
        zoomIndex = findNoLessThan(self.zoom, self.zoomLevels)
        if delta < 0:
            if zoomIndex == 0:
                return
            self.zoom = self.zoomLevels[zoomIndex - 1]
        else:
            if len(self.zoomLevels) <= zoomIndex + 1:
                return
            self.zoom = self.zoomLevels[zoomIndex + 1]
        ds = z0 - 100 / self.zoom
        self.panX += ds * mx
        self.panY += ds * my
        self.updateCanvas()

    def dragStart(self, x, y):
        self.pan0X = self.panX
        self.pan0Y = self.panY
        self.slide0X = self.slideX
        self.slide0Y = self.slideY
        self.dragOriginX = x
        self.dragOriginY = y

    def rightDragMove(self, x, y):
        # we want p0 + m0/s = p1 + m1/s
        # => p1 = p0 + m0/s - m1/s
        # if x is the actual click position and c is the canvas centre then
        # p1 = p0 + (x0-c)/s - (x1-c)/s = p0 + (x0 - x1)/s
        self.panX = math.floor(self.pan0X + (self.dragOriginX - x) * 100 / self.zoom + 0.5)
        self.panY = math.floor(self.pan0Y + (self.dragOriginY - y) * 100 / self.zoom + 0.5)
        self.updateCanvas()

    def dragMove(self, x, y):
        self.slideX = math.floor(self.slide0X + (self.dragOriginX - x) * 100 / self.zoom + 0.5)
        self.slideY = math.floor(self.slide0Y + (self.dragOriginY - y) * 100 / self.zoom + 0.5)
        self.updateCanvas()

    def createCanvas(self):
        self.canvasImage = self.canvas.create_image(0, 0, anchor="nw")

    def updateCanvas(self):
        # fixed image: -pan -> zoom -> crop
        # = -pan -> crop (antizoomed canvas frame) -> zoom
        # = crop (antizoomed canvas frame + pan) -> zoom
        # sliding image: rotate -> slide -> -pan -> zoom -> crop
        # = rotate + slide -> -pan -> crop (antizoomed canvas frame) -> zoom
        # = rotate + slide -> crop (antizoom canvas frame + pan) -> zoom
        self.fixed.update()
        self.sliding.update()
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        tim = TransformableImage(self.fixed, self.panX, self.panY, self.zoom)
        image = tim.get(width, height)
        sim = TransformableImage(
            self.sliding,
            self.panX + self.slideX,
            self.panY + self.slideY,
            self.zoom
        )
        slider = sim.get(width, height)
        image = Image.blend(image, slider, 0.5)
        self.image = ImageTk.PhotoImage(image)
        self.canvas.itemconfigure(self.canvasImage, image=self.image)
        if tim.outdated:
            self.readCurrentLocality()

if len(sys.argv) != 3:
    raise Exception("Need two inputs: ./manimal <fixed-image.czi> <sliding-image.czi>")

root = tk.Tk()
root.grid_columnconfigure(0, weight=1)
root.grid_rowconfigure(0, weight=1)
app = ManimalApplication(root, sys.argv[1], sys.argv[2])
app.master.title("Manimal")
app.mainloop()
