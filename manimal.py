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
    def __init__(self, zim: ZoomableImage, cx: int, cy: int, zoom, angle=0, translation=(0,0)):
        self.zim = zim
        self.cx = cx
        self.cy = cy
        self.zoom = zoom
        self.angle = angle
        self.translation = translation

    def getCropped(self, width, height):
        cx = self.cx
        cy = self.cy
        (leftPil, topPil, widthPil, heightPil) = self.zim.pilPosition
        pil = self.zim.pil
        if self.angle != 0:
            # we need to rotate the loaded portion by angle degrees
            # and transform (cx,cy) by rotating it the same amount
            pil = pil.rotate(self.angle)
            th = self.angle * math.pi / 180
            s = math.sin(th)
            c = math.cos(th)
            # The centre of the pil image in its own coordinates
            pilCx = leftPil + widthPil / 2
            pilCy = topPil + heightPil / 2
            # But now pil has expanded
            widthPil = pil.width
            heightPil = pil.height
            leftPil = pilCx - widthPil / 2
            topPil = pilCy - heightPil / 2
            # The centre of the pil image after being transformed into world co-ordinates
            px = pilCx * c + pilCy * s + self.translation[0]
            py = pilCy * c - pilCx * s + self.translation[1]
            # (x,y) is the position relative to the centre of the image
            x = cx - px
            y = cy - py
            # and finally put this into the (leftPil, topPil, widthPil, heightPil) co-ordinates
            cx = pilCx + x
            cy = pilCy + y
        else:
            cx -= self.translation[0]
            cy -= self.translation[1]
        # position on the image
        zwidth = (width * 100) / self.zoom
        zheight = (height * 100) / self.zoom
        zleft = math.floor(cx - zwidth / 2 + 0.5)
        ztop = math.floor(cy - zheight / 2 + 0.5)
        zwidth = math.floor(zwidth)
        zheight = math.floor(zheight)
        # position of cropped image on screen
        offsetX = 0
        offsetY = 0
        # crops in background pixels
        bbox = self.zim.bbox
        cropped = False
        leftCrop = zleft - leftPil
        if leftCrop < 0:
            offsetX = -leftCrop
            leftCrop = 0
            cropped = bbox.x < leftPil
        topCrop = ztop - topPil
        if topCrop < 0:
            offsetY = -topCrop
            topCrop = 0
            cropped = cropped or bbox.y < topPil
        rightCrop = zleft + zwidth - leftPil
        if widthPil < rightCrop:
            rightCrop = widthPil
            cropped = cropped or leftPil + widthPil < bbox.x + bbox.w
        bottomCrop = ztop + zheight - topPil
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
        bottomIm = min(pil.height, math.floor(bottomCrop * self.zim.zoom / 100))
        leftIm = max(0, math.floor(leftCrop * self.zim.zoom / 100))
        rightIm = min(pil.width, math.floor(rightCrop * self.zim.zoom / 100))
        self.image = pil.resize(
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
        self.grid(row=0, column=0, sticky='nsew')
        self.okButton = tk.Button(self, text='OK', command = self.saveAndQuit)
        master.bind('<Return>', lambda e: self.saveAndQuit())
        self.cancelButton = tk.Button(self, text='cancel', command = self.quit)
        master.bind('<Escape>', lambda e: self.quit())
        self.pinButton = tk.Button(self, text='Pin', relief='raised', command = self.togglePin)
        master.bind('p', lambda e: self.togglePin())
        self.fixed = ZoomableImage(pathlib.Path(fixed))
        self.sliding = ZoomableImage(pathlib.Path(sliding))
        self.canvas = tk.Canvas(self)
        # centres of the images
        centreFixed = self.fixed.centre()
        centreSliding = self.sliding.centre()
        # rotation (in radians) then translation (in image pixels) of the sliding image
        # Order of operations for sliding layer: flip -> rotate -> translate
        self.rotation = 0.0
        self.slideX = centreFixed[0] - centreSliding[0]
        self.slideY = centreFixed[1] - centreSliding[1]
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
        self.okButton.grid(column=2, row=1, sticky="s")
        self.cancelButton.grid(column=1, row=1, sticky="s")
        self.pinButton.grid(column=0, row=1, sticky="s")
        self.canvas.grid(column=0, columnspan=12, row=0, sticky="nsew")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_rowconfigure(2, weight=0)
        self.canvasPin = None
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
        self.readCurrentLocality()
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
            self.pin = None

    def isPinIn(self):
        return self.pinButton.config('relief')[-1] == 'sunken'

    def motion(self, e):
        if self.pin != None or not self.isPinIn():
            return
        self.updateCanvas(pin=(e.x, e.y))

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
        r = self.rotation * math.pi / 180
        s = math.sin(r)
        c = math.cos(r)
        x = self.panX - self.slideX
        y = self.panY - self.slideY
        self.requestQueue.put(ImageReadRequest(
            self.sliding,
            math.floor(radius * 100 / self.zoom),
            self.zoom,
            c * x - s * y,
            c * y + s * x
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

    def screenToFixed(self, x, y):
        fx = math.floor((x - self.canvas.winfo_width() / 2) * self.zoom / 100 + 0.5)
        fy = math.floor((y - self.canvas.winfo_height() / 2) * self.zoom / 100 + 0.5)
        return (self.panX + fx, self.panY + fy)

    def screenToPinOffset(self, x, y):
        (fx,fy) = self.screenToFixed(x, y)
        (px,py) = self.pin
        return (fx - px, fy - py)

    def dragStart(self, x, y):
        self.pan0X = self.panX
        self.pan0Y = self.panY
        self.slide0X = self.slideX
        self.slide0Y = self.slideY
        self.rotation0 = self.rotation
        self.dragOriginX = x
        self.dragOriginY = y
        if self.pin == None and self.isPinIn():
            self.pin = self.screenToFixed(x, y)
            self.updateCanvas()
        if self.pin:
            self.pinOffset = self.screenToPinOffset(x, y)

    def rightDragMove(self, x, y):
        # we want p0 + m0/s = p1 + m1/s
        # => p1 = p0 + m0/s - m1/s
        # if x is the actual click position and c is the canvas centre then
        # p1 = p0 + (x0-c)/s - (x1-c)/s = p0 + (x0 - x1)/s
        self.panX = math.floor(self.pan0X + (self.dragOriginX - x) * 100 / self.zoom + 0.5)
        self.panY = math.floor(self.pan0Y + (self.dragOriginY - y) * 100 / self.zoom + 0.5)
        self.updateCanvas()

    def dragMove(self, x, y):
        if self.pin == None:
            self.slideX = math.floor(self.slide0X - (self.dragOriginX - x) * 100 / self.zoom + 0.5)
            self.slideY = math.floor(self.slide0Y - (self.dragOriginY - y) * 100 / self.zoom + 0.5)
        else:
            # To rotate S around p:
            # S(Rx+t - p) + p = SRx + S(t-p)+p
            (px,py) = self.pinOffset
            (nx,ny) = self.screenToPinOffset(x, y)
            r = math.atan2(py,px) - math.atan2(ny,nx)
            self.rotation = self.rotation0 + r * 180 / math.pi
            sx = self.slide0X - px
            sy = self.slide0Y - py
            s = math.sin(r)
            c = math.cos(r)
            self.slideX = px + c * sx + s * sy
            self.slideY = py + c * sy - s * sx
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
        self.fixed.update()
        self.sliding.update()
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        tim = TransformableImage(self.fixed, self.panX, self.panY, self.zoom)
        image = tim.get(width, height)
        sim = TransformableImage(
            self.sliding,
            self.panX,
            self.panY,
            self.zoom,
            self.rotation,
            (self.slideX, self.slideY)
        )
        slider = sim.get(width, height)
        image = Image.blend(image, slider, 0.5)
        if pin != None:
            (x,y) = pin
            self.canvas.coords(self.pinId, (x,y,x+5,y-15,x+15,y-5))
        elif self.pin != None:
            (x,y) = self.pin
            x = math.floor((x - self.panX) * 100 / self.zoom + width/2)
            y = math.floor((y - self.panY) * 100 / self.zoom + height/2)
            self.canvas.coords(self.pinId, (x,y,x-5,y-15,x+5,y-15))
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
