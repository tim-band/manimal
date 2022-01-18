from aicspylibczi import CziFile
import math
import numpy as np
import pathlib
from PIL import Image, ImageTk
import queue
import skimage.transform as sktr
import skimage.util as skutil
import threading
import tkinter as tk

def findNoLessThan(x, xs):
    for i in range(len(xs)):
        if x <= xs[i]:
            return i
    return len(xs)

def emptyQueue(queue):
    try:
        queue.get(block=False)
    except:
        pass

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
    print("rect: ({0},{1},{2},{3}) scale: {4}".format(top, left, width, height, scale))
    pil = Image.fromarray(skutil.img_as_ubyte(data[0]))
    emptyQueue(queue)
    queue.put((pil, rect, zoom))
    print("read")

class ManimalApplication(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.fixedImageQueue = queue.SimpleQueue()
        self.grid(row=0, column=0, sticky="nsew")
        self.okButton = tk.Button(self, text="OK", command = self.saveAndQuit)
        master.bind('<Return>', lambda e: self.saveAndQuit())
        self.cancelButton = tk.Button(self, text="cancel", command = self.quit)
        master.bind('<Escape>', lambda e: self.quit())
        self.fixedImage = CziFile(pathlib.Path("Ind114CPolTiledMedium.czi"))
        self.slidingImage = CziFile(pathlib.Path("Ind114MicaTiledMedium.czi"))
        self.canvas = tk.Canvas(self)
        fibb = self.fixedImage.get_mosaic_bounding_box()
        sibb = self.slidingImage.get_mosaic_bounding_box()
        # centres of the images
        self.centreFixedX = fibb.x + fibb.w // 2
        self.centreFixedY = fibb.y + fibb.h // 2
        self.centreSlidingX = sibb.x + sibb.w // 2
        self.centreSlidingY = sibb.y + sibb.h // 2
        # rotation (in radians) then translation (in image pixels) of the sliding image
        self.rotation = 0.0
        self.slideX = self.centreFixedX - self.centreSlidingX
        self.slideY = self.centreFixedY - self.centreSlidingY
        # zoom percent
        self.zoomLevels = [5, 10, 25, 50, 100, 200, 400]
        self.zoom = 100
        # where the centre of the view is on the fixed image
        self.panX = self.centreFixedX
        self.panY = self.centreFixedY
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
        self.fixedImagePil = None
        self.fixedImageThread = None
        self.update_idletasks()
        self.readCurrentLocality()
        self.createCanvas()
        self.updateCanvas()

    def readCurrentLocality(self):
        print("read")
        if self.fixedImageThread != None and self.fixedImageThread.is_alive():
            print("busy")
            return
        radius = self.canvas.winfo_width() + self.canvas.winfo_height()
        self.fixedImageThread = threading.Thread(target=readImage, args=(
            self.fixedImageQueue,
            self.fixedImage,
            math.floor(radius * 100 / self.zoom),
            self.zoom,
            self.panX,
            self.panY
        ))
        self.fixedImageThread.start()
        print("go")

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
        print(zoomIndex,self.zoom)
        ds = z0 - 100 / self.zoom
        self.panX += ds * mx
        self.panY += ds * my
        self.updateCanvas()

    def dragStart(self, x, y):
        self.pan0X = self.panX
        self.pan0Y = self.panY
        self.dragOriginX = x
        self.dragOriginY = y

    def dragMove(self, x, y):
        # we want p0 + m0/s = p1 + m1/s
        # => p1 = p0 + m0/s - m1/s
        # if x is the actual click position and c is the canvas centre then
        # p1 = p0 + (x0-c)/s - (x1-c)/s = p0 + (x0 - x1)/s
        self.panX = math.floor(self.pan0X + (self.dragOriginX - x) * 100 / self.zoom + 0.5)
        self.panY = math.floor(self.pan0Y + (self.dragOriginY - y) * 100 / self.zoom + 0.5)
        self.updateCanvas()

    def rightDragMove(self, x, y):
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
        if self.fixedImagePil == None or not self.fixedImageQueue.empty():
            (i, p, z) = self.fixedImageQueue.get()
            while not self.fixedImageQueue.empty():
                (i, p, z) = self.fixedImageQueue.get()
            print("read new image", z)
            self.fixedImagePil = i
            self.fixedPortionPosition = p
            self.fixedImageZoom = z
        # size on screen
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        # position on the fixed background
        zwidth = (width * 100) / self.zoom
        zheight = (height * 100) / self.zoom
        zleft = math.floor(self.panX - zwidth / 2 + 0.5)
        ztop = math.floor(self.panY - zheight / 2 + 0.5)
        zwidth = math.floor(zwidth)
        zheight = math.floor(zheight)
        # position of cropped image on screen
        offsetX = 0
        offsetY = 0
        # crops in background pixels
        cropped = False
        fibb = self.fixedImage.get_mosaic_bounding_box()
        leftCrop = zleft - self.fixedPortionPosition[0]
        if leftCrop < 0:
            offsetX = -leftCrop
            leftCrop = 0
            cropped = fibb.x < self.fixedPortionPosition[0]
        topCrop = ztop - self.fixedPortionPosition[1]
        if topCrop < 0:
            offsetY = -topCrop
            topCrop = 0
            cropped = cropped or fibb.y < self.fixedPortionPosition[1]
        rightCrop = zleft + zwidth - self.fixedPortionPosition[0]
        if self.fixedPortionPosition[2] < rightCrop:
            rightCrop = self.fixedPortionPosition[2]
            cropped = cropped or self.fixedPortionPosition[0] + self.fixedPortionPosition[2] < fibb.x + fibb.w
        bottomCrop = ztop + zheight - self.fixedPortionPosition[1]
        if self.fixedPortionPosition[3] < bottomCrop:
            bottomCrop = self.fixedPortionPosition[3]
            cropped = cropped or self.fixedPortionPosition[1] + self.fixedPortionPosition[3] < fibb.y + fibb.h
        widthCrop = rightCrop - leftCrop
        heightCrop = bottomCrop - topCrop
        # width and height in screen pixels
        canvasX = math.floor(offsetX * self.zoom / 100)
        canvasY = math.floor(offsetY * self.zoom / 100)
        canvasW = math.ceil(widthCrop * self.zoom / 100)
        canvasH = math.ceil(heightCrop * self.zoom / 100)
        # crops in extracted image pixels
        topIm = max(0, math.floor(topCrop * self.fixedImageZoom / 100))
        bottomIm = min(self.fixedImagePil.height, math.floor(bottomCrop * self.fixedImageZoom / 100))
        leftIm = max(0, math.floor(leftCrop * self.fixedImageZoom / 100))
        rightIm = min(self.fixedImagePil.width, math.floor(rightCrop * self.fixedImageZoom / 100))
        image = self.fixedImagePil.resize(
            (canvasW, canvasH),
            resample=Image.NEAREST,
            box=(leftIm, topIm, rightIm, bottomIm)
        )
        self.image = ImageTk.PhotoImage(image)
        self.canvas.coords(self.canvasImage, canvasX, canvasY)
        self.canvas.itemconfigure(self.canvasImage, image=self.image)
        magnification = self.zoom / self.fixedImageZoom
        if (magnification <= 0.5
            or (2 <= magnification and self.fixedImageZoom < 100)
            or cropped):
            self.readCurrentLocality()

root = tk.Tk()
root.grid_columnconfigure(0, weight=1)
root.grid_rowconfigure(0, weight=1)
app = ManimalApplication(root)
app.master.title("Manimal")
app.mainloop()
