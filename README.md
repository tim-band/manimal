# Manimal

Manual Image Alignment tool

## Align by eye

```
./manimal.py -f fixed.czi -a sliding.czi -m matrix.csv
```

Manimal allows you to align two large CZI images by eye.

You can choose the functions of the left button and right button
from the "left mouse button" and "right mouse button" menus.
They will both be set to "move" when the program starts. This
means that either a left button drag or a right button drag
moves the entire image around. If you select "slide" for either
button, then dragging with this button causes just the sliding
image to move.

At the start, this sliding movement will be translation. Click the
"Pin" button to get a pin, then click the image to place the pin.
Dragging with the chosen sliding button now rotate the sliding
image around the pin. Click the "Pin" button again in order to
return to translating the image.

Roll the scroll wheel to zoom in and out.

Click "OK" to output the matrix mapping fixed image points
(in micrometers) to sliding image points (in micrometers)
and close the tool.

Click "cancel" to close the tool without outputting the matrix.

The tool cannot currently cope with non-square pixels, or
fixed and sliding images with different scalings.

The tool can cope with multi-gigabyte images in CZI files.

The output file specified by `-m` or `--matrix` will be filled with
the matrix for transforming the fixed image co-ordinates to
the sliding image co-ordinated (in micrometers).

If not such output file is specified the matrix will be output on
standard out.

## Find Points of Interest

```
./manimal.py -f image.czi -p poi.csv
```

Scroll around the image as above with dragging the moust and
rolling the scroll wheel. Select "POI" for one of the mouse button
functions and you can start clicking this button to add (white)
Point Of Interest markers. Select "Reg. point" to add (yellow)
Registration Point markers. Markers can be dragged around
in any mode. Markers dragged outside the image window are
deleted.

Click "OK" to output a table of markers, one per line in the form:

```
<t>,<x>,<y>
```

where `<t>` is the type: `r` for a registation marker, `i` for a point
of interest marker, `<x>` and `<y>` are the marker's coordinates
in micrometers.

The tool cannot cope with non-square pixels even in this mode where
it would be trivial to support.

## Align an image to POIs

```
./manimal.py -a image.czi -p poi.csv -m matrix.csv
```

Move the image as in "Align by Eye" above, by choosing the
"slide" option for one of the mouse buttons and dragging with
this mouse button. Choose "Add POI" (perhaps for the other
mouse button) if you want to add any more POIs.

The matrix of the transformation from the starting position of
the image to its final position is written into the file given by the
`-m` or `--matrix`, or to standard out if such a file is not given.

# Building the wheel

```sh
python -m pipenv requirements > requirements.txt
python -m pip wheel . -r requirements.txt
```

(or skip the `python -m` if you are in a pipenv shell)

Suppose, for some reason, you wanted to install Manimal on some Windows
machine that has no internet connectivity. In this case you would find a
Windows machine that does have internet connectivity, build the wheels
as above, copy the `.whl` files produced by the lines above onto the
target machine, then install them with `python -m pip install <.whl>`
where `<.whl>` is the wheel file you want to install. The order of
installation is important; one that works is `numpy`, `Pillow`,
`aicspylibczi`, then finally you can install `manimal`.

# The mathematics

We have a number of co-ordinate systems linked by affine transformations.
All these co-ordinate systems are oriented with Y increasing downwards and
X increasing to the right.

## Basics

Rotation anticlockwise by a is represented by

```
(cos a, -sin a)/
(sin a,  cos a)
```

Let us represent an affine transformation by `(L|T)` where `L` is the linear part and `T` is the translation.

### multiplication

```
(L'|T')(L|T) = (L'L|L'T+T')
```

### inverse

`(L'|T')(L|T) = (I|0)` => `L'L = I` and `T' = -L'T` so

```
inv(L'|T') = (inv(L)|-inv(L)T)
```

### rotation about a point

`(I|T)(R|0)(I|-T)` = `(R|(-RT)+T)` =

```
(R|T-RT)
```

## The Spaces

Co-ordinate spaces' origins are in the centre of the bitmaps. `R` is a rotation,
`Z`s are scalings, `F` is horizontal flip (-1, 0 / 0 1). `T`s and `P`s are
translations (or positions)

### mappings between spaces

| Space | Closest space | Transformation to closest space | Transformation to world | Transformation to screen | Transformation from screen |
|--- |--- |--- |--- |--- |--- |
| Screen | World | `(Zv\|Pv)` | `(Zv\|Pv)` | `I` | `I` |
| World | World | `I` | `I` | `(inv(Zv)\|-inv(Zv)Pv)` = `(Zw\|Pw)` | `(Zv\|Pv)` |
| Fixed image cached portion | World | `(Zf\|Pf)` | `(Zf\|Pf)` | `(ZwZf\|Pw+ZwPf)` | `(inv(Zf)Zv\|-inv(Zf)Zv(Pw+ZwPf))` = `(inv(Zf)Zv\|-inv(Zf)(ZvPw+Pf))` = `(inv(Zf)Zv\|inv(Zf)(Pv - Pf))` |
| Sliding image | World | `(RF\|T)` | `(RF\|T)` | `(ZwRF\|Pw+ZwT)` | `(F.inv(R)Zv\|-F.inv(R)Zv(-inv(Zv)Pv+inv(Zv)T))` =  `(F.inv(R)Zv\|F.inv(R)(Pv - T))` |
| Sliding image cached portion | Sliding image | `(Zs\|Ps)` | `(RFZs\|RFPs+T)` | `(ZwZsRF \| Pw + ZwRFPs + ZwT)` | `(inv(ZwZsRF) \| inv(Zs)(F.inv(R)(Pv - T) - Ps))` |
| Rotated flipped cached portion | Sliding cached | `inv(RF)` = `Finv(R)` | `(RFZsFinv(R)\|RFPs+T)` = `(Zs\|RFPs+T)` | `(ZwZs \| Pw + ZwRFPs + ZwT)` | `(inv(Zs)Zv \| inv(Zs)(Pv - T - RFPs))` |

### schema

| Symbol | code |
|--- |--- |
| R | `manimalApplication.rotation` |
| T | `(manimalApplication.panX, manimalApplication.panY)` |
| Zv | `manimalApplication.zoom` |
