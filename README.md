# Manimal

Manual Image Alignment tool

```
./manimal.py fixed.czi sliding.czi
```

Manimal allows you to align two large CZI images by eye.

Use a left-click drag to translate the sliding image around. Click the
"Pin" button to get a pin, then click the image to place the pin.
Left-click drags now rotate the sliding image around the pin. Click
the "Pin" button again in order to return to translating the image.

Use a right-click drag to move the enitre view around, or roll
the scroll wheel to zoom in and out.

Click "OK" to output the matrix mapping fixed image points
(in micrometers) to sliding image points (in micrometers)
and close the tool.

Click "cancel" to close the tool without outputting the matrix.

The tool cannot currently cope with non-square pixels, or
fixed and sliding images with different scalings.

The tool can cope with multi-gigabyte images.

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
