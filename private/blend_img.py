from PIL import Image
import numpy

if __name__ == "__main__":
    d0 = numpy.array( Image.open("depth0.png") )
    d1 = numpy.array( Image.open("depth1.png") )
    mask0 = (d0 != 255)
    mask1 = (d1 != 255)
    mask01 = mask0 & mask1
    m01 = Image.fromarray(mask01)
    m01.save("mask01.png")
    d01 = (d0.astype(numpy.float32) + d1.astype(numpy.float32))*0.5
    print(d01)
    d01 = d01.astype(numpy.uint8)
    d01 = Image.fromarray(d01)
    d01.save("depth01.png")
