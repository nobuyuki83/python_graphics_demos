import numpy
import matplotlib.patches
import matplotlib.pyplot as plt
from del_msh import TriMesh, PolyLoop

if __name__ == "__main__":
    vtx2xy_in = numpy.array([
        [0, 0],
        [1, 0],
        [1, 0.6],
        [0.6, 0.6],
        [0.6, 1.0],
        [0, 1]], dtype=numpy.float32)
    ##
    tri2vtx, vtx2xy = PolyLoop.tesselation2d(vtx2xy_in)
    _, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.triplot(vtx2xy[:, 0], vtx2xy[:, 1], tri2vtx)
    plt.show()
    ##
    xys = TriMesh.sample_many(tri2vtx, vtx2xy, 1000)
    _, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.scatter(xys[:, 0], xys[:, 1])
    ax.add_patch(matplotlib.patches.Polygon(xy=vtx2xy_in, closed=True, fill=False))
    plt.show()
    ##
    tri2vtx, vtx2xy = PolyLoop.tesselation2d(vtx2xy_in,
                                             resolution_edge=0.05,
                                             resolution_face=0.1)
    _, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.triplot(vtx2xy[:, 0], vtx2xy[:, 1], tri2vtx)
    plt.show()
