import math

import moderngl
from PyQt5 import QtWidgets
import numpy

from util_moderngl_qt import DrawerMesh, QGLWidgetViewer3
from del_msh.del_msh import extend_polyedge
from del_msh import TriMesh

if __name__ == "__main__":
    n = 200
    rad = 0.8
    amp = 0.3
    height = 0.3
    lpedge2lpvtx = numpy.ndarray((n, 2), dtype=numpy.uint32)
    lpvtx2xyz = numpy.ndarray((n, 3), dtype=numpy.float64)
    for i in range(0, n):
        lpedge2lpvtx[i, 0] = i
        lpedge2lpvtx[i, 1] = (i + 1) % n
        #
        dtheta = math.pi * 2 / n
        theta = dtheta * i
        rad1 = (rad+math.cos(4*theta)*amp)
        z = math.cos(4*theta)*height
        lpvtx2xyz[i, 0] = rad1 * math.cos(theta)
        lpvtx2xyz[i, 1] = rad1 * math.sin(theta)
        lpvtx2xyz[i, 2] = z

    tri2vtx, vtx2xyz = extend_polyedge(lpvtx2xyz, 0.02, 10)

    with QtWidgets.QApplication([]) as app:
        '''
        drawer_loop = DrawerMesh.Drawer(
            vtx2xyz=lpvtx2xyz.astype(numpy.float32),
            list_elem2vtx=[
                DrawerMesh.ElementInfo(index=lpedge2lpvtx, color=(0, 0, 0), mode=moderngl.LINES)]
        )
        '''
        edge2vtx = TriMesh.edge2vtx(tri2vtx, vtx2xyz.shape[0])
        drawer_trimesh = DrawerMesh.Drawer(
            vtx2xyz=vtx2xyz,
            list_elem2vtx=[
                DrawerMesh.ElementInfo(index=edge2vtx, color=(0, 0, 0), mode=moderngl.LINES),
                DrawerMesh.ElementInfo(index=tri2vtx, color=(1, 1, 1), mode=moderngl.TRIANGLES)]
        )
        win = QGLWidgetViewer3.QtGLWidget_Viewer3([drawer_trimesh])
        win.show()
        app.exec()