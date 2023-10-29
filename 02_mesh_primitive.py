import moderngl
from PyQt5 import QtWidgets
import numpy

from util_moderngl_qt.drawer_mesh import DrawerMesh, ElementInfo
import util_moderngl_qt.qtglwidget_viewer3
import del_msh


def draw_mesh(tri2vtx, vtx2xyz):
    edge2vtx = del_msh.edges_of_uniform_mesh(tri2vtx, vtx2xyz.shape[0])

    with QtWidgets.QApplication([]) as app:
        drawer = DrawerMesh(
            vtx2xyz=vtx2xyz.astype(numpy.float32),
            list_elem2vtx=[
                ElementInfo(index=tri2vtx, color=(1, 0, 0), mode=moderngl.TRIANGLES),
                ElementInfo(index=edge2vtx, color=(0, 0, 0), mode=moderngl.LINES)]
        )
        win = util_moderngl_qt.qtglwidget_viewer3.QtGLWidget_Viewer3([drawer])
        win.show()
        app.exec()


if __name__ == "__main__":
    draw_mesh(*del_msh.torus_meshtri3(0.6, 0.3, 32, 32))
    draw_mesh(*del_msh.capsule_meshtri3(0.1, 0.6, 32, 32, 32))
    draw_mesh(*del_msh.cylinder_closed_end_meshtri3(0.1, 0.8, 32, 32))
    draw_mesh(*del_msh.sphere_meshtri3(1., 32, 32))
