import moderngl
from PyQt5 import QtWidgets

from util_moderngl_qt import DrawerMesh, QGLWidgetViewer3
from del_msh import TriMesh


def draw_mesh(tri2vtx, vtx2xyz):
    edge2vtx = TriMesh.edge2vtx(tri2vtx=tri2vtx, num_vtx=vtx2xyz.shape[0])

    with QtWidgets.QApplication([]) as app:
        drawer = DrawerMesh.Drawer(
            vtx2xyz=vtx2xyz,
            list_elem2vtx=[
                DrawerMesh.ElementInfo(index=tri2vtx, color=(1, 0, 0), mode=moderngl.TRIANGLES),
                DrawerMesh.ElementInfo(index=edge2vtx, color=(0, 0, 0), mode=moderngl.LINES)]
        )
        win = QGLWidgetViewer3.QtGLWidget_Viewer3([drawer])
        win.show()
        app.exec()


if __name__ == "__main__":
    draw_mesh(*TriMesh.hemisphere(radius=1.0, ndiv_longtitude=8))
    draw_mesh(*TriMesh.torus(major_radius=0.4, minor_radius=0.2))
    draw_mesh(*TriMesh.capsule(radius=0.1, height=1.2, ndiv_longtitude=8))
    draw_mesh(*TriMesh.cylinder(radius=0.3, height=1.2))
    draw_mesh(*TriMesh.sphere(radius=1.0))
