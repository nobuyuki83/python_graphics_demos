import pathlib
import moderngl
import numpy
from PyQt5 import QtWidgets
from util_moderngl_qt import DrawerMesh, QGLWidgetViewer3
from del_msh import TriMesh
from del_msh.del_msh import extend_trimesh

if __name__ == "__main__":
    path_file = pathlib.Path('.') / 'asset' / 'bunny_1k.obj'
    tri2vtx, vtx2xyz = TriMesh.load_wavefront_obj(str(path_file), is_centerize=True, normalized_size=1.0)
    vtx2xyz = extend_trimesh(tri2vtx, vtx2xyz.astype(numpy.float64), 0.01, 10)

    with QtWidgets.QApplication([]) as app:
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