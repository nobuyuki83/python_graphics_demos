import numpy
import pathlib
import moderngl
from PyQt5 import QtWidgets
from util_moderngl_qt import DrawerMeshColorMap, QGLWidgetViewer3, Colormap
from del_msh import TriMesh


def main():
    path_file = pathlib.Path('.') / 'asset' / 'bunny_1k.obj'
    tri2vtx, vtx2xyz = TriMesh.load_wavefront_obj(str(path_file), is_centerize=True, normalized_size=1.0)
    vtx2val = (numpy.sin(vtx2xyz[:, 0] * 10.) + 1.) * 0.5

    edge2vtx = TriMesh.edge2vtx(tri2vtx, vtx2xyz.shape[0])
    drawer_edge = DrawerMeshColorMap.Drawer(
        vtx2xyz=vtx2xyz.astype(numpy.float32),
        list_elem2vtx=[
            DrawerMeshColorMap.ElementInfo(index=edge2vtx, color=(0, 0, 0), mode=moderngl.LINES),
            DrawerMeshColorMap.ElementInfo(index=tri2vtx, color=(1, 1, 1), mode=moderngl.TRIANGLES)
        ],
        vtx2val=vtx2val,
        color_map=Colormap.heat()
    )
    with QtWidgets.QApplication([]) as app:
        win = QGLWidgetViewer3.QtGLWidget_Viewer3(
            [drawer_edge])
        win.show()
        app.exec()


if __name__ == "__main__":
    main()
