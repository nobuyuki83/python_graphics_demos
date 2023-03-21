import numpy
import pathlib
import del_msh
import random
import moderngl
import pyrr
from PyQt5 import QtWidgets, QtCore, QtGui
from util_moderngl_qt.drawer_mesh_colormap import Drawer, ElementInfo
import util_moderngl_qt.qtglwidget_viewer3


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        path_file = pathlib.Path('.') / 'asset' / 'bunny_1k.obj'
        self.tri2vtx, self.vtx2xyz = del_msh.load_wavefront_obj_as_triangle_mesh(str(path_file))
        self.vtx2xyz = del_msh.centerize_scale_3d_points(self.vtx2xyz)
        self.vtx2val = (numpy.sin( self.vtx2xyz[:, 0] * 10.) + 1.) * 0.5
        print(self.vtx2val.dtype, self.vtx2val.shape)

        edge2vtx = del_msh.edges_of_uniform_mesh(self.tri2vtx, self.vtx2xyz.shape[0])
        drawer_edge = Drawer(
            vtx2xyz=self.vtx2xyz.astype(numpy.float32),
            list_elem2vtx=[
                ElementInfo(index=edge2vtx, color=(0, 0, 0), mode=moderngl.LINES),
                ElementInfo(index=self.tri2vtx, color=(1, 1, 1), mode=moderngl.TRIANGLES)
            ],
            vtx2val=self.vtx2val,
            color_map=numpy.array([
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.5, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0]])
        )

        super().__init__()
        self.resize(640, 480)
        self.setWindowTitle('Mesh Viewer')
        self.glwidget = util_moderngl_qt.qtglwidget_viewer3.QtGLWidget_Viewer3(
            [drawer_edge])
        self.setCentralWidget(self.glwidget)


def main():
    with QtWidgets.QApplication([]) as app:
        win = MainWindow()
        win.show()
        app.exec()


if __name__ == "__main__":
    main()
