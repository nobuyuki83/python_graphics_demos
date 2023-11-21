from PyQt5 import QtWidgets
from util_moderngl_qt import DrawerMeshUnindex, QGLWidgetViewer3
import numpy
from del_msh.del_msh import MyClass


def main():
    coord = numpy.random.rand(100, 2).astype(numpy.float32)
    coord[:, :] *= 1.8
    coord[:, :] -= 09.
    kdtree2 = MyClass(coord)
    edges = kdtree2.edges()
    print(edges.shape)

    with QtWidgets.QApplication([]) as app:
        drawer = DrawerMeshUnindex.Drawer(elem2node2xyz=edges)
        win = QGLWidgetViewer3.QtGLWidget_Viewer3([drawer])
        win.show()
        app.exec()


if __name__ == "__main__":
    main()
