from PyQt5 import QtWidgets, QtCore, QtGui
from util_moderngl_qt import DrawerMeshUnindex, QGLWidgetViewer3
import del_srch
import numpy

def main():
    coord = numpy.random.rand(100, 2).astype(numpy.float32)
    coord[:, :] *= 1.8
    coord[:, :] -= 0.9
    kdtree2 = del_srch.MyClass(coord)
    edges = kdtree2.edges()
    print(edges.shape)

    with QtWidgets.QApplication([]) as app:
        drawer = DrawerMeshUnindex.Drawer(elem2node2xyz=edges)
        win = QGLWidgetViewer3.QtGLWidget_Viewer3([drawer])
        win.show()
        app.exec()



if __name__ == "__main__":
    main()
