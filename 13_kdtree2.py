from PyQt5 import QtWidgets
from util_moderngl_qt import DrawerMeshUnindex, QGLWidgetViewer3
import numpy
from del_msh import KdTree

def main():
    coord = numpy.random.rand(1000, 2).astype(numpy.float64)
    coord[:, :] *= 1.8
    coord[:, :] -= 0.9
    kdtree = KdTree.build_topology(coord)
    edges = KdTree.build_edge(kdtree, coord)
    print(edges.shape)

    with QtWidgets.QApplication([]) as app:
        drawer = DrawerMeshUnindex.Drawer(elem2node2xyz=edges)
        win = QGLWidgetViewer3.QtGLWidget_Viewer3([drawer])
        win.show()
        app.exec()


if __name__ == "__main__":
    main()
