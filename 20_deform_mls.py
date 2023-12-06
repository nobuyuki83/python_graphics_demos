import pathlib
import math
import numpy
import time
import moderngl
from PyQt5 import QtWidgets
from PyQt5.QtCore import QTimer

from util_moderngl_qt import DrawerMesh, QGLWidgetViewer3, DrawerSpheres
from del_msh import TriMesh, DeformMLS


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        path_file = pathlib.Path('.') / 'asset' / 'bunny_1k.obj'
        self.tri2vtx, self.vtx2xyz = TriMesh.load_wavefront_obj(str(path_file), is_centerize=True, normalized_size=1.)
        self.vtx2xyz_new = self.vtx2xyz.copy()
        print(self.vtx2xyz_new.strides, self.vtx2xyz_new.dtype)
        self.samples_old = numpy.array([
            (-0.3, -0.3, -0.3),
            (+0.3, -0.3, -0.3),
            (-0.3, +0.3, -0.3),
            (-0.3, -0.3, +0.3),
            (-0.3, +0.3, +0.3)], dtype=numpy.float32)
        self.weights = DeformMLS.kernel(self.samples_old, self.vtx2xyz)
        self.mls_data = DeformMLS.precomp(self.samples_old, self.vtx2xyz, self.weights)
        self.samples_new = self.samples_old.copy()

        edge2vtx = TriMesh.edge2vtx(self.tri2vtx, self.vtx2xyz.shape[0])
        self.drawer_edge = DrawerMesh.Drawer(
            vtx2xyz=self.vtx2xyz,
            list_elem2vtx=[
                DrawerMesh.ElementInfo(index=edge2vtx, color=(0, 0, 0), mode=moderngl.LINES),
                DrawerMesh.ElementInfo(index=self.tri2vtx, color=(1, 1, 1), mode=moderngl.TRIANGLES)
            ]
        )

        self.drawer_sphere = DrawerSpheres.Drawer()
        for sample in self.samples_old:
            self.drawer_sphere.list_sphere.append(
                DrawerSpheres.SphereInfo(pos=sample, color=(1.,0.,0.),rad=0.03))

        super().__init__()
        self.resize(640, 480)
        self.setWindowTitle('Mesh Viewer')
        self.glwidget = QGLWidgetViewer3.QtGLWidget_Viewer3(
            [self.drawer_edge, self.drawer_sphere])
        self.setCentralWidget(self.glwidget)
        #
        self.timer = QTimer()
        self.timer.setInterval(30)
        self.timer.timeout.connect(self.step_time)
        self.timer.start()

    def step_time(self):
        # self.time += self.timer.interval() * 1.0e-3
        self.samples_new = self.samples_old.copy()
        self.samples_new[0, 0] += -0.5 + 0.4 * math.sin(time.time() * 2.0)
        self.vtx2xyz_new = self.mls_data.dot(self.samples_new)
        #
        self.drawer_edge.update_position(self.vtx2xyz_new)
        for i_sample in range(self.samples_new.shape[0]):
            self.drawer_sphere.list_sphere[i_sample].pos = self.samples_new[i_sample]
        self.glwidget.update()


def main():
    with QtWidgets.QApplication([]) as app:
        win = MainWindow()
        win.show()
        app.exec()


if __name__ == "__main__":
    main()
