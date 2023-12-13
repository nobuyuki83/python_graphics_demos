import pathlib
#
import numpy
import moderngl
from PyQt5 import QtWidgets, QtCore
#
from util_moderngl_qt import DrawerMesh, QGLWidgetViewer3, DrawerSpheres
from del_msh import TriMesh, DeformMLS, WavefrontObj, Tri


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        path_dir = pathlib.Path('.') / 'asset' / 'shapenet_car1' / 'models'
        obj = WavefrontObj.load(str(path_dir / 'model_normalized.obj'), is_centerize=True, normalized_size=1.8)
        self.tri2vtx = obj.tri2vtxxyz()
        self.vtx2xyz_ini = obj.vtxxyz2xyz
        self.vtx2xyz_def = self.vtx2xyz_ini.copy()
        self.samples_ini = numpy.zeros((0, 3), dtype=numpy.float32)
        self.weights = DeformMLS.kernel(self.samples_ini, self.vtx2xyz_ini)
        self.mls_data = DeformMLS.precomp(self.samples_ini, self.vtx2xyz_ini, self.weights)
        self.samples_def = self.samples_ini.copy()
        self.i_sample_pick = -1

        edge2vtx = TriMesh.edge2vtx(self.tri2vtx, self.vtx2xyz_ini.shape[0])
        self.drawer_edge = DrawerMesh.Drawer(
            vtx2xyz=self.vtx2xyz_def,
            list_elem2vtx=[
                DrawerMesh.ElementInfo(index=edge2vtx, color=(0, 0, 0), mode=moderngl.LINES),
                DrawerMesh.ElementInfo(index=self.tri2vtx, color=(1, 1, 1), mode=moderngl.TRIANGLES)
            ]
        )
        self.drawer_sphere = DrawerSpheres.Drawer()
        for sample in self.samples_ini:
            self.drawer_sphere.list_sphere.append(
                DrawerSpheres.SphereInfo(pos=sample, color=(1., 0., 0.), rad=0.03))

        super().__init__()
        self.resize(640, 480)
        self.setWindowTitle('Mesh Viewer')
        self.glwidget = QGLWidgetViewer3.QtGLWidget_Viewer3(
            [self.drawer_edge, self.drawer_sphere])
        self.glwidget.mousePressCallBack.append(self.mouse_press_callback)
        self.glwidget.mouseMoveCallBack.append(self.mouse_move_callback)
        self.setCentralWidget(self.glwidget)

    def mouse_press_callback(self, event):
        if event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
            return
        if event.modifiers() & QtCore.Qt.KeyboardModifier.AltModifier:
            return
        src, direction = self.glwidget.nav.picking_ray()
        pos_def, tri_index = TriMesh.first_intersection_ray(
            numpy.array(src.xyz).astype(numpy.float32), numpy.array(direction.xyz).astype(numpy.float32),
            self.vtx2xyz_def, self.tri2vtx)
        if tri_index == -1:
            return
        bcoord = Tri.barycentric_coord(
            self.vtx2xyz_def[self.tri2vtx[tri_index, 0]],
            self.vtx2xyz_def[self.tri2vtx[tri_index, 1]],
            self.vtx2xyz_def[self.tri2vtx[tri_index, 2]],
            pos_def)
        pos_ini = self.vtx2xyz_ini[self.tri2vtx[tri_index, 0]] * bcoord[0] \
                  + self.vtx2xyz_ini[self.tri2vtx[tri_index, 1]] * bcoord[1] \
                  + self.vtx2xyz_ini[self.tri2vtx[tri_index, 2]] * bcoord[2]
        self.samples_def = numpy.vstack([self.samples_def, pos_def])
        self.samples_ini = numpy.vstack([self.samples_ini, pos_ini])
        self.weights = DeformMLS.kernel(self.samples_ini, self.vtx2xyz_ini)
        self.mls_data = DeformMLS.precomp(self.samples_ini, self.vtx2xyz_ini, self.weights)
        self.i_sample_pick = self.samples_def.shape[0] - 1
        self.drawer_sphere.list_sphere = []
        for sdef in self.samples_def:
            sinfo = DrawerSpheres.SphereInfo(rad=0.03, pos=sdef, color=(1., 0., 0))
            self.drawer_sphere.list_sphere.append(sinfo)
        self.glwidget.updateGL()

    def mouse_move_callback(self, event):
        if event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
            return
        if event.modifiers() & QtCore.Qt.KeyboardModifier.AltModifier:
            return
        if self.i_sample_pick == -1:
            return
        if self.mls_data is None:
            return
        mvp = self.glwidget.nav.projection_matrix() * self.glwidget.nav.modelview_matrix()
        mvp = numpy.array(mvp).transpose()
        pos_def = numpy.append(self.samples_def[self.i_sample_pick], 1.).astype(numpy.float32)
        trg = numpy.array([self.glwidget.nav.cursor_x, self.glwidget.nav.cursor_y, mvp.dot(pos_def)[2], 1.0],
                          dtype=numpy.float32)
        pos_def_drag = numpy.linalg.inv(mvp).dot(trg)
        self.samples_def[self.i_sample_pick] = pos_def_drag[:3]
        self.vtx2xyz_def = self.mls_data.dot(self.samples_def).astype(numpy.float32)
        self.drawer_edge.update_position(self.vtx2xyz_def)
        self.glwidget.updateGL()


def main():
    with QtWidgets.QApplication([]) as app:
        win = MainWindow()
        win.show()
        app.exec()


if __name__ == "__main__":
    main()
