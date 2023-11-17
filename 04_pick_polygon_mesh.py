from pathlib import Path
import moderngl
import numpy
import pyrr
from PyQt5 import QtWidgets, QtCore

from del_msh import WavefrontObj, TriMesh
from util_moderngl_qt import QGLWidgetViewer3, DrawerMesh
from util_moderngl_qt.drawer_transform import DrawerTransform


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):

        file_path = Path('.') / 'asset' / 'HorseSwap.obj'
        obj = WavefrontObj.load(str(file_path), is_centerize=True, normalized_size=1.8)
        self.vtx2xyz = obj.vtxxyz2xyz
        self.tri2vtx = obj.tris()

        drawer_triquadmesh3 = DrawerMesh.Drawer(
            vtx2xyz=self.vtx2xyz,
            list_elem2vtx=[
                DrawerMesh.ElementInfo(index=obj.edges(), color=(0, 0, 0), mode=moderngl.LINES),
                DrawerMesh.ElementInfo(index=self.tri2vtx, color=(1, 1, 1), mode=moderngl.TRIANGLES)]
        )

        sphere_tri2vtx, sphere_vtx2xyz = TriMesh.sphere()
        self.drawer_sphere = DrawerMesh.Drawer(vtx2xyz=sphere_vtx2xyz, list_elem2vtx=[
            DrawerMesh.ElementInfo(index=sphere_tri2vtx, color=(1., 0., 0.), mode=moderngl.TRIANGLES)])
        self.drawer_sphere = DrawerTransform(self.drawer_sphere)
        self.drawer_sphere.transform = pyrr.Matrix44.from_scale((0.05, 0.05, 0.05))

        QtWidgets.QMainWindow.__init__(self)
        self.resize(640, 480)
        self.setWindowTitle('Mesh Viewer')
        self.glwidget = QGLWidgetViewer3.QtGLWidget_Viewer3(
            [drawer_triquadmesh3, self.drawer_sphere])
        self.glwidget.mousePressCallBack.append(self.mouse_press_callback)
        self.setCentralWidget(self.glwidget)

    def mouse_press_callback(self, event):
        if event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
            return
        if event.modifiers() & QtCore.Qt.KeyboardModifier.AltModifier:
            return
        src, direction = self.glwidget.nav.picking_ray()
        pos, tri_index = TriMesh.first_intersection_ray(
            numpy.array(src.xyz).astype(numpy.float32), numpy.array(direction.xyz).astype(numpy.float32),
            self.vtx2xyz, self.tri2vtx)
        self.drawer_sphere.is_visible = False
        if tri_index != -1:
            self.drawer_sphere.is_visible = True
            self.drawer_sphere.transform = pyrr.Matrix44.from_translation(pos) * pyrr.Matrix44.from_scale(
                (0.03, 0.03, 0.03))
        self.glwidget.updateGL()


if __name__ == "__main__":
    with QtWidgets.QApplication([]) as app:
        win = MainWindow()
        win.show()
        app.exec()
