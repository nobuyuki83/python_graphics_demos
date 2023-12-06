import numpy
import pathlib
import random
import moderngl
import pyrr
from PyQt5 import QtWidgets

from util_moderngl_qt import DrawerMesh, QGLWidgetViewer3, DrawerSpheres
from del_msh import TriMesh


def sample_mesh_uniform(tri2vtx, vtx2xyz):
    tri2area = TriMesh.tri2area(tri2vtx, vtx2xyz)
    cumsum_area = numpy.cumsum(numpy.append(0., tri2area)).astype(numpy.float32)

    rad = 0.1

    samples = []
    for i in range(1000):
        smpl_i = TriMesh.sample(cumsum_area, random.random(), random.random())
        pos_i = TriMesh.position(tri2vtx, vtx2xyz, *smpl_i)
        is_near = False
        for smpl_j in samples:
            pos_j = TriMesh.position(tri2vtx, vtx2xyz, *smpl_j)
            distance = numpy.linalg.norm(pos_i - pos_j)
            if distance < rad:
                is_near = True
                break
        if is_near:
            continue
        samples.append(smpl_i)
    return samples


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        path_file = pathlib.Path('.') / 'asset' / 'bunny_1k.obj'
        self.tri2vtx, self.vtx2xyz = TriMesh.load_wavefront_obj(str(path_file), is_centerize=True, normalized_size=1.)
        samples = sample_mesh_uniform(self.tri2vtx, self.vtx2xyz)

        edge2vtx = TriMesh.edge2vtx(self.tri2vtx, self.vtx2xyz.shape[0])
        drawer_edge = DrawerMesh.Drawer(
            vtx2xyz=self.vtx2xyz,
            list_elem2vtx=[
                DrawerMesh.ElementInfo(index=edge2vtx, color=(0, 0, 0), mode=moderngl.LINES),
                DrawerMesh.ElementInfo(index=self.tri2vtx, color=(1, 1, 1), mode=moderngl.TRIANGLES)
            ]
        )

        sphere_tri2vtx, shere_vtx2xyz = TriMesh.sphere()
        self.drawer_sphere = DrawerMesh.Drawer(vtx2xyz=shere_vtx2xyz, list_elem2vtx=[
            DrawerMesh.ElementInfo(index=sphere_tri2vtx, color=(1., 0., 0.), mode=moderngl.TRIANGLES)])
        self.drawer_sphere = DrawerSpheres.Drawer()
        for sample in samples:
            pos_i = TriMesh.position(self.tri2vtx, self.vtx2xyz, *sample)
            self.drawer_sphere.list_sphere.append(
                DrawerSpheres.SphereInfo(pos=pos_i, color=(1.,0., 0.), rad=0.01)
            )

        super().__init__()
        self.resize(640, 480)
        self.setWindowTitle('Mesh Viewer')
        self.glwidget = QGLWidgetViewer3.QtGLWidget_Viewer3(
            [drawer_edge, self.drawer_sphere])
        self.setCentralWidget(self.glwidget)


def main():
    with QtWidgets.QApplication([]) as app:
        win = MainWindow()
        win.show()
        app.exec()


if __name__ == "__main__":
    main()
