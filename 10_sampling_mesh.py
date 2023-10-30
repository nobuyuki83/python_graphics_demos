import numpy
import pathlib
import del_msh
import random
import moderngl
import pyrr
from PyQt5 import QtWidgets, QtCore, QtGui
from util_moderngl_qt.drawer_mesh import DrawerMesh, ElementInfo
from util_moderngl_qt.drawer_transform_multi import DrawerTransformMulti
import util_moderngl_qt.qtglwidget_viewer3


def position_3d_on_triangle_in_mesh(
        idx_tri: int, r0: float, r1: float,
        tri2vtx, vtx2xyz):
    i0 = tri2vtx[idx_tri][0]
    i1 = tri2vtx[idx_tri][1]
    i2 = tri2vtx[idx_tri][2]
    p0 = vtx2xyz[i0]
    p1 = vtx2xyz[i1]
    p2 = vtx2xyz[i2]
    return r0 * p0 + r1 * p1 + (1.-r0-r1) * p2


def sample_mesh_uniform(tri2vtx, vtx2xyz):
    tri2area = del_msh.areas_of_triangles_of_mesh(tri2vtx, vtx2xyz)
    cumsum_area = numpy.cumsum(numpy.append(0., tri2area)).astype(numpy.float32)

    rad = 0.1

    samples = []
    for i in range(1000):
        smpl_i = del_msh.sample_uniform(cumsum_area, random.random(), random.random())
        pos_i = position_3d_on_triangle_in_mesh(*smpl_i, tri2vtx, vtx2xyz)
        is_near = False
        for smpl_j in samples:
            pos_j = position_3d_on_triangle_in_mesh(*smpl_j, tri2vtx, vtx2xyz)
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
        self.tri2vtx, vtx2xyz = del_msh.load_wavefront_obj_as_triangle_mesh(str(path_file))
        self.vtx2xyz = del_msh.centerize_scale_points(vtx2xyz)
        samples = sample_mesh_uniform(self.tri2vtx, self.vtx2xyz)

        edge2vtx = del_msh.edges_of_uniform_mesh(self.tri2vtx, self.vtx2xyz.shape[0])
        drawer_edge = DrawerMesh(
            vtx2xyz=self.vtx2xyz.astype(numpy.float32),
            list_elem2vtx=[
                ElementInfo(index=edge2vtx, color=(0, 0, 0), mode=moderngl.LINES),
                ElementInfo(index=self.tri2vtx, color=(1, 1, 1), mode=moderngl.TRIANGLES)
            ]
        )

        sphere_tri2vtx, shere_vtx2xyz = del_msh.sphere_meshtri3(1., 32, 32)
        self.drawer_sphere = DrawerMesh(vtx2xyz=shere_vtx2xyz, list_elem2vtx=[
            ElementInfo(index=sphere_tri2vtx, color=(1., 0., 0.), mode=moderngl.TRIANGLES)])
        self.drawer_sphere = DrawerTransformMulti(self.drawer_sphere)
        for sample in samples:
            pos_i = position_3d_on_triangle_in_mesh(*sample, self.tri2vtx, self.vtx2xyz)
            scale = pyrr.Matrix44.from_scale((0.01, 0.01, 0.01))
            translation = pyrr.Matrix44.from_translation(pos_i)
            self.drawer_sphere.list_transform.append(translation*scale)

        super().__init__()
        self.resize(640, 480)
        self.setWindowTitle('Mesh Viewer')
        self.glwidget = util_moderngl_qt.qtglwidget_viewer3.QtGLWidget_Viewer3(
            [drawer_edge, self.drawer_sphere])
        self.setCentralWidget(self.glwidget)


def main():
    with QtWidgets.QApplication([]) as app:
        win = MainWindow()
        win.show()
        app.exec()


if __name__ == "__main__":
    main()