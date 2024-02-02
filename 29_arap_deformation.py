import math
#
import numpy
import moderngl
import pyrr
from PyQt5 import QtWidgets
from PyQt5.QtCore import QTimer
#
from util_moderngl_qt import DrawerMesh, QGLWidgetViewer3
from del_msh import TriMesh
import del_ls


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        tri2vtx, self.vtx2xyz_ini = TriMesh.capsule(
            radius=0.2, height=1.6,
            ndiv_height=15, ndiv_theta=15, ndiv_longtitude=2)
        self.vtx2rot = numpy.zeros((self.vtx2xyz_ini.shape[0], 3, 3))
        self.vtxs0 = []
        self.vtxs1 = []
        for iv in range(self.vtx2xyz_ini.shape[0]):
            y = self.vtx2xyz_ini[iv, 1]
            if y < -0.65:
                self.vtxs0.append(iv)
            if y > +0.65:
                self.vtxs1.append(iv)
        self.vtx2xyz_def = self.vtx2xyz_ini.copy()
        #
        vtx2idx, idx2vtx = TriMesh.vtx2vtx(tri2vtx, self.vtx2xyz_ini.shape[0])
        # print(vtx2idx, idx2vtx)
        self.sparse = del_ls.SparseSquareMatrix(vtx2idx, idx2vtx)
        self.sparse.set_zero()
        TriMesh.merge_hessian_mesh_laplacian(
            tri2vtx, self.vtx2xyz_ini,
            self.sparse.row2idx, self.sparse.idx2col, self.sparse.row2val, self.sparse.idx2val)
        self.r_vec = numpy.zeros_like(self.vtx2xyz_ini)
        self.penalty = 1.0e+2
        self.sparse.row2val[self.vtxs0] += self.penalty
        self.sparse.row2val[self.vtxs1] += self.penalty
        #
        edge2vtx = TriMesh.edge2vtx(tri2vtx, self.vtx2xyz_ini.shape[0])
        self.drawer_edge = DrawerMesh.Drawer(
            vtx2xyz=self.vtx2xyz_def,
            list_elem2vtx=[
                DrawerMesh.ElementInfo(index=edge2vtx, color=(0, 0, 0), mode=moderngl.LINES),
                DrawerMesh.ElementInfo(index=tri2vtx, color=(1, 1, 1), mode=moderngl.TRIANGLES)
            ]
        )

        super().__init__()
        self.resize(640, 480)
        self.setWindowTitle('Mesh Viewer')
        self.glwidget = QGLWidgetViewer3.QtGLWidget_Viewer3(
            [self.drawer_edge])
        self.setCentralWidget(self.glwidget)
        #
        self.timer = QTimer()
        self.timer.setInterval(30)
        self.timer.timeout.connect(self.step_time)
        self.timer.start()
        self.iframe = 0

    def step_time(self):
        iframe = self.iframe
        t0 = pyrr.Matrix44.from_translation((0., -0.6, 0.))
        ry = pyrr.Matrix44.from_y_rotation(0.3 * math.sin(0.05 * iframe))
        rz = pyrr.Matrix44.from_z_rotation(0.8 * math.sin(0.03 * iframe))
        t1 = pyrr.Matrix44.from_translation((0.4 * math.sin(0.03 * iframe), +0.4 + 0.2 * math.cos(0.03 * iframe), 0.))
        # t0 = pyrr.Matrix44.from_translation((0., -0.8, 0.))
        # ry = pyrr.Matrix44.from_y_rotation(0.5 * math.sin(0.05 * iframe))
        # rz = pyrr.Matrix44.from_z_rotation(0.3 * math.sin(0.07 * iframe))
        # t1 = pyrr.Matrix44.from_translation((0.4 * math.sin(0.03 * iframe), +0.6 + 0.3 * math.cos(0.05 * iframe), 0.))
        a = t1 * rz * ry * t0
        a = a.transpose()
        self.vtx2xyz_def[self.vtxs0] = self.vtx2xyz_ini[self.vtxs0]
        # affine transformation
        self.vtx2xyz_def[self.vtxs1] = a.matrix33.dot(self.vtx2xyz_ini[self.vtxs1].transpose()).transpose()
        self.vtx2xyz_def[self.vtxs1] += numpy.array([a.m14, a.m24, a.m34])
        for iter in range(0, 1):
            # the boundary condition are computed
            from del_msh.del_msh import optimal_rotation_for_vertex
            optimal_rotation_for_vertex(
                self.vtx2xyz_ini, self.vtx2xyz_def,
                self.sparse.row2idx, self.sparse.idx2col, self.sparse.idx2val,
                self.vtx2rot)
            # the rotation for vertex is computed
            # the following three lines can be pre-computed
            self.r_vec.fill(0.)
            for i_vtx in range(0, self.vtx2xyz_ini.shape[0]):
                for idx in range(self.sparse.row2idx[i_vtx], self.sparse.row2idx[i_vtx + 1]):
                    j_vtx = self.sparse.idx2col[idx]
                    weight = -self.sparse.idx2val[idx]
                    # todo I don't know why transpose is necessary
                    m_i = weight * 0.5 * (self.vtx2rot[i_vtx] + self.vtx2rot[j_vtx]).transpose()
                    # m_i = weight * numpy.eye(3)
                    r_i = m_i.dot(self.vtx2xyz_ini[i_vtx] - self.vtx2xyz_ini[j_vtx])
                    self.r_vec[i_vtx] += r_i
            # self.sparse.general_mult(+1.0, self.vtx2xyz_ini, +0.0, self.r_vec)
            # self.r_vec[self.vtxs0] -= self.penalty * self.vtx2xyz_ini[self.vtxs0]
            # self.r_vec[self.vtxs1] -= self.penalty * self.vtx2xyz_ini[self.vtxs1]
            #
            self.sparse.general_mult(-1.0, self.vtx2xyz_def, 1.0, self.r_vec)
            self.r_vec[self.vtxs0] += self.penalty * self.vtx2xyz_def[self.vtxs0]
            self.r_vec[self.vtxs1] += self.penalty * self.vtx2xyz_def[self.vtxs1]
            # print(iframe,numpy.linalg.norm(self.r_vec))
            #
            x_vec, conv_hist = self.sparse.solve_cg(self.r_vec, conv_ratio_tol=1.0e-5)
            print("cg_iterations =", len(conv_hist), conv_hist[len(conv_hist) - 1])
            # todo why factor 2.0 comes from (probably it is from each edge is adjacent to two triangles)
            self.vtx2xyz_def += x_vec * 2.0
        #
        self.drawer_edge.update_position(self.vtx2xyz_def)
        self.iframe += 1
        self.glwidget.update()


def main():
    with QtWidgets.QApplication([]) as app:
        win = MainWindow()
        win.show()
        app.exec()


if __name__ == "__main__":
    main()
