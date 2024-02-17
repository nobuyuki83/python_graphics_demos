import numpy
import moderngl
from PyQt5 import QtWidgets
from util_moderngl_qt import DrawerMeshColorMap, QGLWidgetViewer3, Colormap
from del_msh import TriMesh
import del_ls
import del_fem


def main():
    tri2vtx, vtx2xyz = TriMesh.cylinder(radius=0.3, height=1.8, ndiv_height=32)
    vtx2idx, idx2vtx = TriMesh.vtx2vtx(tri2vtx, vtx2xyz.shape[0])
    sparse = del_ls.SparseSquareMatrix(vtx2idx, idx2vtx)
    sparse.set_zero()
    from del_fem.del_fem import merge_hessian_mesh_laplacian_on_trimesh3
    merge_hessian_mesh_laplacian_on_trimesh3(
        tri2vtx, vtx2xyz,
        sparse.row2idx, sparse.idx2col, sparse.row2val, sparse.idx2val)
    r_vec = numpy.zeros((vtx2xyz.shape[0]), dtype=numpy.float64)
    penalty = 1.0e+3
    for iv in range(vtx2xyz.shape[0]):
        y = vtx2xyz[iv][1]
        if y < -0.89:
            r_vec[iv] = penalty * 0.
            sparse.row2val[iv] += penalty
        if y > +0.89:
            r_vec[iv] = penalty * 1.
            sparse.row2val[iv] += penalty
    vtx2val, conv_hist = sparse.solve_cg(r_vec)
    print("cg_iterations =", len(conv_hist))

    edge2vtx = TriMesh.edge2vtx(tri2vtx, vtx2xyz.shape[0])
    drawer_edge = DrawerMeshColorMap.Drawer(
        vtx2xyz=vtx2xyz,
        list_elem2vtx=[
            DrawerMeshColorMap.ElementInfo(index=edge2vtx, color=(0, 0, 0), mode=moderngl.LINES),
            DrawerMeshColorMap.ElementInfo(index=tri2vtx, color=(1, 1, 1), mode=moderngl.TRIANGLES)
        ],
        vtx2val=vtx2val,
        color_map=Colormap.heat()
    )

    with QtWidgets.QApplication([]) as app:
        win = QGLWidgetViewer3.QtGLWidget_Viewer3(
            [drawer_edge])
        win.show()
        app.exec()


if __name__ == "__main__":
    main()
