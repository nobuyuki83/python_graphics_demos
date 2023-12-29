import pathlib
import moderngl
from PyQt5 import QtWidgets
import numpy

from util_moderngl_qt import DrawerMesh, DrawerCylinders, QGLWidgetViewer3
from del_msh import BVH, TriMesh


def main():
    tri2vtx, vtx2xyz0 = TriMesh.sphere(1., ndiv_latitude=32, ndiv_longtitude=32)
    edge2vtx = TriMesh.edge2vtx(tri2vtx, vtx2xyz0.shape[0])
    # print(edge2vtx)
    vtx2uvw = numpy.zeros_like(vtx2xyz0)
    vtx2uvw[:, 0] += -numpy.power(vtx2xyz0[:, 0], 3)
    vtx2xyz1 = vtx2xyz0 + 0.995 * vtx2uvw

    contacting_pair, contacting_coord = TriMesh.contacting_pair(tri2vtx, vtx2xyz1, edge2vtx, 0.01)
    print(contacting_pair, contacting_coord)
    print(contacting_pair.shape, contacting_coord.shape)

    drawer_mesh = DrawerMesh.Drawer(
        vtx2xyz=vtx2xyz1,
        list_elem2vtx=[
            DrawerMesh.ElementInfo(index=edge2vtx, color=(0, 0, 0), mode=moderngl.LINES),
            # DrawerMesh.ElementInfo(index=tri2vtx, color=(1, 1, 0), mode=moderngl.TRIANGLES)
        ]
    )
    #
    drawer_intersection = DrawerCylinders.Drawer()
    drawer_intersection.rad = 0.003
    for ic in range(len(contacting_pair)):
        if contacting_pair[ic, 2] == 1:  # triangle - vertex pair
            it = contacting_pair[ic, 0]
            iv = contacting_pair[ic, 1]
            ap = vtx2xyz1[tri2vtx[it, :], :]
            p = contacting_coord[ic, :3].dot(ap)
            q = vtx2xyz1[iv, :]
            drawer_intersection.list_cylinder.append(
                DrawerCylinders.CylinderInfo(pos0=p, pos1=q))
        elif contacting_pair[ic, 2] == 0:  # edge - edge pair
            ie0 = contacting_pair[ic, 0]
            ie1 = contacting_pair[ic, 1]
            ap0 = vtx2xyz1[edge2vtx[ie0, :], :]
            ap1 = vtx2xyz1[edge2vtx[ie1, :], :]
            p0 = contacting_coord[ic, 0:2].dot(ap0)
            p1 = contacting_coord[ic, 2:4].dot(ap1)
            drawer_intersection.list_cylinder.append(
                DrawerCylinders.CylinderInfo(pos0=ap0[0], pos1=ap0[1], color=(0., 1., 0.)))
            drawer_intersection.list_cylinder.append(
                DrawerCylinders.CylinderInfo(pos0=ap1[0], pos1=ap1[1]))

    with QtWidgets.QApplication([]) as app:
        glwidget = QGLWidgetViewer3.QtGLWidget_Viewer3([drawer_mesh, drawer_intersection])
        glwidget.show()
        app.exec()


if __name__ == "__main__":
    main()
