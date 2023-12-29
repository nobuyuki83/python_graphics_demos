import pathlib
import moderngl
from PyQt5 import QtWidgets
import numpy

from util_moderngl_qt import DrawerMesh, DrawerCylinders, QGLWidgetViewer3
from del_msh import BVH, TriMesh


def main():
    path_file = pathlib.Path('.') / 'asset' / 'bunny_1k.obj'
    tri2vtx, vtx2xyz0 = TriMesh.load_wavefront_obj(str(path_file), is_centerize=True, normalized_size=1.)
    edge2vtx = TriMesh.edge2vtx(tri2vtx, vtx2xyz0.shape[0])
    # print(edge2vtx)
    vtx2uvw = numpy.zeros_like(vtx2xyz0)
    vtx2uvw[:, 0] += - 10 * numpy.power(vtx2xyz0[:, 0], 3) \
                     * numpy.exp(-10 * numpy.power(vtx2xyz0[:, 1], 2)) \
                     * numpy.exp(-10 * numpy.power(vtx2xyz0[:, 2], 2))
    vtx2xyz1 = vtx2xyz0 + 1.0 * vtx2uvw
    bvhnodes, roots = TriMesh.bvhnodes_vtxedgetri(edge2vtx, tri2vtx, vtx2xyz0)
    aabbs = TriMesh.aabb_vtxedgetri(edge2vtx, tri2vtx, vtx2xyz0, bvhnodes, roots, vtx2xyz1=vtx2xyz1)
    pairs, times = TriMesh.ccd_intersection_time(edge2vtx, tri2vtx, vtx2xyz0, vtx2xyz1, bvhnodes, aabbs, roots)
    intersecting_time = numpy.min(times)
    #
    vtx2xyz1 = vtx2xyz0 + intersecting_time * 0.999 * vtx2uvw
    edge2node2xyz, edge2tri = TriMesh.self_intersection(tri2vtx, vtx2xyz1, bvhnodes, aabbs, roots[2])
    assert edge2node2xyz.shape[0] == 0
    #
    vtx2xyz1 = vtx2xyz0 + intersecting_time * 1.001 * vtx2uvw
    edge2node2xyz, edge2tri = TriMesh.self_intersection(tri2vtx, vtx2xyz1, bvhnodes, aabbs, roots[2])
    assert edge2node2xyz.shape[0] != 0

    drawer_mesh = DrawerMesh.Drawer(
        vtx2xyz=vtx2xyz1,
        list_elem2vtx=[
            DrawerMesh.ElementInfo(index=edge2vtx, color=(0, 0, 0), mode=moderngl.LINES),
            # DrawerMesh.ElementInfo(index=tri2vtx, color=(1, 1, 0), mode=moderngl.TRIANGLES)
        ]
    )
    drawer_intersection = DrawerCylinders.Drawer()
    drawer_intersection.rad = 0.005
    for edge in edge2node2xyz[:]:
        drawer_intersection.list_cylinder.append(
            DrawerCylinders.CylinderInfo(pos0=edge[0], pos1=edge[1]))
    #

    with QtWidgets.QApplication([]) as app:
        glwidget = QGLWidgetViewer3.QtGLWidget_Viewer3([drawer_mesh, drawer_intersection])
        glwidget.show()
        app.exec()


if __name__ == "__main__":
    main()
