import pathlib
import moderngl
from PyQt5 import QtWidgets
import numpy

from util_moderngl_qt import DrawerMesh, DrawerCylinders, QGLWidgetViewer3
from del_msh import BVH, TriMesh


def main():
    path_file = pathlib.Path('.') / 'asset' / 'bunny_1k.obj'
    tri2vtx, vtx2xyz = TriMesh.load_wavefront_obj(str(path_file), is_centerize=True, normalized_size=1.)
    vtx2xyz1 = vtx2xyz + numpy.array([0.2, 0.2, 0.2], dtype=numpy.float32)
    tri2vtx, vtx2xyz = TriMesh.merge(tri2vtx, vtx2xyz, tri2vtx, vtx2xyz1)
    bvh, aabb = TriMesh.bvh_aabb(tri2vtx, vtx2xyz)
    edge2node2xyz, edge2tri = BVH.self_intersection_trimesh3(tri2vtx, vtx2xyz, bvh, aabb)
    # print(edge2node2xyz, edge2tri)
    #
    edge2vtx = TriMesh.edge2vtx(tri2vtx, vtx2xyz.shape[0])
    drawer_mesh = DrawerMesh.Drawer(
        vtx2xyz=vtx2xyz,
        list_elem2vtx=[
            DrawerMesh.ElementInfo(index=edge2vtx, color=(0, 0, 0), mode=moderngl.LINES),
            # DrawerMesh.ElementInfo(index=tri2vtx, color=(1, 1, 0), mode=moderngl.TRIANGLES)
        ]
    )
    #
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
