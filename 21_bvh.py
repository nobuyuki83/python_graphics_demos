import pathlib

import numpy
import moderngl
from PyQt5 import QtWidgets

from util_moderngl_qt import DrawerMesh, DrawerMeshUnindex, QGLWidgetViewer3
from del_msh import BVH, TriMesh


def main(is_morton: bool):
    path_file = pathlib.Path('.') / 'asset' / 'bunny_1k.obj'
    tri2vtx, vtx2xyz = TriMesh.load_wavefront_obj(str(path_file), is_centerize=True, normalized_size=1.)
    bvhnodes = TriMesh.bvhnodes_tri(tri2vtx, vtx2xyz, is_morton)
    aabbs = TriMesh.aabbs_tri(tri2vtx, vtx2xyz, bvhnodes)
    #
    edge2vtx = TriMesh.edge2vtx(tri2vtx, vtx2xyz.shape[0])
    drawer_mesh = DrawerMesh.Drawer(
        vtx2xyz=vtx2xyz,
        list_elem2vtx=[
            DrawerMesh.ElementInfo(index=edge2vtx, color=(0, 0, 0), mode=moderngl.LINES),
            DrawerMesh.ElementInfo(index=tri2vtx, color=(1, 1, 0), mode=moderngl.TRIANGLES)
        ]
    )
    #
    aabb_edges = BVH.edges_of_aabb(aabbs)
    drawer_aabb = DrawerMeshUnindex.Drawer(elem2node2xyz=aabb_edges)
    #
    with QtWidgets.QApplication([]) as app:
        glwidget = QGLWidgetViewer3.QtGLWidget_Viewer3([drawer_mesh, drawer_aabb])
        glwidget.show()
        app.exec()


if __name__ == "__main__":
    main(False)
    main(True)
