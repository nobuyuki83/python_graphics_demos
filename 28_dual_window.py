
import pathlib
#
import numpy
import pyrr
import moderngl
from PIL import Image
from PyQt5 import QtWidgets
#
from util_moderngl_qt import DrawerMesh, DrawerViewBoundary, DrawerDepthProjection, \
    OfflineRenderer, QGLWidgetViewer3
from del_msh import TriMesh




def main():
    path_file = pathlib.Path('.') / 'asset' / 'bunny_1k.obj'
    tri2vtx, vtx2xyz = TriMesh.load_wavefront_obj(str(path_file), is_centerize=True, normalized_size=1.8)
    edge2vtx = TriMesh.edge2vtx(tri2vtx, vtx2xyz.shape[0])

    drawer = DrawerMesh.Drawer(
        vtx2xyz=vtx2xyz,
        list_elem2vtx=[
            DrawerMesh.ElementInfo(index=tri2vtx, color=(1., 1., 1.), mode=moderngl.TRIANGLES),
            DrawerMesh.ElementInfo(index=edge2vtx, color=(0., 0., 0.), mode=moderngl.LINES)
        ],
    )

    mvp = pyrr.Matrix44.identity(dtype=numpy.float32)

    offline = OfflineRenderer.OfflineRenderer(width_height=(128, 512))
    drawer.init_gl(offline.ctx)
    offline.start()
    drawer.paint_gl(mvp=mvp)
    rgb = offline.get_rgb()
    depth = offline.get_depth()
    print("rgb", rgb.shape)
    print("depth", depth.shape)

    Image.fromarray(rgb).save("output/rgb0.png")
    Image.fromarray((depth * 255.0).astype(numpy.uint8)).save("output/depth0.png")

    drawer_boundary = DrawerViewBoundary.Drawer(mvp)
    drawer_depth = DrawerDepthProjection.Drawer(depth, mvp)

    def view_change_callback(event):
        mvp = win1.view_transformation_matrix_for_gl()
        offline.start()
        drawer.paint_gl(mvp=mvp)
        depth = offline.get_depth()
        drawer_boundary.mvp_inv = mvp.inverse.copy()
        drawer_depth.update_depth(depth)
        drawer_depth.mvpinv = mvp.inverse.copy()
        win0.update()


    with QtWidgets.QApplication([]) as app:
        win0 = QGLWidgetViewer3.QtGLWidget_Viewer3([drawer, drawer_boundary, drawer_depth])
        win1 = QGLWidgetViewer3.QtGLWidget_Viewer3([drawer])
        win1.viewTransformationChangeCallCack.append(view_change_callback)
        win0.show()
        win1.show()
        app.exec()


if __name__ == "__main__":
    main()


