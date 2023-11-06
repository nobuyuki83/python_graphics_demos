import moderngl
from PyQt5 import QtWidgets
import numpy

from pathlib import Path

from util_moderngl_qt import DrawerMeshNormal, QGLWidgetViewer3
import del_msh
from del_msh import WavefrontObj

if __name__ == "__main__":

    path_dir = Path('.') / 'asset' / 'shapenet_car1' / 'models'
    obj = WavefrontObj.load(str(path_dir / 'model_normalized.obj'))

    idx2uni, uni2vtxxyz, uni2vtxnrm = del_msh.unify_two_indices_of_polygon_mesh(
        obj.elem2idx, obj.idx2vtxxyz, obj.idx2vtxnrm)

    uni2xyz = numpy.ndarray((uni2vtxxyz.shape[0], 3), obj.vtxxyz2xyz.dtype)
    uni2xyz[:, :] = obj.vtxxyz2xyz[uni2vtxxyz[:], :]

    uni2nrm = numpy.ndarray((uni2vtxnrm.shape[0], 3), obj.vtxnrm2nrm.dtype)
    uni2nrm[:, :] = obj.vtxnrm2nrm[uni2vtxnrm[:], :]

    del_msh.centerize_scale_points(uni2xyz, scale=1.8)

    dict_mtl = WavefrontObj.read_material(str(path_dir / obj.mtl_file_name))
    dict_mtl['_default'] = {'Kd': [0., 0., 0.]}

    with QtWidgets.QApplication([]) as app:
        list_elem2vtx = []
        # edge2vtx = del_msh.edges_of_polygon_mesh(obj.elem2idx, idx2uni, uni2xyz.shape[0])
        # list_elem2vtx.append(ElementInfo(index=edge2vtx, color=(0., 0., 0.), mode=moderngl.LINES))
        for mtl_idx, mtl_name in enumerate(obj.mtl2name):
            elem_part2idx_part, idx_part2uni = del_msh.extract_flaged_polygonal_element(
                obj.elem2idx, idx2uni, obj.elem2mtl[:] == mtl_idx)
            if len(elem_part2idx_part) == 1:
                continue
            tri_part2vtx = del_msh.triangles_from_polygon_mesh(elem_part2idx_part, idx_part2uni)
            color = dict_mtl[mtl_name]['Kd']
            list_elem2vtx.append(DrawerMeshNormal.ElementInfo(index=tri_part2vtx, color=color, mode=moderngl.TRIANGLES))
        drawer = DrawerMeshNormal.Drawer(
            vtx2xyz=uni2xyz,
            vtx2nrm=uni2nrm,
            list_elem2vtx=list_elem2vtx
        )
        win = QGLWidgetViewer3.QtGLWidget_Viewer3([drawer])
        win.show()
        app.exec()
