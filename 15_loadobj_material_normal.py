import moderngl
from PyQt5 import QtWidgets
import numpy

from pathlib import Path

from util_moderngl_qt.drawer_mesh_normal import DrawerMeshNormal, ElementInfo
import util_moderngl_qt.qtglwidget_viewer3
import del_msh
from del_msh import WavefrontObj


def read_material(path):
    with open(str(path)) as f:
        dict_mtl = {}
        cur_mtl = {}
        cur_name = ""
        for line in f:
            if line.startswith('#'):
                continue
            words = line.split()
            if len(words) == 2 and words[0] == 'newmtl':
                cur_name = words[1]
                cur_mtl = {}
            if len(words) == 0 and cur_name != "":
                dict_mtl[cur_name] = cur_mtl
            if len(words) == 4 and words[0] == 'Kd':
                cur_mtl['Kd'] = (float(words[1]), float(words[2]), float(words[3]))
    return dict_mtl


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

    dict_mtl = read_material(path_dir / obj.mtl_file_name)
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
            list_elem2vtx.append(ElementInfo(index=tri_part2vtx, color=color, mode=moderngl.TRIANGLES))
        drawer = DrawerMeshNormal(
            vtx2xyz=uni2xyz,
            vtx2nrm=uni2nrm,
            list_elem2vtx=list_elem2vtx
        )
        win = util_moderngl_qt.qtglwidget_viewer3.QtGLWidget_Viewer3([drawer])
        win.show()
        app.exec()
