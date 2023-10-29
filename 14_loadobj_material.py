import moderngl
from PyQt5 import QtWidgets
import numpy

from pathlib import Path

from util_moderngl_qt.drawer_mesh import DrawerMesh, ElementInfo
import util_moderngl_qt.qtglwidget_viewer3
import del_msh

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

    path_dir = Path('.') / 'asset' / 'shapenet_car4' / 'models'
    vtx2xyz, vtx2uv, vtx2nrm, \
        elem2idx, idx2vtx_xyz, idx2vtx_uv, idx2vtx_nrm, \
        elem2group, group2name, \
        elem2mtl, mtl2name, mtl_file_name = del_msh.load_wavefront_obj(str(path_dir / 'model_normalized.obj'))
    vtx2xyz[:, 0] -= (vtx2xyz[:, 0].max() + vtx2xyz[:, 0].min()) * 0.5
    vtx2xyz[:, 1] -= (vtx2xyz[:, 1].max() + vtx2xyz[:, 1].min()) * 0.5
    vtx2xyz[:, 2] -= (vtx2xyz[:, 2].max() + vtx2xyz[:, 2].min()) * 0.5
    idx2vtx_xyz = idx2vtx_xyz.astype(numpy.uint64)

    dict_mtl = read_material(path_dir/ mtl_file_name)
    dict_mtl['_default'] = {'Kd': [0., 0., 0.]}

    with QtWidgets.QApplication([]) as app:
        list_elem2vtx = []
        edge2vtx = del_msh.edges_of_polygon_mesh(elem2idx, idx2vtx_xyz, vtx2xyz.shape[0])
        list_elem2vtx.append(ElementInfo(index=edge2vtx, color=(0.,0.,0.), mode=moderngl.LINES))
        for mtl_idx, mtl_name in enumerate(mtl2name):
            elem_part2idx_part, idx_part2vtx_xyz = del_msh.extract_flaged_polygonal_element(
                elem2idx, idx2vtx_xyz, elem2mtl[:] == mtl_idx)
            if len(elem_part2idx_part) == 1:
                continue
            tri_part2vtx = del_msh.triangles_from_polygon_mesh(elem_part2idx_part, idx_part2vtx_xyz)
            color = dict_mtl[mtl_name]['Kd']
            list_elem2vtx.append(ElementInfo(index=tri_part2vtx, color=color, mode=moderngl.TRIANGLES))

        vtx2nrm_ = vtx2xyz.copy()
        drawer = DrawerMesh(
            vtx2xyz=vtx2xyz,
            list_elem2vtx=list_elem2vtx
        )
        win = util_moderngl_qt.qtglwidget_viewer3.QtGLWidget_Viewer3([drawer])
        win.show()
        app.exec()




