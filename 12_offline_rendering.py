import pathlib
import os
#
import numpy
import pyrr
import moderngl
from PIL import Image
#
from util_moderngl_qt import DrawerMesh
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

    ctx = moderngl.create_context(standalone=True)
    ctx.polygon_offset = 1.1, 4.0
    ctx.enable(moderngl.DEPTH_TEST)
    fbo = ctx.simple_framebuffer((512, 512))
    drawer.init_gl(ctx)
    fbo.use()
    fbo.clear(0.0, 0.0, 1.0, 1.0)
    mvp = pyrr.Matrix44.identity(dtype='f4')
    drawer.paint_gl(mvp=mvp)
    # numpy.ndarray( [fbo.size[0], fbo.size[1], 3], dtype=numpy.float32)
    # print(depth)
    os.makedirs("output", exist_ok=True)
    img = Image.frombytes('RGB', fbo.size, fbo.read(), 'raw', 'RGB')
    img.save("output/rgb.png")
    #
    depth = numpy.frombuffer(fbo.read(attachment=-1, dtype='f4'), dtype=numpy.float32)
    print(depth.shape, fbo.size, depth.shape[0]/(fbo.size[0]*fbo.size[1]))
    depth = depth.reshape((3,fbo.size[0],fbo.size[1]))
    depth = depth.transpose(1,2,0)[:,:,0].copy()
    depth = (depth * 255.0).astype(numpy.uint8)
    depth = Image.fromarray(depth)
    depth.save("output/depth.png")


if __name__ == "__main__":
    main()
