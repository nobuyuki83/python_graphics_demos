import numpy
import pathlib
import pyrr
from util_moderngl_qt import DrawerMesh
import moderngl
from PIL import Image
from del_msh import TriMesh


def main():
    path_file = pathlib.Path('.') / 'asset' / 'bunny_1k.obj'
    tri2vtx, vtx2xyz = TriMesh.load_wavefront_obj(str(path_file))
    edge2vtx = TriMesh.edges(tri2vtx, vtx2xyz.shape[0])

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
    mvp[0][0] = 0.03
    mvp[1][1] = 0.03
    mvp[2][2] = 0.03
    drawer.paint_gl(mvp=mvp)
    # numpy.ndarray( [fbo.size[0], fbo.size[1], 3], dtype=numpy.float32)
    # print(depth)
    img = Image.frombytes('RGB', fbo.size, fbo.read(), 'raw', 'RGB', 0, -1)
    img.save("output/rgb.png")

    depth = numpy.frombuffer(fbo.read(attachment=-1, dtype='f4'), dtype=numpy.float32).reshape(fbo.size)
    depth = (depth * 255.0).astype(numpy.uint8)
    depth = Image.fromarray(depth)
    depth.save("output/depth.png")


if __name__ == "__main__":
    main()
