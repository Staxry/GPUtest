"""SPIR-V compute + rayQueryKHR — без glslang."""

from __future__ import annotations

import struct


def make_rt_spv() -> bytes:
    words: list[int] = []

    def op(code: int, *operands: int) -> None:
        words.append(((1 + len(operands)) << 16) | (code & 0xFFFF))
        words.extend(int(x) & 0xFFFFFFFF for x in operands)

    def str_words(text: str) -> list[int]:
        raw = text.encode("utf-8") + b"\x00"
        raw += b"\x00" * ((4 - len(raw) % 4) % 4)
        return list(struct.unpack("<" + "I" * (len(raw) // 4), raw))

    # Result ids
    void, fn_ty, f32, u32, s32, bty = 1, 2, 3, 4, 5, 6
    v3u, v2s, v3f, v4f = 7, 8, 9, 10
    p_gid_ty, gid = 11, 12
    as_ty, p_as_ty, var_as = 13, 14, 15
    img_ty, p_img_ty, var_img = 16, 17, 18
    ubo_ty, p_ubo_ty, var_ubo = 19, 20, 21
    rq_ty, p_rq_ty = 22, 23
    main, label, rq = 24, 25, 26
    c0u, c1u, c0s, c255 = 27, 28, 29, 30
    c0f, c1f, c01, c100, cz, ch, c2f = 31, 32, 33, 34, 35, 36, 37
    true_c = 38
    gidv, gx, gy, gxf, gyf = 39, 40, 41, 42, 43
    sxi, syi, pix = 44, 45, 46
    img, asv, sz, sx, sy = 47, 48, 49, 50, 51
    fxs, fys, ux, uy, nx, ny = 52, 53, 54, 55, 56, 57
    mx, my, dx, dy = 58, 59, 60, 61
    origin, direction = 62, 63
    loop, merge, cont, proceeded = 64, 65, 66, 67
    itype, hit, shade, color = 68, 69, 70, 71
    bound = 80

    words += [0x07230203, 0x00010500, 0x00010001, bound, 0]
    op(17, 1)
    op(17, 4472)
    op(17, 25)
    op(10, *str_words("SPV_KHR_ray_query"))
    op(14, 0, 1)
    op(15, 5, main, *str_words("main"), gid, var_as, var_img, var_ubo)
    op(16, main, 17, 8, 8, 1)
    op(71, gid, 11, 28)
    op(71, var_as, 33, 0)
    op(71, var_as, 34, 0)
    op(71, var_img, 33, 0)
    op(71, var_img, 34, 1)
    op(71, var_ubo, 33, 0)
    op(71, var_ubo, 34, 2)
    op(71, ubo_ty, 2)
    op(72, ubo_ty, 0, 35, 0)
    op(72, ubo_ty, 1, 35, 4)

    op(19, void)
    op(33, fn_ty, void)
    op(22, f32, 32)
    op(21, u32, 32, 0)
    op(21, s32, 32, 1)
    op(20, bty)
    op(23, v3u, u32, 3)
    op(23, v2s, s32, 2)
    op(23, v3f, f32, 3)
    op(23, v4f, f32, 4)
    op(32, p_gid_ty, 1, v3u)
    op(59, p_gid_ty, gid, 1)
    words += [((2 << 16) | 5341), as_ty]
    op(32, p_as_ty, 0, as_ty)
    op(59, p_as_ty, var_as, 0)
    op(25, img_ty, f32, 1, 0, 0, 0, 2, 4)
    op(32, p_img_ty, 0, img_ty)
    op(59, p_img_ty, var_img, 0)
    op(30, ubo_ty, f32, f32)
    op(32, p_ubo_ty, 2, ubo_ty)
    op(59, p_ubo_ty, var_ubo, 2)
    words += [((2 << 16) | 4472), rq_ty]
    op(32, p_rq_ty, 7, rq_ty)

    op(50, u32, c0u, 0)
    op(50, u32, c1u, 1)
    op(50, s32, c0s, 0)
    op(50, u32, c255, 255)
    op(50, f32, c0f, 0)
    op(50, f32, c1f, 0x3F800000)
    op(50, f32, c01, 0x3C23D70A)
    op(50, f32, c100, 0x42C80000)
    op(50, f32, cz, 0xC1000000)
    op(50, f32, ch, 0x3F000000)
    op(50, f32, c2f, 0x40000000)
    op(41, bty, true_c)

    op(54, void, main, 0, fn_ty)
    op(248, label)
    op(59, p_rq_ty, rq, 7)

    op(61, v3u, gidv, gid)
    op(81, u32, gx, gidv, 0)
    op(81, u32, gy, gidv, 1)
    op(112, f32, gxf, gx)
    op(112, f32, gyf, gy)
    op(124, s32, sxi, gx)
    op(124, s32, syi, gy)
    op(80, v2s, pix, sxi, syi)

    op(61, img_ty, img, var_img)
    op(61, as_ty, asv, var_as)
    op(104, v2s, sz, img)
    op(81, s32, sx, sz, 0)
    op(81, s32, sy, sz, 1)
    op(111, f32, fxs, sx)
    op(111, f32, fys, sy)
    op(129, f32, ux, gxf, ch)
    op(129, f32, uy, gyf, ch)
    op(136, f32, nx, ux, fxs)
    op(136, f32, ny, uy, fys)
    op(133, f32, mx, nx, c2f)
    op(133, f32, my, ny, c2f)
    op(131, f32, dx, mx, c1f)
    op(131, f32, dy, my, c1f)
    op(80, v3f, origin, c0f, c0f, cz)
    op(80, v3f, direction, dx, dy, c1f)

    op(4473, rq, asv, c1u, c255, origin, c01, direction, c100)
    op(249, loop)
    op(248, loop)
    op(246, merge, cont, 0)
    op(249, cont)
    op(248, cont)
    op(4477, bty, proceeded, rq)
    op(250, proceeded, loop, merge)
    op(248, merge)

    op(4479, u32, itype, rq, true_c)
    op(171, bty, hit, itype, c0u)
    op(169, f32, shade, hit, c1f, c0f)
    op(80, v4f, color, shade, shade, c01, c1f)
    op(99, img, pix, color)
    op(253)
    op(56)

    return b"".join(struct.pack("<I", x & 0xFFFFFFFF) for x in words)
