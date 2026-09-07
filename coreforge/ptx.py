"""PTX-ядра: CUDA ALU, тензорные MMA, пропускная способность и случайный доступ к VRAM."""

# Независимые цепочки MMA в одной итерации (иначе тензор ждёт предыдущий результат).
F16_MMA_PER_ITER = 16
TF32_MMA_PER_ITER = 16
FP8_MMA_PER_ITER = 16
# FMA считаем как 2 FLOP; плюс несколько SFU не входят в эту цифру.
ALU_FLOPS_PER_ITER = 72.0

# FLOP на варп за итерацию: MMA m16n8kK = 16*8*K*2
FLOPS_F16_WARP = 16 * 8 * 16 * 2 * F16_MMA_PER_ITER
FLOPS_TF32_WARP = 16 * 8 * 8 * 2 * TF32_MMA_PER_ITER
FLOPS_FP8_WARP = 16 * 8 * 32 * 2 * FP8_MMA_PER_ITER


def _mma_lines(inst: str, chains: int, rounds: int) -> str:
    dests = (
        "{%d0, %d1, %d2, %d3}",
        "{%e0, %e1, %e2, %e3}",
        "{%f0, %f1, %f2, %f3}",
        "{%g0, %g1, %g2, %g3}",
    )[:chains]
    parts: list[str] = []
    for _ in range(rounds):
        for dest in dests:
            parts.append(f"    {inst}")
            parts.append(f"        {dest},")
            parts.append("        {%a0, %a1, %a2, %a3},")
            parts.append("        {%b0, %b1},")
            parts.append(f"        {dest};")
    return "\n".join(parts)


def build_ptx(arch: str, has_fp8: bool) -> str:
    mma_f16 = _mma_lines(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32",
        4,
        F16_MMA_PER_ITER // 4,
    )
    mma_tf32 = _mma_lines(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32",
        4,
        TF32_MMA_PER_ITER // 4,
    )
    mma_fp8 = _mma_lines(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32",
        4,
        FP8_MMA_PER_ITER // 4,
    )

    fp8 = ""
    if has_fp8:
        fp8 = f"""
.visible .entry burn_tensor_fp8(
    .param .u64 out,
    .param .u32 iters
)
{{
    .reg .pred %p;
    .reg .b32 %a0, %a1, %a2, %a3, %b0, %b1;
    .reg .f32 %d0, %d1, %d2, %d3, %e0, %e1, %e2, %e3;
    .reg .f32 %f0, %f1, %f2, %f3, %g0, %g1, %g2, %g3;
    .reg .u32 %i, %t;
    .reg .u64 %pout, %off;

    ld.param.u32 %i, [iters];
    mov.u32 %a0, 0x38383838;
    mov.u32 %a1, 0x38383838;
    mov.u32 %a2, 0x38383838;
    mov.u32 %a3, 0x38383838;
    mov.u32 %b0, 0x38383838;
    mov.u32 %b1, 0x38383838;
    mov.f32 %d0, 0f00000000;
    mov.f32 %d1, 0f00000000;
    mov.f32 %d2, 0f00000000;
    mov.f32 %d3, 0f00000000;
    mov.f32 %e0, 0f3F800000;
    mov.f32 %e1, 0f3F800000;
    mov.f32 %e2, 0f3F800000;
    mov.f32 %e3, 0f3F800000;
    mov.f32 %f0, 0f3F000000;
    mov.f32 %f1, 0f3F000000;
    mov.f32 %f2, 0f3F000000;
    mov.f32 %f3, 0f3F000000;
    mov.f32 %g0, 0f3E800000;
    mov.f32 %g1, 0f3E800000;
    mov.f32 %g2, 0f3E800000;
    mov.f32 %g3, 0f3E800000;

$Lfp8:
{mma_fp8}
    sub.u32 %i, %i, 1;
    setp.ne.u32 %p, %i, 0;
    @%p bra $Lfp8;

    ld.param.u64 %pout, [out];
    mov.u32 %t, %tid.x;
    mul.wide.u32 %off, %t, 4;
    add.u64 %pout, %pout, %off;
    add.f32 %d0, %d0, %e0;
    add.f32 %d0, %d0, %f0;
    add.f32 %d0, %d0, %g0;
    st.global.f32 [%pout], %d0;
    ret;
}}
"""

    return f"""
.version 8.5
.target {arch}
.address_size 64

.visible .entry burn_alu(
    .param .u64 out,
    .param .u32 iters
)
{{
    .reg .pred %p;
    .reg .f32 %a, %b, %c, %d, %e, %s;
    .reg .f32 %p0, %p1, %p2, %p3, %q0, %q1, %q2, %q3;
    .reg .s32 %x, %y, %z;
    .reg .u32 %i, %t, %bx, %n;
    .reg .u64 %pout, %off;

    ld.param.u32 %i, [iters];
    mov.u32 %t, %tid.x;
    mov.u32 %bx, %ctaid.x;
    mad.lo.u32 %n, %bx, 256, %t;
    cvt.rn.f32.u32 %a, %n;
    add.f32 %a, %a, 0f3F800000;
    mov.f32 %b, 0f3F80068E;
    mov.f32 %c, 0f3C23D70A;
    mov.f32 %d, 0f3F800000;
    mov.f32 %e, 0f3F000000;
    mov.f32 %p0, %a;
    mov.f32 %p1, %b;
    mov.f32 %p2, %c;
    mov.f32 %p3, %d;
    mov.f32 %q0, %e;
    mov.f32 %q1, %a;
    mov.f32 %q2, %b;
    mov.f32 %q3, %c;
    mov.s32 %x, 1;
    mov.s32 %y, 3;

$Lalu:
    fma.rn.f32 %p0, %p0, %b, %p1;
    fma.rn.f32 %q0, %q0, %c, %q1;
    fma.rn.f32 %p1, %p1, %c, %p2;
    fma.rn.f32 %q1, %q1, %d, %q2;
    fma.rn.f32 %p2, %p2, %d, %p3;
    fma.rn.f32 %q2, %q2, %e, %q3;
    fma.rn.f32 %p3, %p3, %e, %p0;
    fma.rn.f32 %q3, %q3, %a, %q0;
    fma.rn.f32 %p0, %p1, %p2, %p0;
    fma.rn.f32 %q0, %q1, %q2, %q0;
    fma.rn.f32 %p1, %p2, %p3, %p1;
    fma.rn.f32 %q1, %q2, %q3, %q1;
    fma.rn.f32 %p2, %p3, %p0, %p2;
    fma.rn.f32 %q2, %q3, %q0, %q2;
    fma.rn.f32 %p3, %p0, %p1, %p3;
    fma.rn.f32 %q3, %q0, %q1, %q3;
    fma.rn.f32 %a, %p0, %q0, %a;
    fma.rn.f32 %b, %p1, %q1, %b;
    fma.rn.f32 %c, %p2, %q2, %c;
    fma.rn.f32 %d, %p3, %q3, %d;
    fma.rn.f32 %e, %a, %b, %e;
    fma.rn.f32 %p0, %p0, %q3, %p1;
    fma.rn.f32 %q0, %q0, %p3, %q1;
    fma.rn.f32 %p1, %p1, %q0, %p2;
    fma.rn.f32 %q1, %q1, %p0, %q2;
    fma.rn.f32 %p2, %p2, %q1, %p3;
    fma.rn.f32 %q2, %q2, %p1, %q3;
    fma.rn.f32 %p3, %p3, %q2, %p0;
    fma.rn.f32 %q3, %q3, %p2, %q0;
    fma.rn.f32 %c, %a, %b, %c;
    fma.rn.f32 %a, %c, %b, %a;
    fma.rn.f32 %b, %a, %c, %b;
    sin.approx.f32 %s, %c;
    fma.rn.f32 %a, %s, %b, %a;
    cos.approx.f32 %s, %d;
    fma.rn.f32 %b, %s, %e, %b;
    ex2.approx.f32 %s, %e;
    fma.rn.f32 %d, %s, 0f3DCCCCCD, %d;
    rcp.approx.f32 %s, %b;
    fma.rn.f32 %e, %s, %c, %e;
    mad.lo.s32 %z, %x, %y, %x;
    mad.lo.s32 %x, %z, 17, %y;
    mad.lo.s32 %y, %x, 13, %z;
    sub.u32 %i, %i, 1;
    setp.ne.u32 %p, %i, 0;
    @%p bra $Lalu;

    ld.param.u64 %pout, [out];
    mul.wide.u32 %off, %t, 4;
    add.u64 %pout, %pout, %off;
    add.f32 %c, %c, %p0;
    add.f32 %c, %c, %q0;
    st.global.f32 [%pout], %c;
    ret;
}}

.visible .entry burn_tensor_f16(
    .param .u64 out,
    .param .u32 iters
)
{{
    .reg .pred %p;
    .reg .b32 %a0, %a1, %a2, %a3, %b0, %b1;
    .reg .f32 %d0, %d1, %d2, %d3, %e0, %e1, %e2, %e3;
    .reg .f32 %f0, %f1, %f2, %f3, %g0, %g1, %g2, %g3;
    .reg .u32 %i, %t;
    .reg .u64 %pout, %off;

    ld.param.u32 %i, [iters];
    mov.b32 %a0, 0x3C003C00;
    mov.b32 %a1, 0x3C003C00;
    mov.b32 %a2, 0x3C003C00;
    mov.b32 %a3, 0x3C003C00;
    mov.b32 %b0, 0x3C003C00;
    mov.b32 %b1, 0x3C003C00;
    mov.f32 %d0, 0f3F800000;
    mov.f32 %d1, 0f3F800000;
    mov.f32 %d2, 0f3F800000;
    mov.f32 %d3, 0f3F800000;
    mov.f32 %e0, 0f3F000000;
    mov.f32 %e1, 0f3F000000;
    mov.f32 %e2, 0f3F000000;
    mov.f32 %e3, 0f3F000000;
    mov.f32 %f0, 0f3E800000;
    mov.f32 %f1, 0f3E800000;
    mov.f32 %f2, 0f3E800000;
    mov.f32 %f3, 0f3E800000;
    mov.f32 %g0, 0f3F400000;
    mov.f32 %g1, 0f3F400000;
    mov.f32 %g2, 0f3F400000;
    mov.f32 %g3, 0f3F400000;

$Lf16:
{mma_f16}
    sub.u32 %i, %i, 1;
    setp.ne.u32 %p, %i, 0;
    @%p bra $Lf16;

    ld.param.u64 %pout, [out];
    mov.u32 %t, %tid.x;
    mul.wide.u32 %off, %t, 4;
    add.u64 %pout, %pout, %off;
    add.f32 %d0, %d0, %e0;
    add.f32 %d0, %d0, %f0;
    add.f32 %d0, %d0, %g0;
    st.global.f32 [%pout], %d0;
    ret;
}}

.visible .entry burn_tensor_tf32(
    .param .u64 out,
    .param .u32 iters
)
{{
    .reg .pred %p;
    .reg .b32 %a0, %a1, %a2, %a3, %b0, %b1;
    .reg .f32 %d0, %d1, %d2, %d3, %e0, %e1, %e2, %e3;
    .reg .f32 %f0, %f1, %f2, %f3, %g0, %g1, %g2, %g3;
    .reg .u32 %i, %t;
    .reg .u64 %pout, %off;

    ld.param.u32 %i, [iters];
    mov.b32 %a0, 0x3F800000;
    mov.b32 %a1, 0x3F800000;
    mov.b32 %a2, 0x3F800000;
    mov.b32 %a3, 0x3F800000;
    mov.b32 %b0, 0x3F800000;
    mov.b32 %b1, 0x3F800000;
    mov.f32 %d0, 0f00000000;
    mov.f32 %d1, 0f00000000;
    mov.f32 %d2, 0f00000000;
    mov.f32 %d3, 0f00000000;
    mov.f32 %e0, 0f3F800000;
    mov.f32 %e1, 0f3F800000;
    mov.f32 %e2, 0f3F800000;
    mov.f32 %e3, 0f3F800000;
    mov.f32 %f0, 0f3F000000;
    mov.f32 %f1, 0f3F000000;
    mov.f32 %f2, 0f3F000000;
    mov.f32 %f3, 0f3F000000;
    mov.f32 %g0, 0f3E800000;
    mov.f32 %g1, 0f3E800000;
    mov.f32 %g2, 0f3E800000;
    mov.f32 %g3, 0f3E800000;

$Ltf32:
{mma_tf32}
    sub.u32 %i, %i, 1;
    setp.ne.u32 %p, %i, 0;
    @%p bra $Ltf32;

    ld.param.u64 %pout, [out];
    mov.u32 %t, %tid.x;
    mul.wide.u32 %off, %t, 4;
    add.u64 %pout, %pout, %off;
    add.f32 %d0, %d0, %e0;
    add.f32 %d0, %d0, %f0;
    add.f32 %d0, %d0, %g0;
    st.global.f32 [%pout], %d0;
    ret;
}}

.visible .entry copy_bw(
    .param .u64 src,
    .param .u64 dst,
    .param .u64 nvec,
    .param .u32 rounds
)
{{
    .reg .pred %p, %q;
    .reg .u32 %tix, %bix, %nt, %nb, %r, %stride32;
    .reg .u64 %i, %n, %s, %d, %off, %idx, %ps, %pd, %stride;
    .reg .f32 %f0, %f1, %f2, %f3;

    mov.u32 %tix, %tid.x;
    mov.u32 %bix, %ctaid.x;
    mov.u32 %nt, %ntid.x;
    mov.u32 %nb, %nctaid.x;
    mad.lo.u32 %tix, %bix, %nt, %tix;
    cvt.u64.u32 %i, %tix;
    mul.lo.u32 %stride32, %nt, %nb;
    cvt.u64.u32 %stride, %stride32;
    ld.param.u64 %n, [nvec];
    ld.param.u32 %r, [rounds];
    ld.param.u64 %s, [src];
    ld.param.u64 %d, [dst];

$Lround:
    mov.u64 %idx, %i;
$Lcopy:
    setp.lt.u64 %p, %idx, %n;
    @!%p bra $Lnext;
    mul.lo.u64 %off, %idx, 16;
    add.u64 %ps, %s, %off;
    add.u64 %pd, %d, %off;
    ld.global.v4.f32 {{%f0, %f1, %f2, %f3}}, [%ps];
    fma.rn.f32 %f0, %f0, 0f3F800000, 0f00000000;
    st.global.v4.f32 [%pd], {{%f0, %f1, %f2, %f3}};
    add.u64 %idx, %idx, %stride;
    bra $Lcopy;
$Lnext:
    sub.u32 %r, %r, 1;
    setp.ne.u32 %q, %r, 0;
    @%q bra $Lround;
    ret;
}}

.visible .entry random_walk(
    .param .u64 buf,
    .param .u32 nvec32,
    .param .u32 rounds
)
{{
    .reg .pred %p;
    .reg .u32 %tix, %bix, %nt, %r, %x, %n, %idx32, %tmp;
    .reg .u64 %base, %off;
    .reg .f32 %v0, %v1, %v2, %v3;

    mov.u32 %tix, %tid.x;
    mov.u32 %bix, %ctaid.x;
    mov.u32 %nt, %ntid.x;
    mad.lo.u32 %x, %bix, %nt, %tix;
    add.u32 %x, %x, 2654435769;
    ld.param.u32 %n, [nvec32];
    ld.param.u32 %r, [rounds];
    ld.param.u64 %base, [buf];

$Lrw:
    shl.b32 %tmp, %x, 13;
    xor.b32 %x, %x, %tmp;
    shr.u32 %tmp, %x, 17;
    xor.b32 %x, %x, %tmp;
    shl.b32 %tmp, %x, 5;
    xor.b32 %x, %x, %tmp;
    rem.u32 %idx32, %x, %n;
    mul.wide.u32 %off, %idx32, 16;
    add.u64 %off, %base, %off;
    ld.global.v4.f32 {{%v0, %v1, %v2, %v3}}, [%off];
    fma.rn.f32 %v0, %v0, 0f3F800344, %v1;
    st.global.v4.f32 [%off], {{%v0, %v1, %v2, %v3}};
    sub.u32 %r, %r, 1;
    setp.ne.u32 %p, %r, 0;
    @%p bra $Lrw;
    ret;
}}
{fp8}
"""
