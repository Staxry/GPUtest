"""FSR 1: пресеты и шейдеры EASU + RCAS (алгоритм AMD FidelityFX, MIT)."""

from __future__ import annotations

FSR_MODES = (
    "Выкл",
    "Ultra Quality",
    "Quality",
    "Balanced",
    "Performance",
)

FSR_SCALE = {
    "Выкл": 1.0,
    "Ultra Quality": 1.3,
    "Quality": 1.5,
    "Balanced": 1.7,
    "Performance": 2.0,
}

RENDER_CHOICES = (
    "Авто",
    "960x540",
    "1280x720",
    "1600x900",
    "1920x1080",
)


def parse_wh(text: str) -> tuple[int, int] | None:
    if "x" not in text:
        return None
    a, b = text.lower().split("x", 1)
    try:
        w, h = int(a), int(b)
    except ValueError:
        return None
    if w < 320 or h < 180:
        return None
    return w, h


def render_size(
    out_w: int,
    out_h: int,
    mode: str,
    custom: str = "Авто",
) -> tuple[int, int]:
    out_w = max(320, out_w)
    out_h = max(180, out_h)
    if mode not in FSR_SCALE or mode == "Выкл":
        return out_w, out_h
    picked = parse_wh(custom) if custom and custom != "Авто" else None
    if picked:
        w, h = picked
    else:
        scale = FSR_SCALE[mode]
        w = int(out_w / scale)
        h = int(out_h / scale)
    w = min(max(320, w), out_w)
    h = min(max(180, h), out_h)
    if w >= out_w - 8 and h >= out_h - 8:
        scale = FSR_SCALE[mode]
        w = int(out_w / scale)
        h = int(out_h / scale)
    w -= w % 2
    h -= h % 2
    return max(320, w), max(180, h)


# Компактный порт EASU / RCAS из FidelityFX FSR 1.0 (MIT, AMD GPUOpen).
EASU_SRC = """
#version 430
uniform sampler2D uTex;
uniform vec2 uIn;
uniform vec2 uOut;
out vec4 fragColor;

float lum(vec3 c) { return c.g * 0.5 + (c.r + c.b) * 0.25; }

vec3 tap(vec2 p) {
    return textureLod(uTex, clamp((p + 0.5) / uIn, 0.0, 1.0), 0.0).rgb;
}

void accumulate(inout vec3 acc, inout float wsum, vec2 off, vec2 dir, float len, float lob, float clp, vec3 c) {
    vec2 v = vec2(dot(off, dir), dot(off, vec2(-dir.y, dir.x)));
    float d = v.x * len + v.y * v.y;
    d = min(d, clp);
    float w = lob * d - 1.0;
    w = w * w * w * w;
    acc += c * w;
    wsum += w;
}

void main() {
    vec2 src = (gl_FragCoord.xy - 0.5) * (uIn / uOut);
    vec2 fp = fract(src);
    vec2 base = floor(src);

    vec3 a0 = tap(base + vec2( 0.0, -1.0));
    vec3 a1 = tap(base + vec2( 1.0, -1.0));
    vec3 b0 = tap(base + vec2(-1.0,  0.0));
    vec3 b1 = tap(base + vec2( 0.0,  0.0));
    vec3 b2 = tap(base + vec2( 1.0,  0.0));
    vec3 b3 = tap(base + vec2( 2.0,  0.0));
    vec3 c0 = tap(base + vec2(-1.0,  1.0));
    vec3 c1 = tap(base + vec2( 0.0,  1.0));
    vec3 c2 = tap(base + vec2( 1.0,  1.0));
    vec3 c3 = tap(base + vec2( 2.0,  1.0));
    vec3 d0 = tap(base + vec2( 0.0,  2.0));
    vec3 d1 = tap(base + vec2( 1.0,  2.0));

    float m[12];
    m[0] = lum(a0); m[1] = lum(a1);
    m[2] = lum(b0); m[3] = lum(b1); m[4] = lum(b2); m[5] = lum(b3);
    m[6] = lum(c0); m[7] = lum(c1); m[8] = lum(c2); m[9] = lum(c3);
    m[10] = lum(d0); m[11] = lum(d1);

    vec2 dir;
    dir.x = (m[0] + m[2] + m[6] + m[10]) - (m[1] + m[5] + m[9] + m[11]);
    dir.y = (m[0] + m[1] + m[2] + m[5]) - (m[6] + m[9] + m[10] + m[11]);
    dir += vec2(
        (m[3] + m[6]) - (m[4] + m[9]),
        (m[2] + m[4]) - (m[7] + m[8])
    );
    float llen = max(abs(dir.x), abs(dir.y));
    dir = (llen > 1e-5) ? dir * inversesqrt(max(dot(dir, dir), 1e-8)) : vec2(1.0, 0.0);

    float len = 0.5;
    float stretch = abs(dir.x * dir.y) * 0.35;
    len = mix(1.0, 0.55, stretch);

    vec3 mn = min(min(b1, b2), min(c1, c2));
    vec3 mx = max(max(b1, b2), max(c1, c2));
    float clp = 2.0;
    float lob = 0.5 + 0.35 * (1.0 - stretch);

    vec3 acc = vec3(0.0);
    float wsum = 0.0;
    vec2 f = fp;
    accumulate(acc, wsum, vec2( 0.0 - f.x, -1.0 - f.y), dir, len, lob, clp, a0);
    accumulate(acc, wsum, vec2( 1.0 - f.x, -1.0 - f.y), dir, len, lob, clp, a1);
    accumulate(acc, wsum, vec2(-1.0 - f.x,  0.0 - f.y), dir, len, lob, clp, b0);
    accumulate(acc, wsum, vec2( 0.0 - f.x,  0.0 - f.y), dir, len, lob, clp, b1);
    accumulate(acc, wsum, vec2( 1.0 - f.x,  0.0 - f.y), dir, len, lob, clp, b2);
    accumulate(acc, wsum, vec2( 2.0 - f.x,  0.0 - f.y), dir, len, lob, clp, b3);
    accumulate(acc, wsum, vec2(-1.0 - f.x,  1.0 - f.y), dir, len, lob, clp, c0);
    accumulate(acc, wsum, vec2( 0.0 - f.x,  1.0 - f.y), dir, len, lob, clp, c1);
    accumulate(acc, wsum, vec2( 1.0 - f.x,  1.0 - f.y), dir, len, lob, clp, c2);
    accumulate(acc, wsum, vec2( 2.0 - f.x,  1.0 - f.y), dir, len, lob, clp, c3);
    accumulate(acc, wsum, vec2( 0.0 - f.x,  2.0 - f.y), dir, len, lob, clp, d0);
    accumulate(acc, wsum, vec2( 1.0 - f.x,  2.0 - f.y), dir, len, lob, clp, d1);
    vec3 col = acc / max(wsum, 1e-6);
    col = clamp(col, mn * 0.92, mx * 1.08);
    fragColor = vec4(col, 1.0);
}
"""

RCAS_SRC = """
#version 430
uniform sampler2D uTex;
uniform vec2 uRes;
uniform float uSharp;
out vec4 fragColor;

void main() {
    vec2 uv = gl_FragCoord.xy / uRes;
    vec2 t = 1.0 / uRes;
    vec3 n = textureLod(uTex, uv + vec2(0.0, t.y), 0.0).rgb;
    vec3 e = textureLod(uTex, uv + vec2(t.x, 0.0), 0.0).rgb;
    vec3 s = textureLod(uTex, uv - vec2(0.0, t.y), 0.0).rgb;
    vec3 w = textureLod(uTex, uv - vec2(t.x, 0.0), 0.0).rgb;
    vec3 m = textureLod(uTex, uv, 0.0).rgb;
    float mn = min(min(n.g, e.g), min(s.g, w.g));
    mn = min(mn, min(min(n.r, e.r), min(s.r, w.r)));
    mn = min(mn, min(min(n.b, e.b), min(s.b, w.b)));
    mn = min(mn, min(min(m.r, m.g), m.b));
    float mx = max(max(n.g, e.g), max(s.g, w.g));
    mx = max(mx, max(max(n.r, e.r), max(s.r, w.r)));
    mx = max(mx, max(max(n.b, e.b), max(s.b, w.b)));
    mx = max(mx, max(max(m.r, m.g), m.b));
    float lobe = exp2(-clamp(uSharp, 0.0, 2.0)) * 0.25;
    float hit = min(mn, 1.0 - mx) / max(mx, 1e-4);
    lobe *= clamp(hit * 8.0, 0.0, 1.0);
    vec3 col = (n + e + s + w) * (-lobe) + m * (1.0 + 4.0 * lobe);
    fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
"""

BLIT_SRC = """
#version 430
uniform sampler2D uTex;
uniform vec2 uRes;
out vec4 fragColor;
void main() {
    fragColor = textureLod(uTex, gl_FragCoord.xy / uRes, 0.0);
}
"""
