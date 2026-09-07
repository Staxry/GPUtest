"""Окно шейдерной нагрузки: raymarch + compute."""

from __future__ import annotations

import ctypes
import time
from ctypes import wintypes
from typing import Callable

from coreforge.assets import icon_ico

SCENE_NAMES = (
    "туннель",
    "фрактал",
    "сияние",
    "город",
    "океан",
    "кристаллы",
    "плазма",
)
SCENE_PERIOD = 14.0

user32 = ctypes.WinDLL("user32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
opengl32 = ctypes.WinDLL("opengl32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

WM_DESTROY = 0x0002
WM_SIZE = 0x0005
WM_PAINT = 0x000F
WM_CLOSE = 0x0010
WM_QUIT = 0x0012
WM_ERASEBKGND = 0x0014
WM_SETICON = 0x0080
ICON_SMALL = 0
ICON_BIG = 1
IMAGE_ICON = 1
LR_LOADFROMFILE = 0x0010
PM_REMOVE = 0x0001
CS_OWNDC = 0x0020
WS_OVERLAPPEDWINDOW = 0x00CF0000
WS_VISIBLE = 0x10000000
CW_USEDEFAULT = 0x80000000
PFD_DRAW_TO_WINDOW = 0x00000004
PFD_SUPPORT_OPENGL = 0x00000020
PFD_DOUBLEBUFFER = 0x00000001
PFD_TYPE_RGBA = 0
PFD_MAIN_PLANE = 0
GL_VERTEX_SHADER = 0x8B31
GL_FRAGMENT_SHADER = 0x8B30
GL_COMPUTE_SHADER = 0x91B9
GL_COMPILE_STATUS = 0x8B81
GL_LINK_STATUS = 0x8B82
GL_INFO_LOG_LENGTH = 0x8B84
GL_COLOR_BUFFER_BIT = 0x00004000
GL_TRIANGLES = 0x0004
GL_TEXTURE_2D = 0x0DE1
GL_RGBA8 = 0x8058
GL_RGBA = 0x1908
GL_UNSIGNED_BYTE = 0x1401
GL_TEXTURE_MIN_FILTER = 0x2801
GL_TEXTURE_MAG_FILTER = 0x2800
GL_LINEAR = 0x2601
GL_CLAMP_TO_EDGE = 0x812F
GL_TEXTURE_WRAP_S = 0x2802
GL_TEXTURE_WRAP_T = 0x2803
GL_WRITE_ONLY = 0x88B9
GL_SHADER_IMAGE_ACCESS_BARRIER_BIT = 0x00000020
GL_BLEND = 0x0BE2
GL_SRC_ALPHA = 0x0302
GL_ONE_MINUS_SRC_ALPHA = 0x0303
GL_BGRA = 0x80E1
GL_TEXTURE0 = 0x84C0
WGL_CONTEXT_MAJOR_VERSION_ARB = 0x2091
WGL_CONTEXT_MINOR_VERSION_ARB = 0x2092
WGL_CONTEXT_PROFILE_MASK_ARB = 0x9126
WGL_CONTEXT_CORE_PROFILE_BIT_ARB = 0x00000001

LRESULT = ctypes.c_ssize_t
WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)


class WNDCLASSEXW(ctypes.Structure):
    _fields_ = [
        ("cbSize", ctypes.c_uint),
        ("style", ctypes.c_uint),
        ("lpfnWndProc", WNDPROC),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", wintypes.HINSTANCE),
        ("hIcon", wintypes.HICON),
        ("hCursor", wintypes.HCURSOR),
        ("hbrBackground", wintypes.HBRUSH),
        ("lpszMenuName", wintypes.LPCWSTR),
        ("lpszClassName", wintypes.LPCWSTR),
        ("hIconSm", wintypes.HICON),
    ]


class PIXELFORMATDESCRIPTOR(ctypes.Structure):
    _fields_ = [
        ("nSize", wintypes.WORD),
        ("nVersion", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("iPixelType", ctypes.c_ubyte),
        ("cColorBits", ctypes.c_ubyte),
        ("cRedBits", ctypes.c_ubyte),
        ("cRedShift", ctypes.c_ubyte),
        ("cGreenBits", ctypes.c_ubyte),
        ("cGreenShift", ctypes.c_ubyte),
        ("cBlueBits", ctypes.c_ubyte),
        ("cBlueShift", ctypes.c_ubyte),
        ("cAlphaBits", ctypes.c_ubyte),
        ("cAlphaShift", ctypes.c_ubyte),
        ("cAccumBits", ctypes.c_ubyte),
        ("cAccumRedBits", ctypes.c_ubyte),
        ("cAccumGreenBits", ctypes.c_ubyte),
        ("cAccumBlueBits", ctypes.c_ubyte),
        ("cAccumAlphaBits", ctypes.c_ubyte),
        ("cDepthBits", ctypes.c_ubyte),
        ("cStencilBits", ctypes.c_ubyte),
        ("cAuxBuffers", ctypes.c_ubyte),
        ("iLayerType", ctypes.c_ubyte),
        ("bReserved", ctypes.c_ubyte),
        ("dwLayerMask", wintypes.DWORD),
        ("dwVisibleMask", wintypes.DWORD),
        ("dwDamageMask", wintypes.DWORD),
    ]


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


class MSG(ctypes.Structure):
    _fields_ = [
        ("hwnd", wintypes.HWND),
        ("message", wintypes.UINT),
        ("wParam", wintypes.WPARAM),
        ("lParam", wintypes.LPARAM),
        ("time", wintypes.DWORD),
        ("pt", POINT),
    ]


user32.RegisterClassExW.argtypes = [ctypes.POINTER(WNDCLASSEXW)]
user32.RegisterClassExW.restype = wintypes.ATOM
user32.CreateWindowExW.argtypes = [
    wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID,
]
user32.CreateWindowExW.restype = wintypes.HWND
user32.DefWindowProcW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
user32.DefWindowProcW.restype = LRESULT
user32.PeekMessageW.argtypes = [ctypes.POINTER(MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT, wintypes.UINT]
user32.PeekMessageW.restype = wintypes.BOOL
user32.TranslateMessage.argtypes = [ctypes.POINTER(MSG)]
user32.DispatchMessageW.argtypes = [ctypes.POINTER(MSG)]
user32.GetDC.argtypes = [wintypes.HWND]
user32.GetDC.restype = wintypes.HDC
user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
user32.DestroyWindow.argtypes = [wintypes.HWND]
user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
user32.SetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPCWSTR]
user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
user32.LoadCursorW.argtypes = [wintypes.HINSTANCE, ctypes.c_void_p]
user32.LoadCursorW.restype = wintypes.HCURSOR
user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(RECT)]
user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(RECT)]
user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(POINT)]
user32.SetWindowPos.argtypes = [
    wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.UINT,
]
user32.SetLayeredWindowAttributes.argtypes = [wintypes.HWND, wintypes.DWORD, ctypes.c_ubyte, wintypes.DWORD]
user32.FillRect.argtypes = [wintypes.HDC, ctypes.POINTER(RECT), wintypes.HBRUSH]
user32.DrawTextW.argtypes = [wintypes.HDC, wintypes.LPCWSTR, ctypes.c_int, ctypes.POINTER(RECT), wintypes.UINT]
user32.ValidateRect.argtypes = [wintypes.HWND, ctypes.c_void_p]
user32.ValidateRect.restype = wintypes.BOOL
user32.BeginPaint.argtypes = [wintypes.HWND, ctypes.c_void_p]
user32.BeginPaint.restype = wintypes.HDC
user32.EndPaint.argtypes = [wintypes.HWND, ctypes.c_void_p]
user32.LoadImageW.argtypes = [
    wintypes.HINSTANCE, wintypes.LPCWSTR, wintypes.UINT, ctypes.c_int, ctypes.c_int, wintypes.UINT,
]
user32.LoadImageW.restype = wintypes.HANDLE
user32.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
user32.SendMessageW.restype = LRESULT
gdi32.CreateFontW.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.DWORD,
    wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD,
    wintypes.DWORD, wintypes.DWORD, wintypes.LPCWSTR,
]
gdi32.CreateFontW.restype = wintypes.HFONT
gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
gdi32.SelectObject.restype = wintypes.HGDIOBJ
gdi32.SetTextColor.argtypes = [wintypes.HDC, wintypes.COLORREF]
gdi32.SetBkMode.argtypes = [wintypes.HDC, ctypes.c_int]
gdi32.CreateSolidBrush.argtypes = [wintypes.COLORREF]
gdi32.CreateSolidBrush.restype = wintypes.HBRUSH
gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
gdi32.CreateCompatibleDC.restype = wintypes.HDC
gdi32.DeleteDC.argtypes = [wintypes.HDC]
gdi32.CreateDIBSection.argtypes = [
    wintypes.HDC, ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint,
]
gdi32.CreateDIBSection.restype = wintypes.HBITMAP


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", ctypes.c_uint32),
        ("biWidth", ctypes.c_int32),
        ("biHeight", ctypes.c_int32),
        ("biPlanes", ctypes.c_ushort),
        ("biBitCount", ctypes.c_ushort),
        ("biCompression", ctypes.c_uint32),
        ("biSizeImage", ctypes.c_uint32),
        ("biXPelsPerMeter", ctypes.c_int32),
        ("biYPelsPerMeter", ctypes.c_int32),
        ("biClrUsed", ctypes.c_uint32),
        ("biClrImportant", ctypes.c_uint32),
    ]

WS_EX_LAYERED = 0x00080000
WS_EX_NOACTIVATE = 0x08000000
WS_EX_TOPMOST = 0x00000008
WS_EX_TOOLWINDOW = 0x00000080
WS_POPUP = 0x80000000
LWA_ALPHA = 0x00000002
SWP_NOACTIVATE = 0x0010
SWP_SHOWWINDOW = 0x0040
HWND_TOPMOST = wintypes.HWND(-1)
DT_LEFT = 0x0000
DT_WORDBREAK = 0x0010
TRANSPARENT_BK = 1
HUD_H = 198
GL_FRAMEBUFFER = 0x8D40
GL_COLOR_ATTACHMENT0 = 0x8CE0
GL_FRAMEBUFFER_COMPLETE = 0x8CD5
GL_NEAREST = 0x2600

gdi32.ChoosePixelFormat.argtypes = [wintypes.HDC, ctypes.POINTER(PIXELFORMATDESCRIPTOR)]
gdi32.SetPixelFormat.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.POINTER(PIXELFORMATDESCRIPTOR)]
gdi32.SwapBuffers.argtypes = [wintypes.HDC]

opengl32.wglCreateContext.argtypes = [wintypes.HDC]
opengl32.wglCreateContext.restype = ctypes.c_void_p
opengl32.wglMakeCurrent.argtypes = [wintypes.HDC, ctypes.c_void_p]
opengl32.wglMakeCurrent.restype = wintypes.BOOL
opengl32.wglDeleteContext.argtypes = [ctypes.c_void_p]
opengl32.wglGetProcAddress.argtypes = [ctypes.c_char_p]
opengl32.wglGetProcAddress.restype = ctypes.c_void_p
opengl32.wglGetCurrentContext.restype = ctypes.c_void_p
kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
kernel32.GetModuleHandleW.restype = wintypes.HINSTANCE

IDC_ARROW = 32512
SW_SHOW = 5

VERT_SRC = """
#version 430
void main() {
    vec2 verts[3] = vec2[](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    gl_Position = vec4(verts[gl_VertexID], 0.0, 1.0);
}
"""

HUD_FRAG_SRC = """
#version 430
uniform sampler2D uTex;
uniform vec2 uRes;
uniform float uHudH;
out vec4 fragColor;
void main() {
    float top = uRes.y - uHudH;
    if (gl_FragCoord.y < top) discard;
    vec2 uv = vec2(gl_FragCoord.x / max(uRes.x, 1.0), (gl_FragCoord.y - top) / max(uHudH, 1.0));
    uv.y = 1.0 - uv.y;
    fragColor = texture(uTex, uv);
}
"""

FRAG_SRC = """
#version 430
uniform float uTime;
uniform vec2 uRes;
uniform float uWork;
uniform int uScene;
out vec4 fragColor;

float hash13(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return fract((p.x + p.y) * p.z);
}

float sdBox(vec3 p, vec3 b) {
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

float mapTunnel(vec3 p) {
    p.z += uTime * 2.2;
    float r = 1.15 + 0.12 * sin(p.z * 1.8) + 0.06 * sin(p.z * 4.0 + uTime);
    float tube = length(p.xy) - r;
    float rings = abs(fract(p.z * 0.35) - 0.5) - 0.04;
    float ribs = length(vec2(length(p.xy) - r, rings)) - 0.035;
    vec3 q = p;
    q.xy = abs(q.xy);
    float beams = sdBox(q - vec3(0.95, 0.0, 0.0), vec3(0.04, 0.04, 40.0));
    return min(abs(tube) * 0.55, min(ribs, beams));
}

float mapFractal(vec3 p) {
    p.yz = mat2(cos(uTime * 0.15), -sin(uTime * 0.15), sin(uTime * 0.15), cos(uTime * 0.15)) * p.yz;
    float s = 1.0;
    for (int i = 0; i < 7; i++) {
        p = abs(p) - vec3(0.85, 0.62, 0.48);
        float a = 0.55;
        float c = cos(a), s2 = sin(a);
        p.xy = mat2(c, -s2, s2, c) * p.xy;
        p *= 1.28;
        s *= 1.28;
    }
    return (sdBox(p, vec3(0.55)) * 0.7) / s;
}

float mapCity(vec3 p) {
    vec3 id = floor(p / vec3(2.4, 1.0, 2.4));
    vec3 q = p;
    q.xz = mod(p.xz + 1.2, 2.4) - 1.2;
    float h = 0.35 + 3.2 * hash13(id + vec3(2.1, 7.7, 4.3));
    float d = sdBox(q - vec3(0.0, h, 0.0), vec3(0.62, h, 0.62));
    float ground = p.y + 0.02;
    return min(d, ground);
}

float mapOcean(vec3 p) {
    float w = 0.0;
    vec2 xz = p.xz;
    float amp = 0.28;
    for (int i = 0; i < 6; i++) {
        float f = 1.1 + float(i) * 0.72;
        w += amp * sin(xz.x * f + uTime * (0.7 + float(i) * 0.18));
        w += amp * 0.65 * cos(xz.y * f * 0.85 - uTime * 0.55);
        xz *= mat2(0.8, -0.6, 0.6, 0.8);
        amp *= 0.55;
    }
    return p.y - w;
}

float mapCrystal(vec3 p) {
    float d = 1e3;
    for (int i = 0; i < 7; i++) {
        float a = float(i) * 0.897 + uTime * 0.35;
        vec3 q = p - vec3(cos(a) * 1.05, 0.15 * sin(a * 1.7 + uTime), sin(a) * 1.05);
        q.xy = mat2(cos(a), -sin(a), sin(a), cos(a)) * q.xy;
        d = min(d, sdBox(q, vec3(0.18, 0.55, 0.18)));
    }
    return d * 0.75;
}

float mapScene(vec3 p) {
    if (uScene == 1) return mapFractal(p);
    if (uScene == 3) return mapCity(p);
    if (uScene == 4) return mapOcean(p);
    if (uScene == 5) return mapCrystal(p);
    return mapTunnel(p);
}

vec3 calcNormal(vec3 p) {
    vec2 e = vec2(0.0018, 0.0);
    return normalize(vec3(
        mapScene(p + e.xyy) - mapScene(p - e.xyy),
        mapScene(p + e.yxy) - mapScene(p - e.yxy),
        mapScene(p + e.yyx) - mapScene(p - e.yyx)
    ));
}

vec3 skyColor(vec3 rd) {
    float h = rd.y * 0.5 + 0.5;
    if (uScene == 3) return mix(vec3(0.02, 0.03, 0.07), vec3(0.55, 0.28, 0.12), pow(h, 1.4));
    if (uScene == 4) return mix(vec3(0.75, 0.35, 0.18), vec3(0.08, 0.16, 0.32), h);
    if (uScene == 1) return mix(vec3(0.03, 0.015, 0.02), vec3(0.12, 0.05, 0.03), h);
    return mix(vec3(0.01, 0.02, 0.05), vec3(0.04, 0.08, 0.16), h);
}

vec3 albedoAt(vec3 p, vec3 n) {
    if (uScene == 1) {
        float bands = 0.5 + 0.5 * sin(length(p) * 8.0);
        return mix(vec3(0.55, 0.12, 0.06), vec3(0.95, 0.55, 0.18), bands);
    }
    if (uScene == 3) {
        if (p.y < 0.05) return vec3(0.08, 0.08, 0.09);
        float win = step(0.72, hash13(floor(p * 6.0)));
        return mix(vec3(0.12, 0.13, 0.16), vec3(1.0, 0.72, 0.28), win * 0.85);
    }
    if (uScene == 4) return mix(vec3(0.02, 0.12, 0.22), vec3(0.08, 0.45, 0.55), 0.5 + 0.5 * n.y);
    if (uScene == 5) {
        float f = 0.5 + 0.5 * n.y;
        return mix(vec3(0.35, 0.12, 0.7), vec3(0.15, 0.75, 0.85), f);
    }
    float ring = 0.5 + 0.5 * sin(p.z * 3.0);
    return mix(vec3(0.05, 0.35, 0.7), vec3(0.9, 0.35, 0.1), ring);
}

vec3 aurora(vec2 uv) {
    vec3 col = vec3(0.01, 0.02, 0.05);
    for (int i = 0; i < 7; i++) {
        float fi = float(i);
        vec2 p = uv * (1.15 + fi * 0.28);
        p.x += 0.55 * sin(p.y * 2.4 + uTime * 0.45 + fi);
        float b = exp(-abs(p.x) * (1.6 + fi * 0.35)) * (0.35 + 0.65 * sin(p.y * 3.2 - uTime * 1.1));
        b = max(b, 0.0);
        vec3 tint = mix(vec3(0.05, 0.7, 0.35), vec3(0.45, 0.15, 0.85), fi / 6.0);
        col += tint * b * 0.22;
    }
    float stars = step(0.997, hash13(vec3(uv * 80.0, 2.0)));
    col += vec3(0.7) * stars;
    return col;
}

vec3 plasma(vec2 uv) {
    float v = 0.0;
    int n = int(10.0 + uWork * 14.0);
    for (int i = 0; i < 24; i++) {
        if (i >= n) break;
        v += 0.35 * sin(uv.x * (3.2 + float(i) * 0.35) + uTime * 1.1);
        v += 0.35 * cos(uv.y * (3.8 + float(i) * 0.28) - uTime * 0.8);
        v += 0.25 * sin(length(uv) * (5.0 + float(i) * 0.4) - uTime * 1.4);
    }
    vec3 c = 0.45 + 0.45 * cos(vec3(0.2, 2.1, 4.0) + v);
    return c * 0.72;
}

vec3 shade(vec3 ro, vec3 rd) {
    float t = 0.0;
    bool hit = false;
    int steps = int(56.0 + uWork * 64.0);
    for (int i = 0; i < 128; i++) {
        if (i >= steps) break;
        float d = mapScene(ro + rd * t);
        if (d < 0.002) { hit = true; break; }
        t += max(d, 0.008);
        if (t > 42.0) break;
    }
    vec3 col = skyColor(rd);
    if (hit) {
        vec3 p = ro + rd * t;
        vec3 n = calcNormal(p);
        vec3 l = normalize(vec3(0.45, 0.85, 0.25));
        float diff = clamp(dot(n, l), 0.0, 1.0);
        float wrap = clamp(dot(n, l) * 0.5 + 0.5, 0.0, 1.0);
        float rim = pow(1.0 - clamp(dot(n, -rd), 0.0, 1.0), 2.2);
        float spec = pow(clamp(dot(reflect(-l, n), -rd), 0.0, 1.0), 48.0);
        float ao = clamp(mapScene(p + n * 0.09) / 0.09, 0.15, 1.0);
        vec3 alb = albedoAt(p, n);
        col = alb * (0.10 + 0.55 * wrap + 0.45 * diff) * ao;
        col += vec3(1.0, 0.95, 0.85) * spec * 0.28;
        col += alb * rim * 0.22;
        if (uScene == 3) {
            float win = step(0.72, hash13(floor(p * 6.0)));
            col += vec3(1.0, 0.7, 0.25) * win * 0.25;
        }
    }
    float fog = 1.0 - exp(-t * (uScene == 4 ? 0.035 : 0.055));
    col = mix(col, skyColor(rd), fog);
    return col;
}

void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * uRes) / max(uRes.y, 1.0);
    if (uScene == 2) {
        fragColor = vec4(pow(max(aurora(uv), 0.0), vec3(0.92)), 1.0);
        return;
    }
    if (uScene == 6) {
        fragColor = vec4(pow(max(plasma(uv * 1.8), 0.0), vec3(0.9)), 1.0);
        return;
    }
    vec3 ro, rd;
    if (uScene == 1) {
        ro = vec3(0.0, 0.15, -3.1);
        rd = normalize(vec3(uv, 1.35));
    } else if (uScene == 3) {
        ro = vec3(0.15, 2.4, uTime * 1.15);
        rd = normalize(vec3(uv.x, uv.y - 0.18, 1.25));
    } else if (uScene == 4) {
        ro = vec3(0.0, 1.05, uTime * 0.55);
        rd = normalize(vec3(uv.x, uv.y - 0.12, 1.45));
    } else if (uScene == 5) {
        float ca = uTime * 0.25;
        ro = vec3(sin(ca) * 2.6, 0.55, cos(ca) * 2.6);
        rd = normalize(vec3(uv.x, uv.y + 0.05, 1.4));
        vec3 fwd = normalize(-ro);
        vec3 right = normalize(cross(vec3(0.0, 1.0, 0.0), fwd));
        vec3 up = cross(fwd, right);
        rd = normalize(right * uv.x + up * (uv.y + 0.05) + fwd * 1.35);
    } else {
        ro = vec3(0.0, 0.0, uTime * 1.4);
        rd = normalize(vec3(uv, 1.2));
    }
    vec3 col = shade(ro, rd);
    col = pow(clamp(col, 0.0, 1.0), vec3(0.88));
    fragColor = vec4(col, 1.0);
}
"""

COMP_SRC = """
#version 430
layout(local_size_x = 16, local_size_y = 16) in;
layout(rgba8, binding = 0) uniform writeonly image2D uImg;
uniform float uTime;
uniform float uWork;

void main() {
    ivec2 id = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(uImg);
    if (id.x >= size.x || id.y >= size.y) return;
    vec2 uv = vec2(id) / vec2(size);
    float v = 0.0;
    int n = int(16.0 + uWork * 32.0);
    for (int i = 0; i < 48; i++) {
        if (i >= n) break;
        vec2 p = uv * (8.0 + float(i));
        v += sin(p.x * 6.1 + uTime * 1.7) * cos(p.y * 5.3 - uTime * 1.1);
        v += sin(length(p - 0.5) * 18.0 - uTime * 3.0);
    }
    vec3 c = vec3(0.08, 0.12, 0.18) + vec3(0.4, 0.15, 0.9) * (0.5 + 0.5 * sin(v));
    imageStore(uImg, id, vec4(c, 1.0));
}
"""


class GL:
    def __init__(self) -> None:
        self._keep = []
        self.glCreateShader = self._fn("glCreateShader", ctypes.c_uint, ctypes.c_uint)
        self.glShaderSource = self._fn(
            "glShaderSource", None, ctypes.c_uint, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_int)
        )
        self.glCompileShader = self._fn("glCompileShader", None, ctypes.c_uint)
        self.glGetShaderiv = self._fn("glGetShaderiv", None, ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_int))
        self.glGetShaderInfoLog = self._fn(
            "glGetShaderInfoLog", None, ctypes.c_uint, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_char_p
        )
        self.glCreateProgram = self._fn("glCreateProgram", ctypes.c_uint)
        self.glAttachShader = self._fn("glAttachShader", None, ctypes.c_uint, ctypes.c_uint)
        self.glLinkProgram = self._fn("glLinkProgram", None, ctypes.c_uint)
        self.glGetProgramiv = self._fn("glGetProgramiv", None, ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_int))
        self.glGetProgramInfoLog = self._fn(
            "glGetProgramInfoLog", None, ctypes.c_uint, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_char_p
        )
        self.glUseProgram = self._fn("glUseProgram", None, ctypes.c_uint)
        self.glGetUniformLocation = self._fn("glGetUniformLocation", ctypes.c_int, ctypes.c_uint, ctypes.c_char_p)
        self.glUniform1f = self._fn("glUniform1f", None, ctypes.c_int, ctypes.c_float)
        self.glUniform1i = self._fn("glUniform1i", None, ctypes.c_int, ctypes.c_int)
        self.glUniform2f = self._fn("glUniform2f", None, ctypes.c_int, ctypes.c_float, ctypes.c_float)
        self.glGenVertexArrays = self._fn("glGenVertexArrays", None, ctypes.c_int, ctypes.POINTER(ctypes.c_uint))
        self.glBindVertexArray = self._fn("glBindVertexArray", None, ctypes.c_uint)
        self.glDrawArrays = self._fn("glDrawArrays", None, ctypes.c_uint, ctypes.c_int, ctypes.c_int)
        self.glViewport = self._fn("glViewport", None, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int)
        self.glClear = self._fn("glClear", None, ctypes.c_uint)
        self.glClearColor = self._fn(
            "glClearColor", None, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float
        )
        self.glDeleteShader = self._fn("glDeleteShader", None, ctypes.c_uint)
        self.glGenTextures = self._fn("glGenTextures", None, ctypes.c_int, ctypes.POINTER(ctypes.c_uint))
        self.glBindTexture = self._fn("glBindTexture", None, ctypes.c_uint, ctypes.c_uint)
        self.glTexParameteri = self._fn("glTexParameteri", None, ctypes.c_uint, ctypes.c_uint, ctypes.c_int)
        self.glTexImage2D = self._fn(
            "glTexImage2D", None, ctypes.c_uint, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p,
        )
        self.glBindImageTexture = self._fn(
            "glBindImageTexture", None, ctypes.c_uint, ctypes.c_uint, ctypes.c_int, ctypes.c_bool, ctypes.c_int,
            ctypes.c_uint, ctypes.c_uint,
        )
        self.glDispatchCompute = self._fn("glDispatchCompute", None, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint)
        self.glMemoryBarrier = self._fn("glMemoryBarrier", None, ctypes.c_uint)
        self.glEnable = self._fn("glEnable", None, ctypes.c_uint)
        self.glDisable = self._fn("glDisable", None, ctypes.c_uint)
        self.glBlendFunc = self._fn("glBlendFunc", None, ctypes.c_uint, ctypes.c_uint)
        self.glActiveTexture = self._fn("glActiveTexture", None, ctypes.c_uint)
        self.glTexSubImage2D = self._fn(
            "glTexSubImage2D", None, ctypes.c_uint, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p,
        )
        self.glGenFramebuffers = self._fn("glGenFramebuffers", None, ctypes.c_int, ctypes.POINTER(ctypes.c_uint))
        self.glBindFramebuffer = self._fn("glBindFramebuffer", None, ctypes.c_uint, ctypes.c_uint)
        self.glFramebufferTexture2D = self._fn(
            "glFramebufferTexture2D", None, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_int,
        )
        self.glCheckFramebufferStatus = self._fn("glCheckFramebufferStatus", ctypes.c_uint, ctypes.c_uint)
        self.glDeleteFramebuffers = self._fn("glDeleteFramebuffers", None, ctypes.c_int, ctypes.POINTER(ctypes.c_uint))
        self.glDeleteTextures = self._fn("glDeleteTextures", None, ctypes.c_int, ctypes.POINTER(ctypes.c_uint))
        self.glFinish = self._fn("glFinish", None)

    def _fn(self, name: str, restype, *argtypes):
        addr = opengl32.wglGetProcAddress(name.encode("ascii"))
        if not addr:
            try:
                addr = ctypes.cast(getattr(opengl32, name), ctypes.c_void_p).value
            except AttributeError:
                addr = None
        if not addr:
            raise RuntimeError(f"нет функции OpenGL {name}")
        proto = ctypes.WINFUNCTYPE(restype, *argtypes)
        fn = proto(addr)
        self._keep.append(fn)
        return fn

    def compile(self, src: str, kind: int) -> int:
        sh = self.glCreateShader(kind)
        buf = ctypes.c_char_p(src.encode("utf-8"))
        self.glShaderSource(sh, 1, ctypes.byref(buf), None)
        self.glCompileShader(sh)
        ok = ctypes.c_int()
        self.glGetShaderiv(sh, GL_COMPILE_STATUS, ctypes.byref(ok))
        if not ok.value:
            loglen = ctypes.c_int()
            self.glGetShaderiv(sh, GL_INFO_LOG_LENGTH, ctypes.byref(loglen))
            log = ctypes.create_string_buffer(max(1, loglen.value))
            self.glGetShaderInfoLog(sh, loglen.value, None, log)
            raise RuntimeError(log.value.decode("utf-8", errors="replace"))
        return sh

    def link(self, *shaders: int) -> int:
        prog = self.glCreateProgram()
        for sh in shaders:
            self.glAttachShader(prog, sh)
        self.glLinkProgram(prog)
        ok = ctypes.c_int()
        self.glGetProgramiv(prog, GL_LINK_STATUS, ctypes.byref(ok))
        if not ok.value:
            loglen = ctypes.c_int()
            self.glGetProgramiv(prog, GL_INFO_LOG_LENGTH, ctypes.byref(loglen))
            log = ctypes.create_string_buffer(max(1, loglen.value))
            self.glGetProgramInfoLog(prog, loglen.value, None, log)
            raise RuntimeError(log.value.decode("utf-8", errors="replace"))
        for sh in shaders:
            self.glDeleteShader(sh)
        return prog


class HudTexture:
    """Текст HUD рисуется в DIB и кладётся в текстуру — без отдельного окна и без мерцания."""

    def __init__(self, gl: GL, width: int) -> None:
        self.gl = gl
        self.w = max(320, width)
        self.h = HUD_H
        self._last = ""
        self._bits = ctypes.c_void_p()
        hdr = BITMAPINFOHEADER()
        hdr.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        hdr.biWidth = self.w
        hdr.biHeight = -self.h
        hdr.biPlanes = 1
        hdr.biBitCount = 32
        hdr.biCompression = 0
        self._hdr = hdr
        hdc = gdi32.CreateCompatibleDC(None)
        self._dib = gdi32.CreateDIBSection(hdc, ctypes.byref(hdr), 0, ctypes.byref(self._bits), None, 0)
        self._mem = hdc
        self._old = gdi32.SelectObject(hdc, self._dib) if self._dib else None
        self._brush = gdi32.CreateSolidBrush(0x201410)
        self._font = gdi32.CreateFontW(16, 0, 0, 0, 600, 0, 0, 0, 1, 0, 0, 5, 0, "Segoe UI")
        tex = ctypes.c_uint()
        gl.glGenTextures(1, ctypes.byref(tex))
        self.tex = tex.value
        gl.glBindTexture(GL_TEXTURE_2D, self.tex)
        gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        gl.glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, self.w, self.h, 0, GL_BGRA, GL_UNSIGNED_BYTE, None)

    def resize(self, width: int) -> None:
        width = max(320, width)
        if width == self.w:
            return
        self.close_dib()
        self.w = width
        self._last = ""
        self._bits = ctypes.c_void_p()
        hdr = BITMAPINFOHEADER()
        hdr.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        hdr.biWidth = self.w
        hdr.biHeight = -self.h
        hdr.biPlanes = 1
        hdr.biBitCount = 32
        hdr.biCompression = 0
        self._hdr = hdr
        hdc = gdi32.CreateCompatibleDC(None)
        self._dib = gdi32.CreateDIBSection(hdc, ctypes.byref(hdr), 0, ctypes.byref(self._bits), None, 0)
        self._mem = hdc
        self._old = gdi32.SelectObject(hdc, self._dib) if self._dib else None
        self.gl.glBindTexture(GL_TEXTURE_2D, self.tex)
        self.gl.glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, self.w, self.h, 0, GL_BGRA, GL_UNSIGNED_BYTE, None)

    def update(self, lines: list[str]) -> None:
        text = "\r\n".join(lines or ["—"])
        if text == self._last or not self._mem or not self._bits.value:
            return
        self._last = text
        rc = RECT(0, 0, self.w, self.h)
        user32.FillRect(self._mem, ctypes.byref(rc), self._brush)
        oldf = gdi32.SelectObject(self._mem, self._font)
        gdi32.SetBkMode(self._mem, TRANSPARENT_BK)
        gdi32.SetTextColor(self._mem, 0xE8EDF5)
        pad = RECT(12, 8, self.w - 8, self.h - 6)
        user32.DrawTextW(self._mem, text, -1, ctypes.byref(pad), DT_LEFT | DT_WORDBREAK)
        gdi32.SelectObject(self._mem, oldf)
        nbytes = self.w * self.h * 4
        raw = ctypes.string_at(self._bits.value, nbytes)
        buf = bytearray(raw)
        for i in range(3, nbytes, 4):
            buf[i] = 230
        self._pix = buf
        self._keep = (ctypes.c_ubyte * nbytes).from_buffer(buf)
        self.gl.glBindTexture(GL_TEXTURE_2D, self.tex)
        self.gl.glTexSubImage2D(
            GL_TEXTURE_2D, 0, 0, 0, self.w, self.h, GL_BGRA, GL_UNSIGNED_BYTE, self._keep,
        )

    def close_dib(self) -> None:
        if getattr(self, "_mem", None) and getattr(self, "_old", None):
            gdi32.SelectObject(self._mem, self._old)
        if getattr(self, "_dib", None):
            gdi32.DeleteObject(self._dib)
            self._dib = None
        if getattr(self, "_mem", None):
            gdi32.DeleteDC(self._mem)
            self._mem = None

    def close(self) -> None:
        self.close_dib()
        if getattr(self, "_brush", None):
            gdi32.DeleteObject(self._brush)
            self._brush = None
        if getattr(self, "_font", None):
            gdi32.DeleteObject(self._font)
            self._font = None


_swap_interval = None


def _set_vsync(enabled: bool) -> bool:
    global _swap_interval
    if _swap_interval is None:
        addr = opengl32.wglGetProcAddress(b"wglSwapIntervalEXT")
        if not addr:
            try:
                addr = ctypes.cast(opengl32.wglSwapIntervalEXT, ctypes.c_void_p).value
            except AttributeError:
                addr = None
        if not addr:
            return False
        _swap_interval = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_int)(addr)
    try:
        _swap_interval(1 if enabled else 0)
        return True
    except Exception:
        return False


class ColorRT:
    def __init__(self, gl: GL, w: int, h: int, nearest: bool = False) -> None:
        self.gl = gl
        self.w = 0
        self.h = 0
        tex = ctypes.c_uint()
        fbo = ctypes.c_uint()
        gl.glGenTextures(1, ctypes.byref(tex))
        gl.glGenFramebuffers(1, ctypes.byref(fbo))
        self.tex = tex.value
        self.fbo = fbo.value
        filt = GL_NEAREST if nearest else GL_LINEAR
        gl.glBindTexture(GL_TEXTURE_2D, self.tex)
        gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filt)
        gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filt)
        gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        self.resize(w, h)

    def resize(self, w: int, h: int) -> None:
        w = max(2, w)
        h = max(2, h)
        if w == self.w and h == self.h:
            return
        self.w, self.h = w, h
        self.gl.glBindTexture(GL_TEXTURE_2D, self.tex)
        self.gl.glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        self.gl.glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        self.gl.glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.tex, 0)
        status = self.gl.glCheckFramebufferStatus(GL_FRAMEBUFFER)
        self.gl.glBindFramebuffer(GL_FRAMEBUFFER, 0)
        if status != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"FBO не собран: 0x{status:04x}")

    def bind(self) -> None:
        self.gl.glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)

    def close(self) -> None:
        fbo = ctypes.c_uint(self.fbo)
        tex = ctypes.c_uint(self.tex)
        self.gl.glDeleteFramebuffers(1, ctypes.byref(fbo))
        self.gl.glDeleteTextures(1, ctypes.byref(tex))
        self.fbo = 0
        self.tex = 0


class ShaderStress:
    def __init__(
        self,
        width: int,
        height: int,
        intensity: int,
        stop_event,
        on_fps: Callable[[float], None],
        on_hud: Callable[[], list[str]] | None = None,
        on_scene: Callable[[str], None] | None = None,
        on_log: Callable[[str], None] | None = None,
        fsr_mode: str = "Выкл",
        render_choice: str = "Авто",
        vsync: bool = False,
        fps_cap: float = 0.0,
        on_native_fps: Callable[[float], None] | None = None,
        on_fsr_info: Callable[[str], None] | None = None,
    ) -> None:
        self.width = max(640, width)
        self.height = max(360, height)
        self.intensity = max(10, min(100, intensity))
        self.fsr_mode = fsr_mode or "Выкл"
        self.render_choice = render_choice or "Авто"
        self.vsync = bool(vsync)
        self.fps_cap = float(fps_cap)
        self.stop_event = stop_event
        self.on_fps = on_fps
        self.on_native_fps = on_native_fps or (lambda _v: None)
        self.on_fsr_info = on_fsr_info or (lambda _t: None)
        self.on_hud = on_hud or (lambda: [])
        self.on_scene = on_scene or (lambda _n: None)
        self.on_log = on_log or (lambda _t: None)
        self.hwnd = None
        self._alive = True
        self._cw = self.width
        self._ch = self.height
        self._bg_brush = gdi32.CreateSolidBrush(0x0D0806)
        self._wndproc = WNDPROC(self._proc)

    def request_close(self) -> None:
        self._alive = False
        if self.hwnd:
            user32.PostMessageW(self.hwnd, WM_CLOSE, 0, 0)

    def _proc(self, hwnd, msg, wparam, lparam):
        if msg == WM_ERASEBKGND:
            return 1
        if msg == WM_PAINT:
            user32.ValidateRect(hwnd, None)
            return 0
        if msg == WM_SIZE:
            self._cw = max(1, lparam & 0xFFFF)
            self._ch = max(1, (lparam >> 16) & 0xFFFF)
            return 0
        if msg == WM_CLOSE:
            self._alive = False
            user32.DestroyWindow(hwnd)
            return 0
        if msg == WM_DESTROY:
            self._alive = False
            return 0
        return user32.DefWindowProcW(hwnd, msg, wparam, lparam)

    def run(self) -> None:
        hwnd = None
        hdc = None
        ctx = None
        legacy = None
        hud = None
        rt_lo = None
        rt_hi = None
        try:
            inst = kernel32.GetModuleHandleW(None)
            cls = WNDCLASSEXW()
            cls.cbSize = ctypes.sizeof(WNDCLASSEXW)
            cls.style = CS_OWNDC
            cls.lpfnWndProc = self._wndproc
            cls.hInstance = inst
            cls.hCursor = user32.LoadCursorW(None, ctypes.c_void_p(IDC_ARROW))
            cls.hbrBackground = self._bg_brush
            cls.lpszClassName = "CoreForgeShaderWnd"
            ico = icon_ico()
            if ico:
                self._hicon = user32.LoadImageW(None, str(ico), IMAGE_ICON, 32, 32, LR_LOADFROMFILE)
                self._hiconsm = user32.LoadImageW(None, str(ico), IMAGE_ICON, 16, 16, LR_LOADFROMFILE)
                if self._hicon:
                    cls.hIcon = self._hicon
                if self._hiconsm:
                    cls.hIconSm = self._hiconsm
            atom = user32.RegisterClassExW(ctypes.byref(cls))
            if not atom and ctypes.get_last_error() not in (0, 1410):
                raise RuntimeError(f"RegisterClassExW: {ctypes.get_last_error()}")
            hwnd = user32.CreateWindowExW(
                0,
                "CoreForgeShaderWnd",
                "CoreForge — шейдеры",
                WS_OVERLAPPEDWINDOW | WS_VISIBLE,
                80, 80, self.width, self.height,
                None, None, inst, None,
            )
            if not hwnd:
                raise RuntimeError("не удалось создать окно OpenGL")
            self.hwnd = hwnd
            if getattr(self, "_hicon", None):
                user32.SendMessageW(hwnd, WM_SETICON, ICON_BIG, self._hicon)
            if getattr(self, "_hiconsm", None):
                user32.SendMessageW(hwnd, WM_SETICON, ICON_SMALL, self._hiconsm)
            user32.ShowWindow(hwnd, SW_SHOW)
            hdc = user32.GetDC(hwnd)
            pfd = PIXELFORMATDESCRIPTOR()
            pfd.nSize = ctypes.sizeof(PIXELFORMATDESCRIPTOR)
            pfd.nVersion = 1
            pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER
            pfd.iPixelType = PFD_TYPE_RGBA
            pfd.cColorBits = 32
            pfd.cDepthBits = 24
            pfd.iLayerType = PFD_MAIN_PLANE
            fmt = gdi32.ChoosePixelFormat(hdc, ctypes.byref(pfd))
            gdi32.SetPixelFormat(hdc, fmt, ctypes.byref(pfd))
            legacy = opengl32.wglCreateContext(hdc)
            opengl32.wglMakeCurrent(hdc, legacy)
            wglCreate = opengl32.wglGetProcAddress(b"wglCreateContextAttribsARB")
            if wglCreate:
                factory = ctypes.WINFUNCTYPE(
                    ctypes.c_void_p, wintypes.HDC, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)
                )(wglCreate)
                attribs = (ctypes.c_int * 9)(
                    WGL_CONTEXT_MAJOR_VERSION_ARB, 4,
                    WGL_CONTEXT_MINOR_VERSION_ARB, 3,
                    WGL_CONTEXT_PROFILE_MASK_ARB, WGL_CONTEXT_CORE_PROFILE_BIT_ARB,
                    0, 0, 0,
                )
                ctx = factory(hdc, None, attribs)
                if ctx:
                    opengl32.wglMakeCurrent(hdc, ctx)
                    opengl32.wglDeleteContext(legacy)
                    legacy = None
            gl = GL()
            if _set_vsync(self.vsync):
                self._vsync_applied = self.vsync
                self.on_log("шейдеры: VSync включён, потолок = герцовка монитора" if self.vsync else "шейдеры: VSync выключен")
            else:
                self._vsync_applied = None
                self.on_log("шейдеры: нет wglSwapIntervalEXT, VSync не переключить")
            vs = gl.compile(VERT_SRC, GL_VERTEX_SHADER)
            fs = gl.compile(FRAG_SRC, GL_FRAGMENT_SHADER)
            prog = gl.link(vs, fs)
            loc_time = gl.glGetUniformLocation(prog, b"uTime")
            loc_res = gl.glGetUniformLocation(prog, b"uRes")
            loc_work = gl.glGetUniformLocation(prog, b"uWork")
            loc_scene = gl.glGetUniformLocation(prog, b"uScene")
            hv = gl.compile(VERT_SRC, GL_VERTEX_SHADER)
            hf = gl.compile(HUD_FRAG_SRC, GL_FRAGMENT_SHADER)
            hprog = gl.link(hv, hf)
            hloc_tex = gl.glGetUniformLocation(hprog, b"uTex")
            hloc_res = gl.glGetUniformLocation(hprog, b"uRes")
            hloc_hh = gl.glGetUniformLocation(hprog, b"uHudH")
            rc0 = RECT()
            user32.GetClientRect(hwnd, ctypes.byref(rc0))
            self._cw = max(1, rc0.right - rc0.left)
            self._ch = max(1, rc0.bottom - rc0.top)
            hud = HudTexture(gl, self._cw)
            last_scene = -1
            last_hud = 0.0
            vao = ctypes.c_uint()
            gl.glGenVertexArrays(1, ctypes.byref(vao))
            gl.glBindVertexArray(vao)
            compute_ok = False
            cprog = 0
            tex = ctypes.c_uint()
            try:
                cs = gl.compile(COMP_SRC, GL_COMPUTE_SHADER)
                cprog = gl.link(cs)
                gl.glGenTextures(1, ctypes.byref(tex))
                gl.glBindTexture(GL_TEXTURE_2D, tex)
                gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                gl.glTexImage2D(
                    GL_TEXTURE_2D, 0, GL_RGBA8, self.width, self.height, 0,
                    GL_RGBA, GL_UNSIGNED_BYTE, None,
                )
                compute_ok = True
            except Exception:
                compute_ok = False

            from coreforge.fsr1 import BLIT_SRC, EASU_SRC, RCAS_SRC, render_size

            easu_prog = rcas_prog = blit_prog = 0
            rt_lo = rt_hi = None
            use_fsr = self.fsr_mode != "Выкл"
            if use_fsr:
                try:
                    ev = gl.compile(VERT_SRC, GL_VERTEX_SHADER)
                    ef = gl.compile(EASU_SRC, GL_FRAGMENT_SHADER)
                    easu_prog = gl.link(ev, ef)
                    rv = gl.compile(VERT_SRC, GL_VERTEX_SHADER)
                    rf = gl.compile(RCAS_SRC, GL_FRAGMENT_SHADER)
                    rcas_prog = gl.link(rv, rf)
                    rt_lo = ColorRT(gl, 640, 360, nearest=True)
                    rt_hi = ColorRT(gl, self._cw, self._ch, nearest=False)
                except Exception as exc:
                    self.on_log(f"FSR 1: EASU не собрался ({exc}), билинейный апскейл")
                    easu_prog = 0
                    try:
                        bv = gl.compile(VERT_SRC, GL_VERTEX_SHADER)
                        bf = gl.compile(BLIT_SRC, GL_FRAGMENT_SHADER)
                        blit_prog = gl.link(bv, bf)
                        if rt_lo is None:
                            rt_lo = ColorRT(gl, 640, 360, nearest=False)
                        if rt_hi is None:
                            rt_hi = ColorRT(gl, self._cw, self._ch, nearest=False)
                    except Exception as exc2:
                        self.on_log(f"FSR 1 выключен: {exc2}")
                        use_fsr = False

            gl.glViewport(0, 0, self._cw, self._ch)
            gl.glClearColor(0.02, 0.03, 0.05, 1.0)
            work = self.intensity / 100.0
            frames = 0
            t0 = time.perf_counter()
            last_fps = t0
            last_native = t0
            native_fps = 0.0
            probe_waste = 0.0
            hud.update(self.on_hud())
            msg = MSG()

            def draw_scene(vw: int, vh: int, t: float, scene: int) -> None:
                gl.glViewport(0, 0, vw, vh)
                gl.glUseProgram(prog)
                gl.glUniform1f(loc_time, t)
                gl.glUniform2f(loc_res, float(vw), float(vh))
                gl.glUniform1f(loc_work, work)
                gl.glUniform1i(loc_scene, scene)
                gl.glClear(GL_COLOR_BUFFER_BIT)
                gl.glDrawArrays(GL_TRIANGLES, 0, 3)

            def draw_hud(vw: int, vh: int) -> None:
                gl.glEnable(GL_BLEND)
                gl.glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                gl.glUseProgram(hprog)
                gl.glActiveTexture(GL_TEXTURE0)
                gl.glBindTexture(GL_TEXTURE_2D, hud.tex)
                gl.glUniform1i(hloc_tex, 0)
                gl.glUniform2f(hloc_res, float(vw), float(vh))
                gl.glUniform1f(hloc_hh, float(HUD_H))
                gl.glDrawArrays(GL_TRIANGLES, 0, 3)
                gl.glDisable(GL_BLEND)

            rw, rh = self._cw, self._ch
            while self._alive and not self.stop_event.is_set():
                while user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, PM_REMOVE):
                    if msg.message == WM_QUIT:
                        self._alive = False
                        break
                    user32.TranslateMessage(ctypes.byref(msg))
                    user32.DispatchMessageW(ctypes.byref(msg))
                now = time.perf_counter()
                t = now - t0
                if getattr(self, "_vsync_applied", None) != self.vsync:
                    if _set_vsync(self.vsync):
                        self._vsync_applied = self.vsync
                        self.on_log("шейдеры: VSync включён" if self.vsync else "шейдеры: VSync выключен")
                cw, ch = self._cw, self._ch
                scene = int(t / SCENE_PERIOD) % len(SCENE_NAMES)
                if scene != last_scene:
                    last_scene = scene
                    self.on_scene(SCENE_NAMES[scene])
                want_w, want_h = render_size(cw, ch, self.fsr_mode, self.render_choice)
                fsr_on = use_fsr and (want_w < cw or want_h < ch) and rt_lo is not None
                rw, rh = (want_w, want_h) if fsr_on else (cw, ch)
                if now - last_hud >= 0.35:
                    hud.resize(cw)
                    hud.update(self.on_hud())
                    last_hud = now
                    if fsr_on:
                        self.on_fsr_info(f"FSR 1 {self.fsr_mode} · {rw}×{rh} → {cw}×{ch}")
                    else:
                        self.on_fsr_info("")
                if compute_ok:
                    gl.glUseProgram(cprog)
                    cloc = gl.glGetUniformLocation(cprog, b"uTime")
                    wloc = gl.glGetUniformLocation(cprog, b"uWork")
                    gl.glUniform1f(cloc, t)
                    gl.glUniform1f(wloc, work)
                    gl.glBindImageTexture(0, tex.value, 0, False, 0, GL_WRITE_ONLY, GL_RGBA8)
                    gx = (self.width + 15) // 16
                    gy = (self.height + 15) // 16
                    gl.glDispatchCompute(gx, gy, 1)
                    gl.glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)
                if fsr_on:
                    rt_lo.resize(rw, rh)
                    rt_lo.bind()
                    draw_scene(rw, rh, t, scene)
                    gl.glBindFramebuffer(GL_FRAMEBUFFER, 0)
                    if easu_prog and rt_hi is not None:
                        rt_hi.resize(cw, ch)
                        rt_hi.bind()
                        gl.glViewport(0, 0, cw, ch)
                        gl.glUseProgram(easu_prog)
                        gl.glActiveTexture(GL_TEXTURE0)
                        gl.glBindTexture(GL_TEXTURE_2D, rt_lo.tex)
                        gl.glUniform1i(gl.glGetUniformLocation(easu_prog, b"uTex"), 0)
                        gl.glUniform2f(gl.glGetUniformLocation(easu_prog, b"uIn"), float(rw), float(rh))
                        gl.glUniform2f(gl.glGetUniformLocation(easu_prog, b"uOut"), float(cw), float(ch))
                        gl.glDrawArrays(GL_TRIANGLES, 0, 3)
                        gl.glBindFramebuffer(GL_FRAMEBUFFER, 0)
                        gl.glViewport(0, 0, cw, ch)
                        gl.glUseProgram(rcas_prog)
                        gl.glActiveTexture(GL_TEXTURE0)
                        gl.glBindTexture(GL_TEXTURE_2D, rt_hi.tex)
                        gl.glUniform1i(gl.glGetUniformLocation(rcas_prog, b"uTex"), 0)
                        gl.glUniform2f(gl.glGetUniformLocation(rcas_prog, b"uRes"), float(cw), float(ch))
                        gl.glUniform1f(gl.glGetUniformLocation(rcas_prog, b"uSharp"), 0.22)
                        gl.glDrawArrays(GL_TRIANGLES, 0, 3)
                    else:
                        gl.glViewport(0, 0, cw, ch)
                        gl.glUseProgram(blit_prog)
                        gl.glActiveTexture(GL_TEXTURE0)
                        gl.glBindTexture(GL_TEXTURE_2D, rt_lo.tex)
                        gl.glUniform1i(gl.glGetUniformLocation(blit_prog, b"uTex"), 0)
                        gl.glUniform2f(gl.glGetUniformLocation(blit_prog, b"uRes"), float(cw), float(ch))
                        gl.glDrawArrays(GL_TRIANGLES, 0, 3)
                    if now - last_native >= 2.4 and rt_hi is not None:
                        probe_n = 5
                        rt_hi.resize(cw, ch)
                        gl.glFinish()
                        pt0 = time.perf_counter()
                        for _ in range(probe_n):
                            rt_hi.bind()
                            draw_scene(cw, ch, t, scene)
                        gl.glFinish()
                        dt = max(1e-4, time.perf_counter() - pt0)
                        probe_waste += dt
                        native_fps = probe_n / dt
                        self.on_native_fps(native_fps)
                        last_native = now
                        gl.glBindFramebuffer(GL_FRAMEBUFFER, 0)
                else:
                    gl.glBindFramebuffer(GL_FRAMEBUFFER, 0)
                    draw_scene(cw, ch, t, scene)
                    native_fps = 0.0
                draw_hud(cw, ch)
                gdi32.SwapBuffers(hdc)
                cap = self.fps_cap if not self.vsync else 0.0
                if cap > 0:
                    leftover = (1.0 / cap) - (time.perf_counter() - now)
                    if leftover > 0.0005:
                        time.sleep(leftover)
                frames += 1
                if now - last_fps >= 0.4:
                    fps = frames / max(1e-4, now - last_fps - probe_waste)
                    probe_waste = 0.0
                    self.on_fps(fps)
                    if fsr_on and native_fps > 1:
                        title = (
                            f"CoreForge — шейдеры · {SCENE_NAMES[scene]} · "
                            f"FSR1 {self.fsr_mode} · натив {native_fps:.0f} · FSR {fps:.0f} FPS"
                        )
                    else:
                        title = f"CoreForge — шейдеры · {SCENE_NAMES[scene]} · {fps:.0f} FPS"
                    user32.SetWindowTextW(hwnd, title)
                    frames = 0
                    last_fps = now
        except Exception as exc:
            self.on_fps(0.0)
            self.on_log(f"шейдеры: {exc}")
        finally:
            if rt_lo:
                try:
                    rt_lo.close()
                except Exception:
                    pass
            if rt_hi:
                try:
                    rt_hi.close()
                except Exception:
                    pass
            if hud:
                hud.close()
            if hdc and hwnd:
                opengl32.wglMakeCurrent(hdc, None)
            if ctx:
                opengl32.wglDeleteContext(ctx)
            if legacy:
                opengl32.wglDeleteContext(legacy)
            if hdc and hwnd:
                user32.ReleaseDC(hwnd, hdc)
            if hwnd:
                user32.DestroyWindow(hwnd)
            self.hwnd = None
