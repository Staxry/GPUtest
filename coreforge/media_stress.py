"""Нагрузка NVENC / NVDEC — движки Video Encode и Video Decode в диспетчере задач."""

from __future__ import annotations

import ctypes
import struct
import threading
import time
from ctypes import POINTER, byref, c_int, c_uint32, c_void_p
from typing import Callable


NV_ENC_SUCCESS = 0
NV_ENC_ERR_NEED_MORE_INPUT = 17
NV_ENC_DEVICE_TYPE_CUDA = 1
NV_ENC_BUFFER_FORMAT_NV12 = 0x1
NV_ENC_PIC_STRUCT_FRAME = 0x01

NVENC_ERR = {
    1: "нет NVENC",
    2: "устройство не поддерживает NVENC",
    3: "неверный CUDA-контекст",
    8: "неверный параметр",
    12: "неподдерживаемый параметр",
    15: "неверная версия структуры",
    17: "нужно больше кадров",
}


def _struct_ver(api_ver: int, ver: int, extra: bool = False) -> int:
    value = (api_ver | (ver << 16) | (0x7 << 28)) & 0xFFFFFFFF
    if extra:
        value |= 0x80000000
    return value


def _nvenc_err(code: int) -> str:
    return NVENC_ERR.get(code, f"код {code}")


class GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", c_uint32),
        ("Data2", ctypes.c_ushort),
        ("Data3", ctypes.c_ushort),
        ("Data4", ctypes.c_ubyte * 8),
    ]


def _guid(d1: int, d2: int, d3: int, d4: tuple[int, ...]) -> GUID:
    g = GUID()
    g.Data1 = d1
    g.Data2 = d2
    g.Data3 = d3
    for i, b in enumerate(d4):
        g.Data4[i] = b
    return g


H264_GUID = _guid(0x6BC82762, 0x4E63, 0x4CA4, (0xAA, 0x85, 0x1E, 0x50, 0xF3, 0x21, 0xF6, 0xBF))
PRESET_P1 = _guid(0xFC0A8D3E, 0x45F8, 0x4CF8, (0x80, 0xC2, 0x99, 0x84, 0x71, 0x39, 0x07, 0x13))
PRESET_HP = _guid(0x60E4C59F, 0xE846, 0x4484, (0xA5, 0x6D, 0xCD, 0x45, 0xBE, 0x9F, 0xDD, 0xF6))
PRESET_DEFAULT = _guid(0xB2DFB705, 0x4EBD, 0x4C49, (0x9B, 0x5F, 0x24, 0xA7, 0x77, 0xD3, 0xE5, 0x87))


class NvEncFns(ctypes.Structure):
    _fields_ = [("version", c_uint32), ("reserved", c_uint32)] + [
        (name, c_void_p)
        for name in (
            "nvEncOpenEncodeSession",
            "nvEncGetEncodeGUIDCount",
            "nvEncGetEncodeProfileGUIDCount",
            "nvEncGetEncodeProfileGUIDs",
            "nvEncGetEncodeGUIDs",
            "nvEncGetInputFormatCount",
            "nvEncGetInputFormats",
            "nvEncGetEncodeCaps",
            "nvEncGetEncodePresetCount",
            "nvEncGetEncodePresetGUIDs",
            "nvEncGetEncodePresetConfig",
            "nvEncInitializeEncoder",
            "nvEncCreateInputBuffer",
            "nvEncDestroyInputBuffer",
            "nvEncCreateBitstreamBuffer",
            "nvEncDestroyBitstreamBuffer",
            "nvEncEncodePicture",
            "nvEncLockBitstream",
            "nvEncUnlockBitstream",
            "nvEncLockInputBuffer",
            "nvEncUnlockInputBuffer",
            "nvEncGetEncodeStats",
            "nvEncGetSequenceParams",
            "nvEncRegisterAsyncEvent",
            "nvEncUnregisterAsyncEvent",
            "nvEncMapInputResource",
            "nvEncUnmapInputResource",
            "nvEncDestroyEncoder",
            "nvEncInvalidateRefFrames",
            "nvEncOpenEncodeSessionEx",
            "nvEncRegisterResource",
            "nvEncUnregisterResource",
            "nvEncReconfigureEncoder",
            "reserved1",
            "nvEncCreateMVBuffer",
            "nvEncDestroyMVBuffer",
            "nvEncRunMotionEstimationOnly",
            "nvEncGetLastErrorString",
            "nvEncSetIOCudaStreams",
            "nvEncGetEncodePresetConfigEx",
            "nvEncGetSequenceParamEx",
        )
    ]


class OpenEx(ctypes.Structure):
    _fields_ = [
        ("version", c_uint32),
        ("deviceType", c_uint32),
        ("device", c_void_p),
        ("reserved", c_void_p),
        ("apiVersion", c_uint32),
        ("reserved1", c_uint32 * 253),
        ("reserved2", c_void_p * 64),
    ]


class CreateIn(ctypes.Structure):
    _fields_ = [
        ("version", c_uint32),
        ("width", c_uint32),
        ("height", c_uint32),
        ("memoryHeap", c_uint32),
        ("bufFmt", c_uint32),
        ("reserved", c_uint32),
        ("inputBuffer", c_void_p),
        ("pSysMemBuffer", c_void_p),
        ("reserved1", c_uint32 * 58),
        ("reserved2", c_void_p * 63),
    ]


class CreateOut(ctypes.Structure):
    _fields_ = [
        ("version", c_uint32),
        ("size", c_uint32),
        ("memoryHeap", c_uint32),
        ("reserved", c_uint32),
        ("bitstreamBuffer", c_void_p),
        ("bitstreamBufferPtr", c_void_p),
        ("reserved1", c_uint32 * 58),
        ("reserved2", c_void_p * 64),
    ]


class LockIn(ctypes.Structure):
    _fields_ = [
        ("version", c_uint32),
        ("doNotWait", c_uint32),
        ("inputBuffer", c_void_p),
        ("bufferDataPtr", c_void_p),
        ("pitch", c_uint32),
        ("reserved1", c_uint32 * 251),
        ("reserved2", c_void_p * 64),
    ]


class LockBits(ctypes.Structure):
    _fields_ = [
        ("version", c_uint32),
        ("doNotWait", c_uint32),
        ("outputBitstream", c_void_p),
        ("sliceOffsets", c_void_p),
        ("frameIdx", c_uint32),
        ("hwEncodeStatus", c_uint32),
        ("numSlices", c_uint32),
        ("bitstreamSizeInBytes", c_uint32),
        ("outputTimeStamp", ctypes.c_uint64),
        ("outputDuration", ctypes.c_uint64),
        ("bitstreamBufferPtr", c_void_p),
        ("pictureType", c_uint32),
        ("pictureStruct", c_uint32),
        ("frameAvgQP", c_uint32),
        ("frameSatd", c_uint32),
        ("ltrFrameIdx", c_uint32),
        ("ltrFrameBitmap", c_uint32),
        ("reserved", c_uint32 * 236),
        ("reserved2", c_void_p * 64),
    ]


class InitParams(ctypes.Structure):
    _fields_ = [
        ("version", c_uint32),
        ("encodeGUID", GUID),
        ("presetGUID", GUID),
        ("encodeWidth", c_uint32),
        ("encodeHeight", c_uint32),
        ("darWidth", c_uint32),
        ("darHeight", c_uint32),
        ("frameRateNum", c_uint32),
        ("frameRateDen", c_uint32),
        ("enableEncodeAsync", c_uint32),
        ("enablePTD", c_uint32),
        ("flags", c_uint32),
        ("privDataSize", c_uint32),
        ("reserved0", c_uint32),
        ("privData", c_void_p),
        ("encodeConfig", c_void_p),
        ("maxEncodeWidth", c_uint32),
        ("maxEncodeHeight", c_uint32),
        ("meHints", c_uint32 * 8),
        ("tuningInfo", c_uint32),
        ("bufferFormat", c_uint32),
        ("numStateBuffers", c_uint32),
        ("outputStatsLevel", c_uint32),
        ("reserved1", c_uint32 * 284),
        ("reserved2", c_void_p * 64),
    ]


class PicParams(ctypes.Structure):
    _fields_ = [
        ("version", c_uint32),
        ("inputWidth", c_uint32),
        ("inputHeight", c_uint32),
        ("inputPitch", c_uint32),
        ("encodePicFlags", c_uint32),
        ("frameIdx", c_uint32),
        ("inputTimeStamp", ctypes.c_uint64),
        ("inputDuration", ctypes.c_uint64),
        ("inputBuffer", c_void_p),
        ("outputBitstream", c_void_p),
        ("completionEvent", c_void_p),
        ("bufferFmt", c_uint32),
        ("pictureStruct", c_uint32),
        ("pictureType", c_uint32),
        ("codecPicParams", ctypes.c_ubyte * 2048),
        ("reserved", ctypes.c_ubyte * 2048),
    ]


def _fn(ptr, restype, *argtypes):
    if not ptr:
        raise RuntimeError("пустой указатель NVENC")
    return ctypes.WINFUNCTYPE(restype, *argtypes)(ptr)


class MediaStress:
    def __init__(self, stop_event, encode: bool, decode: bool, intensity: int, on_log: Callable[[str], None], on_fps=None) -> None:
        self.stop = stop_event
        self.encode = encode
        self.decode = decode
        self.intensity = intensity
        self.on_log = on_log
        self.on_fps = on_fps or (lambda *_a: None)
        self._alive = True

    def request_close(self) -> None:
        self._alive = False

    def run(self) -> None:
        try:
            self._run()
        except Exception as exc:
            self.on_log(f"NVENC/NVDEC: {exc}")
            while self._alive and not self.stop.is_set():
                time.sleep(0.3)

    def _run(self) -> None:
        from coreforge.cuda_api import Cuda

        cuda = Cuda()
        dec_stop = threading.Event()
        dec_thread = None
        enc = None
        try:
            w, h = 1280, 720
            if self.intensity >= 80:
                w, h = 1920, 1080
            need_enc = self.encode or self.decode
            bitstream = b""
            if need_enc:
                enc = NvEncoder(cuda, w, h)
                self.on_log(f"NVENC {w}x{h} H.264")
                for i in range(8):
                    chunk = enc.encode_frame(i)
                    if chunk:
                        bitstream += chunk
                self.on_log(f"NVENC: поток {len(bitstream)} байт")
            if self.decode:
                if not bitstream:
                    raise RuntimeError("нет H.264 потока для NVDEC")
                dec = NvDecoderLoop(bitstream, w, h)
                self.on_log("NVDEC: повтор H.264 в отдельном контексте")
                dec_thread = threading.Thread(target=dec.loop, args=(self.stop, dec_stop), daemon=True)
                dec_thread.start()
            frames = 0
            last = time.perf_counter()
            idx = 12
            while self._alive and not self.stop.is_set():
                if enc and self.encode:
                    enc.encode_frame(idx)
                    idx += 1
                    frames += 1
                else:
                    time.sleep(0.01)
                now = time.perf_counter()
                if now - last >= 0.5:
                    self.on_fps(frames / (now - last), 0.0)
                    frames = 0
                    last = now
            if enc:
                enc.close()
                enc = None
        finally:
            dec_stop.set()
            if dec_thread:
                dec_thread.join(timeout=2.0)
            if enc:
                try:
                    enc.close()
                except Exception:
                    pass
            cuda.close()


class NvEncoder:
    def __init__(self, cuda, width: int, height: int) -> None:
        self.cuda = cuda
        self.w = width
        self.h = height
        self.lib = ctypes.WinDLL("nvEncodeAPI64.dll")
        self.lib.NvEncodeAPIGetMaxSupportedVersion.argtypes = [POINTER(c_uint32)]
        self.lib.NvEncodeAPIGetMaxSupportedVersion.restype = c_int
        self.lib.NvEncodeAPICreateInstance.argtypes = [POINTER(NvEncFns)]
        self.lib.NvEncodeAPICreateInstance.restype = c_int
        ver = c_uint32()
        rc = self.lib.NvEncodeAPIGetMaxSupportedVersion(byref(ver))
        if rc != 0:
            raise RuntimeError(f"NvEncodeAPIGetMaxSupportedVersion {_nvenc_err(rc)}")
        major, minor = ver.value >> 4, ver.value & 0xF
        self.api_ver = (major | (minor << 24)) & 0xFFFFFFFF
        self.fns = NvEncFns()
        self.fns.version = _struct_ver(self.api_ver, 2)
        rc = self.lib.NvEncodeAPICreateInstance(byref(self.fns))
        if rc != 0:
            raise RuntimeError(f"NvEncodeAPICreateInstance {_nvenc_err(rc)}")
        self.encoder = c_void_p()
        params = OpenEx()
        params.version = _struct_ver(self.api_ver, 1)
        params.deviceType = NV_ENC_DEVICE_TYPE_CUDA
        params.device = cuda.context
        params.apiVersion = self.api_ver
        open_ex = _fn(self.fns.nvEncOpenEncodeSessionEx, c_int, POINTER(OpenEx), POINTER(c_void_p))
        rc = open_ex(byref(params), byref(self.encoder))
        if rc != 0:
            raise RuntimeError(f"nvEncOpenEncodeSessionEx {_nvenc_err(rc)}")
        self.codec = self._pick_codec()
        self.preset = self._pick_preset()
        self._init_encoder()
        self._make_buffers()

    def _pick_codec(self) -> GUID:
        count = c_uint32()
        get_n = _fn(self.fns.nvEncGetEncodeGUIDCount, c_int, c_void_p, POINTER(c_uint32))
        rc = get_n(self.encoder, byref(count))
        if rc != 0 or count.value < 1:
            return H264_GUID
        guids = (GUID * count.value)()
        got = c_uint32()
        get = _fn(self.fns.nvEncGetEncodeGUIDs, c_int, c_void_p, POINTER(GUID), c_uint32, POINTER(c_uint32))
        rc = get(self.encoder, guids, count.value, byref(got))
        if rc != 0 or got.value < 1:
            return H264_GUID
        for i in range(got.value):
            if guids[i].Data1 == H264_GUID.Data1:
                return guids[i]
        return guids[0]

    def _pick_preset(self) -> GUID:
        count = c_uint32()
        get_n = _fn(self.fns.nvEncGetEncodePresetCount, c_int, c_void_p, GUID, POINTER(c_uint32))
        rc = get_n(self.encoder, self.codec, byref(count))
        wanted = (PRESET_P1.Data1, PRESET_HP.Data1, PRESET_DEFAULT.Data1)
        if rc == 0 and count.value:
            guids = (GUID * count.value)()
            got = c_uint32()
            get = _fn(self.fns.nvEncGetEncodePresetGUIDs, c_int, c_void_p, GUID, POINTER(GUID), c_uint32, POINTER(c_uint32))
            if get(self.encoder, self.codec, guids, count.value, byref(got)) == 0:
                for target in wanted:
                    for i in range(got.value):
                        if guids[i].Data1 == target:
                            return guids[i]
                if got.value:
                    return guids[0]
        return PRESET_P1

    def _init_encoder(self) -> None:
        cfg = (ctypes.c_ubyte * 16384)()
        # NV_ENC_PRESET_CONFIG: version, reserved, presetCfg (NV_ENC_CONFIG)
        struct.pack_into("<I", cfg, 0, _struct_ver(self.api_ver, 5, extra=True))
        struct.pack_into("<I", cfg, 8, _struct_ver(self.api_ver, 9, extra=True))
        rc = 1
        if self.fns.nvEncGetEncodePresetConfigEx:
            get_ex = _fn(
                self.fns.nvEncGetEncodePresetConfigEx,
                c_int,
                c_void_p,
                GUID,
                GUID,
                c_uint32,
                c_void_p,
            )
            rc = get_ex(self.encoder, self.codec, self.preset, 1, ctypes.addressof(cfg))
            if rc != 0:
                rc = get_ex(self.encoder, self.codec, self.preset, 2, ctypes.addressof(cfg))
        self._cfg_keep = cfg

        ip = InitParams()
        ip.version = _struct_ver(self.api_ver, 7, extra=True)
        ip.encodeGUID = self.codec
        ip.presetGUID = self.preset
        ip.encodeWidth = self.w
        ip.encodeHeight = self.h
        ip.darWidth = self.w
        ip.darHeight = self.h
        ip.frameRateNum = 60
        ip.frameRateDen = 1
        ip.enablePTD = 1
        ip.maxEncodeWidth = self.w
        ip.maxEncodeHeight = self.h
        ip.tuningInfo = 1
        if rc == 0:
            ip.encodeConfig = ctypes.addressof(cfg) + 8
        init_fn = _fn(self.fns.nvEncInitializeEncoder, c_int, c_void_p, c_void_p)
        irc = init_fn(self.encoder, byref(ip))
        if irc != 0:
            ip.tuningInfo = 2
            irc = init_fn(self.encoder, byref(ip))
        if irc != 0:
            raise RuntimeError(f"nvEncInitializeEncoder {_nvenc_err(irc)}")
        self._init_keep = ip

    def _make_buffers(self) -> None:
        self.nbuf = 4
        self.inputs: list[int] = []
        self.outputs: list[int] = []
        self._slot = 0
        create_in = _fn(self.fns.nvEncCreateInputBuffer, c_int, c_void_p, POINTER(CreateIn))
        create_out = _fn(self.fns.nvEncCreateBitstreamBuffer, c_int, c_void_p, POINTER(CreateOut))
        for _ in range(self.nbuf):
            cin = CreateIn()
            cin.version = _struct_ver(self.api_ver, 2)
            cin.width = self.w
            cin.height = self.h
            cin.bufFmt = NV_ENC_BUFFER_FORMAT_NV12
            rc = create_in(self.encoder, byref(cin))
            if rc != 0:
                raise RuntimeError(f"nvEncCreateInputBuffer {_nvenc_err(rc)}")
            self.inputs.append(cin.inputBuffer)
            cout = CreateOut()
            cout.version = _struct_ver(self.api_ver, 1)
            rc = create_out(self.encoder, byref(cout))
            if rc != 0:
                raise RuntimeError(f"nvEncCreateBitstreamBuffer {_nvenc_err(rc)}")
            self.outputs.append(cout.bitstreamBuffer)
        self.input = self.inputs[0]
        self.output = self.outputs[0]
        self._lock_in = _fn(self.fns.nvEncLockInputBuffer, c_int, c_void_p, POINTER(LockIn))
        self._unlock_in = _fn(self.fns.nvEncUnlockInputBuffer, c_int, c_void_p, c_void_p)
        self._encode = _fn(self.fns.nvEncEncodePicture, c_int, c_void_p, POINTER(PicParams))
        self._lock_out = _fn(self.fns.nvEncLockBitstream, c_int, c_void_p, POINTER(LockBits))
        self._unlock_out = _fn(self.fns.nvEncUnlockBitstream, c_int, c_void_p, c_void_p)

    def encode_frame(self, idx: int) -> bytes:
        slot = self._slot % self.nbuf
        self._slot += 1
        self.input = self.inputs[slot]
        self.output = self.outputs[slot]
        li = LockIn()
        li.version = _struct_ver(self.api_ver, 1)
        li.inputBuffer = self.input
        rc = self._lock_in(self.encoder, byref(li))
        if rc != 0:
            return b""
        pitch = li.pitch or self.w
        if pitch < self.w or pitch > self.w * 8:
            pitch = self.w
        ptr = li.bufferDataPtr
        addr = ptr if isinstance(ptr, int) else (ptr.value or 0)
        y_size = pitch * self.h
        uv_size = pitch * self.h // 2
        phase = (idx * 17) & 255
        if addr:
            ctypes.memset(addr, phase, y_size)
            ctypes.memset(addr + y_size, (phase + 64) & 255, uv_size)
        self._unlock_in(self.encoder, self.input)
        pic = PicParams()
        pic.version = _struct_ver(self.api_ver, 7, extra=True)
        pic.inputWidth = self.w
        pic.inputHeight = self.h
        pic.inputPitch = pitch
        pic.frameIdx = idx
        pic.inputTimeStamp = idx
        pic.inputBuffer = self.input
        pic.outputBitstream = self.output
        pic.bufferFmt = NV_ENC_BUFFER_FORMAT_NV12
        pic.pictureStruct = NV_ENC_PIC_STRUCT_FRAME
        rc = self._encode(self.encoder, byref(pic))
        if rc not in (NV_ENC_SUCCESS, NV_ENC_ERR_NEED_MORE_INPUT):
            pic.version = _struct_ver(self.api_ver, 4, extra=True)
            rc = self._encode(self.encoder, byref(pic))
        if rc not in (NV_ENC_SUCCESS, NV_ENC_ERR_NEED_MORE_INPUT):
            return b""
        if rc == NV_ENC_ERR_NEED_MORE_INPUT:
            return b""
        lb = LockBits()
        lb.version = _struct_ver(self.api_ver, 2, extra=True)
        lb.outputBitstream = self.output
        rc = self._lock_out(self.encoder, byref(lb))
        if rc != 0:
            return b""
        data = b""
        size = int(lb.bitstreamSizeInBytes or 0)
        ptr = lb.bitstreamBufferPtr
        addr = ptr if isinstance(ptr, int) else (ptr.value if ptr else 0)
        if addr and 0 < size <= 8 * 1024 * 1024:
            data = ctypes.string_at(addr, size)
        self._unlock_out(self.encoder, self.output)
        return data

    def close(self) -> None:
        if not getattr(self, "encoder", None):
            return
        try:
            pic = PicParams()
            pic.version = _struct_ver(self.api_ver, 7, extra=True)
            pic.encodePicFlags = 0x8
            self._encode(self.encoder, byref(pic))
        except Exception:
            pass
        try:
            destroy_in = _fn(self.fns.nvEncDestroyInputBuffer, c_int, c_void_p, c_void_p) if self.fns.nvEncDestroyInputBuffer else None
            destroy_out = _fn(self.fns.nvEncDestroyBitstreamBuffer, c_int, c_void_p, c_void_p) if self.fns.nvEncDestroyBitstreamBuffer else None
            for buf in getattr(self, "inputs", []):
                if destroy_in and buf:
                    destroy_in(self.encoder, buf)
            for buf in getattr(self, "outputs", []):
                if destroy_out and buf:
                    destroy_out(self.encoder, buf)
        except Exception:
            pass
        if self.fns.nvEncDestroyEncoder:
            try:
                _fn(self.fns.nvEncDestroyEncoder, c_int, c_void_p)(self.encoder)
            except Exception:
                pass
        self.encoder = None


class CuvidParserParams(ctypes.Structure):
    _fields_ = [
        ("CodecType", c_uint32),
        ("ulMaxNumDecodeSurfaces", c_uint32),
        ("ulClockRate", c_uint32),
        ("ulErrorThreshold", c_uint32),
        ("ulMaxDisplayDelay", c_uint32),
        ("uReserved1", c_uint32),
        ("pUserData", c_void_p),
        ("pfnSequenceCallback", c_void_p),
        ("pfnDecodePicture", c_void_p),
        ("pfnDisplayPicture", c_void_p),
        ("pvReserved", c_void_p * 5),
        ("pExtVideoInfo", c_void_p),
    ]


class CuvidPacket(ctypes.Structure):
    _fields_ = [
        ("flags", c_uint32),
        ("payload_size", c_uint32),
        ("payload", c_void_p),
        ("timestamp", ctypes.c_int64),
    ]


class CuvidCreate(ctypes.Structure):
    _fields_ = [
        ("ulWidth", c_uint32),
        ("ulHeight", c_uint32),
        ("ulNumDecodeSurfaces", c_uint32),
        ("CodecType", c_uint32),
        ("ChromaFormat", c_uint32),
        ("ulCreationFlags", c_uint32),
        ("bitDepthMinus8", c_uint32),
        ("ulIntraDecodeOnly", c_uint32),
        ("ulMaxWidth", c_uint32),
        ("ulMaxHeight", c_uint32),
        ("Reserved1", c_uint32),
        ("display_area", ctypes.c_short * 4),
        ("OutputFormat", c_uint32),
        ("DeinterlaceMode", c_uint32),
        ("ulTargetWidth", c_uint32),
        ("ulTargetHeight", c_uint32),
        ("ulNumOutputSurfaces", c_uint32),
        ("vidLock", c_void_p),
        ("target_rect", ctypes.c_short * 4),
        ("enableHistogram", c_uint32),
        ("Reserved2", c_uint32 * 4),
    ]


SEQ_CB = ctypes.CFUNCTYPE(c_int, c_void_p, c_void_p)
DEC_CB = ctypes.CFUNCTYPE(c_int, c_void_p, c_void_p)
DISP_CB = ctypes.CFUNCTYPE(c_int, c_void_p, c_void_p)


class NvDecoderLoop:
    def __init__(self, bitstream: bytes, width: int, height: int) -> None:
        self.bits = bitstream
        self.w = width
        self.h = height
        self.lib = ctypes.WinDLL("nvcuvid.dll")
        self.parser = c_void_p()
        self.decoder = c_void_p()

    def loop(self, stop, extra_stop) -> None:
        from coreforge.cuda_api import Cuda

        cuda = Cuda()
        try:
            self._setup()
            pkt = CuvidPacket()
            pkt.payload = ctypes.cast(self._buf, c_void_p)
            pkt.payload_size = len(self.bits)
            while not stop.is_set() and not extra_stop.is_set():
                self.lib.cuvidParseVideoData(self.parser, byref(pkt))
        except Exception:
            pass
        finally:
            if self.parser:
                self.lib.cuvidDestroyVideoParser(self.parser)
            if self.decoder:
                self.lib.cuvidDestroyDecoder(self.decoder)
            cuda.close()

    def _setup(self) -> None:
        L = self.lib
        L.cuvidCreateVideoParser.argtypes = [POINTER(c_void_p), POINTER(CuvidParserParams)]
        L.cuvidCreateVideoParser.restype = c_int
        L.cuvidParseVideoData.argtypes = [c_void_p, POINTER(CuvidPacket)]
        L.cuvidParseVideoData.restype = c_int
        L.cuvidDestroyVideoParser.argtypes = [c_void_p]
        L.cuvidCreateDecoder.argtypes = [POINTER(c_void_p), POINTER(CuvidCreate)]
        L.cuvidCreateDecoder.restype = c_int
        L.cuvidDecodePicture.argtypes = [c_void_p, c_void_p]
        L.cuvidDecodePicture.restype = c_int
        L.cuvidDestroyDecoder.argtypes = [c_void_p]
        self._seq = SEQ_CB(self._on_seq)
        self._dec = DEC_CB(self._on_dec)
        self._disp = DISP_CB(self._on_disp)
        p = CuvidParserParams()
        p.CodecType = 4
        p.ulMaxNumDecodeSurfaces = 8
        p.ulErrorThreshold = 100
        p.ulMaxDisplayDelay = 1
        p.pfnSequenceCallback = ctypes.cast(self._seq, c_void_p)
        p.pfnDecodePicture = ctypes.cast(self._dec, c_void_p)
        p.pfnDisplayPicture = ctypes.cast(self._disp, c_void_p)
        if L.cuvidCreateVideoParser(byref(self.parser), byref(p)) != 0:
            raise RuntimeError("cuvidCreateVideoParser")
        self._buf = ctypes.create_string_buffer(self.bits)
        pkt = CuvidPacket()
        pkt.payload = ctypes.cast(self._buf, c_void_p)
        pkt.payload_size = len(self.bits)
        L.cuvidParseVideoData(self.parser, byref(pkt))

    def _on_seq(self, _user, _fmt) -> int:
        if self.decoder:
            return 8
        info = CuvidCreate()
        info.ulWidth = self.w
        info.ulHeight = self.h
        info.ulNumDecodeSurfaces = 8
        info.CodecType = 4
        info.ChromaFormat = 1
        info.ulMaxWidth = self.w
        info.ulMaxHeight = self.h
        info.OutputFormat = 0
        info.ulTargetWidth = self.w
        info.ulTargetHeight = self.h
        info.ulNumOutputSurfaces = 4
        if self.lib.cuvidCreateDecoder(byref(self.decoder), byref(info)) != 0:
            return 0
        return 8

    def _on_dec(self, _user, pic) -> int:
        if self.decoder:
            self.lib.cuvidDecodePicture(self.decoder, pic)
        return 1

    def _on_disp(self, _user, _info) -> int:
        return 1
