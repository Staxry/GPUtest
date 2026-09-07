# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[('assets', 'assets')],
    hiddenimports=['coreforge', 'coreforge.engine', 'coreforge.ui', 'coreforge.rt_stress', 'coreforge.gl_stress', 'coreforge.vk_compute', 'coreforge.media_worker', 'coreforge.media_stress', 'coreforge.copy_stress', 'coreforge.adl_api', 'coreforge.sensors', 'coreforge.vendor', 'coreforge.assets', 'coreforge.fsr1'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='CoreForge',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['assets/coreforge.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='CoreForge',
)
