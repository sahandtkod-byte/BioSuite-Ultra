# -*- mode: python ; coding: utf-8 -*-

import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

a = Analysis(
    ['run.py'],  
    pathex=[],
    binaries=[],
    datas=collect_data_files('matplotlib') + collect_data_files('scipy') + collect_data_files('numpy') + collect_data_files('customtkinter'),
    hiddenimports=[
        'numpy._core._exceptions',
        'numpy._core._multiarray_umath',
        'scipy._lib._ccallback_c',
        'scipy._lib._ccallback',
        'scipy._lib._threadsafety',
        'scipy._lib._disjoint_set',
        'scipy.spatial.distance',
        'scipy.spatial._distance_wrap',
        'scipy.spatial._hausdorff',
        'scipy.special._ufuncs_cxx',
        'scipy.special._specfun',
        'scipy.integrate._odepack',
        'scipy.integrate._quadpack',
        'scipy.linalg._fblas',
        'scipy.linalg._flapack',
        'sklearn.utils._typedefs',
        'sklearn.utils._heap',
        'sklearn.utils._sorting',
        'sklearn.utils._vector_sentinel',
        'customtkinter',
        'pydantic',
        'pydantic.deprecated',
        'pydantic_core',
        'multiprocessing',
        'matplotlib.backends.backend_tkagg',
        'PIL',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='BioSuite',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  
    icon='api.png',  
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
