@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d "%~dp0"
cl.exe /nologo /O2 /W3 /D_CRT_SECURE_NO_WARNINGS bench.c cuda_drv.c cpu_ops.c /Fe:bench.exe
