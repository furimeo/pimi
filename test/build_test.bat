@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d "%~dp0"
cl.exe /nologo /O2 /W3 /D_CRT_SECURE_NO_WARNINGS /I..\include /I..\src test_runtime.c ..\src\runtime.c ..\src\cuda_drv.c ..\src\tensor.c ..\src\ops.c /Fe:test_runtime.exe
