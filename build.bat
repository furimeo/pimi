@echo off
cl.exe /nologo /O2 /W3 /D_CRT_SECURE_NO_WARNINGS /Iinclude /Isrc src\main.c src\model.c src\tokenizer.c src\runtime.c src\cuda_drv.c src\tensor.c src\ops.c /Fe:pimi.exe
del *.obj 2>nul
