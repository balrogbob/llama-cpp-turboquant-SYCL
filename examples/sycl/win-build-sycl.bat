
::  MIT license
::  Copyright (C) 2024 Intel Corporation
::  SPDX-License-Identifier: MIT


set "ROOT_DIR=%~dp0..\.."
for %%I in ("%ROOT_DIR%") do set "ROOT_DIR=%%~fI"
set "BUILD_DIR=%ROOT_DIR%\build"

if not exist "%BUILD_DIR%" (
    mkdir "%BUILD_DIR%" || goto ERROR
)

@call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64 --force
if %errorlevel% neq 0 goto ERROR

::  for FP16
::  faster for long-prompt inference
::  cmake -G "MinGW Makefiles" .. -DLLAMA_OPENSSL=OFF -DGGML_SYCL=ON -DCMAKE_CXX_COMPILER=icx -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DGGML_SYCL_F16=ON

::  for FP32
cmake -S "%ROOT_DIR%" -B "%BUILD_DIR%" -G "Ninja" -DLLAMA_OPENSSL=OFF -DGGML_SYCL=ON -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=icx -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release
if %errorlevel% neq 0 goto ERROR

::  build all binary
::  clean-first keeps this script from reusing stale SYCL objects after kernel edits
cmake --build "%BUILD_DIR%" --clean-first -j
if %errorlevel% neq 0 goto ERROR

exit /B 0

:ERROR
echo comomand error: %errorlevel%
exit /B %errorlevel%
