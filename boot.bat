@echo off

msbuild "RayTracing(Cuda).sln" /p:Configuration=Release /p:Platform=x64 /m /verbosity:minimal

x64\Release\"RayTracing(Cuda).exe" > image.ppm




