# 使用Torch

该目录是一个使用Pytorch的例子。包括通过C/C++ API或Python API使用Torch。

## 1 编译

```
# 需要编译整个根目录
cd /Users/zhengchenyu/work/github/pytorch
mkdir build
cd build
cmake -GNinja -DBUILD_PYTHON=True -DBUILD_TEST=True -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=/Users/zhengchenyu/work/github/pytorch/torch -DCMAKE_PREFIX_PATH=/opt/anaconda3/envs/ml/lib/python3.12/site-packages -DJAVA_HOME=/Library/Java/JavaVirtualMachines/jdk-1.8.jdk/Contents/Home -DPython_EXECUTABLE=/opt/anaconda3/envs/ml/bin/python -DTORCH_BUILD_VERSION=2.4.0a0+gitd990dad -DUSE_LOW_IMPACT_MONITORING=True -DUSE_NUMPY=True /Users/zhengchenyu/work/github/pytorch
cmake --build . --config Debug
```
