# 使用Torch

该目录是一个使用Pytorch的例子。包括通过C/C++ API或Python API使用Torch。

## 1 编译

```
# 需要编译整个根目录
export MACOSX_DEPLOYMENT_TARGET=11.0  
export USE_OPENMP=1  
export USE_PYTORCH_METAL=1
export USE_FLASH_ATTENTIO=0
mkdir -p build && cd build
cmake -GNinja \
  -DUSE_FLASH_ATTENTION=0 \
  -DBUILD_CUSTOM_PROTOBUF=ON \
  -DCAFFE2_LINK_LOCAL_PROTOBUF=ON \
  -DCAFFE2_CUSTOM_PROTOC_EXECUTABLE=/Users/zhengchenyu/work/github/pytorch/third_party/protobuf/src/protoc \
  -DUSE_DISTRIBUTED=OFF \
  -DUSE_GLOO=ON \
  -DBUILD_PYTHON=ON \
  -DBUILD_TEST=OFF \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_INSTALL_PREFIX=/Users/zhengchenyu/work/github/pytorch/torch \
  -DCMAKE_PREFIX_PATH=/opt/anaconda3/envs/pytorch-build/lib/python3.12/site-packages \
  -DJAVA_HOME=/Library/Java/JavaVirtualMachines/temurin-17.jdk/Contents/Home \
  -DPython_EXECUTABLE=/opt/anaconda3/envs/pytorch-build/bin/python \
  -DTORCH_BUILD_VERSION=2.8.0.dev \
  -DUSE_LOW_IMPACT_MONITORING=True \
  -DUSE_NUMPY=True ..
N_JOBS=$(sysctl -n hw.ncpu)
cmake --build . --config Debug -j${N_JOBS}
```
