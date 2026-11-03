#include <torch/extension.h>
#include "local_aggregate.h"
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("forward", &GaussianVoxelSplatCUDA);
    module.def("backward", &GaussianVoxelSplatBackwardCUDA);
}
