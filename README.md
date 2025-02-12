> [!IMPORTANT]
> This project has been archived and is no longer maintained. The vulkan
> extension `VK_NV_displacement_micromap` is no longer available.
>
> We recommend exploring [**NVIDIA RTX Mega
> Geometry**](https://developer.nvidia.com/blog/nvidia-rtx-mega-geometry-now-available-with-new-vulkan-samples/),
> which can provide similar functionality with greater flexibility. See
> [vk_tessellated_clusters](https://github.com/nvpro-samples/vk_tessellated_clusters),
> which demonstrates raytracing displacement with Vulkan.

# NVIDIA Displacement Micro-Map SDK

For more information about NVIDIA Micro-Mesh Technology visit [NVIDIA developer](https://developer.nvidia.com/rtx/ray-tracing/micro-mesh)

The libraries in this repository aid 3D asset content creation using displacement Micro-Maps.
Please refer to the [NVIDIA Displacement-MicroMap-Toolkit](https://github.com/NVIDIAGameWorks/Displacement-MicroMap-Toolkit) to see them being used.

This SDK provides a low-level API meant for embedding in other applications and tools.
It has a C-style API as well as an API agnostic GPU interface to facilitate this. 
As a result it is sometimes a bit less easy to use. All functionality is provided
through the `micromesh` namespace and it makes frequent use of the `micromesh::ArrayInfo`
structure, which allows it to pass data as a pointer & stride combination. All user visible data
is allocated by the user, so some operations are executed in two steps where a *micromeshOpSomethingBegin* returns
the sizing required, while *micromeshOpSomethingEnd* completes it. One can also abort such 
operations with *micromeshOpContextAbort*. The `micromesh::Context` therefore is stateful
but fairly lightweight, in case you want to create one per thread. Right now there is also some rudimentary
automatic threading within the context.

- [`micromesh_core`](/micromesh_core/README.md): Library for basic data structures, utilities and operations to create or modify micromap and micromesh data.
- [`micromesh_displacement_compression`](/micromesh_displacement_compression/README.md): Library that handles the compression of displacement micromaps.
- [`micromesh_displacement_remeshing`](/micromesh_displacement_remeshing/README.md): Library for GPU-based remeshing (currently only a Vulkan/SPIR-V based implementation for the GPU exists).

## About the Latest Release

Version 2.1

- Add OpTessellateMesh_input::pfnProvideTriangleVertices for control over the
  output location.
- Add micromeshOpChangeLayoutPacked for format conversion
- Fixed bugs in compressor bit packing

Version 2.0

- API break: micromesh::OpGrowTriangleSelection_input::topology is now a const array
- Fix UB in compressor due to possible negative shifts
- Step towards remesher determinism

## Support Contact

Feel free to file issues directly on the GitHub page or reach out to NVIDIA at
<displacedmicromesh-sdk-support@nvidia.com>
