//
// Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//

#pragma once

#include "micromesh_types.h"

namespace micromesh
{
namespace gpu
{

//////////////////////////////////////////////////////////////////////////
/// WARNING: the gpu interface is still subject to change

enum ShaderType : uint32_t
{
    eShaderCompute,
    eShaderMaxTypes,
};

enum ShaderCodeType : uint32_t
{
    eShaderCodeGLSL,
    eShaderCodeHLSL,
    eShaderCodeSPIRV,
    eShaderCodeDXIL,
    eShaderCodePTX,
    eShaderCodeInvalid,
    eShaderCodeMaxTypes = eShaderCodeInvalid,
};

struct AvailableShaderCodeTypes
{
    bool isAvailable[eShaderCodeMaxTypes];
};

struct ShaderCode
{
    ShaderType     type      = eShaderMaxTypes;
    ShaderCodeType codeType  = eShaderCodeMaxTypes;
    const void*    data      = nullptr;
    uint64_t       size      = 0;
    const char*    entryName = nullptr;
    const char*    fileName  = nullptr;
};

enum class PipelineType : uint32_t
{
    eCompute,
    eGraphics,
};

enum class DescriptorType : uint32_t
{
    eInvalid,
    //
    eConstantBuffer,
    // byte address buffer in dx / ssbo in vk
    eBufferRead,
    // byte address buffer UAV in dx / ssbo in vk
    eBufferReadWrite,
};

struct DescriptorRangeInfo
{
    DescriptorType descriptorType    = DescriptorType::eInvalid;
    uint32_t       baseRegisterIndex = 0;
    uint32_t       descriptorCount   = 0;
};

struct DescriptorSetAllocationInfo
{
    // overall number of bindings required
    uint32_t setMaxCount = 0;

    uint32_t constantBufferMaxCount = 0;
    uint32_t bufferMaxCount         = 0;
    uint32_t storageBufferMaxCount  = 0;

    // uint32_t structuredBufferMaxCount        = 0;
    // uint32_t storageStructuredBufferMaxCount = 0;
    // uint32_t staticSamplerMaxCount           = 0;
    // uint32_t textureMaxCount                 = 0;
    // uint32_t storageTextureMaxCount          = 0;
};

struct ConstantBufferInfo
{
    uint32_t registerIndex = 0;
    uint32_t maxDataSize   = 0;
};

struct BufferAllocInfo
{
    uint64_t size;
};

struct ImageAllocInfo
{
    uint16_t width;
    uint16_t height;
    uint16_t depth;
    uint16_t mips;
    uint16_t layers;
};

struct ResourceAllocInfo
{
    DescriptorType type;
    union
    {
        BufferAllocInfo buffer;
        ImageAllocInfo  image;
    };
};

struct PersistentResourceInfo
{
    uint32_t           scratchPersistentCount;
    ResourceAllocInfo* scratchPersistentAllocs;
};

struct PipelineLayoutInfo
{
    // sets (1<<pipelineType) bit per pipelineType
    // all resources must be visible to alll shaders stages within a pipeline type
    uint32_t pipelineTypesUsed = 0;

    // if true then must add global constant buffer,
    // from `SetupInfo::globalConstantBuffer`
    bool hasGlobalConstants = true;

    // if non-zero then uses local constants
    // (push constants / root constants)
    uint32_t localPushConstantSize = 0;

    // all other resource bindings used by this layout
    const DescriptorRangeInfo* descriptorRanges;
    uint32_t                   descriptorRangeCount;
};

// Information on the pipelines provided by the SDK
struct PipelineInfo
{
    PipelineType type;
    // how many sources from the next array are used
    uint32_t     sourceCount;
    ShaderCode   sources[ShaderType::eShaderMaxTypes];
    const char*  debugName           = nullptr;
    uint32_t     pipelineLayoutIndex = 0;
};

//  Information on the user-provided pipelines, which the app
//  defines entirely
struct UserPipelineInfo
{
    PipelineType type;
    const char*  debugName           = nullptr;
};

// These offsets are expected to be applied on top of the DescriptorRangeInfo::baseRegisterIndex and ConstantBufferInfo::registerIndex;
// when spir-v is used and the appropriate bool is active (If the source was native HLSL, this can happen).
struct SPIRVBindingOffsets
{
    uint32_t samplerOffset;
    uint32_t textureOffset;
    uint32_t constantBufferOffset;
    uint32_t storageTextureAndBufferOffset;
};

//////////////////////////////////////////////////////////////////////////
//
//  // pseudo code how below is used
//
//  SetupInfo   setupInfo;
//  result = mygpuGetOperationSetupInfo(&setupInfo, eShaderCodeGLSL);
//
//  if (result != eSuccess) ...
//
//  PipelineInfo pipeInfos[setupInfo.pipelineCount];
//  mygpuGetPipelineInfo(setupInfo.pipelineCount, pipeInfos, eShaderCodeGLSL);
//

struct SetupInfo
{
    // how many pipelines this operation needs
    uint32_t pipelineCount = 0;
    // how many pipelines provided by user
    uint32_t userPipelineCount = 0;

    // how many pipeline layouts
    uint32_t pipelineLayoutCount = 0;

    // sets (1<<pipelineType) bit per pipelineType
    uint32_t pipelineTypesUsed = 0;

    // maximum over all pipeline layouts' descriptorRangeCount
    uint32_t descriptorRangeMaxCountPerLayout = 0;

    DescriptorSetAllocationInfo descriptorSetAllocationInfo;

    // global constant buffer (if bufferSize != 0)
    ResourceAllocInfo globalConstantBuffer;

    // how many scratch resources are needed per task
    uint32_t scratchTaskCount;
    // how many scratch resources are needed task-independent
    uint32_t scratchPersistentCount;

    // how many readback resources are ever needed
    uint32_t readResourcesMaxCount = 0;

    // if false then no need to use the offsets
    bool                spirvBindingOffsetsUsed = false;
    SPIRVBindingOffsets spirvBindingOffsets;
};


//////////////////////////////////////////////////////////////////////////
//
//  // pseudo code for command buffer generation and execution
//
//  CommandSequenceInfo seq;
//  seq.previousReadData      = nullptr;
//  seq.userData              = myUserData;
//  seq.pfnGenerateGpuCommand = myCmdCallback;
//
//  while(mygpuSequenceContinue(opContext, &seq) == eContinue)
//  {
//      // submit command buffer work from myCmdCallback
//
//      // the callback would have set this up to react on
//      // `eReadResources`
//      if (myUserData.requiresReadData) {
//          // do readback and setup data for next sequence
//          seq.previousReadData = ...;
//      }
//      else {
//          seq.previousReadData = nullptr;
//      }
//  }
//

enum class CommandType : uint32_t
{
    // CmdBindPipeline
    eBindPipeline,

    // CmdBindUserPipeline
    // these pipelines are coming from the developer
    // the operation expects certain work to be done there
    eBindUserPipeline,

    // CmdBindResources
    eBindResources,

    // CmdClearResources
    eClearResources,

    // CmdReadResources
    eReadResources,

    // CmdGlobalConstants
    eGlobalConstants,
    // CmdLocalConstants
    eLocalConstants,
    // CmdBarrier
    eBarrier,

    // CmdDispatch
    eDispatch,
    // CmdDispatchIndirect
    eDispatchIndirect,

    // CmdBeginLabel, for debugging
    eBeginLabel,
    // nothing
    eEndLabel,
};

// developer is expected to react on the commands accordingly
// it is guaranteed all state is provided prior dispatches
typedef void (*PFN_generateGpuCommand)(CommandType cmdType, const void* cmdData, void* userData);


// following use template enums, because every gpu operation has its own
// enum type for the specific resources and user pipelines it uses.

template <class TresourceEnum = uint32_t>
struct ResourceInfo
{
    // operation specific enums
    union
    {
        TresourceEnum resourceEnum;
        uint32_t      resourceIndex;
    };
    // uint32_t staticSamplerIndex;
};

template <class TresourceEnum = uint32_t>
struct ReadResourceData
{
    uint32_t                     resourceCount;
    ResourceInfo<TresourceEnum>* resources;
    void**                       resourceDatas;
    uint64_t*                    resourceDataSizes;
};

template <class TresourceEnum = uint32_t>
struct CommandSequenceInfo
{
    void*                                  userData;
    PFN_generateGpuCommand                 pfnGenerateGpuCommand = nullptr;
    const ReadResourceData<TresourceEnum>* previousReadData      = nullptr;
};

struct CmdBindPipeline
{
    // operation specific pipeline
    uint32_t pipelineIndex;
};

template <class TuserPipeEnum = uint32_t>
struct CmdBindUserPipeline
{
    // operation specific enums
    // we expect the user to do certain work
    union
    {
        TuserPipeEnum userPipelineEnum;
        uint32_t      userPipelineIndex;
    };
};

template <class TresourceEnum = uint32_t>
struct CmdBindResources
{
    // operation specific pipeline layout
    uint32_t pipelineLayoutIndex;

    // flattened resources to match all ranges within
    // pipelineLayout
    uint32_t                           resourceCount;
    const ResourceInfo<TresourceEnum>* resources;
};

template <class TresourceEnum = uint32_t>
struct CmdClearResources
{
    // operation specific enums
    uint32_t                           resourceCount;
    const ResourceInfo<TresourceEnum>* resources;
    uint32_t clearValue;
};

template <class TresourceEnum = uint32_t>
struct CmdReadResources
{
    // operation specific enums
    uint32_t                           resourceCount;
    const ResourceInfo<TresourceEnum>* resources;
};

struct CmdGlobalConstants
{
    uint32_t    byteSize;
    const void* data;
};

struct CmdLocalConstants
{
    // operation specific pipeline
    uint32_t    pipelineLayoutIndex;
    uint32_t    byteSize;
    const void* data;
};

enum BarrierBits : uint32_t
{
    eBarrierNone,
    eBarrierBufferBit       = 1 << 1,
    eBarrierImageBit        = 1 << 2,
    eBarrierColorBit        = 1 << 3,
    eBarrierDepthStencilBit = 1 << 4,
    eBarrierIndirectBit     = 1 << 5,
};

struct CmdBarrier
{
    uint32_t readBits;
    uint32_t writeBits;
};

struct CmdDispatch
{
    uint32_t gridX;
    uint32_t gridY;
    uint32_t gridZ;
};

template <class TresourceEnum = uint32_t>
struct CmdDispatchIndirect
{
    ResourceInfo<TresourceEnum> indirectBuffer;
    uint64_t                    indirectBufferOffset;
};

struct CmdBeginLabel
{
    const char* labelName;
};

}  // namespace gpu

}  // namespace micromesh