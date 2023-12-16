//
// Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//

#include <micromesh/micromesh_operations.h>
#include <micromesh/micromesh_utils.h>
#include <micromesh_internal/micromesh_context.h>
#include <micromesh_internal/micromesh_containers.h>

#include <mutex>
#include <algorithm>
#include <limits.h>
#include <string.h>
#include <cfloat>

namespace micromesh
{
template <typename T>
inline void micromapValueFromPointer(MicromapValue& value, const T* ptr, uint32_t channelCount)
{
    T* val = reinterpret_cast<T*>(value.value_float);

    val[0] = ptr[0 % channelCount];
    val[1] = ptr[1 % channelCount];
    val[2] = ptr[2 % channelCount];
    val[3] = ptr[3 % channelCount];
}

template <typename T>
inline void micromapValueToPointer(const MicromapValue& value, T* ptr, uint32_t channelCount)
{
    const T* val = reinterpret_cast<const T*>(value.value_float);

#if 1
    uint32_t channelMax = (channelCount - 1) & 3;  // aid compiler to know this is 0..3
    for(uint32_t c = 0; c <= channelMax; c++)
    {
        ptr[c] = val[c];
    }
#else
    ptr[0 % channelCount] = val[0 % channelCount];
    ptr[1 % channelCount] = val[1 % channelCount];
    ptr[2 % channelCount] = val[2 % channelCount];
    ptr[3 % channelCount] = val[3 % channelCount];
#endif
}

template <class T>
inline void micromapValueMin(MicromapValue& minval, const MicromapValue& otherval)
{
    T*       val   = reinterpret_cast<T*>(minval.value_float);
    const T* other = reinterpret_cast<const T*>(otherval.value_float);

    val[0] = std::min(val[0], other[0]);
    val[1] = std::min(val[1], other[1]);
    val[2] = std::min(val[2], other[2]);
    val[3] = std::min(val[3], other[3]);
}

template <class T>
inline void micromapValueMax(MicromapValue& maxval, const MicromapValue& otherval)
{
    T*       val   = reinterpret_cast<T*>(maxval.value_float);
    const T* other = reinterpret_cast<const T*>(otherval.value_float);

    val[0] = std::max(val[0], other[0]);
    val[1] = std::max(val[1], other[1]);
    val[2] = std::max(val[2], other[2]);
    val[3] = std::max(val[3], other[3]);
}

template <class T>
static void processMinMaxs(OpContext ctx, const Micromap* input, uint32_t channelCount, OpComputeTriangleMinMaxs_output* output)
{
    container::vector<MicromapValue> threadMins(ctx->getThreadCount(), output->globalMin);
    container::vector<MicromapValue> threadMaxs(ctx->getThreadCount(), output->globalMax);

    bool useArrays = arrayIsValid(output->triangleMins) && !arrayIsEmpty(output->triangleMins)
                     && arrayIsValid(output->triangleMaxs) && !arrayIsEmpty(output->triangleMaxs);

    uint32_t threadCount = ctx->parallel_item_ranges(
        input->triangleSubdivLevels.count, [&](uint64_t idxFirst, uint64_t idxLast, uint32_t threadIdx, void* userData) {
            for(uint64_t idx = idxFirst; idx < idxLast; idx++)
            {
                uint32_t subdivLevel = arrayGetV<uint16_t>(input->triangleSubdivLevels, idx);
                uint32_t valueFirst  = arrayGetV<uint32_t>(input->triangleValueIndexOffsets, idx);

                uint32_t count = subdivLevelGetCount(subdivLevel, input->frequency);

                MicromapValue localMin = output->globalMin;
                MicromapValue localMax = output->globalMax;

                for(uint32_t i = 0; i < count; i++)
                {
                    const T* __restrict valuePtr = arrayGet<T>(input->values, valueFirst + i);
                    MicromapValue value;

                    micromapValueFromPointer(value, valuePtr, channelCount);
                    micromapValueMin<T>(localMin, value);
                    micromapValueMax<T>(localMax, value);
                }

                micromapValueMin<T>(threadMins[threadIdx], localMin);
                micromapValueMax<T>(threadMaxs[threadIdx], localMax);

                if(useArrays)
                {
                    micromapValueToPointer(localMin, arrayGet<T>(output->triangleMins, idx), channelCount);
                    micromapValueToPointer(localMax, arrayGet<T>(output->triangleMaxs, idx), channelCount);
                }
            }
        });

    output->globalMin = threadMins[0];
    output->globalMax = threadMaxs[0];

    for(uint32_t i = 1; i < threadCount; i++)
    {
        micromapValueMin<T>(output->globalMin, threadMins[i]);
        micromapValueMax<T>(output->globalMax, threadMaxs[i]);
    }
}


MICROMESH_API Result MICROMESH_CALL micromeshOpComputeTriangleMinMaxs(OpContext ctx, const Micromap* input, OpComputeTriangleMinMaxs_output* output)
{
    Result result;

    CHECK_CTX_NONNULL(ctx);
    CHECK_NONNULL(ctx, input);
    CHECK_NONNULL(ctx, output);
    CHECK_CTX_BEGIN(ctx);

    FormatInfo inputFormatInfo;
    result = micromeshFormatGetInfo(input->values.format, &inputFormatInfo);

    if(result != Result::eSuccess || inputFormatInfo.isCompressedOrPacked || inputFormatInfo.channelType == ChannelType::eUndefined)
    {
        LOGE(ctx, "input->values.format must be uncompressed");
        return Result::eInvalidFormat;
    }

    uint32_t channelCount = inputFormatInfo.channelCount;

    switch(inputFormatInfo.channelType)
    {
    case ChannelType::eUint:
    case ChannelType::eUnorm:

        output->globalMax.value_uint32[0] = 0;
        output->globalMax.value_uint32[1] = 0;
        output->globalMax.value_uint32[2] = 0;
        output->globalMax.value_uint32[3] = 0;

        output->globalMin.value_uint32[0] = 0xFFFFFFFF;
        output->globalMin.value_uint32[1] = 0xFFFFFFFF;
        output->globalMin.value_uint32[2] = 0xFFFFFFFF;
        output->globalMin.value_uint32[3] = 0xFFFFFFFF;

        if(inputFormatInfo.channelBitCount == 8)
        {
            processMinMaxs<uint8_t>(ctx, input, channelCount, output);
        }
        else if(inputFormatInfo.channelBitCount == 16 || inputFormatInfo.channelBitCount == 11)
        {
            processMinMaxs<uint16_t>(ctx, input, channelCount, output);
        }
        else if(inputFormatInfo.channelBitCount == 32)
        {
            processMinMaxs<uint32_t>(ctx, input, channelCount, output);
        }
        break;
    case ChannelType::eSint:
    case ChannelType::eSnorm:

        if(inputFormatInfo.channelBitCount == 8)
        {
            output->globalMax.value_int8[0] = -0x7F;
            output->globalMax.value_int8[1] = -0x7F;
            output->globalMax.value_int8[2] = -0x7F;
            output->globalMax.value_int8[3] = -0x7F;

            output->globalMin.value_int8[0] = 0x7F;
            output->globalMin.value_int8[1] = 0x7F;
            output->globalMin.value_int8[2] = 0x7F;
            output->globalMin.value_int8[3] = 0x7F;

            processMinMaxs<int8_t>(ctx, input, channelCount, output);
        }
        else if(inputFormatInfo.channelBitCount == 16)
        {
            output->globalMax.value_int16[0] = -0x7FFF;
            output->globalMax.value_int16[1] = -0x7FFF;
            output->globalMax.value_int16[2] = -0x7FFF;
            output->globalMax.value_int16[3] = -0x7FFF;

            output->globalMin.value_int16[0] = 0x7FFF;
            output->globalMin.value_int16[1] = 0x7FFF;
            output->globalMin.value_int16[2] = 0x7FFF;
            output->globalMin.value_int16[3] = 0x7FFF;

            processMinMaxs<int16_t>(ctx, input, channelCount, output);
        }
        else if(inputFormatInfo.channelBitCount == 32)
        {
            output->globalMax.value_int32[0] = -0x7FFFFFFF;
            output->globalMax.value_int32[1] = -0x7FFFFFFF;
            output->globalMax.value_int32[2] = -0x7FFFFFFF;
            output->globalMax.value_int32[3] = -0x7FFFFFFF;

            output->globalMin.value_int32[0] = 0x7FFFFFFF;
            output->globalMin.value_int32[1] = 0x7FFFFFFF;
            output->globalMin.value_int32[2] = 0x7FFFFFFF;
            output->globalMin.value_int32[3] = 0x7FFFFFFF;

            processMinMaxs<int32_t>(ctx, input, channelCount, output);
        }
        break;

    case ChannelType::eSfloat:

        output->globalMax.value_float[0] = -FLT_MAX;
        output->globalMax.value_float[1] = -FLT_MAX;
        output->globalMax.value_float[2] = -FLT_MAX;
        output->globalMax.value_float[3] = -FLT_MAX;

        output->globalMin.value_float[0] = FLT_MAX;
        output->globalMin.value_float[1] = FLT_MAX;
        output->globalMin.value_float[2] = FLT_MAX;
        output->globalMin.value_float[3] = FLT_MAX;

        processMinMaxs<float>(ctx, input, channelCount, output);

        break;
    }

    return Result::eSuccess;
}

struct OpPackingPayload
{
    ArrayInfo inputValues;
    ArrayInfo outputValues;
    uint32_t  channelCount = 1;
};

template <class Tfrom, class Tto, uint32_t SHIFT>
static void processPacking(uint64_t idxFirst, uint64_t idxLast, uint32_t threadIndex, void* userData)
{
    OpPackingPayload* payload      = reinterpret_cast<OpPackingPayload*>(userData);
    uint32_t          channelCount = payload->channelCount;

    for(uint64_t idx = idxFirst; idx < idxLast; idx++)
    {
        const Tfrom* __restrict valueIn = arrayGet<Tfrom>(payload->inputValues, idx);
        Tto* __restrict valueOut        = arrayGet<Tto>(payload->outputValues, idx);

        valueOut[0 % channelCount] = Tto(valueIn[0 % channelCount] >> SHIFT);
        valueOut[1 % channelCount] = Tto(valueIn[1 % channelCount] >> SHIFT);
        valueOut[2 % channelCount] = Tto(valueIn[2 % channelCount] >> SHIFT);
        valueOut[3 % channelCount] = Tto(valueIn[3 % channelCount] >> SHIFT);
    }
}

MICROMESH_API Result MICROMESH_CALL micromeshOpLowerBit(OpContext ctx, const Micromap* input, Micromap* output)
{
    CHECK_CTX_NONNULL(ctx);
    CHECK_NONNULL(ctx, input);
    CHECK_NONNULL(ctx, output);
    CHECK_CTX_BEGIN(ctx);

    FormatInfo inputFormatInfo;
    FormatInfo outputFormatInfo;
    Result     result;

    result = micromeshFormatGetInfo(input->values.format, &inputFormatInfo);
    if(result != Result::eSuccess)
    {
        return result;
    }

    result = micromeshFormatGetInfo(output->values.format, &outputFormatInfo);
    if(result != Result::eSuccess)
    {
        return result;
    }

    if(inputFormatInfo.channelCount != outputFormatInfo.channelCount
       || inputFormatInfo.channelBitCount < outputFormatInfo.channelBitCount
       || inputFormatInfo.channelType != outputFormatInfo.channelType || inputFormatInfo.isCompressedOrPacked
       || outputFormatInfo.isCompressedOrPacked || outputFormatInfo.valueCount != 1 || inputFormatInfo.valueCount != 1)
    {
        return Result::eInvalidFormat;
    }

    OpPackingPayload payload;
    payload.inputValues  = input->values;
    payload.outputValues = output->values;
    payload.channelCount = inputFormatInfo.channelCount;

    OpContext_s::FnParallelRanges fnBatchRanges;

    switch(inputFormatInfo.channelType)
    {
    case ChannelType::eUnorm:
    case ChannelType::eUint:
        if(inputFormatInfo.channelBitCount == 32 && outputFormatInfo.channelBitCount == 16)
        {
            fnBatchRanges = processPacking<uint32_t, uint16_t, 16>;
        }
        else if(inputFormatInfo.channelBitCount == 32 && outputFormatInfo.channelBitCount == 8)
        {
            fnBatchRanges = processPacking<uint32_t, uint8_t, 24>;
        }
        else if(inputFormatInfo.channelBitCount == 16 && outputFormatInfo.channelBitCount == 11)
        {
            fnBatchRanges = processPacking<uint16_t, uint16_t, 5>;
        }
        else if(inputFormatInfo.channelBitCount == 16 && outputFormatInfo.channelBitCount == 8)
        {
            fnBatchRanges = processPacking<uint16_t, uint8_t, 8>;
        }
        else
        {
            return Result::eFailure;
        }
        break;
    case ChannelType::eSnorm:
    case ChannelType::eSint:
        if(inputFormatInfo.channelBitCount == 32 && outputFormatInfo.channelBitCount == 16)
        {
            fnBatchRanges = processPacking<int32_t, int16_t, 16>;
        }
        else if(inputFormatInfo.channelBitCount == 32 && outputFormatInfo.channelBitCount == 8)
        {
            fnBatchRanges = processPacking<int32_t, int8_t, 24>;
        }
        else if(inputFormatInfo.channelBitCount == 16 && outputFormatInfo.channelBitCount == 8)
        {
            fnBatchRanges = processPacking<int16_t, int8_t, 8>;
        }
        else
        {
            return Result::eFailure;
        }
        break;
    default:
        return Result::eFailure;
    }

    assert(fnBatchRanges);

    ctx->parallel_item_ranges(input->values.count, fnBatchRanges, &payload);

    return Result::eSuccess;
}

MICROMESH_API Result MICROMESH_CALL micromeshOpPack(OpContext ctx, const Micromap* input, MicromapPacked* output)
{
    CHECK_CTX_NONNULL(ctx);
    CHECK_NONNULL(ctx, input);
    CHECK_NONNULL(ctx, output);
    CHECK_CTX_BEGIN(ctx);

    if(input->values.format == Format::eR11_unorm_pack16 && output->values.format == Format::eR11_unorm_packed_align32)
    {
        ctx->parallel_item_ranges(output->triangleSubdivLevels.count, [&](uint64_t idxFirst, uint64_t idxLast,
                                                                          uint32_t threadIndex, void* userData) {
            for(uint64_t idx = idxFirst; idx < idxLast; idx++)
            {
                uint32_t valueIdxIn              = arrayGetV<uint32_t>(input->triangleValueIndexOffsets, idx);
                uint32_t valueIdxOut             = arrayGetV<uint32_t>(output->triangleValueByteOffsets, idx);
                uint32_t* __restrict triangleOut = arrayGet<uint32_t>(output->values, valueIdxOut);

                uint32_t count = subdivLevelGetCount(arrayGetV<uint16_t>(output->triangleSubdivLevels, idx), output->frequency);

                for(uint32_t i = 0; i < count; i++)
                {
                    uint16_t value = arrayGetV<uint16_t>(input->values, valueIdxIn + i);
                    packedWriteR11UnormPackedAlign32(triangleOut, i, value);
                }
            }
        });

        return Result::eSuccess;
    }
    else if(input->values.format == Format::eR16_unorm && output->values.format == Format::eR11_unorm_packed_align32)
    {
        ctx->parallel_item_ranges(output->triangleSubdivLevels.count, [&](uint64_t idxFirst, uint64_t idxLast,
                                                                          uint32_t threadIndex, void* userData) {
            for(uint64_t idx = idxFirst; idx < idxLast; idx++)
            {
                uint32_t valueIdxIn              = arrayGetV<uint32_t>(input->triangleValueIndexOffsets, idx);
                uint32_t valueIdxOut             = arrayGetV<uint32_t>(output->triangleValueByteOffsets, idx);
                uint32_t* __restrict triangleOut = arrayGet<uint32_t>(output->values, valueIdxOut);

                uint32_t count = subdivLevelGetCount(arrayGetV<uint16_t>(output->triangleSubdivLevels, idx), output->frequency);

                for(uint32_t i = 0; i < count; i++)
                {
                    uint16_t value = arrayGetV<uint16_t>(input->values, valueIdxIn + i);
                    packedWriteR11UnormPackedAlign32(triangleOut, i, value >> 5);
                }
            }
        });

        return Result::eSuccess;
    }

    return Result::eFailure;
}

MICROMESH_API Result MICROMESH_CALL micromeshOpUnpack(OpContext ctx, const MicromapPacked* input, Micromap* output)
{
    CHECK_CTX_NONNULL(ctx);
    CHECK_NONNULL(ctx, input);
    CHECK_NONNULL(ctx, output);
    CHECK_CTX_BEGIN(ctx);

    if(input->values.format == Format::eR11_unorm_packed_align32 && output->values.format == Format::eR11_unorm_pack16)
    {
        ctx->parallel_item_ranges(output->triangleSubdivLevels.count, [&](uint64_t idxFirst, uint64_t idxLast,
                                                                          uint32_t threadIndex, void* userData) {
            for(uint64_t idx = idxFirst; idx < idxLast; idx++)
            {
                uint32_t valueIdxIn                   = arrayGetV<uint32_t>(input->triangleValueByteOffsets, idx);
                uint32_t valueIdxOut                  = arrayGetV<uint32_t>(output->triangleValueIndexOffsets, idx);
                const uint32_t* __restrict triangleIn = arrayGet<uint32_t>(input->values, valueIdxIn);

                uint32_t count = subdivLevelGetCount(arrayGetV<uint16_t>(output->triangleSubdivLevels, idx), output->frequency);

                for(uint32_t i = 0; i < count; i++)
                {
                    uint16_t value = packedReadR11UnormPackedAlign32(triangleIn, i);
                    arraySetV<uint16_t>(output->values, valueIdxOut + i, value);
                }
            }
        });

        return Result::eSuccess;
    }
    else if(input->values.format == Format::eR11_unorm_packed_align32 && output->values.format == Format::eR8_unorm)
    {
        ctx->parallel_item_ranges(output->triangleSubdivLevels.count, [&](uint64_t idxFirst, uint64_t idxLast,
                                                                          uint32_t threadIndex, void* userData) {
            for(uint64_t idx = idxFirst; idx < idxLast; idx++)
            {
                uint32_t valueIdxIn                   = arrayGetV<uint32_t>(input->triangleValueByteOffsets, idx);
                uint32_t valueIdxOut                  = arrayGetV<uint32_t>(output->triangleValueIndexOffsets, idx);
                const uint32_t* __restrict triangleIn = arrayGet<uint32_t>(input->values, valueIdxIn);

                uint32_t count = subdivLevelGetCount(arrayGetV<uint16_t>(output->triangleSubdivLevels, idx), output->frequency);

                for(uint32_t i = 0; i < count; i++)
                {
                    uint16_t value = packedReadR11UnormPackedAlign32(triangleIn, i);
                    arraySetV<uint8_t>(output->values, valueIdxOut + i, uint8_t(value >> 3));
                }
            }
        });

        return Result::eSuccess;
    }

    return Result::eFailure;
}


MICROMESH_API Result MICROMESH_CALL micromeshOpChangeLayout(OpContext ctx, const MicromapLayout* newLayout, Micromap* modified)
{
    CHECK_CTX_NONNULL(ctx);
    CHECK_NONNULL(ctx, newLayout);
    CHECK_NONNULL(ctx, modified);
    CHECK_CTX_BEGIN(ctx);

    Result result;

    uint32_t maxValues = subdivLevelGetCount(modified->maxSubdivLevel, modified->frequency);

    const MicromapLayout* oldLayout = &modified->layout;
    FormatInfo            formatInfo;
    result = micromeshFormatGetInfo(modified->values.format, &formatInfo);

    if(result != Result::eSuccess || formatInfo.isCompressedOrPacked)
    {
        return Result::eInvalidFormat;
    }


    uint32_t                   byteSize = formatInfo.byteSize;
    container::vector<uint8_t> mapValues(ctx->getThreadCount() * maxValues * byteSize);

    ctx->parallel_item_ranges(modified->triangleSubdivLevels.count, [&](uint64_t idxFirst, uint64_t idxLast,
                                                                        uint32_t threadIndex, void* userData) {
        for(uint64_t idx = idxFirst; idx < idxLast; idx++)
        {
            uint32_t valueOffset = arrayGetV<uint32_t>(modified->triangleValueIndexOffsets, idx);
            uint32_t subdivLevel = arrayGetV<uint16_t>(modified->triangleSubdivLevels, idx);

            uint32_t numVertices  = subdivLevelGetVertexCount(subdivLevel);
            uint32_t numTriangles = subdivLevelGetTriangleCount(subdivLevel);

            uint32_t numSegmentsPerEdge = subdivLevelGetSegmentCount(subdivLevel);
            uint32_t numVtxPerEdge      = subdivLevelGetSegmentCount(subdivLevel) + 1;

            uint8_t* tempValues = mapValues.data() + (threadIndex * maxValues * byteSize);

            uint32_t valueCount = 0;

            // read from values into scratch per-thread data
            // from old layout into new layout ordering

            if(modified->frequency == Frequency::ePerMicroVertex)
            {
                valueCount = numVertices;

                for(uint32_t u = 0; u < numVtxPerEdge; u++)
                {
                    for(uint32_t v = 0; v < numVtxPerEdge - u; v++)
                    {
                        uint32_t    oldIdx = oldLayout->pfnGetMicroVertexIndex(u, v, subdivLevel, oldLayout->userData);
                        uint32_t    newIdx = newLayout->pfnGetMicroVertexIndex(u, v, subdivLevel, newLayout->userData);
                        const void* oldValues = arrayGet<void>(modified->values, valueOffset + oldIdx);

                        memcpy(tempValues + (newIdx * byteSize), oldValues, byteSize);
                    }
                }
            }
            else
            {
                valueCount = numTriangles;

                for(uint32_t u = 0; u < numSegmentsPerEdge; u++)
                {
                    for(uint32_t v = 0; v < numSegmentsPerEdge - u; v++)
                    {
                        {
                            uint32_t oldIdx = oldLayout->pfnGetMicroTriangleIndex(u, v, 0, subdivLevel, oldLayout->userData);
                            uint32_t newIdx = newLayout->pfnGetMicroTriangleIndex(u, v, 0, subdivLevel, newLayout->userData);
                            const void* oldValues = arrayGet<void>(modified->values, valueOffset + oldIdx);

                            memcpy(tempValues + (newIdx * byteSize), oldValues, byteSize);
                        }
                        if(v != numSegmentsPerEdge - u - 1)
                        {
                            uint32_t oldIdx = oldLayout->pfnGetMicroTriangleIndex(u, v, 1, subdivLevel, oldLayout->userData);
                            uint32_t newIdx = newLayout->pfnGetMicroTriangleIndex(u, v, 1, subdivLevel, newLayout->userData);
                            const void* oldValues = arrayGet<void>(modified->values, valueOffset + oldIdx);

                            memcpy(tempValues + (newIdx * byteSize), oldValues, byteSize);
                        }
                    }
                }
            }

            // write from per-thread, now in new order, back to values
            for(uint32_t newIdx = 0; newIdx < valueCount; newIdx++)
            {
                void* newValues = arrayGet<void>(modified->values, valueOffset + newIdx);
                memcpy(newValues, tempValues + (newIdx * byteSize), byteSize);
            }
        }
    });

    modified->layout = *newLayout;

    return Result::eSuccess;
}

MICROMESH_API Result MICROMESH_CALL micromeshOpChangeLayoutPacked(OpContext ctx, const MicromapLayout* newLayout, MicromapPacked* modified)
{
    CHECK_CTX_NONNULL(ctx);
    CHECK_NONNULL(ctx, newLayout);
    CHECK_NONNULL(ctx, modified);
    CHECK_CTX_BEGIN(ctx);

    uint32_t maxValues = subdivLevelGetCount(modified->maxSubdivLevel, modified->frequency);

    const MicromapLayout* oldLayout = &modified->layout;

    if((modified->values.format != Format::eR11_unorm_packed_align32 || modified->values.byteStride != 1))
    {
        return Result::eInvalidFormat;
    }

    uint32_t                   threadValues = packedCountBytesR11UnormPackedAlign32(maxValues);
    container::vector<uint8_t> mapValues(ctx->getThreadCount() * threadValues);

    ctx->parallel_item_ranges(modified->triangleSubdivLevels.count, [&](uint64_t idxFirst, uint64_t idxLast,
                                                                        uint32_t threadIndex, void* userData) {
        for(uint64_t idx = idxFirst; idx < idxLast; idx++)
        {
            uint32_t valueOffset = arrayGetV<uint32_t>(modified->triangleValueByteOffsets, idx);
            uint32_t subdivLevel = arrayGetV<uint16_t>(modified->triangleSubdivLevels, idx);

            uint32_t numVertices  = subdivLevelGetVertexCount(subdivLevel);
            uint32_t numTriangles = subdivLevelGetTriangleCount(subdivLevel);

            uint32_t numSegmentsPerEdge = subdivLevelGetSegmentCount(subdivLevel);
            uint32_t numVtxPerEdge      = subdivLevelGetSegmentCount(subdivLevel) + 1;

            uint8_t* tempValues = mapValues.data() + (threadIndex * threadValues);

            uint32_t valueCount = 0;

            // read from values into scratch per-thread data
            // from old layout into new layout ordering

            void* oldValues = arrayGet<void>(modified->values, valueOffset);

            if(modified->frequency == Frequency::ePerMicroVertex)
            {
                valueCount = numVertices;

                for(uint32_t u = 0; u < numVtxPerEdge; u++)
                {
                    for(uint32_t v = 0; v < numVtxPerEdge - u; v++)
                    {
                        uint32_t oldIdx = oldLayout->pfnGetMicroVertexIndex(u, v, subdivLevel, oldLayout->userData);
                        uint32_t newIdx = newLayout->pfnGetMicroVertexIndex(u, v, subdivLevel, newLayout->userData);

                        uint32_t oldValue = packedReadR11UnormPackedAlign32(oldValues, oldIdx);
                        packedWriteR11UnormPackedAlign32(tempValues, newIdx, oldValue);
                    }
                }
            }
            else
            {
                valueCount = numTriangles;

                for(uint32_t u = 0; u < numSegmentsPerEdge; u++)
                {
                    for(uint32_t v = 0; v < numSegmentsPerEdge - u; v++)
                    {
                        {
                            uint32_t oldIdx = oldLayout->pfnGetMicroTriangleIndex(u, v, 0, subdivLevel, oldLayout->userData);
                            uint32_t newIdx = newLayout->pfnGetMicroTriangleIndex(u, v, 0, subdivLevel, newLayout->userData);

                            uint32_t oldValue = packedReadR11UnormPackedAlign32(oldValues, oldIdx);
                            packedWriteR11UnormPackedAlign32(tempValues, newIdx, oldValue);
                        }
                        if(v != numSegmentsPerEdge - u - 1)
                        {
                            uint32_t oldIdx = oldLayout->pfnGetMicroTriangleIndex(u, v, 1, subdivLevel, oldLayout->userData);
                            uint32_t newIdx = newLayout->pfnGetMicroTriangleIndex(u, v, 1, subdivLevel, newLayout->userData);

                            uint32_t oldValue = packedReadR11UnormPackedAlign32(oldValues, oldIdx);
                            packedWriteR11UnormPackedAlign32(tempValues, newIdx, oldValue);
                        }
                    }
                }
            }

            // write from per-thread, now in new order, back to values
            memcpy(oldValues, tempValues, packedCountBytesR11UnormPackedAlign32(valueCount));
        }
    });

    modified->layout = *newLayout;

    return Result::eSuccess;
}

MICROMESH_API Result MICROMESH_CALL micromeshOpSwizzle(OpContext ctx, const OpSwizzle_input* input, Micromap* modified)
{
    CHECK_CTX_NONNULL(ctx);
    CHECK_NONNULL(ctx, input);
    CHECK_NONNULL(ctx, modified);
    CHECK_ARRAYVALID(ctx, input->triangleSwizzle);
    CHECK_CTX_BEGIN(ctx);

    Result result;

    uint32_t maxValues = subdivLevelGetCount(modified->maxSubdivLevel, modified->frequency);

    const MicromapLayout* layout = &modified->layout;
    FormatInfo            formatInfo;
    result = micromeshFormatGetInfo(modified->values.format, &formatInfo);

    if(result != Result::eSuccess || formatInfo.isCompressedOrPacked)
    {
        return Result::eInvalidFormat;
    }


    uint32_t                   byteSize = formatInfo.byteSize;
    container::vector<uint8_t> mapValues(ctx->getThreadCount() * maxValues * byteSize);

    ctx->parallel_item_ranges(modified->triangleSubdivLevels.count, [&](uint64_t idxFirst, uint64_t idxLast,
                                                                        uint32_t threadIndex, void* userData) {
        for(uint64_t idx = idxFirst; idx < idxLast; idx++)
        {
            uint32_t valueOffset = arrayGetV<uint32_t>(modified->triangleValueIndexOffsets, idx);
            uint32_t subdivLevel = arrayGetV<uint16_t>(modified->triangleSubdivLevels, idx);

            uint32_t numVertices  = subdivLevelGetVertexCount(subdivLevel);
            uint32_t numTriangles = subdivLevelGetTriangleCount(subdivLevel);

            uint32_t numSegmentsPerEdge = subdivLevelGetSegmentCount(subdivLevel);
            uint32_t numVtxPerEdge      = subdivLevelGetSegmentCount(subdivLevel) + 1;

            uint8_t* tempValues = mapValues.data() + (threadIndex * maxValues * byteSize);

            uint32_t valueCount = 0;

            union
            {
                TriangleSwizzle swizzle;
                uint8_t         swizzleBits;
            };

            swizzleBits = arrayGetV<uint8_t>(input->triangleSwizzle, idx) >> input->swizzleBitShift;

            // read from values into scratch per-thread data
            // from old layout into new layout ordering

            if(modified->frequency == Frequency::ePerMicroVertex)
            {
                valueCount = numVertices;

                for(uint32_t u = 0; u < numVtxPerEdge; u++)
                {
                    for(uint32_t v = 0; v < numVtxPerEdge - u; v++)
                    {
                        BaryUV_uint16 swizzledUV = micromeshUVGetSwizzled({uint16_t(u), uint16_t(v)}, subdivLevel, swizzle);

                        uint32_t oldIdx = layout->pfnGetMicroVertexIndex(u, v, subdivLevel, layout->userData);
                        uint32_t newIdx = layout->pfnGetMicroVertexIndex(swizzledUV.u, swizzledUV.v, subdivLevel, layout->userData);
                        const void* oldValues = arrayGet<void>(modified->values, valueOffset + oldIdx);

                        memcpy(tempValues + (newIdx * byteSize), oldValues, byteSize);
                    }
                }
            }
            else
            {
                valueCount = numTriangles;

                for(uint32_t u = 0; u < numSegmentsPerEdge; u++)
                {
                    for(uint32_t v = 0; v < numSegmentsPerEdge - u; v++)
                    {
                        {
                            BaryUV_uint16 swizzledUV = micromeshUVGetSwizzled({uint16_t(u), uint16_t(v)}, subdivLevel, swizzle);

                            uint32_t oldIdx = layout->pfnGetMicroTriangleIndex(u, v, 0, subdivLevel, layout->userData);
                            uint32_t newIdx =
                                layout->pfnGetMicroTriangleIndex(swizzledUV.u, swizzledUV.v, 0, subdivLevel, layout->userData);
                            const void* oldValues = arrayGet<void>(modified->values, valueOffset + oldIdx);

                            memcpy(tempValues + (newIdx * byteSize), oldValues, byteSize);
                        }
                        if(v != numSegmentsPerEdge - u - 1)
                        {
                            BaryUV_uint16 swizzledUV = micromeshUVGetSwizzled({uint16_t(u), uint16_t(v)}, subdivLevel, swizzle);

                            uint32_t oldIdx = layout->pfnGetMicroTriangleIndex(u, v, 1, subdivLevel, layout->userData);
                            uint32_t newIdx =
                                layout->pfnGetMicroTriangleIndex(swizzledUV.u, swizzledUV.v, 1, subdivLevel, layout->userData);

                            const void* oldValues = arrayGet<void>(modified->values, valueOffset + oldIdx);

                            memcpy(tempValues + (newIdx * byteSize), oldValues, byteSize);
                        }
                    }
                }
            }

            // write from per-thread, now in new order, back to values
            for(uint32_t newIdx = 0; newIdx < valueCount; newIdx++)
            {
                void* newValues = arrayGet<void>(modified->values, valueOffset + newIdx);
                memcpy(newValues, tempValues + (newIdx * byteSize), byteSize);
            }
        }
    });

    return Result::eSuccess;
}


MICROMESH_API uint32_t MICROMESH_CALL micromeshGetBlockFormatUsageReserveCount(const MicromapCompressed* compressed)
{
    if(compressed->values.format == Format::eDispC1_r11_unorm_block)
    {
        return 4 * (compressed->maxSubdivLevel + 1);
    }
    else if(compressed->values.format == Format::eOpaC1_rx_uint_block)
    {
        return 3 * (compressed->maxSubdivLevel + 1);
    }

    return 0;
}

MICROMESH_API Result MICROMESH_CALL micromeshOpComputeBlockFormatUsages(OpContext                               ctx,
                                                                        const OpComputeBlockFormatUsages_input* input,
                                                                        OpComputeBlockFormatUsages_output*      output)
{
    CHECK_CTX_NONNULL(ctx);
    CHECK_NONNULL(ctx, input);
    CHECK_NONNULL(ctx, output);
    CHECK_CTX_BEGIN(ctx);

    if(output->reservedUsageCount != micromeshGetBlockFormatUsageReserveCount(input->compressed))
    {
        LOGE(ctx, "output->reservedUsageCount is invalid for provided input->compressed");
        return Result::eInvalidValue;
    }

    bool useMappings = arrayIsValid(input->meshTriangleMappings) && !arrayIsEmpty(input->meshTriangleMappings);
    const MicromapCompressed* compressed = input->compressed;

    uint64_t triangleCount = useMappings ? input->meshTriangleMappings.count : compressed->triangleSubdivLevels.count;

    uint32_t maxSudbdivLevels = compressed->maxSubdivLevel + 1;
    uint32_t maxBlockFormats  = 1;
    if(compressed->values.format == Format::eDispC1_r11_unorm_block)
    {
        maxBlockFormats = 4;
    }
    else if(compressed->values.format == Format::eOpaC1_rx_uint_block)
    {
        maxBlockFormats = 3;
    }

    for(uint32_t i = 0; i < output->reservedUsageCount; i++)
    {
        output->pUsages[i].count       = 0;
        output->pUsages[i].subdivLevel = i / maxBlockFormats;
        output->pUsages[i].blockFormat = i % maxBlockFormats;
    }

    for(uint64_t i = 0; i < triangleCount; i++)
    {
        uint32_t tri = useMappings ? arrayGetV<uint32_t>(input->meshTriangleMappings, i) : uint32_t(i);

        uint32_t subdivLevel = arrayGetV<uint16_t>(compressed->triangleSubdivLevels, tri) % maxSudbdivLevels;
        uint32_t blockFormat = arrayGetV<uint16_t>(compressed->triangleBlockFormats, tri) % maxBlockFormats;

        output->pUsages[subdivLevel * maxBlockFormats + blockFormat].count++;
    }

    // compaction
    uint32_t outIdx = 0;
    for(uint32_t i = 0; i < output->reservedUsageCount; i++)
    {
        BlockFormatUsage usage = output->pUsages[i];
        if(usage.count)
        {
            output->pUsages[outIdx++] = usage;
        }
    }
    output->usageCount = outIdx;

    return Result::eSuccess;
}

}  // namespace micromesh
