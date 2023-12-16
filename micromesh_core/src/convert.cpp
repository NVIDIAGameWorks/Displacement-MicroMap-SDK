//
// Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//

#include <inttypes.h>
#include <micromesh_internal/micromesh_context.h>
#include <micromesh/micromesh_utils.h>

namespace micromesh
{
struct OpQuantizedPayload
{
    MicromapValueFloatExpansion   ex;
    MicromapValueFloatExpansion   outputEx;
    ArrayInfo                     inputValues;
    ArrayInfo                     outputValues;
    uint32_t                      channelCount = 1;
    OpContext_s::FnParallelRanges fnBatchRanges;

    template <class T, uint32_t MAX, bool UNSIGNED>
    static void processFloatToQuantized(uint64_t idxFirst, uint64_t idxLast, uint32_t threadIndex, void* userData)
    {
        OpQuantizedPayload*         payload      = reinterpret_cast<OpQuantizedPayload*>(userData);
        uint32_t                    channelCount = payload->channelCount;
        uint32_t                    channelMax   = (channelCount - 1) & 3;  // aid compiler knowing this value is 0...3
        MicromapValueFloatExpansion ex           = payload->ex;

        for(uint64_t idx = idxFirst; idx < idxLast; idx++)
        {
            const float* __restrict in = arrayGet<float>(payload->inputValues, idx);
            T* __restrict out          = arrayGet<T>(payload->outputValues, idx);

            for(uint32_t c = 0; c <= channelMax; ++c)
            {
                out[c] = T(((in[c] - (UNSIGNED ? ex.bias[c] : 0)) * ex.scale[c]) * float(MAX));
            }
        }
    }

    template <class T, uint32_t MAX, bool UNSIGNED>
    static void processQuantizedToFloat(uint64_t idxFirst, uint64_t idxLast, uint32_t threadIndex, void* userData)
    {
        OpQuantizedPayload*         payload      = reinterpret_cast<OpQuantizedPayload*>(userData);
        uint32_t                    channelCount = payload->channelCount;
        uint32_t                    channelMax   = (channelCount - 1) & 3;  // aid compiler knowing this value is 0...3
        MicromapValueFloatExpansion ex           = payload->ex;

        for(uint64_t idx = idxFirst; idx < idxLast; idx++)
        {
            const T* __restrict in = arrayGet<T>(payload->inputValues, idx);
            float* __restrict out  = arrayGet<float>(payload->outputValues, idx);

            for(uint32_t c = 0; c <= channelMax; ++c)
            {
                out[c] = (float(in[c]) / float(MAX)) * ex.scale[c] + (UNSIGNED ? ex.bias[c] : 0);
            }
        }
    }

    template <class T_SRC, uint32_t MAX_SRC, class T_DST, uint32_t MAX_DST>
    static void processQuantizedToQuantized(uint64_t idxFirst, uint64_t idxLast, uint32_t threadIndex, void* userData)
    {
        OpQuantizedPayload* payload      = reinterpret_cast<OpQuantizedPayload*>(userData);
        uint32_t            channelCount = payload->channelCount;
        uint32_t            channelMax   = (channelCount - 1) & 3;  // aid compiler knowing this value is 0...3

        for(uint64_t idx = idxFirst; idx < idxLast; idx++)
        {
            const T_SRC* __restrict in = arrayGet<T_SRC>(payload->inputValues, idx);
            T_DST* __restrict out      = arrayGet<T_DST>(payload->outputValues, idx);


            for(uint32_t c = 0; c <= channelMax; ++c)
            {
                float temp = (float(in[c]) / float(MAX_SRC));
                out[c]     = T_DST(temp * float(MAX_DST));
            }
        }
    }

    template <class T>
    static void processCopy(uint64_t idxFirst, uint64_t idxLast, uint32_t threadIndex, void* userData)
    {
        OpQuantizedPayload* payload      = reinterpret_cast<OpQuantizedPayload*>(userData);
        uint32_t            channelCount = payload->channelCount;
        uint32_t            channelMax   = (channelCount - 1) & 3;  // aid compiler knowing this value is 0...3

        for(uint64_t idx = idxFirst; idx < idxLast; idx++)
        {
            const T* __restrict in = arrayGet<T>(payload->inputValues, idx);
            T* __restrict out      = arrayGet<T>(payload->outputValues, idx);

            for(uint32_t c = 0; c <= channelMax; ++c)
            {
                out[c] = in[c];
            }
        }
    }

    // Because ctx may be nullptr, we use a message callback directly here.
    Result initFloatToQuantized(OpContext                       ctx,
                                const OpFloatToQuantized_input* settings,
                                const ArrayInfo*                input,
                                const ArrayInfo*                output,
                                const MessageCallbackInfo*      callbacks)
    {
        if(input->count != output->count)
        {
            MLOGE(callbacks, "input and output count must match");
            return Result::eInvalidRange;
        }

        FormatInfo inputFormatInfo;
        Result     result = micromeshFormatGetInfo(input->format, &inputFormatInfo);

        if(result != Result::eSuccess || inputFormatInfo.channelType != ChannelType::eSfloat
           || inputFormatInfo.isCompressedOrPacked || inputFormatInfo.channelBitCount != 32)
        {
            MLOGE(callbacks, "input->floatMicromap->values.format must be uncompressed 32 bit SFLOAT");
            return Result::eInvalidFormat;
        }

        FormatInfo outputFormatInfo;
        result = micromeshFormatGetInfo(output->format, &outputFormatInfo);
        if(result != Result::eSuccess
           || (outputFormatInfo.channelType != ChannelType::eSfloat && outputFormatInfo.channelType != ChannelType::eSnorm
               && outputFormatInfo.channelType != ChannelType::eUnorm)
           || outputFormatInfo.isCompressedOrPacked || outputFormatInfo.channelCount != inputFormatInfo.channelCount
           || (outputFormatInfo.channelType == ChannelType::eSfloat && outputFormatInfo.channelBitCount != 32))
        {
            MLOGE(callbacks,
                  "output->values.format must be uncompressed SFLOAT (only 32bit), UNORM, NORM of same channel count");
            return Result::eInvalidFormat;
        }

        // Compute output MicromapValueFloatExpansion.
        // We apply an affine transformation to the values so that they fit
        // inside [0, 1] for unsigned formats and [-1, 1] (preserving 0) for
        // signed formats; the bias and scale from this quantization satisfies
        //
        // inputValue ~= quantizedScale * outputValue + quantizedBias.
        //
        // Then we compose this with the input's float expansion to get the
        // output float expansion:
        //
        //    inputScale * inputValue + inputBias
        // ~= inputScale * (quantizedScale * outputValue + quantizedBias) + inputBias
        // == (inputScale * quantizedScale) * outputValue + (inputScale * quantizedBias + inputBias)
        const bool isUnsigned = (outputFormatInfo.channelType == ChannelType::eUnorm)
                                || (outputFormatInfo.channelType == ChannelType::eSfloat && settings->outputUnsignedSfloat);
        for(int c = 0; c < 4; c++)
        {
            const float quantizedBias = isUnsigned ? settings->globalMin.value_float[c] : 0.0f;
            const float quantizedScale =
                isUnsigned ?
                    (settings->globalMax.value_float[c] - settings->globalMin.value_float[c]) :
                    std::max(std::abs(settings->globalMax.value_float[c]), std::abs(settings->globalMin.value_float[c]));

            // ex here isn't a standard bias/scale; it's only for quantization,
            // and it uses 1/scale to avoid division inside processFloatToQuantized.
            ex.bias[c]  = quantizedBias;
            ex.scale[c] = (quantizedScale < settings->scaleThreshold) ? 0.0f : (1.0f / quantizedScale);

            const float inputBias  = settings->floatMicromap->valueFloatExpansion.bias[c];
            const float inputScale = settings->floatMicromap->valueFloatExpansion.scale[c];
            outputEx.bias[c]       = inputScale * quantizedBias + inputBias;
            outputEx.scale[c]      = inputScale * quantizedScale;
        }

        // convert
        switch(outputFormatInfo.channelType)
        {
        case ChannelType::eSfloat:
            if(settings->outputUnsignedSfloat)
            {
                fnBatchRanges = processFloatToQuantized<float, 1, true>;
            }
            else
            {
                fnBatchRanges = processFloatToQuantized<float, 1, false>;
            }
            break;
        case ChannelType::eSnorm:
            if(outputFormatInfo.channelBitCount == 8)
            {
                fnBatchRanges = processFloatToQuantized<int8_t, 0x7F, false>;
            }
            else if(outputFormatInfo.channelBitCount == 16)
            {
                fnBatchRanges = processFloatToQuantized<int16_t, 0x7FFF, false>;
            }
            else
            {
                MLOGE(callbacks, "invalid output format");
                return Result::eInvalidFormat;
            }
            break;
        case ChannelType::eUnorm:
            if(outputFormatInfo.channelBitCount == 8)
            {
                fnBatchRanges = processFloatToQuantized<uint8_t, 0xFF, true>;
            }
            else if(outputFormatInfo.channelBitCount == 11 && outputFormatInfo.valueCount == 1)
            {
                fnBatchRanges = processFloatToQuantized<uint16_t, 0x7FF, true>;
            }
            else if(outputFormatInfo.channelBitCount == 16)
            {
                fnBatchRanges = processFloatToQuantized<uint16_t, 0xFFFF, true>;
            }
            else
            {
                MLOGE(callbacks, "invalid output format");
                return Result::eInvalidFormat;
            }
            break;
        }

        channelCount = inputFormatInfo.channelCount;
        inputValues  = *input;
        outputValues = *output;

        return Result::eSuccess;
    }

    // Because ctx may be nullptr, we use a message callback directly here.
    Result initQuantizedToFloat(OpContext                   ctx,
                                bool                        keepExpansion,
                                MicromapValueFloatExpansion inputEx,
                                const ArrayInfo*            input,
                                const ArrayInfo*            output,
                                const MessageCallbackInfo*  callbacks)
    {
        if(input->count != output->count)
        {
            MLOGE(callbacks, "input and output count must match");
            return Result::eInvalidRange;
        }

        FormatInfo inputFormatInfo;
        Result     result = micromeshFormatGetInfo(input->format, &inputFormatInfo);

        if(result != Result::eSuccess
           || (inputFormatInfo.channelType != ChannelType::eSfloat && inputFormatInfo.channelType != ChannelType::eSnorm
               && inputFormatInfo.channelType != ChannelType::eUnorm)
           || inputFormatInfo.isCompressedOrPacked
           || (inputFormatInfo.channelType == ChannelType::eSfloat && inputFormatInfo.channelBitCount != 32))
        {
            MLOGE(callbacks,
                  "input->quantizedMicromap->values.format must be uncompressed SFLOAT (32-bit), UNORM, NORM");
            return Result::eInvalidFormat;
        }

        FormatInfo outputFormatInfo;
        result = micromeshFormatGetInfo(output->format, &outputFormatInfo);
        if(result != Result::eSuccess || outputFormatInfo.channelType != ChannelType::eSfloat || outputFormatInfo.isCompressedOrPacked
           || outputFormatInfo.channelBitCount != 32 || outputFormatInfo.channelCount != inputFormatInfo.channelCount)
        {
            MLOGE(callbacks, "output->values.format must be uncompressed 32-bit SFLOAT of same channel count");
            return Result::eInvalidFormat;
        }

        outputEx = keepExpansion ? inputEx : MicromapValueFloatExpansion();

        // source expansion handling
        ex = keepExpansion ? MicromapValueFloatExpansion() : inputEx;

        // convert
        switch(inputFormatInfo.channelType)
        {
        case ChannelType::eSfloat:
            fnBatchRanges = processQuantizedToFloat<float, 1, true>;
            break;
        case ChannelType::eSnorm:
            if(inputFormatInfo.channelBitCount == 8)
            {
                fnBatchRanges = processQuantizedToFloat<int8_t, 0x7F, false>;
            }
            else if(inputFormatInfo.channelBitCount == 16)
            {
                fnBatchRanges = processQuantizedToFloat<int16_t, 0x7FFF, false>;
            }
            else
            {
                MLOGE(callbacks, "invalid output format");
                return Result::eInvalidFormat;
            }
            break;
        case ChannelType::eUnorm:
            if(inputFormatInfo.channelBitCount == 8)
            {
                fnBatchRanges = processQuantizedToFloat<uint8_t, 0xFF, true>;
            }
            else if(inputFormatInfo.channelBitCount == 11)
            {
                fnBatchRanges = processQuantizedToFloat<uint16_t, 0x7FF, true>;
            }
            else if(inputFormatInfo.channelBitCount == 16)
            {
                fnBatchRanges = processQuantizedToFloat<uint16_t, 0xFFFF, true>;
            }
            else
            {
                MLOGE(callbacks, "invalid input format");
                return Result::eInvalidFormat;
            }
            break;
        default:
            MLOGE(callbacks, "invalid input format");
            return Result::eInvalidFormat;
        }

        channelCount = inputFormatInfo.channelCount;
        inputValues  = *input;
        outputValues = *output;

        return Result::eSuccess;
    }

    Result initQuantizedToQuantized(OpContext ctx, const ArrayInfo* input, const ArrayInfo* output)
    {
        if(input->count != output->count)
        {
            LOGE(ctx, "input and output count must match");
            return Result::eInvalidRange;
        }

        FormatInfo inputFormatInfo;
        Result     result = micromeshFormatGetInfo(input->format, &inputFormatInfo);

        FormatInfo outputFormatInfo;
        Result     resultOut = micromeshFormatGetInfo(output->format, &outputFormatInfo);

        if((result != Result::eSuccess || resultOut != Result::eSuccess)
           || (inputFormatInfo.channelType != outputFormatInfo.channelType)
           || (inputFormatInfo.isCompressedOrPacked || outputFormatInfo.isCompressedOrPacked))
        {
            LOGE(ctx, "formats must be uncompressed, same SFLOAT/UNORM/SNORM channel Type");
            return Result::eInvalidFormat;
        }

        if(inputFormatInfo.channelCount != outputFormatInfo.channelCount)
        {
            LOGE(ctx, "fromats must have same channel count");
            return Result::eInvalidFormat;
        }


        // convert
        switch(inputFormatInfo.channelType)
        {
        case ChannelType::eSfloat:
            fnBatchRanges = processCopy<float>;
            break;
        case ChannelType::eSnorm:
            if(inputFormatInfo.channelBitCount == 8)
            {
                switch(outputFormatInfo.channelBitCount)
                {
                case 8:
                    fnBatchRanges = processCopy<int8_t>;
                    break;
                case 16:
                    fnBatchRanges = processQuantizedToQuantized<int8_t, 0x7F, int16_t, 0x7FFF>;
                    break;
                default:
                    LOGE(ctx, "invalid output format");
                    return Result::eInvalidFormat;
                }
            }
            else if(inputFormatInfo.channelBitCount == 16)
            {
                switch(outputFormatInfo.channelBitCount)
                {
                case 8:
                    fnBatchRanges = processQuantizedToQuantized<int16_t, 0x7FFF, int8_t, 0x7F>;
                    break;
                case 16:
                    fnBatchRanges = processCopy<int16_t>;
                    break;
                default:
                    LOGE(ctx, "invalid output format");
                    return Result::eInvalidFormat;
                }
            }
            else
            {
                LOGE(ctx, "invalid input format");
                return Result::eInvalidFormat;
            }
            break;
        case ChannelType::eUnorm:
            if(inputFormatInfo.channelBitCount == 8)
            {
                switch(outputFormatInfo.channelBitCount)
                {
                case 8:
                    fnBatchRanges = processCopy<uint8_t>;
                    break;
                case 11:
                    fnBatchRanges = processQuantizedToQuantized<uint8_t, 0xFF, uint16_t, 0x7FF>;
                    break;
                case 16:
                    fnBatchRanges = processQuantizedToQuantized<uint8_t, 0xFF, uint16_t, 0xFFFF>;
                    break;
                default:
                    LOGE(ctx, "invalid output format");
                    return Result::eInvalidFormat;
                }
            }
            else if(inputFormatInfo.channelBitCount == 11)
            {
                switch(outputFormatInfo.channelBitCount)
                {
                case 8:
                    fnBatchRanges = processQuantizedToQuantized<uint16_t, 0x7FF, uint8_t, 0xFF>;
                    break;
                case 11:
                    fnBatchRanges = processCopy<uint16_t>;
                    break;
                case 16:
                    fnBatchRanges = processQuantizedToQuantized<uint16_t, 0x7FF, uint16_t, 0xFFFF>;
                    break;
                default:
                    LOGE(ctx, "invalid output format");
                    return Result::eInvalidFormat;
                }
            }
            else if(inputFormatInfo.channelBitCount == 16)
            {
                switch(outputFormatInfo.channelBitCount)
                {
                case 8:
                    fnBatchRanges = processQuantizedToQuantized<uint16_t, 0xFFFF, uint8_t, 0xFF>;
                    break;
                case 11:
                    fnBatchRanges = processQuantizedToQuantized<uint16_t, 0xFFFF, uint16_t, 0x7FF>;
                    break;
                case 16:
                    fnBatchRanges = processCopy<uint16_t>;
                    break;
                default:
                    LOGE(ctx, "invalid output format");
                    return Result::eInvalidFormat;
                }
            }
            else
            {
                LOGE(ctx, "invalid input format");
                return Result::eInvalidFormat;
            }
            break;
        default:
            LOGE(ctx, "invalid input format");
            return Result::eInvalidFormat;
        }

        channelCount = inputFormatInfo.channelCount;
        inputValues  = *input;
        outputValues = *output;

        return Result::eSuccess;
    }
};


MICROMESH_API Result MICROMESH_CALL micromeshOpFloatToQuantized(OpContext ctx, const OpFloatToQuantized_input* input, Micromap* output)
{
    Result result;

    CHECK_CTX_NONNULL(ctx);
    CHECK_NONNULL(ctx, input);
    CHECK_NONNULL(ctx, input->floatMicromap);
    CHECK_NONNULL(ctx, output);
    CHECK_ARRAYVALID(ctx, input->floatMicromap->values);
    CHECK_ARRAYVALID(ctx, output->values);
    CHECK_CTX_BEGIN(ctx);

    const Micromap* inputMap = input->floatMicromap;

    OpQuantizedPayload payload;
    result = payload.initFloatToQuantized(ctx, input, &input->floatMicromap->values, &output->values, &ctx->m_messageCallbackInfo);
    if(result != Result::eSuccess)
    {
        return result;
    }

    if(inputMap->triangleSubdivLevels.count != output->triangleSubdivLevels.count
       || inputMap->triangleValueIndexOffsets.count != output->triangleValueIndexOffsets.count)
    {
        LOGE(ctx, "input map and output map must have all arrays with same count");
        return Result::eInvalidRange;
    }

    if(!arrayIsEqual(inputMap->triangleSubdivLevels, output->triangleSubdivLevels))
    {
        ctx->arrayCopy<uint16_t>(output->triangleSubdivLevels, inputMap->triangleSubdivLevels);
    }

    if(!arrayIsEqual(inputMap->triangleValueIndexOffsets, output->triangleValueIndexOffsets))
    {
        ctx->arrayCopy<uint32_t>(output->triangleValueIndexOffsets, inputMap->triangleValueIndexOffsets);
    }

    output->layout              = inputMap->layout;
    output->frequency           = inputMap->frequency;
    output->minSubdivLevel      = inputMap->minSubdivLevel;
    output->maxSubdivLevel      = inputMap->maxSubdivLevel;
    output->valueFloatExpansion = payload.outputEx;

    assert(payload.fnBatchRanges);
    ctx->parallel_item_ranges(inputMap->values.count, payload.fnBatchRanges, &payload);

    return Result::eSuccess;
}

MICROMESH_API Result MICROMESH_CALL micromeshFloatToQuantizedValues(const OpFloatToQuantized_input* settings,
                                                                    const ArrayInfo*                floatInput,
                                                                    ArrayInfo*                      output,
                                                                    MicromapValueFloatExpansion*    outputExpansion,
                                                                    const MessageCallbackInfo*      callbacks)
{
    OpQuantizedPayload payload;
    Result             result = payload.initFloatToQuantized(nullptr, settings, floatInput, output, callbacks);
    if(result != Result::eSuccess)
    {
        return result;
    }

    payload.fnBatchRanges(0, floatInput->count, 0, &payload);

    *outputExpansion = payload.outputEx;

    return Result::eSuccess;
}

MICROMESH_API Result MICROMESH_CALL micromeshOpFloatToQuantizedPacked(OpContext ctx, const OpFloatToQuantized_input* input, MicromapPacked* output)
{
    Result result;

    CHECK_CTX_NONNULL(ctx);
    CHECK_NONNULL(ctx, input);
    CHECK_NONNULL(ctx, input->floatMicromap);
    CHECK_NONNULL(ctx, output);
    CHECK_ARRAYVALID(ctx, input->floatMicromap->values);
    CHECK_ARRAYVALID(ctx, output->values);
    CHECK_CTX_BEGIN(ctx);

    const Micromap* inputMap = input->floatMicromap;


    FormatInfo inputFormatInfo;
    result = micromeshFormatGetInfo(input->floatMicromap->values.format, &inputFormatInfo);

    if(result != Result::eSuccess || inputFormatInfo.channelType != ChannelType::eSfloat
       || inputFormatInfo.isCompressedOrPacked || inputFormatInfo.channelBitCount != 32)
    {
        LOGE(ctx, "input->floatMicromap->values.format must be uncompressed 32 bit SFLOAT");
        return Result::eInvalidFormat;
    }

    FormatInfo outputFormatInfo;
    result = micromeshFormatGetInfo(output->values.format, &outputFormatInfo);
    if(result != Result::eSuccess || output->values.format != Format::eR11_unorm_packed_align32)
    {
        LOGE(ctx, "output->values.format must be eR11_unorm_packed_align32");
        return Result::eInvalidFormat;
    }

    if(inputMap->triangleSubdivLevels.count != output->triangleSubdivLevels.count
       || inputMap->triangleValueIndexOffsets.count != output->triangleValueByteOffsets.count
       || inputMap->values.count != output->values.count)
    {
        LOGE(ctx, "input map and output map must have all arrays with same count");
        return Result::eInvalidRange;
    }

    if(!arrayIsEqual(inputMap->triangleSubdivLevels, output->triangleSubdivLevels))
    {
        ctx->arrayCopy<uint16_t>(output->triangleSubdivLevels, inputMap->triangleSubdivLevels);
    }

    output->layout         = inputMap->layout;
    output->frequency      = inputMap->frequency;
    output->minSubdivLevel = inputMap->minSubdivLevel;
    output->maxSubdivLevel = inputMap->maxSubdivLevel;

    MicromapValueFloatExpansion ex;

    if(outputFormatInfo.channelType == ChannelType::eUnorm
       || (outputFormatInfo.channelType == ChannelType::eSfloat && input->outputUnsignedSfloat))
    {
        ex.bias[0]  = input->globalMin.value_float[0];
        ex.bias[1]  = input->globalMin.value_float[1];
        ex.bias[2]  = input->globalMin.value_float[2];
        ex.bias[3]  = input->globalMin.value_float[3];
        ex.scale[0] = input->globalMax.value_float[0] - input->globalMin.value_float[0];
        ex.scale[1] = input->globalMax.value_float[1] - input->globalMin.value_float[1];
        ex.scale[2] = input->globalMax.value_float[2] - input->globalMin.value_float[2];
        ex.scale[3] = input->globalMax.value_float[3] - input->globalMin.value_float[3];
    }
    else
    {
        ex.bias[0] = 0.0f;
        ex.bias[1] = 0.0f;
        ex.bias[2] = 0.0f;
        ex.bias[3] = 0.0f;

        ex.scale[0] = std::max(std::abs(input->globalMax.value_float[0]), std::abs(input->globalMin.value_float[0]));
        ex.scale[1] = std::max(std::abs(input->globalMax.value_float[1]), std::abs(input->globalMin.value_float[1]));
        ex.scale[2] = std::max(std::abs(input->globalMax.value_float[2]), std::abs(input->globalMin.value_float[2]));
        ex.scale[3] = std::max(std::abs(input->globalMax.value_float[3]), std::abs(input->globalMin.value_float[3]));
    }

    output->valueFloatExpansion = ex;

    OpQuantizedPayload payload;
    payload.ex           = ex;
    payload.inputValues  = inputMap->values;
    payload.outputValues = output->values;
    payload.channelCount = inputFormatInfo.channelCount;
    payload.ex.scale[0]  = (ex.scale[0] < input->scaleThreshold) ? 0.0f : (1.0f / ex.scale[0]);
    payload.ex.scale[1]  = (ex.scale[1] < input->scaleThreshold) ? 0.0f : (1.0f / ex.scale[1]);
    payload.ex.scale[2]  = (ex.scale[2] < input->scaleThreshold) ? 0.0f : (1.0f / ex.scale[2]);
    payload.ex.scale[3]  = (ex.scale[3] < input->scaleThreshold) ? 0.0f : (1.0f / ex.scale[3]);

    // special case
    if(output->values.format == Format::eR11_unorm_packed_align32)
    {
        uint32_t dataIndex = 0;
        for(uint64_t idx = 0; idx < output->triangleSubdivLevels.count; idx++)
        {
            arraySetV(output->triangleValueByteOffsets, idx, dataIndex);
            dataIndex += packedCountBytesR11UnormPackedAlign32(
                subdivLevelGetCount(arrayGetV<uint16_t>(output->triangleSubdivLevels, idx), output->frequency));
        }

        ctx->parallel_item_ranges(output->triangleSubdivLevels.count, [&](uint64_t idxFirst, uint64_t idxLast,
                                                                          uint32_t threadIndex, void* userData) {
            for(uint64_t idx = idxFirst; idx < idxLast; idx++)
            {
                uint32_t valueIdxIn              = arrayGetV<uint32_t>(inputMap->triangleValueIndexOffsets, idx);
                uint32_t valueIdxOut             = arrayGetV<uint32_t>(output->triangleValueByteOffsets, idx);
                uint32_t* __restrict triangleOut = arrayGet<uint32_t>(output->values, valueIdxOut);

                uint32_t count = subdivLevelGetCount(arrayGetV<uint16_t>(output->triangleSubdivLevels, idx), output->frequency);

                for(uint32_t i = 0; i < count; i++)
                {
                    float    in    = arrayGetV<float>(inputMap->values, valueIdxIn + i);
                    uint16_t value = uint16_t(((in - ex.bias[0]) * payload.ex.scale[0]) * float(0x7FF));
                    packedWriteR11UnormPackedAlign32(triangleOut, i, value);
                }
            }
        });

        return Result::eSuccess;
    }

    return Result::eInvalidFormat;
}

MICROMESH_API Result MICROMESH_CALL micromeshOpQuantizedToFloat(OpContext ctx, const OpQuantizedToFloat_input* input, Micromap* output)
{
    Result result;

    CHECK_CTX_NONNULL(ctx);
    CHECK_NONNULL(ctx, input);
    CHECK_NONNULL(ctx, input->quantizedMicromap);
    CHECK_NONNULL(ctx, output);
    CHECK_ARRAYVALID(ctx, input->quantizedMicromap->values);
    CHECK_ARRAYVALID(ctx, output->values);
    CHECK_CTX_BEGIN(ctx);

    const Micromap* inputMap = input->quantizedMicromap;

    OpQuantizedPayload payload;
    result = payload.initQuantizedToFloat(ctx, input->outputKeepFloatExpansion, inputMap->valueFloatExpansion,
                                          &inputMap->values, &output->values, &ctx->m_messageCallbackInfo);

    if(inputMap->triangleSubdivLevels.count != output->triangleSubdivLevels.count
       || inputMap->triangleValueIndexOffsets.count != output->triangleValueIndexOffsets.count)
    {
        LOGE(ctx, "input map and output map must have all arrays with same count");
        return Result::eInvalidRange;
    }

    if(!arrayIsEqual(inputMap->triangleSubdivLevels, output->triangleSubdivLevels))
    {
        ctx->arrayCopy<uint16_t>(output->triangleSubdivLevels, inputMap->triangleSubdivLevels);
    }

    if(!arrayIsEqual(inputMap->triangleValueIndexOffsets, output->triangleValueIndexOffsets))
    {
        ctx->arrayCopy<uint32_t>(output->triangleValueIndexOffsets, inputMap->triangleValueIndexOffsets);
    }

    output->layout              = inputMap->layout;
    output->frequency           = inputMap->frequency;
    output->minSubdivLevel      = inputMap->minSubdivLevel;
    output->maxSubdivLevel      = inputMap->maxSubdivLevel;
    output->valueFloatExpansion = payload.outputEx;


    assert(payload.fnBatchRanges);
    ctx->parallel_item_ranges(inputMap->values.count, payload.fnBatchRanges, &payload);

    return Result::eSuccess;
}

MICROMESH_API Result MICROMESH_CALL micromeshQuantizedToFloatValues(bool             outputKeepFloatExpansion,
                                                                    const ArrayInfo* input,
                                                                    const MicromapValueFloatExpansion* inputExpansion,
                                                                    // the output's arrays must be properly sized
                                                                    // their contents will be filled
                                                                    ArrayInfo*                   output,
                                                                    MicromapValueFloatExpansion* outputExpansion,
                                                                    const MessageCallbackInfo*   callbacks)
{
    OpQuantizedPayload payload;
    Result result = payload.initQuantizedToFloat(nullptr, outputKeepFloatExpansion, *inputExpansion, input, output, callbacks);
    if(result != Result::eSuccess)
    {
        return result;
    }

    payload.fnBatchRanges(0, input->count, 0, &payload);

    *outputExpansion = payload.outputEx;

    return Result::eSuccess;
}

// chains in-place value conversion from quantized to float then float to quantized
// `inputIntermediate.floatMicromap` is ignored
MICROMESH_API Result MICROMESH_CALL micromeshOpQuantizedToQuantized(OpContext ctx,
                                                                    Micromap* input,
                                                                    // the output's arrays must be properly sized
                                                                    // their contents will be filled
                                                                    Micromap* output)
{
    Result result;

    CHECK_CTX_NONNULL(ctx);
    CHECK_NONNULL(ctx, input);
    CHECK_NONNULL(ctx, output);
    CHECK_ARRAYVALID(ctx, input->values);
    CHECK_ARRAYVALID(ctx, output->values);
    CHECK_CTX_BEGIN(ctx);

    OpQuantizedPayload payload;
    result = payload.initQuantizedToQuantized(ctx, &input->values, &output->values);
    if(result != Result::eSuccess)
    {
        return result;
    }

    if(input->triangleSubdivLevels.count != output->triangleSubdivLevels.count
       || input->triangleValueIndexOffsets.count != output->triangleValueIndexOffsets.count)
    {
        LOGE(ctx, "input map and output map must have all arrays with same count");
        return Result::eInvalidRange;
    }

    if(!arrayIsEqual(input->triangleSubdivLevels, output->triangleSubdivLevels))
    {
        ctx->arrayCopy<uint16_t>(output->triangleSubdivLevels, input->triangleSubdivLevels);
    }

    if(!arrayIsEqual(input->triangleValueIndexOffsets, output->triangleValueIndexOffsets))
    {
        ctx->arrayCopy<uint32_t>(output->triangleValueIndexOffsets, input->triangleValueIndexOffsets);
    }

    output->layout              = input->layout;
    output->frequency           = input->frequency;
    output->minSubdivLevel      = input->minSubdivLevel;
    output->maxSubdivLevel      = input->maxSubdivLevel;
    output->valueFloatExpansion = input->valueFloatExpansion;


    assert(payload.fnBatchRanges);
    ctx->parallel_item_ranges(input->values.count, payload.fnBatchRanges, &payload);

    return Result::eSuccess;
}


MICROMESH_API Result MICROMESH_CALL micromeshOpQuantizedPackedToFloat(OpContext                             ctx,
                                                                      const OpQuantizedPackedToFloat_input* input,
                                                                      Micromap*                             output)
{
    Result result;

    CHECK_CTX_NONNULL(ctx);
    CHECK_NONNULL(ctx, input);
    CHECK_NONNULL(ctx, input->quantizedMicromap);
    CHECK_NONNULL(ctx, output);
    CHECK_ARRAYVALID(ctx, input->quantizedMicromap->values);
    CHECK_ARRAYVALID(ctx, output->values);
    CHECK_CTX_BEGIN(ctx);

    const MicromapPacked* inputMap = input->quantizedMicromap;

    FormatInfo inputFormatInfo;
    result = micromeshFormatGetInfo(input->quantizedMicromap->values.format, &inputFormatInfo);

    if(result != Result::eSuccess || inputMap->values.format != Format::eR11_unorm_packed_align32)
    {
        LOGE(ctx, "input->quantizedMicromap->values.format must be uncompressed eR11_unorm_packed_align32");
        return Result::eInvalidFormat;
    }

    FormatInfo outputFormatInfo;
    result = micromeshFormatGetInfo(output->values.format, &outputFormatInfo);
    if(result != Result::eSuccess || outputFormatInfo.channelType != ChannelType::eSfloat || outputFormatInfo.isCompressedOrPacked
       || outputFormatInfo.channelBitCount != 32 || outputFormatInfo.channelCount != inputFormatInfo.channelCount)
    {
        LOGE(ctx, "output->values.format must be uncompressed 32-bit SFLOAT of same channel count");
        return Result::eInvalidFormat;
    }

    if(inputMap->triangleSubdivLevels.count != output->triangleSubdivLevels.count
       || inputMap->triangleValueByteOffsets.count != output->triangleValueIndexOffsets.count
       || inputMap->values.count != output->values.count)
    {
        LOGE(ctx, "input map and output map must have all arrays with same count");
        return Result::eInvalidRange;
    }

    if(!arrayIsEqual(inputMap->triangleSubdivLevels, output->triangleSubdivLevels))
    {
        ctx->arrayCopy<uint16_t>(output->triangleSubdivLevels, inputMap->triangleSubdivLevels);
    }

    output->layout         = inputMap->layout;
    output->frequency      = inputMap->frequency;
    output->minSubdivLevel = inputMap->minSubdivLevel;
    output->maxSubdivLevel = inputMap->maxSubdivLevel;

    MicromapValueFloatExpansion ex = inputMap->valueFloatExpansion;
    if(input->outputKeepFloatExpansion)
    {
        output->valueFloatExpansion = ex;
        // keep inputs in natural [0,1] or [-1,1]
        // so reset applied expansion
        ex = MicromapValueFloatExpansion();
    }
    else
    {
        // output has scale/bias applied already, reset here
        output->valueFloatExpansion = MicromapValueFloatExpansion();
    }

    // special case
    if(inputMap->values.format == Format::eR11_unorm_packed_align32)
    {
        uint32_t dataIndex = 0;
        for(uint64_t idx = 0; idx < output->triangleSubdivLevels.count; idx++)
        {
            arraySetV(output->triangleValueIndexOffsets, idx, dataIndex);
            dataIndex += subdivLevelGetCount(arrayGetV<uint16_t>(output->triangleSubdivLevels, idx), output->frequency);
        }

        ctx->parallel_item_ranges(output->triangleSubdivLevels.count, [&](uint64_t idxFirst, uint64_t idxLast,
                                                                          uint32_t threadIndex, void* userData) {
            for(uint64_t idx = idxFirst; idx < idxLast; idx++)
            {
                uint32_t valueIdxIn                   = arrayGetV<uint32_t>(inputMap->triangleValueByteOffsets, idx);
                uint32_t valueIdxOut                  = arrayGetV<uint32_t>(output->triangleValueIndexOffsets, idx);
                const uint32_t* __restrict triangleIn = arrayGet<uint32_t>(inputMap->values, valueIdxIn);

                uint32_t count = subdivLevelGetCount(arrayGetV<uint16_t>(output->triangleSubdivLevels, idx), output->frequency);

                for(uint32_t i = 0; i < count; i++)
                {
                    uint32_t value = packedReadR11UnormPackedAlign32(triangleIn, i);
                    float    out   = (float(value) / float(0x7FF)) * ex.scale[0] + ex.bias[0];
                    arraySetV<float>(output->values, valueIdxOut + i, out);
                }
            }
        });

        return Result::eSuccess;
    }

    return Result::eInvalidFormat;
}

MICROMESH_API Result MICROMESH_CALL micromeshQuantizedPackedToFloatValues(bool             outputKeepFloatExpansion,
                                                                          const ArrayInfo* input,
                                                                          const MicromapValueFloatExpansion* inputExpansion,
                                                                          // the output's arrays must be properly sized
                                                                          // their contents will be filled
                                                                          ArrayInfo*                   output,
                                                                          MicromapValueFloatExpansion* outputExpansion,
                                                                          const MessageCallbackInfo*   callbacks)
{
    CHECK_NONNULLM(callbacks, input);
    CHECK_NONNULLM(callbacks, inputExpansion);
    CHECK_NONNULLM(callbacks, output);
    CHECK_NONNULLM(callbacks, outputExpansion);

    if(input->count != output->count)
    {
        MLOGE(callbacks, "input->count (%" PRIu64 ") and output->count (%" PRIu64 ") must match.", input->count, output->count);
        return Result::eInvalidRange;
    }

    // This code is special-cased to only handle one input and one output format:
    if(input->format != Format::eR11_unorm_packed_align32)
    {
        MLOGE(callbacks, "input->format must be Format::eR11_unorm_packed_align32.");
        return Result::eInvalidFormat;
    }

    if(output->format != Format::eR32_sfloat)
    {
        MLOGE(callbacks, "output->format must be Format::eR32_sfloat.");
        return Result::eInvalidFormat;
    }

    if(input->count > std::numeric_limits<uint32_t>::max())
    {
        MLOGE(callbacks, "input->count (%" PRIu64 ") was too large to be stored in an unsigned 32-bit integer.", input->count);
        return Result::eInvalidRange;
    }

    MicromapValueFloatExpansion ex = *inputExpansion;
    if(outputKeepFloatExpansion)
    {
        *outputExpansion = ex;
        // keep inputs in natural [0,1] or [-1,1]
        // so reset applied expansion
        ex = MicromapValueFloatExpansion();
    }
    else
    {
        // output has scale/bias applied already, reset here
        *outputExpansion = MicromapValueFloatExpansion();
    }

    for(uint32_t i = 0; i < static_cast<uint32_t>(input->count); i++)
    {
        uint16_t value = packedReadR11UnormPackedAlign32(input->data, i);
        float    out   = (float(value) / float(0x7FF)) * ex.scale[0] + ex.bias[0];
        arraySetV<float>(*output, i, out);
    }

    return Result::eSuccess;
}

}  // namespace micromesh
