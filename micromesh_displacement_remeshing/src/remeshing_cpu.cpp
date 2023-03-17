/*
* Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <micromesh/micromesh_displacement_remeshing.h>

namespace micromesh
{
MICROMESH_API uint32_t MICROMESH_CALL micromeshOpRemeshingGetScratchCount(OpContext ctx, const OpRemeshing_settings* settings)
{
    return 0;
}

MICROMESH_API Result MICROMESH_CALL micromeshOpRemeshingBegin(OpContext                   ctx,
                                                              const OpRemeshing_settings* settings,
                                                              const OpRemeshing_input*    input,
                                                              OpRemeshing_output*         output)
{
    return Result::eFailure;
}

MICROMESH_API Result MICROMESH_CALL micromeshOpRemeshingEnd(OpContext ctx, const OpRemeshing_input* input, OpRemeshing_output* output)
{
    return Result::eFailure;
}
}  // namespace micromesh