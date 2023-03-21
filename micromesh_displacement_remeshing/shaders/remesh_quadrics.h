/*
* Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
* 
* Set base values on the vertices. If QSLIM-type methods would be used,
* the quadrics initialization would go here
*/

#undef MAIN_NAME
#define MAIN_NAME remeshQuadrics


#include "remesh_common.h"

MAIN
{
    if(RM_DATA(currentState).errorState != eRemesherErrorNone)
        return;

    uint index = uint(gl_GlobalInvocationID.x);
    if(index >= RM_DATA(scratchMetadata).uncompactedTriangleCount)
        return;

    if(!rtGetIsValid(index))
    {
        return;
    }
    uvec3 indices = siGetDedupTriangle(index);

    RM_DATA(scratchTriangleDescs)[index] = ~uint64_t(0);
    rtSetEdgeIndices(index, uvec3(~0u));

    for(uint i = 0; i < 3; i++)
    {
        rtSetPreviousTriangle(index, i, rvAtomicExchLastTriangle(indices[i], encodePreviousTriangle(index, i)));
    }
}
