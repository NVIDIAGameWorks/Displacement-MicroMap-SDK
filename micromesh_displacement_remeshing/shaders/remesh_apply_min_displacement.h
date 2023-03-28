/*
* Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.


For each vertex, find and store the minimum and maximum displacement values 
stored in its adjacent triangles

*/


#ifndef __cplusplus
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#endif

#include "remesh_common.h"

#undef MAIN_NAME
#define MAIN_NAME remeshApplyMinDisplacement


// Find the min and max displacement for the vertex at vertexIndex so that displacement
// encompasses any displacement within the bounds of the triangle at triangleIndex
vec2 findMinMaxDisplacementFromTriangle(RM_DATA_ARG uint vertexIndex, uint triangleIndex)
{
    float triMin = rtGetMinDisplacement(triangleIndex);
    float triMax = rtGetMaxDisplacement(triangleIndex);
    return vec2(triMin, triMax);
}


// Find the min and max displacement values stored in the
// triangles adjacent to the vertex. Returns vec2(min, max)
vec2 findMinMaxDisplacementInFan(RM_DATA_ARG uint vertexIndex)
{
    uint encodedLastTriangleIndex = rvGetLastTriangle(vertexIndex);
    vec2 d                        = vec2(1e34f, -1e34f);
    bool isAllIgnored             = true;
    // If this function is called the vertex must be connected to at least one triangle
    if(encodedLastTriangleIndex == ~0u)
    {
        RM_DATA(currentState).errorState = eRemesherErrorDebug;
        RM_DATA(currentState).debug.x    = vertexIndex;
        return vec2(-1, 1);
    }

    int  counter = 0;
    bool updated = false;
    // Iterate over adjacent triangles
    while(encodedLastTriangleIndex != ~0u)
    {
        uvec2 lastTri           = decodePreviousTriangle(encodedLastTriangleIndex);
        uint  lastTriangleIndex = lastTri.x;


        {
            vec2 triDisp = findMinMaxDisplacementFromTriangle(RM_DATA_VAL vertexIndex, lastTriangleIndex);

            if(abs(triDisp.x) < 1e10 && abs(triDisp.y) < 1e10)
            {
                d.x     = min(triDisp.y, min(d.x, triDisp.x));
                d.y     = max(triDisp.x, max(d.y, triDisp.y));
                updated = true;
            }
            isAllIgnored = false;
        }


        uint localVertexIndex    = lastTri.y;
        encodedLastTriangleIndex = rtGetPreviousTriangle(lastTriangleIndex, localVertexIndex);
    }
    if(updated)
    {
        return d;
    }

    return vec2(uintBitsToFloat(~0u));
}


MAIN
{
    if(RM_DATA(currentState).errorState != eRemesherErrorNone)
        return;


    uint index = gl_GlobalInvocationID.x;
    if(index >= RM_DATA(scratchMetadata).uncompactedTriangleCount)
        return;

    if(!rtGetIsValid(index))
    {
        return;
    }
    uvec3 dedupIndices = siGetDedupTriangle(index);
    if(!isValid(dedupIndices))
    {
        return;
    }

    for(uint i = 0; i < 3; i++)
    {
        uint originalIndex = RM_DATA(triangles)[3 * index + i];
        uint dedupIndex    = dedupIndices[i];
        if(dedupIndex == ~0u)
        {
            rvAtomicMinDisplacement(originalIndex, 0.f);
            rvAtomicMaxDisplacement(originalIndex, 0.f);
            continue;
        }
        vec2 minMaxDisplacement = findMinMaxDisplacementInFan(RM_DATA_VAL dedupIndex);
        if(minMaxDisplacement.x == uintBitsToFloat(~0u))
        {
            rvAtomicMinDisplacement(originalIndex, 0.f);
            rvAtomicMaxDisplacement(originalIndex, 0.f);
            continue;
        }


        if(abs(minMaxDisplacement.x) < 1e10 && abs(minMaxDisplacement.y) < 1e10)
        {
            rvAtomicMinDisplacement(originalIndex, minMaxDisplacement.x);
            rvAtomicMaxDisplacement(originalIndex, minMaxDisplacement.y);
        }
        else
        {
            rvAtomicMinDisplacement(originalIndex, 0.f);
            rvAtomicMaxDisplacement(originalIndex, 0.f);
        }
    }
}
