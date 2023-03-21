/*
* Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
* 
* 
* Generate displacement micromesh-related subdivision level based on 
* the number of original vertices represented by each simplified triangle
* 
*/


#undef MAIN_NAME
#define MAIN_NAME remeshGenerateSubdivisionInfo

#include "remesh_common.h"

// Compute the subdivision level corresponding to a given vertex count
uint subdivisionLevel(uint vertexCount)
{
    uint targetCount = vertexCount;
    uint l           = 1;
    while(true)
    {
        if(targetCount < subdivLevelGetVertexCount(l))
            return l;
        l++;
    }
}

// For a given triangle and edge, find the other triangle adjacent to that edge
uint findOtherTriangle(RM_DATA_ARG uint triIndex, uint edgeIndex)
{
    uint vertexIndex = reGetVertices(edgeIndex).x;

    uint encodedLastTriangleIndex = rvGetLastTriangle(vertexIndex);
    while(encodedLastTriangleIndex != ~0u)
    {
        uvec2 lastTri           = decodePreviousTriangle(encodedLastTriangleIndex);
        uint  lastTriangleIndex = lastTri.x;
        if(lastTriangleIndex != triIndex)
        {
            uvec3 triEdges = rtGetEdgeIndices(lastTriangleIndex);
            for(uint i = 0; i < 3; i++)
            {
                if(triEdges[i] == edgeIndex && rtGetIsValid(lastTriangleIndex))
                    return lastTriangleIndex;
            }
        }
        uint localVertexIndex    = lastTri.y;
        encodedLastTriangleIndex = rtGetPreviousTriangle(lastTriangleIndex, localVertexIndex);
    }
    return ~0u;
}

// Find the other triangles adjacent to the edges of the triangle at triIndex
uvec3 findNeighbors(RM_DATA_ARG uint triIndex)
{
    uvec3 edges = rtGetEdgeIndices(triIndex);
    uvec3 res   = uvec3(~0u);
    for(uint i = 0; i < 3; i++)
    {
        uint edgeIndex = edges[i];
        res[i]         = findOtherTriangle(RM_DATA_VAL triIndex, edgeIndex);
    }

    return res;
}

// Find the maximum subdivision level of the immediate neighbors of the triangle at index
uint findMaxNeighborSubd(RM_DATA_ARG uint index)
{
    uvec3 neighbors = findNeighbors(RM_DATA_VAL index);
    uint  subd      = 0;
    for(uint i = 0; i < 3; i++)
    {
        if(neighbors[i] != ~0u)
            subd = max(subd, rmmGetCurrentSubdivisionLevel(neighbors[i]));
    }
    return subd;
}

// The edges of the remesher are ordered with vertices (0,1), (0,2), (1,2) but
// the meshops::DeviceMesh expects edge flags ordered (0,1), (1,2), (0,2)
uint reorderEdgeFlags(uint flags)
{
    uint res;
    res = (flags & 0x1) | ((flags & 0x2) << 1) | ((flags & 0x4) >> 1);
    return res;
}

// Compute the edge flags depending on the subd level of the neighbors
uint computeEdgeDecimationFlags(RM_DATA_ARG uint triIndex)
{
    uint level = rmmGetCurrentSubdivisionLevel(triIndex);
    if(level == 0)
        return 0;
    uvec3 neighbors = findNeighbors(RM_DATA_VAL triIndex);
    uint  res       = 0;
    for(uint i = 0; i < 3; i++)
    {
        if(neighbors[i] == ~0u)
            continue;
        uint l = rmmGetCurrentSubdivisionLevel(neighbors[i]);

        if(l < level)
            res |= (1 << i);
    }
    return reorderEdgeFlags(res);
}

MAIN
{
    if(RM_DATA(currentState).errorState != eRemesherErrorNone)
        return;

    uint index = uint(gl_GlobalInvocationID.x);

    if(RM_CONSTANTS.iterationIndex > 12)
    {
        RM_DATA(currentState).errorState = eRemesherErrorInvalidConstantValue;
        return;
    }


    if(RM_CONSTANTS.iterationIndex < 12)
    {
        if(index >= RM_DATA(scratchMetadata).uncompactedTriangleCount)
            return;

        if(!rtGetIsValid(index))
        {
            if(RM_CONSTANTS.iterationIndex == 0)
            {
                rmmSetTargetSubdivisionLevel(index, 0);
                rmmSetCurrentSubdivisionLevel(index, 0);
            }
            return;
        }


        // First iteration, initialize the subd level per triangle based on the number of
        // representative vertices
        // FIXME: count the displacement map texels as well
        if(RM_CONSTANTS.iterationIndex == 0)
        {
            uint subd = min(RM_CONSTANTS.clampedSubdLevel, subdivisionLevel(rtGetAggregatedCounter(index)));
            rmmSetTargetSubdivisionLevel(index, subd);
            rmmSetCurrentSubdivisionLevel(index, 0);
            return;
        }

        // Final iteration, set the decimation flags for the edges of the triangles neighboring
        // triangles with lower subd level
        if(RM_CONSTANTS.iterationIndex == 11)
        {
            rmmSetDecimationFlags(index, computeEdgeDecimationFlags(RM_DATA_VAL index));
            rmmCleanup(index);

            return;
        }

        // Get the current subd level, the maximum subd level of the neighbors, and potentially
        // raise the subd level to have a max difference of 1 with the neighbors
        uint currentSubd = rmmGetCurrentSubdivisionLevel(index);
        uint targetSubd  = findMaxNeighborSubd(RM_DATA_VAL index);

        if(targetSubd > (currentSubd + 1))
        {
            rmmSetTargetSubdivisionLevel(index, targetSubd - 1);
        }
        else
        {
            rmmSetTargetSubdivisionLevel(index, rmmGetCurrentSubdivisionLevel(index));
        }
    }
    else  // Iteration 12, reencode the min/max displacement bounds into actual (min, max-min) floats
    {
        if(index >= RM_CONSTANTS.vertexCount)
            return;

        rvFinalizeMinMaxDisplacement(index);
    }
}
