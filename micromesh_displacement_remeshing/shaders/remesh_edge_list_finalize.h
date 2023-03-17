/*
* Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
* 
* Finalize the edge list by assigning the unique edge indices to the triangles
* Also uses the reference counter of the edges to further determine their status
* 
*/


#undef MAIN_NAME
#define MAIN_NAME remeshEdgeListFinalize

#include "remesh_common.h"

// Mark the vertices on the mesh edge as being on a discontinuity
void markBorderTriangleEdges(RM_DATA_ARG uvec2 edgeIndices, uvec3 triangleIndices)
{
    // Find the vertex that is not on the edge
    uint nonEdgeVertex;
    for(uint i = 0; i < 3; i++)
    {
        if(triangleIndices[i] != edgeIndices.x && triangleIndices[i] != edgeIndices.y)
        {
            nonEdgeVertex = triangleIndices[i];
            break;
        }
    }
    // For decimation the edge vertices are marked as being on a discontinuity,
    // which will still allow for a collapse along the edge. For relaxation the
    // edge vertices are fixed
    uint borderFlags = RM_V_MARKED | RM_V_EDGE;

    // If the vertex not lying on the edge is on a discontinuity the edge
    // vertices are then at the intersection of the attribute discontinuity and
    // the mesh edge. The vertices are then marked as fixed
    if(hasFlag(rvGetFlags(nonEdgeVertex), RM_V_MARKED))
    {
        borderFlags |= RM_V_FIXED;
    }
    rvAtomicAddFlag(edgeIndices.x, borderFlags);
    rvAtomicAddFlag(edgeIndices.y, borderFlags);
}


// Compute the area covered by the triangle in UV space
float getUVArea(RM_DATA_ARG uint triIndex)
{
    uvec3 indices;
    vec2  texCoord[3];
    for(uint i = 0; i < 3; i++)
    {
        indices[i]  = RM_DATA(triangles)[3 * triIndex + i];
        texCoord[i] = cvGetOutputTexCoord(indices[i]);
    }

    return 0.5f
           * abs(texCoord[0].x * (texCoord[1].y - texCoord[2].y) + texCoord[1].x * (texCoord[2].y - texCoord[0].y)
                 + texCoord[2].x * (texCoord[0].y - texCoord[1].y));
}


MAIN
{
    if(RM_DATA(currentState).errorState != eRemesherErrorNone)
        return;

    uint index = uint(gl_GlobalInvocationID.x);

    if(index >= RM_DATA(scratchMetadata).uncompactedTriangleCount)
        return;

    if(!rtGetIsValid(index))
        return;
    uvec3 indices = siGetDedupTriangle(index);

    uvec2 edges[3];
    edges[0] = uvec2(indices[0], indices[1]);
    edges[1] = uvec2(indices[0], indices[2]);
    edges[2] = uvec2(indices[2], indices[1]);

    for(uint i = 0; i < 3; i++)
    {
        edges[i] = uvec2(min(edges[i].x, edges[i].y), max(edges[i].x, edges[i].y));
    }


    uvec3 edgeIndices = uvec3(~0u);

    for(uint i = 0; i < 3; i++)
    {
        uint h = siGetHashIndex(3 * index + i);
        if(h == RM_H_NOT_SET)
        {
            continue;
        }
        edgeIndices[i] = rhGetStoredIndex(h);

        {
            // If the decimation has a limit on how many original vertices are represented within a single
            // simplified triangle, store the maximum original vertex counter of the adjacent triangles
            // in the edge cost. The cost itself will be deduced in the edge cost propagation kernel
            if(RM_CONSTANTS.clampedSubdLevel != ~0u)
                reAtomicMaxCost(edgeIndices[i], float(rtGetAggregatedCounter(index)));
            // If a target displacement map resolution is provided, we compute the number of texels covered
            // by the resulting triangle, and store the maximum texel count in the edge cost for later use
            if(RM_CONSTANTS.dispMapResolution.x > 0 && RM_CONSTANTS.dispMapResolution.y > 0)
            {
                float uvArea     = getUVArea(RM_DATA_VAL index);
                float texelCount = uvArea * (RM_CONSTANTS.dispMapResolution.x * RM_CONSTANTS.dispMapResolution.y);
                if(RM_CONSTANTS.clampedSubdLevel != ~0u)
                    reAtomicMaxCost(edgeIndices[i], texelCount);
            }
        }

        // If an edge is referenced only once it lies on the edge of the mesh. We then mark the corresponding vertices
        // as being on a discontinuity
        if(rhGetRefCounter(h) == 1)
        {
            markBorderTriangleEdges(RM_DATA_VAL edges[i], indices);
        }
        if(rhGetRefCounter(h) > 2)
        {
            rvAtomicAddFlag(edges[i].x, RM_V_DEBUG);
            rvAtomicAddFlag(edges[i].y, RM_V_DEBUG);
        }
    }

    rtSetEdgeIndices(index, edgeIndices);

    rtResetAggregatedCounter(index);
}
