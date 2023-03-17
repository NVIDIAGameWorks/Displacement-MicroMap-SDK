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
* For each collapsed edge, flag the deduplicated vertices for collapse
*/


#undef MAIN_NAME
#define MAIN_NAME remeshCollapseFlag

#include "remesh_common.h"


uint unpackEdgeIndex(uint64_t edgeDesc)
{
    u32vec2 e = unpack32(edgeDesc);
    return e.x;
}


// Unpack the minimum edge cost descriptor of the triangle at triIndex, and check whether
// it corresponds to edgeIndex
bool isMinEdge(RM_DATA_ARG uint triIndex, uint edgeIndex)
{
    uint minEdge = unpackEdgeIndex(RM_DATA(scratchTriangleDescs)[triIndex]);
    if(minEdge != ~0u)
        return minEdge == edgeIndex;
    return true;
}


// Check whether the edge at edgeIndex is the one with lowest cost stored in its adjacent triangles
bool isCheapestInConnectedTriangles(RM_DATA_ARG uint vertex, float cost, uint edgeIndex)
{
    uint encodedLastTriangleIndex = rvGetLastTriangle(vertex);

    while(encodedLastTriangleIndex != ~0u)
    {
        uvec2 lastTri           = decodePreviousTriangle(encodedLastTriangleIndex);
        uint  lastTriangleIndex = lastTri.x;
        if(hasEdge(lastTriangleIndex, edgeIndex) && !isMinEdge(RM_DATA_VAL lastTriangleIndex, edgeIndex))
            return false;
        uint localVertexIndex    = lastTri.y;
        encodedLastTriangleIndex = rtGetPreviousTriangle(lastTriangleIndex, localVertexIndex);
    }
    return true;
}

bool isCheapest(RM_DATA_ARG uint edgeIndex)
{
    // Check whether this edge has the lowest cost in its connected triangles
    return isCheapestInConnectedTriangles(RM_DATA_VAL reGetVertices(edgeIndex).x, reGetCost(edgeIndex), edgeIndex);
}


MAIN
{
    if(RM_DATA(currentState).errorState != eRemesherErrorNone)
        return;

    uint index = uint(gl_GlobalInvocationID.x);
    if(index >= RM_DATA(scratchMetadata).edgeCount)
        return;


    uvec2 edgeVertices = reGetVertices(index);
    if(edgeVertices.x == ~0u || edgeVertices.y == ~0u)
        return;

    bool isNonManifold = false;

    if(!isNonManifold)
    {
        // The error threshold for decimation is provided by the user through constants. For relaxation
        // the cost of an edge is the change of anisotropy in its neighborhood after shortening, the change
        // being normalized around 1 (= no change).
        float errorThreshold = (RM_CONSTANTS.remeshingMode == eDecimate) ? RM_CONSTANTS.errorThreshold : 1.f;

        if(reGetCost(index) > errorThreshold)
            return;

        // Check the validity of the collapse operation
        bvec2 isFlaggedVertex      = bvec2(false);
        bvec2 isDoubleMarked       = bvec2(false);
        bvec2 isIntersectionVertex = bvec2(false);
        for(uint i = 0; i < 2; i++)
        {
            uint flags     = rvGetFlags(edgeVertices[i]);
            bool isFlagged = hasFlag(flags, RM_V_MARKED);
            if(isFlagged)
            {
                isFlaggedVertex[i] = true;
                if(hasFlag(flags, RM_V_DOUBLE_MARKED))
                    isDoubleMarked[i] = true;
                isIntersectionVertex[i] = hasFlag(flags, RM_V_FIXED);
            }
        }
        bool isBlockedEdge;
        isBlockedEdge = (isFlaggedVertex[0] != isFlaggedVertex[1]) || (isIntersectionVertex[0] || isIntersectionVertex[1])
                        || (isDoubleMarked[0] && isDoubleMarked[1]);


        if(isBlockedEdge)
            return;
    }


    // If the edge is the one with the lowest cost of its neighborhood, flag the deduplicated vertices for merging
    if(isNonManifold || isCheapest(RM_DATA_VAL index))
    {
        rvSetDedupMerged(min(edgeVertices.x, edgeVertices.y), max(edgeVertices.x, edgeVertices.y));
    }
}