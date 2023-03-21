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

// Intersect a ray starting at v in direction d with a plane passing through anchor with normal n
float intersectDirectionTrianglePlane(vec3 v, vec3 d, vec3 n, vec3 anchor)
{
    float nDotD = dot(n, d);
    if(nDotD != 0.f)
        return (dot(n, anchor) - dot(n, v)) / nDotD;
    return FLT_MAX;
}

// Find the min and max displacement for the vertex at vertexIndex so that displacement
// encompasses any displacement within the bounds of the triangle at triangleIndex
vec2 findMinMaxDisplacementFromTriangle(RM_DATA_ARG uint vertexIndex, uint triangleIndex)
{
    uint localIndex;
    vec3 v[3];
    vec3 d[3];

    vec3 vMax[3], vMin[3];

    float triMin = rtGetMinDisplacement(triangleIndex);
    float triMax = rtGetMaxDisplacement(triangleIndex);

    uvec3 indices = siGetDedupTriangle(triangleIndex);

    for(uint i = 0; i < 3; i++)
    {
        v[i] = cvGetOutputPosition(indices[i]);
        d[i] = cvGetVertexDirection(indices[i]);
        if(indices[i] == vertexIndex)
        {
            localIndex = i;
        }
        vMin[i] = v[i] + d[i] * triMin;
        vMax[i] = v[i] + d[i] * triMax;
    }

    vec2 displacement = vec2(triMin, triMax);

    for(uint i = 0; i < 3; i++)
    {
        if(i != localIndex)
        {
            // Intersect the plane containing the min and max displaced vertex with a ray
            // starting at the vertex of interest along its displacement direction
            // This is the more conservative estimate for max (resp. min) displacement on convex (resp. concave)
            // areas
            float d0 = 0.f;
            float d1 = 0.f;
            // The ray-plane intersection will output very large results if the displacement directions
            // of the vertex and the vertex of interest are nearly orthogonal. In this case we revert
            // to using the other estimate below
            if(abs(dot(d[localIndex], d[i])) > 0.1f)
            {
                d0 = intersectDirectionTrianglePlane(v[localIndex], d[localIndex], d[i], vMin[i]);
                d1 = intersectDirectionTrianglePlane(v[localIndex], d[localIndex], d[i], vMax[i]);
            }

            // Project the min and max displaced vertex along
            // the displacement direction of the vertex of interest
            // This is the more conservative estimate for min (resp. max) displacement on convex (resp. concave)
            // areas
            vec3  minDisp = v[i] + triMin * d[i] - v[localIndex];
            vec3  maxDisp = v[i] + triMax * d[i] - v[localIndex];
            float d2      = dot(d[localIndex], minDisp);
            float d3      = dot(d[localIndex], maxDisp);

            // Take the min and max of both calculations
            // This is required to account for the surface of original triangles whose vertices
            // lie across several decimated triangles
            float minD = min(d3, min(d2, min(d0, d1)));
            float maxD = max(d3, max(d2, max(d0, d1)));

            if(minD != FLT_MAX)
                displacement.x = min(displacement.x, minD);
            if(maxD != FLT_MAX)
                displacement.y = max(displacement.y, maxD);
        }
    }
    return displacement;
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
