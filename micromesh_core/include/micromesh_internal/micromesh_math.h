//
// Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//

#include <cmath>
#include <cfloat>
#include <micromesh/micromesh_types.h>

namespace micromesh
{
namespace math
{
inline Vector_float_3 cross(Vector_float_3 a, Vector_float_3 b)
{
    Vector_float_3 ret;
    ret.x = a.y * b.z - a.z * b.y;
    ret.y = a.z * b.x - a.x * b.z;
    ret.z = a.x * b.y - a.y * b.x;
    return ret;
}

inline float dot(Vector_float_3 a, Vector_float_3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline float length(Vector_float_3 v)
{
    return sqrtf(dot(v, v));
}

inline Vector_float_3 normalize(Vector_float_3 v)
{
    float norm = sqrtf(dot(v, v));
    if(norm > FLT_EPSILON)
        norm = 1.0f / norm;
    else
        norm = 0.0f;
    v.x *= norm;
    v.y *= norm;
    v.z *= norm;
    return v;
}
}  // namespace math

inline Vector_float_3 makeVector_float_3(Vector_float_2 in)
{
    return {in.x, in.y, 0};
}

inline Vector_float_4 makeVector_float_4(Vector_float_2 in)
{
    return {in.x, in.y, 0, 0};
}

inline Vector_float_4 makeVector_float_4(Vector_float_3 in)
{
    return {in.x, in.y, in.z, 0};
}

inline Vector_float_3 operator+(Vector_float_3 lhs, const Vector_float_3& rhs)
{
    Vector_float_3 ret;
    ret.x = lhs.x + rhs.x;
    ret.y = lhs.y + rhs.y;
    ret.z = lhs.z + rhs.z;
    return ret;
}

inline Vector_float_3 operator-(Vector_float_3 lhs, const Vector_float_3& rhs)
{
    Vector_float_3 ret;
    ret.x = lhs.x - rhs.x;
    ret.y = lhs.y - rhs.y;
    ret.z = lhs.z - rhs.z;
    return ret;
}

inline Vector_float_3 operator*(Vector_float_3 lhs, const Vector_float_3& rhs)
{
    Vector_float_3 ret;
    ret.x = lhs.x * rhs.x;
    ret.y = lhs.y * rhs.y;
    ret.z = lhs.z * rhs.z;
    return ret;
}

inline Vector_float_3 operator/(Vector_float_3 lhs, const Vector_float_3& rhs)
{
    Vector_float_3 ret;
    ret.x = lhs.x / rhs.x;
    ret.y = lhs.y / rhs.y;
    ret.z = lhs.z / rhs.z;
    return ret;
}

inline Vector_float_3 operator+(Vector_float_3 lhs, float rhs)
{
    Vector_float_3 ret;
    ret.x = lhs.x + rhs;
    ret.y = lhs.y + rhs;
    ret.z = lhs.z + rhs;
    return ret;
}

inline Vector_float_3 operator-(Vector_float_3 lhs, float rhs)
{
    Vector_float_3 ret;
    ret.x = lhs.x - rhs;
    ret.y = lhs.y - rhs;
    ret.z = lhs.z - rhs;
    return ret;
}

inline Vector_float_3 operator*(Vector_float_3 lhs, float rhs)
{
    Vector_float_3 ret;
    ret.x = lhs.x * rhs;
    ret.y = lhs.y * rhs;
    ret.z = lhs.z * rhs;
    return ret;
}

inline Vector_float_3 operator/(Vector_float_3 lhs, float rhs)
{
    Vector_float_3 ret;
    ret.x = lhs.x / rhs;
    ret.y = lhs.y / rhs;
    ret.z = lhs.z / rhs;
    return ret;
}

inline Vector_float_2 operator+(Vector_float_2 lhs, const Vector_float_2& rhs)
{
    Vector_float_2 ret;
    ret.x = lhs.x + rhs.x;
    ret.y = lhs.y + rhs.y;
    return ret;
}

inline Vector_float_2 operator-(Vector_float_2 lhs, const Vector_float_2& rhs)
{
    Vector_float_2 ret;
    ret.x = lhs.x - rhs.x;
    ret.y = lhs.y - rhs.y;
    return ret;
}

inline Vector_float_2 operator*(Vector_float_2 lhs, const Vector_float_2& rhs)
{
    Vector_float_2 ret;
    ret.x = lhs.x * rhs.x;
    ret.y = lhs.y * rhs.y;
    return ret;
}

inline Vector_float_2 operator/(Vector_float_2 lhs, const Vector_float_2& rhs)
{
    Vector_float_2 ret;
    ret.x = lhs.x / rhs.x;
    ret.y = lhs.y / rhs.y;
    return ret;
}

inline Vector_float_2 operator+(Vector_float_2 lhs, float rhs)
{
    Vector_float_2 ret;
    ret.x = lhs.x + rhs;
    ret.y = lhs.y + rhs;
    return ret;
}

inline Vector_float_2 operator-(Vector_float_2 lhs, float rhs)
{
    Vector_float_2 ret;
    ret.x = lhs.x - rhs;
    ret.y = lhs.y - rhs;
    return ret;
}

inline Vector_float_2 operator*(Vector_float_2 lhs, float rhs)
{
    Vector_float_2 ret;
    ret.x = lhs.x * rhs;
    ret.y = lhs.y * rhs;
    return ret;
}

inline Vector_float_2 operator/(Vector_float_2 lhs, float rhs)
{
    Vector_float_2 ret;
    ret.x = lhs.x / rhs;
    ret.y = lhs.y / rhs;
    return ret;
}

struct Vector_int32_2
{
    int32_t x;
    int32_t y;

    inline int32_t& operator[](size_t idx)
    {
        assert(idx < 2);
        return (&x)[idx];
    }
    inline const int32_t& operator[](size_t idx) const
    {
        assert(idx < 2);
        return (&x)[idx];
    }
};


inline Vector_int32_2 operator+(Vector_int32_2 lhs, const Vector_int32_2& rhs)
{
    Vector_int32_2 ret;
    ret.x = lhs.x + rhs.x;
    ret.y = lhs.y + rhs.y;
    return ret;
}

inline Vector_int32_2 operator-(Vector_int32_2 lhs, const Vector_int32_2& rhs)
{
    Vector_int32_2 ret;
    ret.x = lhs.x - rhs.x;
    ret.y = lhs.y - rhs.y;
    return ret;
}

inline Vector_int32_2 operator*(Vector_int32_2 lhs, const Vector_int32_2& rhs)
{
    Vector_int32_2 ret;
    ret.x = lhs.x * rhs.x;
    ret.y = lhs.y * rhs.y;
    return ret;
}

inline Vector_int32_2 operator/(Vector_int32_2 lhs, const Vector_int32_2& rhs)
{
    Vector_int32_2 ret;
    ret.x = lhs.x / rhs.x;
    ret.y = lhs.y / rhs.y;
    return ret;
}

inline Vector_int32_2 operator+(Vector_int32_2 lhs, int32_t rhs)
{
    Vector_int32_2 ret;
    ret.x = lhs.x + rhs;
    ret.y = lhs.y + rhs;
    return ret;
}

inline Vector_int32_2 operator-(Vector_int32_2 lhs, int32_t rhs)
{
    Vector_int32_2 ret;
    ret.x = lhs.x - rhs;
    ret.y = lhs.y - rhs;
    return ret;
}

inline Vector_int32_2 operator*(Vector_int32_2 lhs, int32_t rhs)
{
    Vector_int32_2 ret;
    ret.x = lhs.x * rhs;
    ret.y = lhs.y * rhs;
    return ret;
}

inline Vector_int32_2 operator/(Vector_int32_2 lhs, int32_t rhs)
{
    Vector_int32_2 ret;
    ret.x = lhs.x / rhs;
    ret.y = lhs.y / rhs;
    return ret;
}

}  // namespace micromesh
