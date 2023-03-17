# micromesh_core

New core library for micromesh / micromap data processing.

Other `micromesh_...` libraries will provide additional functionality and depend on it.

## Goals

- "stable" C-style interface, just passing structs to API functions.
- Utility inline functions allow C++ types.
- Try to use simple data types / structs
- No allocation management for user data
- No serialization of user data
- No additional dependencies

## Public Data Structures (`micromesh/micromesh_types.h`)

Within are the most common data types used to communicate with the library.

At the core lies the `micromesh::ArrayInfo` and its various "typed" variants (e.g. `micromesh::ArrayInfo_uint32`).
This allows passing values and properties as pointer & stride combination so that the library can operate on
data that can be embedded in the users' structs. 

**Note:** To avoid having a lot of types in the API the `ArrayInfo` stores a non-const `void*` data pointer. Which means
some of the utility functions do a `const_cast`. When ArrayInfo structs are passed const, then the data pointer is used
const only as well.

The most commonly used interface will be the `micromesh::Micromap` struct, which contains information
for a typical micromap using several `micromesh::ArrayInfo` for its properties.

``` c++
micromesh::Micromap micromap;

// different ways to init arrays
// manually construct / set fields
micromap.values = {myvalues.data(), myvalues.size(), micromesh::Format:eR32_sfloat, sizeof(float)};
// use a typed array (_type suffix) constructor which implicitly sets ArrayInfo::format and `byteStride`. 
// In this example it would be to `Format:eR32_sfloat` and `byteStride = sizeof(float)`
micromap.values = micromesh::ArrayInfo_float(myvalues.data(), myvalues.size());

// for already typed ones might just set data alone
struct MyTriangle {
  uint16_t subdivLevel;
  uint16_t foo;
};
std::vector<MyTriangle> triangles(100);
// micromap.triangleSubdivLevels is ArrayInfo_uint16
arraySetData(micromap.triangleSubdivLevels, &triangles[0].subdivLevel, triangles.size(), sizeof(Triangle));

// can also derive information from containers with .size(), .data() and value_type.
std::vector<uint32_t> myOffsets(100);
// this assert that `sizeof(myOffsets's value_type)` matches sizeof(uint32_t)
// as micromap.triangleValueIndexOffsets is ArrayInfo_uint32.
arraySetDataVec(micromap.triangleValueIndexOffsets, myOffsets);

```

`micromesh::MeshTopology` is another frequently used interface to provide information about connectivity
within a triangle mesh. It is, for example, used to ensure watertight values along triangle edges within micromaps.

Other structs in this header only exist for documentation purposes to give information about how certain variable
and struct field names are used in a standardized fashion.

## Utilities (`micromesh/micromesh_utils.h`)

Several functions to work with the data structures or setup default interfaces. Some of these may not be required
by users, given they target development of other micromesh libraries. For example, we do not recommend users to
iterate data using the various `array` functions, given users bring their own data structures
for storage.

## Operations (`micromesh/micromesh_operations.h`)

- `micromesh::OpContext` implements some very simple multi-thread support to accelerate larger operations. A context is provided to all threaded operations. 
  Any non-context functions are thread-safe.

Most complex operations tend to use following pattern (replacing `<Name>`):

- `struct micromesh::Op<Name>_input`: Interface struct for read-only inputs to an operation
- `struct micromesh::Op<Name>_output`: Interface struct for written outputs of an operation
- `Result micromesh::micromeshOp<Name>Begin(context, &input, &output)`: The **begin function** prepares the outputs and provides information about allocation sizes. Users should check the documentation which output fields are written and must be reacted upon (most commonly the `micromesh::ArrayInfo::count`). The begin function must be followed by corresponding end function or aborted via `micromeshOpContextAbort(context)`.
- `Result micromesh::micromeshOp<Name>End(context, &output)`: The **end function** now assumes that all pointers in output are valid and properly sized to complete the operation.
- `callbacks` passed to the operations must be thread-safe. They will be passed  a `userData` pointer as well as a `threadIndex` that lies within the number of threads that the context is allowed to use.

## GPU Operation Interfaces (`micromesh/micromesh_gpu.h`)

- Experimental design, work in progress
- API agnostic interface that allows to run certain library operations on GPU.
- All API calling must be done by the users, as well as uploads/downloads etc.
- The library operations that do have GPU support, will expose functions to query detailed information 
  about which shaders/pipelines and bindings are expected to be set and what kind of draws or dispatches to execute for each pass.

## Extended format types (`micromesh/micromesh_format_types.h`)

For those interested in the use of `auto` etc. The various `micromesh::Format` enums can be mapped to structs and vice versa.
This header is not used by the library itself, nor is it mandatory for the users. 

## Tests (`tests/test_core.cpp`)

[This file](tests/test_core.cpp) doesn't test all functions yet, but shows a bit how the API and utilities are used.

## Internals (`micromesh_internal/...`)

These files are meant solely for micromesh library development and should
never be included by users of this or other micromesh libraries.


