# micromesh_displacement_compression

This library compresses and decompresses UNORM11 barycentric micromesh data into compact block formats, much like [block-compressing](https://www.reedbeta.com/blog/understanding-bcn-texture-compression-formats/) a texture.

These block formats use a hierarchical compression scheme designed for uniform triangle subdivision, so different levels of detail can be read without re-encoding. Each level of detail is expressed by subdividing the previous level, then applying corrections scaled by an appropriate shift value. Please refer to the supplemental slides for illustrated details.

To compress displacements over a mesh, you'll probably want to use the `micromeshOpCompressDisplacement{Begin, End}` API. Although each micromap triangle can have a different subdivision level, these functions will ensure all displacement is watertight so long as adjacent faces' subdivision levels differ by 0 or 1. To use it, fill `OpCompressDisplacement_settings` and `OpCompressDisplacement_input`, and call `micromeshOpCompressDisplacementBegin`. This will tell you the amount of memory you must allocate within `MicromapCompressed`. Once you've allocated that memory and set `MicromapCompressed`'s `data` pointers, call `micromeshOpCompressDisplacementEnd()` to fill the `MicromapCompressed` data. (See below for a more concrete example.)

This library also contains `micromeshCompressDisplacementBlock()` and `micromeshDecompressDisplacementBlock()`, which are the low-level core of the displacement encoder: they compress and decompress blocks of UNORM11 values to compressed block formats.

## Porting guide from displacement_encoder to micromesh_displacement_compression

Previously, this library was available as a standalone [displacement_encoder](https://gitlab-master.nvidia.com/barycentric_displacement/displacement_encoder) library. It's now part of the standard micromesh SDK, and has a number of nice improvements:

* **Faster**: Mesh encoding is now parallelized -- this alone gave roughly a 2X speedup to the micromesh optimizer. Future optimization work will be done on micromesh_displacement_compression.
* **Lower memory usage**: Internal structures now use 16-bit unsigned types in triangular arrays, while displacement_encoder used `addr` indexing with 32-bit types.
* **Simplified API**: The API no longer has support for mixed subtriangle formats, which means apps no longer have to call `MeshEncoder:getPrimitiveBlocksInfo()` and `MeshEncoder::fillPrimitiveBlocksData()`. `MeshTopology` is now part of `micromesh_core` and shared with several other functions.
* **Improved error messages**: Forgot to set a struct member? You'll quickly get a message callback with a detailed explanation, like using the Vulkan Validation Layers.

However, there are a number of breaking changes — for instance:

* Encoding and packing are combined in the public API: `micromeshCompressDisplacementBlock()` is roughly equivalent to `BlockEncoder::encodeValues()`, followed by `BlockEncoder::packMicroMeshData()`. `micromeshDecompressDisplacementBlock()` is roughly equivalent to `BlockEncoder::decodeValues()`.
* Previously, each micromap triangle would have a `BasePrimitive` struct specifying the value offset, mesh triangle, and subdivision level. Now, value offsets and subdivision levels are contained within `Micromap`, while the micromap-to-mesh-triangle mapping is now a mesh-to-micromap-triangle mapping, given by `OpCompressDisplacement_input::meshTriangleMappings`.
* Compression control modes other than PSNR have been removed, since they weren't used very often. I typically use a PSNR of 35 dB on fitted data and 50 dB on unfitted data.
* `Micromap` and `MicromapCompressed` are not generally compatible with helper classes such as `DisplacementBaryFromEncoder`. 

One of the most useful resources for using the new encoder API is the [micromesh optimizer source code](https://gitlab-master.nvidia.com/barycentric_displacement/partner_tools/-/blob/neilbickford/optimizer/micromesh_optimizer/lib/main_optimize.cpp). (Depending on what you're doing, you may want to use the high-level micromesh optimizer, instead of the displacement encoder directly!) If you're currently using `DisplacementBaryFromEncoder`, this includes a `DisplacementBary14FromEncoderSDK` class that uses this encoder to generate `nv::DisplacementBary` structs suitable for use with version 14 bary functions. It also includes a `MeshTopologyBuilder` class that manages storage much like `nv::MeshTopology`.

All tests and examples have also been updated to use the new API.

Note that the internal `displacement_mesh_codec.h` header may look a lot like the old `MeshEncoder` API, but please don't use it directly! This internal header assumes its inputs are correct (validation logic is mostly in the public API, so you won't get messages if things are wrong), and there's no stability guarantees on internal headers.

If you run into any trouble using the new API, let me (Neil Bickford) know! I'm happy to provide tips for using the new API, and maybe there's a way we can improve it or make porting easier.

## Examples

### Encoding and Decoding a Block with `BlockEncoder`

We currently support three `BlockFormat`s, members of the `eDispC1_r11_unorm_block` family.

* `eR11_unorm_lvl3_pack512` encodes 45 scalar UNORM11 values for the vertices of 64 microtriangles (i.e. a triangle split in half 3 times - subdivision level 3) in 64 bytes. It's a lossless format (11 bits * 45 values = 495 bits < 64 * 8 bits), but requires one byte per triangle.
* `eR11_unorm_lvl4_pack1024` encodes 153 scalar UNORM11 values for the vertices of 256 microtriangles in 128 bytes. This format is lossy, but gives very good perceptual quality: corrections for levels 0, 1, and 2 are lossless; level 3 uses 10-bit quantized corrections, and level 4 uses 5-bit quantized corrections.
* `eR11_unorm_lvl5_pack1024` encodes 561 scalar UNORM11 values for the vertices of 1024 triangles in 128 bytes. Corrections for levels 0 and 1 are lossless, while levels 2, 3, 4, and 5 have corrections quantized to 8 bits, 4 bits, 2 bits, and 1 bit, respectively.

Both compressed block formats have shift bits for the interior and three edges for each compressed level that allow high-frequency details to be represented. However, when used for baked displacement, we've noticed that subdividing a triangle at level n >= 4 and encoding using `R11_UNORM_LVL4_PACK1024` usually gives perceptually better results than subdividing at level n-1 and encoding using the lossless format, or subdividing at level n+1 and encoding using `R11_UNORM_LVL5_PACK1024` (though there can be space improvements to the latter).

Here's how to use `BlockEncoder` to encode a single subdivision level 4 block using `R11_UNORM_LVL4_PACK1024` based on data from a real model. Note that in most cases, one won't use `micromeshCompressDisplacementBlock()` directly — rather, one will use `micromeshOpCompressDisplacement{Begin,End}` to encode an entire mesh at once and handle watertightness issues (such as when two adjacent base triangles or subtriangles have different subdivision levels or formats).

```c++
// Specify the values for the block.
// These are stored in an u16 array in u-major order: a triangular array,
// u vertically, v horizontally.
// In general code, this should have subdivLevelGetVertexCount(subdivLevel)
// elements and be filled using the functions from a MicromeshLayout,
// instead of hardcoding this information.
const uint32_t                  subdivLevel = 4;
const std::array<uint16_t, 153> referenceValues{
1319, 1339, 1321, 1423, 1375, 1241, 1309, 1243, 1087, 1045, 593, 273, 435, 615, 627, 807, 947,
1435, 1343, 1369, 1431, 1349, 1261, 1297, 1119, 981,  941,  467, 311, 467, 619, 647, 849,
1465, 1359, 1377, 1427, 1323, 1281, 1237, 1077, 1067, 767,  445, 259, 421, 661, 689,
1451, 1433, 1363, 1405, 1355, 1257, 1119, 1067, 913,  635,  473, 163, 423, 709,
1385, 1385, 1351, 1443, 1329, 1155, 967,  1043, 825,  649,  249, 107, 543,
1395, 1347, 1351, 1419, 1265, 991,  1039, 867,  691,  595,  410, 225,
1421, 1339, 1333, 1373, 1135, 953,  969,  739,  713,  383,  161,
1329, 1275, 1307, 1313, 875,  997,  815,  759,  611,  183,
1235, 1205, 1315, 1167, 963,  815,  757,  815,  551,
1165, 1189, 1235, 1021, 977,  795,  879,  803,
1069, 1083, 1257, 965,  785,  897,  941,
1047, 1123, 1097, 907,  893,  1037,
1009, 1099, 903,  899,  1065,
989,  887,  1021, 1095,
839,  1021, 1137,
945,  1201,
1185};

// Define the compression settings we'll use.
DisplacementBlock_settings settings;
micromeshLayoutInitStandard(&settings.decompressedLayout, StandardLayoutType::eUmajor);  // Use a U-major layout.
settings.decompressedFormat = Format::eR11_unorm_pack16;  // Our input is 11-bit values in a u16 array
settings.subdivLevel        = 4;                          // at subdivision level 4
settings.compressedFormat = Format::eDispC1_r11_unorm_block;  // Compress to a format in the eDispC1_r11_unorm_block family
settings.compressedBlockFormatDispC1 = BlockFormatDispC1::eR11_unorm_lvl4_pack1024;  // The level-4, 1024-byte format

// Allocate space for the compressed block.
// Here we'll show general usage of micromeshBlockFormatDispC1GetInfo():
FormatInfo blockFmtInfo{};
if(micromeshBlockFormatDispC1GetInfo(settings.compressedBlockFormatDispC1, &blockFmtInfo) != Result::eSuccess)
{
    // This should never happen, but often good to check:
    PRINT_AND_ASSERT_FALSE("Failed to get format info!\n");
    return false;
}
std::vector<uint8_t> packedBlock(blockFmtInfo.byteSize);

uint64_t scratchSize = micromeshGetDisplacementBlockScratchSize(&settings);
std::vector<uint8_t> scratchData(scratchSize);

// Encode!
const Result encodeResult =
    micromeshCompressDisplacementBlock(&settings, scratchSize, scratchData.data(), referenceValues.data(), packedBlock.data(), &messenger);
// Also good to check for errors here; usually failures will be due to
// incorrect settings, so we shouldn't see errors here.
TEST_SUCCESS(encodeResult);
```

Here are what the contents of `packedBlock` look like, starting with bit 1023, down to bit 0. I've grouped and annotated the bits to show the important components of a compressed block.

```

┌Top two bits are always 0
│  ┌Shift bits for level 3: {interior, edge 0, edge 1, edge 2} = {0, 0, 0, 0}
│  │       ┌Shift bits for level 4: {5, 3, 4, 4}
├┐ ├─┬─┬─┐ ├───┬───┬───┬──
00 0 0 0 0 100 100 011 101 (1006)
Bit 1005 is unused:
                    0
Bits 465-1004: 3*T_8 subdivision level 4 corrections, 5 bits each
                     00000 00010000 (992)
01111100 01100001 10000100 01100101 (960)
11110111 00100011 11010001 10110111 (928)
11100000 00000001 00110111 10110011 (896)
01100010 11111110 00100000 11111111 (864)
01011111 00000100 10000101 00011000 (832)
01111111 11110001 00000100 00000000 (800)
11101110 00000100 00010100 00000000 (768)
10000001 00010111 10111101 11010000 (736)
10000100 00011101 00010000 00111101 (704)
11110001 10010011 11000001 11101000 (672)
10111110 01011101 10000111 10100001 (640)
11110000 00000010 00101111 00000111 (608)
11100011 00000111 10110100 00011111 (576)
10000000 00000011 00011000 00000001 (544)
11111111 11111100 00100000 00010000 (512)
10000010 00100010 10001100 01011110 (480)
00101111 10000011

Bits 165-464: 3*T_4 subdivision level 3 corrections, 10 bits each
                  11011110 01111001 (448)
00011110 00000000 11011111 10100011 (416)
11101011 00000000 11001100 11111101 (384)
00100110 11100011 01100100 00101000 (352)
00000001 01111111 00000001 10011011 (320)
11010111 10010110 10000111 10011011 (288)
00011111 10101000 10010111 11001001 (256)
01111100 01100000 11101000 10011100 (224)
00110111 10000110 01111111 11010000 (192)
11100011 11111110 11111100 110

Bits 66-164: 3*T_2 subdivision level 2 corrections, 11 bits each
                              00000 (160)
00011011 10011001 01011011 10101110 (128)
01101110 00110001 01000010 00110000 (96)
01101100 00010101 00000010 101100

Bits 33-65: 3*T_1 subdivision level 1 corrections, 11 bits each
                                 11 (64)
11110111 11011111 11011111 1010010  (33)

Bits 0-32: 33 bits for level 0 corrections (anchors)
                                  0 (32)
01110110011 10010100001 10100100111 (0)
=947        =1185       =1319
```

Note that the three level 0 corrections are the values at the corners of the triangle.

Finally, we can use `BlockEncoder` to decode packed blocks as well (though this can be done faster on the GPU). Here's an example:

```c++
std::vector<uint16_t> decodedValues(subdivLevelGetVertexCount(subdivLevel));
// We can use the same settings to reverse the process!
const DisplacementBlock_settings decompressionSettings = settings;
TEST_SUCCESS(micromeshDecompressDisplacementBlock(&decompressionSettings, scratchSize, scratchData.data(), packedBlock.data(), decodedValues.data(), &messenger));
```

The decoded values are pretty close to our input!

```
Block values (u down, v to the right):
1319 1336 1321 1428 1375 1246 1309 1246 1087 1048 593  274  435  611  627  803  947
1432 1329 1381 1440 1349 1252 1305 1130 981  926  455  312  460  620  658  850
1465 1357 1377 1414 1323 1280 1237 1088 1067 756  445  273  421  651  689
1449 1445 1364 1401 1358 1251 1134 1081 914  635  475  175  418  712
1385 1400 1351 1436 1329 1148 967  1056 825  633  249  108  543
1395 1354 1342 1427 1264 987  1032 865  705  609  397  224
1421 1345 1333 1362 1135 956  969  745  713  373  161
1328 1284 1292 1321 889  998  799  767  600  180
1235 1211 1315 1171 963  828  757  814  551
1168 1192 1222 1014 970  803  881  810
1069 1067 1257 957  785  895  941
1047 1133 1112 908  893  1035
1009 1084 903  888  1065
988  871  1020 1101
839  1020 1137
948  1193
1185
```

### Encoding a Mesh with `micromeshOpCompressDisplacement`

Here's an example showing how to encode a full mesh using the `micromeshOpCompressDisplacement{Begin, End}` functions. Our mesh will have only two triangles, but one triangle will have subdivision level 4 and the second will have subdivision level 5.

First, we define the connectivity of the mesh by making an index buffer and creating a `MeshTopology` structure from it. This takes a while, but you can use similar code or a helper function to create `MeshTopology` structs for use with other parts of the micromesh SDK.

```c++
// Build the mesh topology for the following mesh:
// 0-1
//  2-3
constexpr size_t                           NUM_TRIANGLES = 2;
constexpr size_t                           NUM_VERTICES  = 4;
std::array<Vector_uint32_3, NUM_TRIANGLES> indices{0, 1, 2, 1, 3, 2};

// Allocate space for triangleEdges, vertexEdgeRanges, and vertexTriangleRanges
MeshTopology topo{};
arraySetDataVec((ArrayInfo&)topo.triangleVertices, indices);
topo.triangleVertices.count = NUM_TRIANGLES;
topo.triangleEdges.count    = NUM_TRIANGLES;
std::vector<Vector_uint32_3> topoTriangleEdgesStorage(topo.triangleEdges.count);
topo.triangleEdges.data = topoTriangleEdgesStorage.data();

topo.vertexEdgeRanges.count = NUM_VERTICES;
std::vector<Range32> topoVertexEdgeRangesStorage(NUM_VERTICES);
topo.vertexEdgeRanges.data = topoVertexEdgeRangesStorage.data();

topo.vertexTriangleRanges.count = NUM_VERTICES;
std::vector<Range32> topoVertexTriangleRangesStorage(NUM_VERTICES);
topo.vertexTriangleRanges.data = topoVertexTriangleRangesStorage.data();

// Fill those 3 arrays and get sizes for remaining MeshTopology arrays
TEST_SUCCESS(micromeshOpBuildMeshTopologyBegin(context, &topo));


// Allocate remaining output.
std::vector<uint32_t> topoVertexTriangleConnectionsStorage(topo.vertexTriangleConnections.count);
topo.vertexTriangleConnections.data = topoVertexTriangleConnectionsStorage.data();

std::vector<uint32_t> topoVertexEdgeConectionsStorage(topo.vertexEdgeConnections.count);
topo.vertexEdgeConnections.data = topoVertexEdgeConectionsStorage.data();

std::vector<uint32_t> topoEdgeVerticesStorage(topo.edgeVertices.count * 2);
topo.edgeVertices.data = topoEdgeVerticesStorage.data();

std::vector<Range32> topoEdgeTriangleRangesStorage(topo.edgeTriangleRanges.count);
topo.edgeTriangleRanges.data = topoEdgeTriangleRangesStorage.data();

std::vector<uint32_t> topoEdgeTriangleConnectionsStorage(topo.edgeTriangleConnections.count);
topo.edgeTriangleConnections.data = topoEdgeTriangleConnectionsStorage.data();

// Okay, now build the topology!
TEST_SUCCESS(micromeshOpBuildMeshTopologyEnd(context, &topo));
```

Next, we specify the subdivision level of each triangle, and generate random value data.

```c++
// We'll have two triangles with subdiv levels 4 and 5, and the values
// will be random 11-bit UNORM values - including mismatches at edges!
Micromap uncompressed{};
micromeshLayoutInitStandard(&uncompressed.layout, StandardLayoutType::eUmajor);
uncompressed.frequency = Frequency::ePerMicroVertex;
for(int i = 0; i < 4; i++)
{
    uncompressed.valueFloatExpansion.scale[i] = 1.0f;
}
uncompressed.maxSubdivLevel = 5;

std::array<uint16_t, NUM_TRIANGLES> subdivLevels{4, 5};
uncompressed.triangleSubdivLevels.count = subdivLevels.size();
uncompressed.triangleSubdivLevels.data  = subdivLevels.data();

std::array<uint32_t, NUM_TRIANGLES> uncompressedIndexOffsets{0, subdivLevelGetVertexCount(subdivLevels[0])};
uncompressed.triangleValueIndexOffsets.count = uncompressedIndexOffsets.size();
uncompressed.triangleValueIndexOffsets.data  = uncompressedIndexOffsets.data();

std::vector<uint16_t>                   values(subdivLevelGetVertexCount(4) +
                                               subdivLevelGetVertexCount(5));
std::default_random_engine              gen;
std::uniform_int_distribution<uint16_t> dist(0, (1 << 11) - 1);
for(uint16_t& v : values)
{
    v = dist(gen);
}
uncompressed.values.count      = values.size();
uncompressed.values.data       = values.data();
uncompressed.values.byteStride = sizeof(uint16_t);
uncompressed.values.format     = Format::eR11_unorm_pack16;
```

Note that if we rendered this mesh using microdisplacement as is, we'd get seams along the common edge, since the triangles have different displacement values there! Because the index buffer has a common edge and the triangles have different subdivision levels, though, the mesh encoder will fix this discrepancy as a side effect of encoding. This isn't guaranteed if the triangles have the same subdivision level - don't rely on this to fix watertightness issues for you (use `micromeshOpSanitizeEdgeValues()` instead).

Here I have it validate the outputs for watertightness, but this can be set to `false` to save time.

```c++
// We now have the uncompressed data. Encode it!
OpCompressDisplacement_input input{};
input.data                   = &uncompressed;
input.topology               = &topo;
input.compressedFormatFamily = Format::eDispC1_r11_unorm_block;

// Rely on the encoder fixing cracks
OpCompressDisplacement_settings settings{};
settings.validateInputs  = false;
settings.validateOutputs = true;

MicromapCompressed compressed{};

TEST_SUCCESS(micromeshOpCompressDisplacementBegin(context, &settings, &input, &compressed));
```

`micromeshOpCompressDisplacementBegin()` has now set all the fields of `MicromapCompressed`, except the `data` pointers. We must allocate the space to write the compressed data, then call `micromeshOpCompressDisplacementEnd()` to finish compression.

```c++
// Allocate the output. values.{count, byteStride, format} are already set by Begin.
std::vector<uint8_t> compressedData(compressed.values.count * compressed.values.byteStride);
compressed.values.data = compressedData.data();

// The different .count fields are set by Begin:
compressed.triangleSubdivLevels.data = subdivLevels.data();

std::vector<uint32_t> compressedValueByteOffsets(compressed.triangleValueByteOffsets.count);
compressed.triangleValueByteOffsets.data = compressedValueByteOffsets.data();

std::vector<uint16_t> compressedBlockFormats(compressed.triangleBlockFormats.count);
compressed.triangleBlockFormats.data = compressedBlockFormats.data();

// Finish compression.
TEST_SUCCESS(micromeshOpCompressDisplacementEnd(context, &compressed));
```

And we're done!

## How the Mesh Encoder Works

During the encoding process an input micromap triangle can be split into multiple subtriangles, each represented by one block format.

To achieve watertightness, the encoder must ensure inner edges (between sub-triangles of the same base-triangle) can decode to matching values. This causes the encoding to run in phases, where edge values are propagated between triangles after encoding. This iterative process is currently found in `MeshEncoder::batchEncode`.

A single triangle is split until the quality criteria are met. See the `MeshEncoder::Triangle` constructors and `MeshEncoder::Triangle::encode` for more details. If split, each triangle will point to 4 child triangles. This spans a tree from the base-triangle as root and only leaves will contain data.

This relies on a few mathematical properties. Firstly, the edges must be encoded without influence from the  values in the interior of each triangle: this makes it so that if two adjacent triangles with matching edges are encoded independently, they'll make the same decisions about how to encode the edge. Secondly, we must consider formats in a specific order such that any triangle encoded using an earlier format is also representable in a later format.

## Future Work

The encoder interface currently does not allow for partial updates (e.g. in a sculpting application one might want to update only a subset of triangles and their ring-1 connections).

The encoder assumes the input UNORM11 values are already chosen to maximize fidelity. There is currently no feedback loop between defining the subdivision level of a base-triangle and the actual data it represents (via sampling from a high-res data source), and how well it can be compressed. Fitting a shell tightly around the intended displacement and using the full UNORM11 range (i.e. using direction bounds) is important to minimize quantization artifacts.
