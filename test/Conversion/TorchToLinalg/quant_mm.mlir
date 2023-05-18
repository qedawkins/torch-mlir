// RUN: torch-mlir-opt <%s -convert-torch-to-linalg -split-input-file -verify-diagnostics | FileCheck %s

module attributes {torch.debug_module_name = "CustomOpExampleModule"} {
  func.func @forward(%lhs: !torch.vtensor<[1,?,4096],f32>, %packed_weights: !torch.vtensor<[4096,2048],ui8>, %scales: !torch.vtensor<[4096,32,1],f32>, %zpoints: !torch.vtensor<[4096,32,1],f32>) -> !torch.vtensor<[1,?,4096],f32> {
    %unpackedtypewidth = torch.constant.int 4
    %groupsize = torch.constant.int 128
    %0 = torch.operator "brevitas.matmul_rhs_group_quant"(%lhs, %packed_weights, %scales, %zpoints, %unpackedtypewidth, %groupsize) : (!torch.vtensor<[1,?,4096],f32>, !torch.vtensor<[4096,2048],ui8>, !torch.vtensor<[4096,32,1],f32>, !torch.vtensor<[4096,32,1],f32>, !torch.int, !torch.int) -> !torch.vtensor<[1,?,4096],f32>
    return %0 : !torch.vtensor<[1,?,4096],f32>
  }
}

// module attributes {torch.debug_module_name = "CustomOpExampleModule"} {
//   func.func @forward(%lhs: !torch.vtensor<[1,1,11008],f32>, %packed_weights: !torch.vtensor<[4096,11008],ui8>, %scales: !torch.vtensor<[4096,86,1],f32>, %zpoints: !torch.vtensor<[4096,86,1],f32>) -> !torch.vtensor<[1,1,4096],f32> {
//     %unpackedtypewidth = torch.constant.int 8
//     %groupsize = torch.constant.int 128
//     %0 = torch.operator "brevitas.matmul_rhs_group_quant"(%lhs, %packed_weights, %scales, %zpoints, %unpackedtypewidth, %groupsize) : (!torch.vtensor<[1,1,11008],f32>, !torch.vtensor<[4096,11008],ui8>, !torch.vtensor<[4096,86,1],f32>, !torch.vtensor<[4096,86,1],f32>, !torch.int, !torch.int) -> !torch.vtensor<[1,1,4096],f32>
//     return %0 : !torch.vtensor<[1,1,4096],f32>
//   }
// }

// #map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// #map1 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
// #map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
// #map3 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>
// #map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
// module attributes {torch.debug_module_name = "CustomOpExampleModule"} {
//   func.func @forward(%arg0: !torch.vtensor<[1,1,11008],f32>, %arg1: !torch.vtensor<[4096,11008],ui8>, %arg2: !torch.vtensor<[4096,86,1],f32>, %arg3: !torch.vtensor<[4096,86,1],f32>) -> !torch.vtensor<[1,1,4096],f32> {
//     %0 = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[1,1,11008],f32> -> tensor<1x1x11008xf32>
//     %1 = torch_c.to_builtin_tensor %arg1 : !torch.vtensor<[4096,11008],ui8> -> tensor<4096x11008xi8>
//     %2 = torch_c.to_builtin_tensor %arg2 : !torch.vtensor<[4096,86,1],f32> -> tensor<4096x86x1xf32>
//     %3 = torch_c.to_builtin_tensor %arg3 : !torch.vtensor<[4096,86,1],f32> -> tensor<4096x86x1xf32>
//     %int8 = torch.constant.int 8
//     %int128 = torch.constant.int 128
//     %expanded = tensor.expand_shape %0 [[0], [1], [2, 3]] : tensor<1x1x11008xf32> into tensor<1x1x86x128xf32>
//     %4 = tensor.empty() : tensor<4096x86x128xf32>
//     %expanded_0 = tensor.expand_shape %1 [[0], [1, 2]] : tensor<4096x11008xi8> into tensor<4096x86x128xi8>
//     %cst = arith.constant 0.000000e+00 : f32
//     %5 = tensor.empty() : tensor<1x1x4096xf32>
//     %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x1x4096xf32>) -> tensor<1x1x4096xf32>
//     %7 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_0, %2, %3 : tensor<4096x86x128xi8>, tensor<4096x86x1xf32>, tensor<4096x86x1xf32>) outs(%4 : tensor<4096x86x128xf32>) {
//     ^bb0(%in: i8, %in_1: f32, %in_2: f32, %out: f32):
//       %10 = arith.extui %in : i8 to i32
//       %11 = arith.uitofp %10 : i32 to f32
//       %12 = arith.subf %11, %in_2 : f32
//       %13 = arith.mulf %12, %in_1 : f32
//       linalg.yield %13 : f32
//     } -> tensor<4096x86x128xf32>
//     %8 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%expanded, %7 : tensor<1x1x86x128xf32>, tensor<4096x86x128xf32>) outs(%6 : tensor<1x1x4096xf32>) {
//     ^bb0(%in: f32, %in_1: f32, %out: f32):
//       %10 = arith.mulf %in, %in_1 : f32
//       %11 = arith.addf %10, %out : f32
//       linalg.yield %11 : f32
//     } -> tensor<1x1x4096xf32>
//     %cast = tensor.cast %8 : tensor<1x1x4096xf32> to tensor<1x1x4096xf32>
//     %9 = torch_c.from_builtin_tensor %cast : tensor<1x1x4096xf32> -> !torch.vtensor<[1,1,4096],f32>
//     return %9 : !torch.vtensor<[1,1,4096],f32>
//   }
// }