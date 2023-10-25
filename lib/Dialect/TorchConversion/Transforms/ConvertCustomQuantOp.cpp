//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/Passes.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {
class ConvertCustomQuantizedMatmulOp : public OpConversionPattern<OperatorOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(OperatorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getName().str() != "quant.matmul_rhs_group_quant") {
      return failure();
    }
    Location loc = op->getLoc();
    if (failed(verifyLinalgCompatibleTypes(op, rewriter))) {
      return failure();
    }

    // get inputs: lhs, rhsQuant, scales, zps
    Value lhs = adaptor.getOperands()[0];
    auto lhsType = lhs.getType().cast<RankedTensorType>();
    if (!lhsType) {
      return failure();
    }
    auto lhsShape = lhsType.getShape();
    int lhsReductDimSize = lhsShape.back();

    Value rhsQuant = adaptor.getOperands()[1];
    auto rhsType = rhsQuant.getType().cast<RankedTensorType>();
    if (!rhsType) {
      return failure();
    }
    auto rhsShape = rhsType.getShape();
    int rhsReductDimSize = rhsShape.back();
    Type rhsElementType = rhsType.getElementType();

    Value scales = adaptor.getOperands()[2];
    Value zps = adaptor.getOperands()[3];
    Value unpackedTypeWidth = adaptor.getOperands()[4];
    Value groupSize = adaptor.getOperands()[5];

    auto getConstantIntegerFromDefiningOp = [](Value operand,
                                               int &extractedInt) {
      auto castOp = dyn_cast<mlir::UnrealizedConversionCastOp>(operand.getDefiningOp());
      if (!castOp) {
        return failure();
      }
      auto constOp =
          dyn_cast<Torch::ConstantIntOp>(castOp.getOperand(0).getDefiningOp());
      if (!constOp) {
        return failure();
      }
      extractedInt = constOp.getValue();
      return success();
    };

    int gs;
    if (failed(getConstantIntegerFromDefiningOp(groupSize, gs))) {
      return failure();
    }
    int unpackedBitWidth;
    if (failed(getConstantIntegerFromDefiningOp(unpackedTypeWidth, unpackedBitWidth))) {
      return failure();
    }
    if (unpackedBitWidth !=
        static_cast<int>(rhsElementType.getIntOrFloatBitWidth())) {
      return failure();
    }

    // get outputs
    Type newResultType = getTypeConverter()->convertType(op.getType(0));
    auto resultType = newResultType.cast<RankedTensorType>();
    if (!resultType) {
      return failure();
    }
    auto resultShape = resultType.getShape();
    Type elementType = resultType.getElementType();

    // expand lhs
    std::vector<int64_t> lhsExpandedShape;
    SmallVector<ReassociationIndices, 4> lhsReassociation;
    int lhsRank = lhsShape.size();
    for (int i = 0, e = lhsRank - 1; i < e; i++) {
      lhsExpandedShape.push_back(lhsShape[i]);
      lhsReassociation.push_back({i});
    }
    lhsExpandedShape.push_back(lhsReductDimSize / gs);
    lhsExpandedShape.push_back(gs);
    lhsReassociation.push_back({lhsRank - 1, lhsRank});
    RankedTensorType lhsExpandedType =
        RankedTensorType::get(lhsExpandedShape, elementType);
    Value lhsExpanded = rewriter.create<tensor::ExpandShapeOp>(
      loc, lhsExpandedType, lhs, lhsReassociation);

    // expand rhs
    std::vector<int64_t> rhsExpandedShape = {rhsShape[0], rhsReductDimSize/gs, gs};
    RankedTensorType rhsExpandedType = RankedTensorType::get(rhsExpandedShape, rhsElementType);
    SmallVector<ReassociationIndices, 4> rhsReassociation = {{0}, {1, 2}};
    Value rhsExpanded = rewriter.create<tensor::ExpandShapeOp>(
      loc, rhsExpandedType, rhsQuant, rhsReassociation);
    Value cst0 = rewriter.create<arith::ConstantOp>(
      loc, FloatAttr::get(elementType, 0.0));

    Value emptyDequant = rewriter.create<tensor::EmptyOp>(
      loc, rhsExpandedShape, elementType);
    SmallVector<Value> dynDims;
    for (int i = 0; i < lhsType.getRank(); i++) {
      if (lhsType.isDynamicDim(i)) {
        dynDims.push_back(rewriter.create<tensor::DimOp>(loc, lhs, i));
      }
    }
    Value empty = rewriter.create<tensor::EmptyOp>(
      loc, resultShape, elementType, dynDims);
    Value output = rewriter.create<linalg::FillOp>(
      loc, cst0, empty).getResult(0);

    MLIRContext *context = rewriter.getContext();
    SmallVector<AffineExpr> dimList(lhsRank + 2);
    bindDimsList(context, MutableArrayRef{dimList});
    auto d0 = dimList[0];
    auto d1 = dimList[1];
    auto d2 = dimList[2];
    auto c0 = rewriter.getAffineConstantExpr(0);
    // Construct maps for the dequantization.
    int64_t scaleRank = scales.getType().cast<RankedTensorType>().getRank();
    int64_t zpRank = zps.getType().cast<RankedTensorType>().getRank();
    auto map = AffineMap::get(3, 0, {d0, d1, d2}, context);
    auto mapScale =
        AffineMap::get(3, 0,
                       scaleRank == 3 ? SmallVector<AffineExpr>{d0, d1, c0}
                                      : SmallVector<AffineExpr>{d0, d1},
                       context);
    auto mapZp =
        AffineMap::get(3, 0,
                       zpRank == 3 ? SmallVector<AffineExpr>{d0, d1, c0}
                                   : SmallVector<AffineExpr>{d0, d1},
                       context);

    SmallVector<AffineExpr> lhsExprs;
    SmallVector<AffineExpr> rhsExprs;
    SmallVector<AffineExpr> outExprs;
    for (int i = 0, e = lhsRank + 2; i < e; ++i) {
      auto expr = dimList[i];
      if (i != lhsRank - 1)
        lhsExprs.push_back(expr);
      if (i >= lhsRank - 1)
        rhsExprs.push_back(expr);
      if (i <= lhsRank - 1)
        outExprs.push_back(expr);
    }
    auto map2 = AffineMap::get(lhsRank + 2, 0, lhsExprs, context);
    auto map3 = AffineMap::get(lhsRank + 2, 0, rhsExprs, context);
    auto map4 = AffineMap::get(lhsRank + 2, 0, outExprs, context);
    SmallVector<AffineMap, 4> dqIndexingMaps = {map, mapScale, mapZp, map};
    SmallVector<AffineMap, 4> matIndexingMaps = {map2, map3, map4};

    SmallVector<utils::IteratorType> dequantIteratorTypes(3, utils::IteratorType::parallel);
    SmallVector<utils::IteratorType> matmulIteratorTypes(
        lhsRank, utils::IteratorType::parallel);
    matmulIteratorTypes.push_back(utils::IteratorType::reduction);
    matmulIteratorTypes.push_back(utils::IteratorType::reduction);

    Value rhsDequant =
        rewriter
            .create<linalg::GenericOp>(
                loc, emptyDequant.getType(),
                ValueRange{rhsExpanded, scales, zps}, emptyDequant,
                /*indexingMaps=*/dqIndexingMaps,
                /*iteratorTypes=*/dequantIteratorTypes,
                [&](OpBuilder &b, Location loc, ValueRange args) {
                  Value w = args[0], scale = args[1], zeroPoint = args[2];
                  Value extw = b.create<arith::ExtUIOp>(loc, rewriter.getI32Type(), w);
                  Value fp_extw = b.create<arith::UIToFPOp>(
                      loc, rewriter.getF32Type(), extw);
                  Value shifted = b.create<arith::SubFOp>(loc, fp_extw, zeroPoint);
                  Value dqw = b.create<arith::MulFOp>(loc, shifted, scale);
                  b.create<linalg::YieldOp>(loc, dqw);
                })
            .getResult(0);

    Value matmulDequant =
        rewriter
            .create<linalg::GenericOp>(
                loc, output.getType(),
                ValueRange{lhsExpanded, rhsDequant}, output,
                /*indexingMaps=*/matIndexingMaps,
                /*iteratorTypes=*/matmulIteratorTypes,
                [&](OpBuilder &b, Location loc, ValueRange args) {
                  Value l = args[0], r = args[1], out = args[2];
                  Value pd = b.create<arith::MulFOp>(loc, l, r);
                  Value ac = b.create<arith::AddFOp>(loc, pd, out);
                  b.create<linalg::YieldOp>(loc, ac);
                })
            .getResult(0);

    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, matmulDequant);
    return success();
  }
};
} // namespace

namespace {
class ConvertCustomQuantOpPass
    : public TorchConversion::ConvertCustomQuantOpBase<ConvertCustomQuantOpPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<linalg::LinalgDialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<Torch::TorchDialect>();
    TorchConversion::getBackendTypeConversionDependentDialects(registry);
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<linalg::LinalgDialect, func::FuncDialect,
                           tensor::TensorDialect, arith::ArithDialect,
                           Torch::TorchDialect>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    TorchConversion::setupBackendTypeConversion(target, typeConverter);

    RewritePatternSet patterns(context);
    target.addIllegalOp<OperatorOp>();
    patterns.add<ConvertCustomQuantizedMatmulOp>(typeConverter, context);

    if (failed(
            applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::TorchConversion::createConvertCustomQuantOpPass() {
  return std::make_unique<ConvertCustomQuantOpPass>();
}
