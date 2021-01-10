package com.example.objectdetectionapp

import android.annotation.SuppressLint
import android.graphics.Bitmap
import android.graphics.RectF
import android.media.Image
import android.util.Log
import android.util.Size
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.Rot90Op

typealias ObjectDetectorCallback = (image: List<DetectionObject>) -> Unit

/**
 * CameraXの物体検知の画像解析ユースケース
 * @param yuvToRgbConverter カメラ画像のImageバッファYUV_420_888からRGB形式に変換する
 * @param interpreter tfliteモデルを操作するライブラリ
 * @param labels 正解ラベルのリスト
 * @param resultViewSize 結果を表示するsurfaceViewのサイズ
 * @param listener コールバックで解析結果のリストを受け取る
 */
class ObjectDetector(
    private val yuvToRgbConverter: YuvToRgbConverter,
    private val interpreter: Interpreter,
    private val labels: List<String>,
    private val resultViewSize: Size,
    private val listener: ObjectDetectorCallback
) : ImageAnalysis.Analyzer {

    companion object {
        // モデルのinputとoutputサイズ
        private const val IMG_SIZE_X = 300
        private const val IMG_SIZE_Y = 300
        private const val MAX_DETECTION_NUM = 10

        // 今回使うtfliteモデルは量子化済みなのでnormalize関連は127.5fではなく以下の通り
        private const val NORMALIZE_MEAN = 0f
        private const val NORMALIZE_STD = 1f

        // 検出結果のスコアしきい値
        private const val SCORE_THRESHOLD = 0.5f
    }

    private var imageRotationDegrees: Int = 0
    private val tfImageProcessor by lazy {
        ImageProcessor.Builder()
            .add(ResizeOp(IMG_SIZE_X, IMG_SIZE_Y, ResizeOp.ResizeMethod.BILINEAR)) // モデルのinputに合うように画像のリサイズ
            .add(Rot90Op(-imageRotationDegrees / 90)) // 流れてくるImageProxyは90度回転しているのでその補正
            .add(NormalizeOp(NORMALIZE_MEAN, NORMALIZE_STD)) // normalization関連
            .build()
    }

    private val tfImageBuffer = TensorImage(DataType.UINT8)

    // 検出結果のバウンディングボックス [1:10:4]
    // バウンディングボックスは [top, left, bottom, right] の形
    private val outputBoundingBoxes: Array<Array<FloatArray>> = arrayOf(
        Array(MAX_DETECTION_NUM) {
            FloatArray(4)
        }
    )

    // 検出結果のクラスラベルインデックス [1:10]
    private val outputLabels: Array<FloatArray> = arrayOf(
        FloatArray(MAX_DETECTION_NUM)
    )

    // 検出結果の各スコア [1:10]
    private val outputScores: Array<FloatArray> = arrayOf(
        FloatArray(MAX_DETECTION_NUM)
    )

    // 検出した物体の数(今回はtflite変換時に設定されているので 10 (一定))
    private val outputDetectionNum: FloatArray = FloatArray(1)

    // 検出結果を受け取るためにmapにまとめる
    private val outputMap = mapOf(
        0 to outputBoundingBoxes,
        1 to outputLabels,
        2 to outputScores,
        3 to outputDetectionNum
    )

    // cameraXから流れてくるプレビューのimageを物体検知モデルに入れて推論する
    @SuppressLint("UnsafeExperimentalUsageError")
    override fun analyze(image: ImageProxy) {
        if (image.image == null) return
        imageRotationDegrees = image.imageInfo.rotationDegrees
        val detectedObjectList = detect(image.image!!)
        listener(detectedObjectList)
        image.close()
    }

    // 画像をYUV -> RGB bitmap -> tensorflowImage -> tensorflowBufferに変換して推論し結果をリストとして出力
    private fun detect(targetImage: Image): List<DetectionObject> {
        val targetBitmap = Bitmap.createBitmap(targetImage.width, targetImage.height, Bitmap.Config.ARGB_8888)
        yuvToRgbConverter.yuvToRgb(targetImage, targetBitmap) // rgbに変換
        tfImageBuffer.load(targetBitmap)
        val tensorImage = tfImageProcessor.process(tfImageBuffer)

        //tfliteモデルで推論の実行
        interpreter.runForMultipleInputsOutputs(arrayOf(tensorImage.buffer), outputMap)

        // 推論結果を整形してリストにして返す
        val detectedObjectList = arrayListOf<DetectionObject>()
        loop@ for (i in 0 until outputDetectionNum[0].toInt()) {
            val score = outputScores[0][i]
            val label = labels[outputLabels[0][i].toInt()]
            val boundingBox = RectF(
                outputBoundingBoxes[0][i][1] * resultViewSize.width,
                outputBoundingBoxes[0][i][0] * resultViewSize.height,
                outputBoundingBoxes[0][i][3] * resultViewSize.width,
                outputBoundingBoxes[0][i][2] * resultViewSize.height
            )

            // しきい値よりも大きいもののみ追加
            if (score >= SCORE_THRESHOLD) {
                detectedObjectList.add(
                    DetectionObject(
                        score = score,
                        label = label,
                        boundingBox = boundingBox
                    )
                )
            } else {
                // 検出結果はスコアの高い順にソートされたものが入っているので、しきい値を下回ったらループ終了
                break@loop
            }
        }
        return detectedObjectList.take(4)
    }
}
