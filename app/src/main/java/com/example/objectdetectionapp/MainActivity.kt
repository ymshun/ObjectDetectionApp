package com.example.objectdetectionapp

import android.Manifest
import android.content.res.AssetFileDescriptor
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import kotlinx.android.synthetic.main.activity_main.*
import org.tensorflow.lite.Interpreter
import permissions.dispatcher.NeedsPermission
import permissions.dispatcher.RuntimePermissions
import java.io.BufferedReader
import java.io.FileInputStream
import java.io.InputStream
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors


@RuntimePermissions
class MainActivity : AppCompatActivity() {
    companion object {
        private const val MODEL_FILE_NAME = "ssd_mobilenet_v1.tflite"
        private const val LABEL_FILE_NAME = "coco_dataset_labels.txt"
    }

    private lateinit var overlaySurfaceView: OverlaySurfaceView
    private lateinit var cameraExecutor: ExecutorService

    // tfliteモデルを扱うためのラッパーを含んだinterpreter
    private val interpreter: Interpreter by lazy {
        Interpreter(loadModel())
    }

    // カメラのYUV画像をRGBに変換するコンバータ
    private val yuvToRgbConverter: YuvToRgbConverter by lazy {
        YuvToRgbConverter(this)
    }

    // モデルの正解ラベルリスト
    private val labels: List<String> by lazy {
        loadLabels()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        overlaySurfaceView = OverlaySurfaceView(resultView)

        cameraExecutor = Executors.newSingleThreadExecutor()
        setupCameraWithPermissionCheck()
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        onRequestPermissionsResult(requestCode, grantResults)
    }

    @NeedsPermission(Manifest.permission.CAMERA)
    fun setupCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            // プレビューユースケース
            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(cameraView.surfaceProvider)
                }

            // 背面カメラを使用
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            // 画像解析(今回は物体検知)のユースケース
            val imageAnalyzer = ImageAnalysis.Builder()
                .setTargetRotation(cameraView.display.rotation)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST) // 最新のcameraのプレビュー画像だけをを流す
                .build()
                .also {
                    it.setAnalyzer(
                        cameraExecutor,
                        ObjectDetector(
                            yuvToRgbConverter,
                            interpreter,
                            labels,
                            Size(resultView.width, resultView.height)
                        ) { detectedObjectList ->
                            overlaySurfaceView.draw(detectedObjectList)
                        }
                    )
                }

            try {
                cameraProvider.unbindAll()

                // 各ユースケースをcameraXにバインドする
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalyzer)

            } catch (exc: Exception) {
                Log.e("ERROR: Camera", "Use case binding failed", exc)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    // tfliteモデルをassetsから読み込む
    private fun loadModel(fileName: String = MODEL_FILE_NAME): ByteBuffer {
        lateinit var modelBuffer: ByteBuffer
        var file: AssetFileDescriptor? = null
        try {
            file = assets.openFd(fileName)
            val inputStream = FileInputStream(file.fileDescriptor)
            val fileChannel = inputStream.channel
            modelBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, file.startOffset, file.declaredLength)
        } catch (e: Exception) {
            Toast.makeText(this, "モデルファイル読み込みエラー", Toast.LENGTH_SHORT).show()
            finish()
        } finally {
            file?.close()
        }
        return modelBuffer
    }

    // モデルの正解ラベルデータをassetsから取得
    private fun loadLabels(fileName: String = LABEL_FILE_NAME): List<String> {
        var labels = listOf<String>()
        var inputStream: InputStream? = null
        try {
            inputStream = assets.open(fileName)
            val reader = BufferedReader(InputStreamReader(inputStream))
            labels = reader.readLines()
        } catch (e: Exception) {
            Toast.makeText(this, "txtファイル読み込みエラー", Toast.LENGTH_SHORT).show()
            finish()
        } finally {
            inputStream?.close()
        }
        return labels
    }
}
