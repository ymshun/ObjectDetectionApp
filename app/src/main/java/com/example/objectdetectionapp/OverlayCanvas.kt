package com.example.objectdetectionapp

import android.graphics.*
import android.view.SurfaceHolder
import android.view.SurfaceView

/**
 * 検出結果を表示する透過surfaceView
 */
class OverlaySurfaceView(surfaceView: SurfaceView) :
    SurfaceView(surfaceView.context), SurfaceHolder.Callback {

    init {
        surfaceView.holder.addCallback(this)
        surfaceView.setZOrderOnTop(true)
    }

    private var surfaceHolder = surfaceView.holder
    private val paint = Paint()
    private val pathColorList = listOf(Color.RED, Color.GREEN, Color.CYAN, Color.BLUE)

    override fun surfaceCreated(holder: SurfaceHolder) {
        // surfaceViewを透過させる
        surfaceHolder.setFormat(PixelFormat.TRANSPARENT)
    }

    override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {
    }

    override fun surfaceDestroyed(holder: SurfaceHolder) {
    }

    /**
     * surfaceViewに物体検出結果を表示
     */
    fun draw(detectedObjectList: List<DetectionObject>) {
        // surfaceHolder経由でキャンバス取得(onStop時にもdrawされてしまいexception発生の可能性があるのでnullableにして以下扱ってます)
        val canvas: Canvas? = surfaceHolder.lockCanvas()
        // 前に描画していたものをクリア
        canvas?.drawColor(0, PorterDuff.Mode.CLEAR)

        detectedObjectList.mapIndexed { i, detectionObject ->
            // バウンディングボックスの表示
            paint.apply {
                color = pathColorList[i]
                style = Paint.Style.STROKE
                strokeWidth = 7f
                isAntiAlias = false
            }
            canvas?.drawRect(detectionObject.boundingBox, paint)

            // ラベルとスコアの表示
            paint.apply {
                style = Paint.Style.FILL
                isAntiAlias = true
                textSize = 77f
            }
            canvas?.drawText(
                detectionObject.label + " " + "%,.2f".format(detectionObject.score * 100) + "%",
                detectionObject.boundingBox.left,
                detectionObject.boundingBox.top - 5f,
                paint
            )
        }

        surfaceHolder.unlockCanvasAndPost(canvas ?: return)
    }
}
