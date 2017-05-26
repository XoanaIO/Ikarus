package io.xoana.ikarus

import io.xoana.ikarus.Model
import koma.*
import koma.matrix.*
import org.junit.Assert
import org.junit.Assert.*
import org.junit.Test
import java.awt.Graphics2D
import java.awt.image.BufferedImage
import java.io.File
import java.nio.file.Path
import java.util.*

/**
 * Created by jcatrambone on 5/25/17.
 */
class ModelTest {
	@Test
	fun verifySaveRestore() {
		val modelA = Model(1, 20);
		modelA.addConvLayer(1, 5, 1, 3, Model.Activation.NONE)
		modelA.addDenseLayer(100, Model.Activation.TANH)
		modelA.addDenseLayer(100, Model.Activation.RELU)

		val modelB = Model(1, 1)
		println(modelA.serializeToString())
		modelB.restoreFromString(modelA.serializeToString())

		assertEquals(modelA.outputNode!!.rows, modelB.outputNode!!.rows)
		assertEquals(modelA.outputNode!!.columns, modelB.outputNode!!.columns)
	}

	@Test
	fun convolutionTest() {
		// If the MNIST data doesn't work we'll skip this test.
		val mnistIn = File("asdf")
		org.junit.Assume.assumeTrue(mnistIn.isFile() && mnistIn.canRead())

		val shapeDetector = Model(128, 128)
		shapeDetector.addConvLayer(3, 3, 2, 2, Model.Activation.TANH)
		shapeDetector.addConvLayer(3, 3, 2, 2, Model.Activation.TANH)
		shapeDetector.addConvLayer(3, 3, 2, 2, Model.Activation.TANH)
		shapeDetector.addConvLayer(3, 3, 2, 2, Model.Activation.TANH)
		shapeDetector.addFlattenLayer()
		shapeDetector.addDenseLayer(128, Model.Activation.TANH)
		shapeDetector.addDenseLayer(2, Model.Activation.SIGMOID)

		// Train model.  First run is a bit slow.
		val random = Random()
		for(i in 0 until 1000) {
			//val img = BufferedImage(128, 128, BufferedImage.TYPE_BYTE_GRAY)
			//val gfx:Graphics2D = img.graphics as Graphics2D
			//gfx.drawOval(32, 32, 64, 64)
			val hasCircle = random.nextBoolean()
			val hasSquare = random.nextBoolean()
			// Generate a sample
			var x = zeros(128, 128)
			var y = doubleArrayOf(0.0, 0.0)
			if(hasCircle) {
				y[0] = 1.0
				// Pick a random place and draw a circle.
				val circleX = random.nextInt(32)+32
				val circleY = random.nextInt(32)+32
				val rad = random.nextInt(16)
				x = x.mapIndexed({i,j,v -> if(((i-circleX).pow(2.0) + (j-circleY).pow(2.0)).sqrt() < rad) { 1.0 } else { 0.0 } })
			}
			if(hasSquare) {
				y[1] = 1.0
				val squareX = random.nextInt(96)
				val squareY = random.nextInt(96)
				val side = random.nextInt(32)
				x = x.mapIndexed({i,j,v ->
					if(i > squareY && i < squareY+side && j > squareX && j < squareX+side) {
						1.0
					} else if(i==squareY || i == squareY+side || j == squareX || j == squareX+side) {
						0.0 // Draw a black outline.
					} else {
						v
					}
				})
			}
			// Train sample
			shapeDetector.fit(x.getDoubleData(), y, 0.01, Model.Loss.SQUARED)

			// Loss calc
			val tempLoss = (shapeDetector.predict(x.getDoubleData())[0]-y[0]).abs() + (shapeDetector.predict(x.getDoubleData())[1]-y[1]).abs()
			println("Last loss: $tempLoss")
		}
	}
}
