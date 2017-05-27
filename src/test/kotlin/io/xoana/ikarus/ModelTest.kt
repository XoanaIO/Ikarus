package io.xoana.ikarus

import io.xoana.ikarus.models.Model
import org.junit.Assert.*
import org.junit.Test
import java.io.DataInputStream
import java.io.File
import java.io.FileInputStream
import java.io.IOException
import java.util.*
import java.util.zip.GZIPInputStream

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
		modelA.fit(DoubleArray(20), DoubleArray(100), 0.1, Model.Loss.ABS)

		val modelB = Model(1, 1)
		println(modelA.serializeToString())
		modelB.restoreFromString(modelA.serializeToString())

		assertEquals(modelA.outputNode!!.rows, modelB.outputNode!!.rows)
		assertEquals(modelA.outputNode!!.columns, modelB.outputNode!!.columns)
	}

	@Throws(IOException::class)
	private fun loadMNISTExamples(filename: String): Array<DoubleArray> {
		val images: Array<DoubleArray>

		val image_in = DataInputStream(GZIPInputStream(FileInputStream(filename)))

		val magicNumber = image_in.readInt()
		assert(magicNumber == 0x00000803) // 2051 for training images.  2049 for training labels.
		val imageCount = image_in.readInt()
		val rows = image_in.readInt()
		val columns = image_in.readInt()
		// Images are row-wise, which is great because so is our model.
		images = Array(imageCount) { DoubleArray(rows * columns) }
		for (imageNumber in 0..imageCount - 1) {
			for (r in 0..rows - 1) {
				for (c in 0..columns - 1) {
					images[imageNumber][c + r * columns] = (image_in.readUnsignedByte() / 255.0f).toDouble()
				}
			}
		}

		return images
	}

	@Throws(IOException::class)
	private fun loadMNISTLabels(filename: String): Array<DoubleArray> {
		val labels: Array<DoubleArray> // One-hot.
		val labels_in = DataInputStream(GZIPInputStream(FileInputStream(filename)))

		// Read the labels.
		val magicNumber = labels_in.readInt()
		assert(magicNumber == 0x00000801)
		val labelCount = labels_in.readInt()
		labels = Array(labelCount) { DoubleArray(10) }
		for (labelNumber in 0..labelCount - 1) {
			val label = labels_in.readUnsignedByte()
			labels[labelNumber][label] = 1.0
		}

		return labels
	}

	@Test
	@Throws(IOException::class)
	fun testMNIST() {
		// If the MNIST data doesn't work we'll skip this test.
		val mnistDataIn = File("train-images-idx3-ubyte.gz")
		org.junit.Assume.assumeTrue(mnistDataIn.isFile() && mnistDataIn.canRead())
		val mnistLabelIn = File("train-images-idx3-ubyte.gz")
		org.junit.Assume.assumeTrue(mnistLabelIn.isFile() && mnistLabelIn.canRead())

		val ITERATION_COUNT = 100000
		val BATCH_SIZE = 10
		val REPORT_INTERVAL = 100
		val model: Model
		val images = loadMNISTExamples("train-images-idx3-ubyte.gz")
		val labels = loadMNISTLabels("train-labels-idx1-ubyte.gz")

		// Verify we've got all the data and labels.
		assert(images.size == labels.size)

		val imageCount = images.size
		val rows = 28
		val columns = 28

		// Build and train our model.
		model = Model(rows, columns)
		model.addConvLayer(3, 3, 2, 2, Model.Activation.RELU)
		model.addConvLayer(3, 3, 2, 2, Model.Activation.RELU)
		model.addFlattenLayer()
		model.addDenseLayer(64, Model.Activation.RELU)
		model.addDenseLayer(32, Model.Activation.TANH)
		model.addDenseLayer(10, Model.Activation.SIGMOID)

		// Split up the training data into target and test.
		// Start by shuffling the data.
		val random = Random()
		for (i in 0..imageCount - 1) {
			// Randomly assign another index to this value.
			val swapTarget = random.nextInt(imageCount - i) + i
			val tempImage = images[i]
			images[i] = images[swapTarget]
			images[swapTarget] = tempImage

			val tempLabel = labels[i]
			labels[i] = labels[swapTarget]
			labels[swapTarget] = tempLabel
		}

		// Pick a cutoff.  80% training?
		var learningRate = 0.1
		val trainingCutoff = (imageCount * 0.8f).toInt()
		for (i in 0..ITERATION_COUNT - 1) {
			val batch = Array(BATCH_SIZE) { DoubleArray(images[0].size) }
			val target = Array(BATCH_SIZE) { DoubleArray(labels[0].size) }
			// Pick N items at random.
			for (j in 0..BATCH_SIZE - 1) {
				val ex = random.nextInt(trainingCutoff)
				batch[j] = images[ex]
				target[j] = labels[ex]
			}
			// Train the model for an iteration.
			model.fitBatch(batch, target, learningRate, Model.Loss.SQUARED)
			// Check if we should report:
			if (i % REPORT_INTERVAL == 0) {
				learningRate *= 0.99999f
				// Select an example from the test set.
				val ex = trainingCutoff + random.nextInt(imageCount - trainingCutoff)
				val guess = model.predict(images[ex])
				// Display the image on the left and the guesses on the right.
				for (r in 0..rows - 1) {
					// Show the image.
					for (c in 0..columns - 1) {
						if (images[ex][c + r * columns] > 0.5f) {
							print("#")
						} else {
							print(".")
						}
					}

					// For each of our guesses, display some pretty graphs.
					if (r < 10) {
						if (labels[ex][r] > 0) {
							print(" [C]")
						} else {
							print(" [_]")
						}
						print(" $r: ")
						var m = 0
						while (m < guess[r] * 10) {
							print("#")
							m++
						}
					} else if (r == 10) {
						print(" ITERATION: $i   LEARNING RATE: $learningRate")
					}
					println()
				}
				println()
			}
		}

		// Save the model to a file.
		model.serializeToString()
	}

	@Test
	@Throws(IOException::class)
	fun testGenerateMNIST() {
		// If the MNIST data doesn't work we'll skip this test.
		val mnistIn = File("train-images-idx3-ubyte.gz")
		org.junit.Assume.assumeTrue(mnistIn.isFile() && mnistIn.canRead())

		val ITERATION_COUNT = 1000000
		val BATCH_SIZE = 10
		val REPORT_INTERVAL = 1000
		val NOISE_LEVEL = 0.1
		val model: Model

		val rows = 28
		val columns = 28

		// Build and train our model.
		model = Model(rows, columns)
		model.addConvLayer(4, 4, 2, 2, Model.Activation.TANH)
		model.addConvLayer(3, 3, 2, 2, Model.Activation.TANH)
		model.addFlattenLayer()
		model.addDenseLayer(64, Model.Activation.TANH)
		model.addDenseLayer(20, Model.Activation.TANH) // Representation.
		model.addDenseLayer(36, Model.Activation.TANH)
		model.addReshapeLayer(6, 6)
		model.addDeconvLayer(3, 3, 2, 2, Model.Activation.RELU)
		model.addDeconvLayer(4, 4, 2, 2, Model.Activation.SIGMOID)

		// Load data.
		val images = loadMNISTExamples("train-images-idx3-ubyte.gz")
		val imageCount = images.size

		// Split up the training data into target and test.
		// Start by shuffling the data.
		val random = Random()
		for (i in 0..imageCount - 1) {
			// Randomly assign another index to this value.
			val swapTarget = random.nextInt(imageCount - i) + i
			val tempImage = images[i]
			images[i] = images[swapTarget]
			images[swapTarget] = tempImage
		}

		// Pick a cutoff.  80% training?
		var learningRate = 0.001
		val trainingCutoff = (imageCount * 0.8f).toInt()
		for (i in 0..ITERATION_COUNT - 1) {
			val examples = Array(BATCH_SIZE) { DoubleArray(images[0].size) }
			val labels = Array(BATCH_SIZE) { DoubleArray(images[0].size) }
			// Pick N items at random.
			for (j in 0..BATCH_SIZE - 1) {
				val ex = random.nextInt(trainingCutoff)
				labels[j] = images[ex]
				// Make a copy of the data.
				examples[j] = DoubleArray(images[0].size, { k -> random.nextGaussian()*NOISE_LEVEL + images[ex][k] })
				/*
				System.arraycopy(labels[j], 0, examples[j], 0, examples[j].size)
				// Add noise to it.
				for (k in 0..examples[j].size - 1) {
					examples[j][k] += random.nextGaussian() * NOISE_LEVEL
				}
				*/
			}
			// Train the model for an iteration.
			model.fit(examples, labels, learningRate, Model.Loss.ABS)
			// Check if we should report:
			if (i % REPORT_INTERVAL == 0) {
				//learningRate *= 0.999;
				// Select an example from the test set.
				val ex = trainingCutoff + random.nextInt(imageCount - trainingCutoff)
				val guess = model.predict(images[ex])
				// Display the image on the left and the guesses on the right.
				for (r in 0..rows - 1) {
					// Show the image.
					for (c in 0..columns - 1) {
						if (images[ex][c + r * columns] > 0.8f) {
							print("█")
						} else if (images[ex][c + r * columns] > 0.6f) {
							print("▓")
						} else if (images[ex][c + r * columns] > 0.4f) {
							print("▒")
						} else if (images[ex][c + r * columns] > 0.2f) {
							print("░")
						} else {
							print(".")
						}
					}
					print(" | ")
					for (c in 0..columns - 1) {
						if (guess[c + r * columns] > 0.8f) {
							print("█")
						} else if (guess[c + r * columns] > 0.6f) {
							print("▓")
						} else if (guess[c + r * columns] > 0.4f) {
							print("▒")
						} else if (guess[c + r * columns] > 0.2f) {
							print("░")
						} else {
							print(".")
						}
					}
					println()
				}
				println()
			}
		}

		// Save the model to a file.
		model.serializeToString()
	}
}
