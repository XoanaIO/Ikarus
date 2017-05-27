package io.xoana.ikarus.models

import io.xoana.ikarus.Graph
import koma.*
import koma.matrix.*

import java.util.*
import java.util.stream.IntStream

import io.xoana.ikarus.nodes.*
import io.xoana.ikarus.optimizers.SGD

/**
 * Created by jcatrambone on 5/25/17.
 */
open class Model(inputRows: Int, inputColumns: Int) : Graph() {
	//public enum Optimizer { SGD, MOMENTUM, ADAGRAD };
	enum class Activation {
		NONE, TANH, SIGMOID, RELU
	}

	enum class Loss {
		ABS, SQUARED, BINARY_CROSS_ENTROPY
	}

	private var inputNode: Node? = null // Only one input to our compute graph.
	var outputNode: Node? = null
		private set // Keeps track of the last later.

	// Used for training.
	private var targetNode: Node? = null
	private var lossNode: Node? = null
	private val trainableVariables: MutableList<VariableNode>

	init {
		inputNode = InputNode(inputRows, inputColumns) // Gotta' resize.
		outputNode = inputNode
		trainableVariables = ArrayList()
	}

	private fun randomWeight(rows: Int, columns: Int): VariableNode {
		val w = VariableNode(rows, columns, 0.1)
		w.name = "variable"
		trainableVariables.add(w)
		return w
	}

	private fun xavierWeight(rows: Int, columns: Int): VariableNode {
		// Xavier says 2 + (n_in + n_out).
		val scaling = 2.0f / rows.toDouble() // Based on a recent paper by He, Rang, Zhen, and Sun.
		val w = VariableNode(rows, columns, scaling)
		w.name = "variable"
		trainableVariables.add(w)
		return w
	}

	override fun restoreFromString(s: String) {
		super.restoreFromString(s)
		// We populate the names with some special tags when we start training.
		// Then we restore and use the names to put back the values.
		for (n in this.nodes) {
			if (n.name.startsWith("variable")) {
				this.trainableVariables.add(n as VariableNode)
			} else if (n.name.startsWith("input")) {
				this.inputNode = n
			} else if (n.name.startsWith("output")) {
				this.outputNode = n
			} else if (n.name.startsWith("loss")) {
				this.lossNode = n
			} else if (n.name.startsWith("target")) {
				this.targetNode = n
			}
		}
	}

	private fun finalizeNetwork(loss: Loss) {
		// If this is the first time we've run fit, we'll need to make our loss node.
		if (targetNode == null || lossNode == null) {
			// TODO: Sanity check if target node size changed.
			targetNode = InputNode(outputNode!!.rows, outputNode!!.columns)
			val diff = SubtractNode(outputNode!!, targetNode!!)
			when (loss) {
				Loss.ABS -> lossNode = AbsNode(diff)
				Loss.SQUARED -> lossNode = PowerNode(diff, 2.0)
			}
			lossNode = CollapseSumNode(lossNode!!) // Roll up into a single value.
			addNode(lossNode!!)
			// Need these for save/restore.
			inputNode!!.name = "input"
			outputNode!!.name = "output"
			targetNode!!.name = "target"
			lossNode!!.name = "loss"
		}
	}

	fun fit(x: DoubleArray, y: DoubleArray, learningRate: Double, loss: Loss) {
		finalizeNetwork(loss)
		assert(x.size == this.inputNode!!.rows * this.inputNode!!.columns)

		// Calculate the difference and apply the gradient.
		val inputFeed = mapOf<Node, Matrix<Double>>(
			inputNode!! to create(x, inputNode!!.rows, inputNode!!.columns),
			targetNode!! to create(y, targetNode!!.rows, targetNode!!.columns)
		)

		// Minimize loss
		val optimizer = SGD(this, this.trainableVariables.toTypedArray(), learningRate)
		optimizer.minimize(lossNode!!, inputFeed)
	}

	fun fit(x: Array<DoubleArray>, y: Array<DoubleArray>, learningRate: Double, loss: Loss) {
		for (i in x.indices) {
			fit(x[i], y[i], learningRate, loss)
		}
	}

	/***
	 * fitBatch, unlike fit, calculates the gradient for all examples before applying it to the model's variables.
	 * @param x
	 * @param y
	 * @param learningRate
	 * @param loss
	 */
	fun fitBatch(x: Array<DoubleArray>, y: Array<DoubleArray>, learningRate: Double, loss: Loss) {
		finalizeNetwork(loss)

		// This will accumulate our gradients below.
		val grads = Array<Array<Matrix<Double>>>(targetNode!!.id+1, { _ -> arrayOf<Matrix<Double>>() } )

		// Calculate all the gradients in parallel.
		IntStream.range(0, x.size).parallel().forEach { i ->
			val inputFeed = mapOf<Node, Matrix<Double>>(
				inputNode!! to create(x[i], inputNode!!.rows, inputNode!!.columns),
				targetNode!! to create(y[i], targetNode!!.rows, targetNode!!.columns)
			)
			grads[i] = getGradient(inputFeed, null, lossNode!!)
		}

		// Apply the gradients, scaled, to each of the learning variables.
		// Dividing by x.length will average our gradients.
		for (n in trainableVariables) {
			var accumulator = grads[0][n.id]
			for (j in 1..x.size - 1) {
				accumulator += grads[j][n.id]
			}
			n.data -= (learningRate*accumulator)/x.size.toDouble()
		}
	}

	fun predict(x: DoubleArray): DoubleArray {
		val inputMap = mapOf<Node, DoubleArray>(inputNode!! to x)
		return getOutput(inputMap, outputNode!!)
	}

	fun predict(x: Array<DoubleArray>): Array<DoubleArray> {
		val result = Array(x.size) { DoubleArray(outputNode!!.rows * outputNode!!.columns) }
		for (i in x.indices) {
			result[i] = predict(x[i])
		}
		return result
	}

	private fun makeActivationNode(n: Node, act: Activation): Node? {
		when (act) {
			Activation.NONE -> return n
			Activation.RELU -> return ReLUNode(n)
			Activation.SIGMOID -> return SigmoidNode(n)
			Activation.TANH -> return TanhNode(n)
		}
	}

	fun addDenseLayer(hiddenSize: Int, act: Activation) {
		assert(outputNode!!.rows === 1) // TODO: Throw error.
		val w = xavierWeight(outputNode!!.columns, hiddenSize)
		val b = randomWeight(1, hiddenSize)
		val prod = AddNode(MatrixMultiplyNode(outputNode!!, w), b)

		outputNode = makeActivationNode(prod, act)
		addNode(outputNode!!)
	}

	fun addConvLayer(kernelHeight: Int, kernelWidth: Int, yStride: Int, xStride: Int, act: Activation) {
		val kernel = xavierWeight(kernelHeight, kernelWidth)
		val conv = Convolution2DNode(outputNode!!, kernel, yStride, xStride)
		val bias = randomWeight(conv.rows, conv.columns)
		val prod = AddNode(conv, bias)
		outputNode = makeActivationNode(prod, act)
		addNode(outputNode!!)
	}

	fun addFlattenLayer() {
		outputNode = ReshapeNode(outputNode!!, 1, -1)
		addNode(outputNode!!)
	}

	fun addReshapeLayer(height: Int, width: Int) {
		outputNode = ReshapeNode(outputNode!!, height, width)
		addNode(outputNode!!)
	}

	fun addDeconvLayer(kernelHeight: Int, kernelWidth: Int, yStride: Int, xStride: Int, act: Activation) {
		val kernel = xavierWeight(kernelHeight, kernelWidth)
		val deconv = Deconvolution2DNode(outputNode!!, kernel, yStride, xStride)
		val bias = randomWeight(deconv.rows, deconv.columns) // TODO: Verify these dimensions are correct.
		val prod = AddNode(deconv, bias)
		outputNode = makeActivationNode(prod, act)
		addNode(outputNode!!)
	}
}
