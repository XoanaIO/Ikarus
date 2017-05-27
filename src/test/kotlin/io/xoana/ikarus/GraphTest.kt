package io.xoana.ikarus

import io.xoana.ikarus.nodes.*

import koma.matrix.*
import koma.*

import org.junit.Test
import org.junit.Assert.*
import org.junit.runner.RunWith
import java.util.*

//@RunWith(SpringRunner::class)
class GraphTest {
	@Test
	fun forwardModeTestSin() {
		val g = Graph()
		val xn = InputNode(2, 5)
		val xs = SinNode(xn)
		g.addNode(xs)
		val inputFeed = mapOf<Node, DoubleArray>(xn to doubleArrayOf(-1.0, -0.5, 0.0, 0.5, 1.0, 10.0, 5.0, 0.0, 5.0, 10.0));
		assertArrayEquals(g.getOutput(inputFeed, xs), doubleArrayOf(
			-0.8414709848078965, -0.479425538604203, 0.0, 0.479425538604203, 0.8414709848078965,
			-0.5440211108893699, -0.9589242746631385, 0.0, -0.9589242746631385, -0.5440211108893699
		), 1e-3)
	}

	@Test
	fun reverseModeTestSin() {
		val g = Graph()
		val xn = InputNode(2, 5)
		val xs = SinNode(xn)
		g.addNode(xs)
		val inputFeed = mapOf<Node, Matrix<Double>>(xn to mat[
			-1.0, -0.5, 0, 0.5, 1 end
			10.0, 5.0, 0, 5.0, 10.0
		]);
		assertArrayEquals(g.getGradient(inputFeed, null, xs)[xn.id].getDoubleData(), doubleArrayOf(
			0.5403023058681398, 0.8775825618903728, 1.0, 0.8775825618903728, 0.5403023058681398,
			-0.8390715290764524, 0.2836621854632263, 1.0, 0.2836621854632263, -0.8390715290764524
		), 1e-3)
	}

	@Test
	fun learnXOR() {
		val learningRate = 0.1
		val g = Graph()
		val x = InputNode(4, 2)
		val w1 = VariableNode(2, 5)
		val h = TanhNode(MatrixMultiplyNode(x, w1))
		val w2 = VariableNode(5, 1)
		val out = MatrixMultiplyNode(h, w2)

		val target = InputNode(4, 1)
		val loss = PowerNode(SubtractNode(out, target), 2.0)

		g.addNode(loss)
		for(i in 0 until 1000) {
			// Calculate the gradients.
			val inputFeed = mapOf<Node,Matrix<Double>>(
				x to mat[
					0, 0 end
					0, 1 end
					1, 0 end
					1, 1],
				target to mat[0.0, 1.0, 1.0, 0.0].T
			)
			val grads = g.getGradient(inputFeed, null, loss)

			// Apply gradients to weights.
			w1.data -= learningRate*grads[w1.id]
			w2.data -= learningRate*grads[w2.id]

			// Evaluate our success.
			var accum = 0.0
			var exLoss = g.getOutput(mapOf<Node,DoubleArray>(
					x to doubleArrayOf(0.0, 0.0)
			), out)
			print("0^0: ${exLoss[0]}\t")
			accum += (0.0 - exLoss[0]).abs()

			exLoss = g.getOutput(mapOf<Node,DoubleArray>(
					x to doubleArrayOf(0.0, 1.0)
			), out)
			print("0^1: ${exLoss[0]}\t")
			accum += (1.0-exLoss[0]).abs()

			exLoss = g.getOutput(mapOf<Node,DoubleArray>(
					x to doubleArrayOf(1.0, 0.0)
			), out)
			print("1^0: ${exLoss[0]}\t")
			accum += (1.0-exLoss[0]).abs()

			exLoss = g.getOutput(mapOf<Node,DoubleArray>(
					x to doubleArrayOf(1.0, 1.0)
			), out)
			print("1^1: ${exLoss[0]}\t")
			accum += (0.0 - exLoss[0]).abs()
			
			println("Loss: ${accum}")
		}
	}
}
