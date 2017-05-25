package io.xoana.ikarus

import io.xoana.ikarus.nodes.*

import koma.matrix.*
import koma.*

import org.junit.Test
import org.junit.Assert.*
import org.junit.runner.RunWith
import java.util.*

//@RunWith(SpringRunner::class)
class NodeTests {
	fun testGradient(n: Node, domain: DoubleArray, dx: Double, threshold: Double) {
		n.id = 0 // Hack in case we index by this.
		// Do a forward pass on the node with a matrix of three rows.
		// In a verticle line we've got DX.
		// forward: Matrix[] args -> Matrix.
		// Assumes single operator to the matrix.
		val arg = zeros(3, domain.size)
		for (i in domain.indices) {
			arg.set(0, i, domain[i] - dx)
			arg.set(1, i, domain[i])
			arg.set(2, i, domain[i] + dx)
		}
		val fwd = n.forward(arrayOf<Matrix<Double>>(arg))

		// f'(x) = f(x+dx) - f(x-dx) / (2*dx)
		val numericalGradient = DoubleArray(domain.size)
		for (i in domain.indices) {
			numericalGradient[i] = (fwd[2, i] - fwd[0, i]) / (2.0f * dx)
		}

		// Calculate the exact gradient.
		val grad = n.reverse(arrayOf<Matrix<Double>>(arg), ones(fwd.numRows(), fwd.numCols()))[0]
		val calculatedGradient = grad[1..1, 0..9].getDoubleData()

		// Dump output.
		println(n.javaClass.canonicalName + " gradient:")
		println(" - Numerical gradient: " + Arrays.toString(numericalGradient))
		println(" - Calculated gradient: " + Arrays.toString(calculatedGradient))

		// Gradient error magnitude.
		for (i in domain.indices) {
			if (calculatedGradient[i] == 0.0 && numericalGradient[i] == 0.0) {
				continue // If there's no grad, we're okay.
			}
			val p = calculatedGradient[i]
			val q = numericalGradient[i]
			org.junit.Assert.assertTrue("Gradient order less than threshold?  $p vs $q", Math.abs(p - q) / Math.max(p, q) < threshold)
		}

		// Calculate and check the gradient order.
		//org.junit.Assert.assertArrayEquals(numericalGradient, calculatedGradient, threshold);
	}

	@Test
	fun testGrad() {
		val x = InputNode(1, 10)
		var values = doubleArrayOf(-10.0, -5.0, -2.0, -1.0, -0.1, 0.1, 1.0, 2.0, 5.0, 10.0)

		testGradient(TanhNode(x), values, 0.01, 0.1)
		testGradient(SigmoidNode(x), values, 0.01, 0.5)
		testGradient(InverseNode(x), values, 0.0001, 0.2)
		testGradient(ExpNode(x), values, 0.01, 0.2)
		testGradient(PowerNode(x, 2.0), values, 0.01, 0.2)
		testGradient(NegateNode(x), values, 0.01, 0.2)

		// These require non-negative values.
		values = doubleArrayOf(0.1, 1.0, 2.0, 5.0, 10.0, 100.0, 1000.0, 10000.0, 1000000.0, 1000000000.0)
		testGradient(LogNode(x), values, 0.01, 0.2)

		// Has a discontinuity at zero.
		values = doubleArrayOf(-10.0, -5.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 5.0, 10.0)
		testGradient(AbsNode(x), values, 0.01, 0.2)
		//testGradient(ReLUNode(x), values, 0.01, 0.2)
	}

}
