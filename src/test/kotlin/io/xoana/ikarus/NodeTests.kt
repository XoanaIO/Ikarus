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

	@Test
	fun testShapeChanges() {
		// Verical test.
		val x1 = InputNode(2, 3)
		val x2 = InputNode(4, 3)
		val vs = VStackNode(x1, x2)
		val g = Graph()
		g.addNode(vs)

		assertEquals(vs.rows, 6)
		assertEquals(vs.columns, 3)

		val x1Data = mat[
			1, 2, 3 end
			4, 5, 6
		]

		val x2Data = mat[
			9, 8, 7 end
			6, 5, 4 end
			3, 2, 1 end
			0, 1, 2
		]

		val feedDict = mapOf<Node,Matrix<Double>>(x1 to x1Data, x2 to x2Data)

		val grads = g.getGradient(feedDict, null, vs)

		assertEquals(grads[x1.id].numRows(), x1.rows)
		assertEquals(grads[x1.id].numCols(), x1.columns)
		assertEquals(grads[x2.id].numRows(), x2.rows)
		assertEquals(grads[x2.id].numCols(), x2.columns)
		assertEquals(grads[x1.id].elementSum(), x1.rows*x1.columns.toDouble(), 1e-8) // 1 for each element.
		assertEquals(grads[x2.id].elementSum(), x2.rows*x2.columns.toDouble(), 1e-8)

		// Horizontal concat test.
		val x3 = InputNode(3, 2)
		val x4 = InputNode(3, 4)
		val hs = HStackNode(x3, x4)
		val g2 = Graph()
		g2.addNode(hs)

		assertEquals(hs.rows, 3)
		assertEquals(hs.columns, 6)

		val grads2 = g2.getGradient(mapOf<Node,Matrix<Double>>(
			x3 to mat[1, 2 end 3, 4 end 5, 6],
			x4 to mat[9, 8, 7, 6 end 5, 4, 3, 2 end 1, 0, 0, 0]
		), null, hs)

		assertEquals(grads2[x3.id].numRows(), x3.rows)
		assertEquals(grads2[x3.id].numCols(), x3.columns)
		assertEquals(grads2[x4.id].numRows(), x4.rows)
		assertEquals(grads2[x4.id].numCols(), x4.columns)
	}

}
