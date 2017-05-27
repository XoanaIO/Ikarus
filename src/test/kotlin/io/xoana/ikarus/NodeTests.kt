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

		// Not valid for values less than zero
		values = doubleArrayOf(0.1, 1.0, 2.0, 5.0, 10.0, 100.0, 1000.0, 10000.0, 1000000.0, 1000000000.0)
		testGradient(ReLUNode(x), values, 0.001, 0.2)
	}

	@Test
	fun testBinaryOperations() {
		val x1 = InputNode(5, 5)
		val x2 = InputNode(5, 5)
		val sum = AddNode(x1, x2)
		var g = Graph()
		g.addNode(sum)


	}

	@Test
	fun testShapeChanges() {
		// Verical test.
		val x1 = InputNode(2, 3)
		val x2 = InputNode(4, 3)
		val vs = VStackNode(x1, x2)
		var g = Graph()
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

		val feedDict = mapOf<Node, Matrix<Double>>(x1 to x1Data, x2 to x2Data)

		val grads = g.getGradient(feedDict, null, vs)

		assertEquals(grads[x1.id].numRows(), x1.rows)
		assertEquals(grads[x1.id].numCols(), x1.columns)
		assertEquals(grads[x2.id].numRows(), x2.rows)
		assertEquals(grads[x2.id].numCols(), x2.columns)
		assertEquals(grads[x1.id].elementSum(), x1.rows * x1.columns.toDouble(), 1e-8) // 1 for each element.
		assertEquals(grads[x2.id].elementSum(), x2.rows * x2.columns.toDouble(), 1e-8)

		// Horizontal concat test.
		val x3 = InputNode(3, 2)
		val x4 = InputNode(3, 4)
		val hs = HStackNode(x3, x4)
		g = Graph()
		g.addNode(hs)

		assertEquals(hs.rows, 3)
		assertEquals(hs.columns, 6)

		val grads2 = g.getGradient(mapOf<Node, Matrix<Double>>(
				x3 to mat[1, 2 end 3, 4 end 5, 6],
				x4 to mat[9, 8, 7, 6 end 5, 4, 3, 2 end 1, 0, 0, 0]
		), null, hs)

		assertEquals(grads2[x3.id].numRows(), x3.rows)
		assertEquals(grads2[x3.id].numCols(), x3.columns)
		assertEquals(grads2[x4.id].numRows(), x4.rows)
		assertEquals(grads2[x4.id].numCols(), x4.columns)

		// Row-sum test.
		val x5 = InputNode(2, 10)
		val rs = RowSumNode(x5)
		g = Graph()
		g.addNode(rs)

		val feedDict3 = mapOf<Node, Matrix<Double>>(
			x5 to mat[
				1, 2, 3, 4, 5, 6, 7, 8, 9, 10 end
				-1, -2, -3, -4, -5, -6, -7, -8, -9, -10
			]
		)
		val fwd3 = g.getOutput(mapOf<Node, DoubleArray>(x5 to feedDict3[x5]!!.getDoubleData()), rs)
		val grads3 = g.getGradient(feedDict3, null, rs)

		assertEquals(fwd3.size, 2) // Two rows
		assertEquals(fwd3[0], 55.0, 1e-6)
		assertEquals(fwd3[1], -55.0, 1e-6)
		assertArrayEquals(grads3[x5.id].getDoubleData(), ones(2,10).getDoubleData(), 1e-6)
	}

	@Test
	fun testSerialization() {
		val v = VariableNode(5, 5, 0.5)
		val u = Node.fromString(v.toString(), listOf<Node>())
		assertArrayEquals(v.data.getDoubleData(), (u as VariableNode).data.getDoubleData(), 1.0e-6)
	}

	@Test
	fun testConvolution() {
		val input = InputNode(200,200)
		val kernel = VariableNode(10, 10)
		val convNode = Convolution2DNode(input, kernel, 10, 10)
		val g = Graph()
		g.addNode(convNode)

		kernel.data = ones(10, 10)

		assertEquals(convNode.rows, 20)
		assertEquals(convNode.columns, 20)

		val res = g.getOutput(mapOf<Node,DoubleArray>(
			input to DoubleArray(200*200, { i -> if(i%200 < 20 && i/200 < 20) { 1.0 } else { 0.0 }}) // Color top-left as 'white'.
		), convNode)

		val expectation = mat[
			25, 50, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 end
			50, 100, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 end
			25, 50, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 end
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 end
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 end
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 end
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 end
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 end
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 end
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 end
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 end
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 end
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 end
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 end
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 end
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 end
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 end
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 end
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 end
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
		]

		assertArrayEquals(res, expectation.getDoubleData(), 1.0e-6)
	}

	@Test
	fun testVAE() {
		val input = InputNode(100, 100)
		val kernel = VariableNode(5, 5)
		val convNode = Convolution2DNode(input, kernel, 3, 3)
		val act1 = TanhNode(convNode)
		val flat = ReshapeNode(act1, 1, -1)
		val weightFlat
	}
}
