package io.xoana.ikarus.nodes

import koma.*
import koma.matrix.*

/**
 * Created by jcatrambone on 5/25/17.
 */
class Convolution2DNode(input: Node, kernel: Node, var rowStride: Int, var columnStride: Int) : Node(
		(input.rows - kernel.rows) / rowStride + 1,
		(input.columns - kernel.columns) / columnStride + 1,
		arrayOf(input, kernel)
	) {
	// Performs a shallow 2D convolution on the input node.
	// W1 H1 D1
	// K = num filters.
	// F = spatial extent.
	// S = stride.
	// P = padding.
	// W2 = (W1 - F + 2P)/S + 1
	// H2 = (H1 - F + 2P)/S + 1
	// D = k
	// For the 2D convolution, the number of filters is restricted to 1.

	// F = kernel.width

	override fun forward(args: Array<Matrix<Double>>): Matrix<Double> {
		val output = zeros(this.rows, this.columns)
		val input = args[0]
		val kernel = args[1]
		// For each filter, sum the element-wise product with the input volume and assign it to the output.
		for (r in 0..this.rows - 1) {
			for (c in 0..this.columns - 1) {
				// Should add padding.
				val inRCenter = r * rowStride
				val inCCenter = c * columnStride
				var accumulator = 0.0
				// Center kernel at r,c
				for (rk in 0..kernel.numRows() - 1) {
					for (ck in 0..kernel.numCols() - 1) {
						// Delta position
						val inR = rk - kernel.numRows() / 2 + inRCenter
						val inC = ck - kernel.numCols() / 2 + inCCenter
						if (inR >= 0 && inR < input.numRows() && inC >= 0 && inC < input.numCols()) {
							accumulator += input.get(inR, inC) * kernel.get(rk, ck)
						}
					}
				}
				output.set(r, c, accumulator)
			}
		}
		return output
	}

	override fun reverse(forward: Array<Matrix<Double>>, adjoint: Matrix<Double>, cachedOutput: Matrix<Double>?): Array<Matrix<Double>> {
		// If this were c = i*k, then i_adj = c_adj*k and k_adj = c_adj*i
		// Instead, treat this as much bigger and apply the region in question for each adjoint.
		val inputAdjoint = zeros(forward[0].numRows(), forward[0].numCols())
		val kernelAdjoint = zeros(forward[1].numRows(), forward[1].numCols())
		val input = forward[0]
		val kernel = forward[1]
		// For each filter, sum the element-wise product with the input volume and assign it to the output.
		for (r in 0..this.rows - 1) {
			for (c in 0..this.columns - 1) {
				// Center kernel at r,c
				for (rk in 0..kernel.numRows() - 1) {
					for (ck in 0..kernel.numCols() - 1) {
						val outRow = rk - kernel.numRows() / 2 + r * rowStride
						val outColumn = ck - kernel.numCols() / 2 + c * columnStride
						if (outRow >= 0 && outRow < inputAdjoint.numRows() && outColumn >= 0 && outColumn < inputAdjoint.numCols()) {
							inputAdjoint.set(outRow, outColumn, inputAdjoint.get(outRow, outColumn) + adjoint.get(r, c) * kernel.get(rk, ck))
							kernelAdjoint.set(rk, ck, kernelAdjoint.get(rk, ck) + adjoint.get(r, c) * input.get(outRow, outColumn))
						}
					}
				}
			}
		}
		return arrayOf<Matrix<Double>>(inputAdjoint, kernelAdjoint)
	}

	// Used to augment serialization.
	override fun dataToString(): String {
		return rowStride.toString() + "," + columnStride
	}

	override fun stringToData(s: String) {
		val tokens = s.split(",".toRegex()).dropLastWhile { it.isEmpty() }.toTypedArray()
		this.rowStride = Integer.parseInt(tokens[0])
		this.columnStride = Integer.parseInt(tokens[1])
	}
}


class Deconvolution2DNode(input:Node, kernel:Node, var rowStride:Int, var columnStride:Int) : Node(
		(input.rows - 1) * rowStride + kernel.rows,
		(input.columns - 1) * columnStride + kernel.columns,
		arrayOf(input, kernel)
) {
	internal var padding = 0
		// Performs a shallow 2D convolution on the input node.
		// W1 H1 D1
		// K = num filters.
		// F = spatial extent.
		// S = stride.
		// P = padding.
		// W2 = (W1 - F + 2P)/S + 1
		// H2 = (H1 - F + 2P)/S + 1
		// D = k
		// For the 2D convolution, the number of filters is restricted to 1.

		// F = kernel.width
		// We have W2 and need to determine W1 from the parameters.
		// W2 = ((W1 - F + 2P)/S) + 1
		// W2 - 1 = (W1 - F + 2P)/S
		// S*(W2-1) = W1 - F + 2P
		// S*(W2-1) - 2P + F = W1


	override fun forward(args: Array<Matrix<Double>>): Matrix<Double> {
		val output = zeros(this.rows, this.columns)
		val input = args[0]
		val kernel = args[1]
		for (inRow in 0..input.numRows() - 1) {
			for (inCol in 0..input.numCols() - 1) {
				// inRow/Col gives us our position on the convolution object.
				// Calculate the center on our output image from our position on the convoluted input.
				val outcenterRow = inRow * rowStride
				val outcenterCol = inCol * columnStride

				// Iterate over the kernel and use that to apply our output.
				for (kRow in 0..kernel.numRows() - 1) {
					for (kCol in 0..kernel.numCols() - 1) {
						val outRow = outcenterRow - kernel.numRows() / 2 + kRow
						val outCol = outcenterCol - kernel.numCols() / 2 + kCol

						if (outRow >= 0 && outRow < output.numRows() && outCol >= 0 && outCol < output.numCols()) {
							output.set(outRow, outCol, output.get(outRow, outCol) + kernel.get(kRow, kCol) * input.get(inRow, inCol))
						}
					}
				}
			}
		}
		return output
	}

	override fun reverse(forward: Array<Matrix<Double>>, adjoint: Matrix<Double>, cachedOutput: Matrix<Double>?): Array<Matrix<Double>> {
		// If this were c = i*k, then i_adj = c_adj*k and k_adj = c_adj*i
		// Instead, treat this as much bigger and apply the region in question for each adjoint.
		val inputAdjoint = zeros(forward[0].numRows(), forward[0].numCols())
		val kernelAdjoint = zeros(forward[1].numRows(), forward[1].numCols())
		val input = forward[0]
		val kernel = forward[1]

		for (inRow in 0..input.numRows() - 1) {
			for (inCol in 0..input.numCols() - 1) {
				// inRow/Col gives us our position on the convolution object.
				// Calculate the center on our output image from our position on the convoluted input.
				val outcenterRow = inRow * rowStride
				val outcenterCol = inCol * columnStride

				// Iterate over the kernel and use that to apply our output.
				for (kRow in 0..kernel.numRows() - 1) {
					for (kCol in 0..kernel.numCols() - 1) {
						val outRow = outcenterRow - kernel.numRows() / 2 + kRow
						val outCol = outcenterCol - kernel.numCols() / 2 + kCol

						if (outRow >= 0 && outRow < adjoint.numRows() && outCol >= 0 && outCol < adjoint.numCols()) {
							// output.set(outRow, outCol, output.get(outRow, outCol) + kernel.get(kRow, kCol)*input.get(inRow, inCol));
							// Given ourput = input * kernel, and our adjoint applies to the output, 
							// adj(input) += adj(output)*kernel
							// adj(kernel) += adj(output)*input
							inputAdjoint.set(inRow, inCol, inputAdjoint.get(inRow, inCol) + adjoint.get(outRow, outCol) * kernel.get(kRow, kCol))
							kernelAdjoint.set(kRow, kCol, kernelAdjoint.get(kRow, kCol) + adjoint.get(outRow, outCol) * input.get(inRow, inCol))
						}
					}
				}
			}
		}

		// For each filter, sum the element-wise product with the input volume and assign it to the output.
		return arrayOf<Matrix<Double>>(inputAdjoint, kernelAdjoint)
	}

	// Used to augment serialization.
	override fun dataToString(): String {
		return padding.toString() + "," + rowStride + "," + columnStride
	}

	override fun stringToData(s: String) {
		val tokens = s.split(",".toRegex()).dropLastWhile { it.isEmpty() }.toTypedArray()
		this.padding = Integer.parseInt(tokens[0])
		this.rowStride = Integer.parseInt(tokens[1])
		this.rowStride = Integer.parseInt(tokens[2])
	}
}

