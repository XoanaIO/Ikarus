package io.xoana.ikarus.nodes;

import koma.*;
import koma.matrix.Matrix;

abstract class Node(val rows:Int, val columns:Int, val inputs:Array<Node>) {
	abstract fun forward(args:Array<Matrix>): Matrix;
	abstract fun reverse(forward: Array<Matrix>, adjoint: Matrix): Array<Matrix>;
}

class InputNode(rows, columns) : Node(rows, columns, arrayOf()) {
	override fun forward(args:Array<Matrix>): Matrix {
		return args[0]!!
	}

	override fun reverse(forward: Array<Matrix>, adjoint: Matrix): Array<Matrix> {
		return arrayOf()
	}
}

class UnaryNode(input: Node, f: (Double)->Double, df: (Double)->Double) : Node(input.rows, input.columns, arrayOf(input)) {
	// Forward: z := f(x)
	// adj_x += (adj_z dot df(x))
	override fun forward(args:Array<Matrix>): Matrix {
		return x.mapMat { f(it) }
	}

	override fun reverse(forward: Array<Matrix>, adjoint: Matrix): Array<Matrix> {
		return forward.forEach( fwd -> {
			fwd.mapMat { df(it) } emul adjoint
		})
	}
}
