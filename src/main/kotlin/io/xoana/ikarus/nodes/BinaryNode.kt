package io.xoana.ikarus.nodes;

import koma.*;
import koma.matrix.*

import io.xoana.ikarus.*;

class MatrixMultiply(left:Node, right:Node) : Node(left.rows, right.columns, arrayOf(left, right)) {
	override fun forward(args:Array<Matrix<Double>>): Matrix<Double> {
		return args[0]*args[1]
	}

	override fun reverse(forward: Array<Matrix<Double>>, adjoint: Matrix<Double>): Array<Matrix<Double>> {
		return arrayOf(
			adjoint * forward[1].T,
			forward[0].T * adjoint
		)
	}
}

class AddNode(left:Node, right:Node) : Node(left.rows, left.columns, arrayOf(left, right)) {
	override fun forward(args:Array<Matrix<Double>>): Matrix<Double> {
		return args[0] + args[1]
	}

	override fun reverse(forward: Array<Matrix<Double>>, adjoint: Matrix<Double>): Array< Matrix<Double> > {
		return arrayOf(forward[1].elementTimes(adjoint), forward[0].elementTimes(adjoint))
	}
}
