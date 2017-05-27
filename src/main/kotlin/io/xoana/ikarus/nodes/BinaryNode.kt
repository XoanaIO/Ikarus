package io.xoana.ikarus.nodes;

import koma.*;
import koma.matrix.*

import io.xoana.ikarus.*;

class AddNode(left:Node, right:Node) : Node(left.rows, left.columns, arrayOf(left, right)) {
	override fun forward(args:Array<Matrix<Double>>): Matrix<Double> {
		return args[0] + args[1]
	}

	override fun reverse(forward: Array<Matrix<Double>>, adjoint: Matrix<Double>, cachedOutput: Matrix<Double>?): Array< Matrix<Double> > {
		return arrayOf(adjoint, adjoint)
	}
}

class MatrixMultiplyNode(left:Node, right:Node) : Node(left.rows, right.columns, arrayOf(left, right)) {
	override fun forward(args:Array<Matrix<Double>>): Matrix<Double> {
		return args[0]*args[1]
	}

	override fun reverse(forward: Array<Matrix<Double>>, adjoint: Matrix<Double>, cachedOutput: Matrix<Double>?): Array<Matrix<Double>> {
		return arrayOf(
				adjoint * forward[1].T,
				forward[0].T * adjoint
		)
	}
}

class ElementMultiplyNode(left:Node, right:Node) : Node(left.rows, left.columns, arrayOf(left, right)) {
	override fun forward(args:Array<Matrix<Double>>): Matrix<Double> {
		return args[0] emul args[1]
	}

	override fun reverse(forward: Array<Matrix<Double>>, adjoint: Matrix<Double>, cachedOutput: Matrix<Double>?): Array<Matrix<Double>> {
		return arrayOf(
				adjoint emul forward[1],
				adjoint emul forward[0]
		)
	}
}

class SubtractNode(left:Node, right:Node) : Node(left.rows, left.columns, arrayOf(left, right)) {
	override fun forward(args:Array<Matrix<Double>>): Matrix<Double> {
		return args[0] - args[1]
	}

	override fun reverse(forward: Array<Matrix<Double>>, adjoint: Matrix<Double>, cachedOutput: Matrix<Double>?): Array< Matrix<Double> > {
		return arrayOf(adjoint, -adjoint)
	}
}