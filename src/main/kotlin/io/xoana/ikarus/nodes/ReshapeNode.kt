package io.xoana.ikarus.nodes

import koma.matrix.*
import koma.*

/**
 * Created by jcatrambone on 5/24/17.
 */

class RepeatNode(input:Node, hRepeats: Int, vRepeats: Int) : Node(input.rows*vRepeats, input.columns*hRepeats, arrayOf(input)) {
	override fun forward(args:Array<Matrix<Double>>): Matrix<Double> {
		TODO()
	}

	override fun reverse(forward: Array<Matrix<Double>>, adjoint: Matrix<Double>, cachedOutput: Matrix<Double>?): Array<Matrix<Double>> {
		TODO()
	}
}

class HStackNode(left:Node, right:Node) : Node(left.rows, left.columns+right.columns, arrayOf(left, right)) {
	override fun forward(args:Array<Matrix<Double>>): Matrix<Double> {
		TODO()
	}

	override fun reverse(forward: Array<Matrix<Double>>, adjoint: Matrix<Double>, cachedOutput: Matrix<Double>?): Array<Matrix<Double>> {
		TODO()
	}
}

class VStackNode(left:Node, right:Node) : Node(left.rows+right.rows, left.columns, arrayOf(left, right)) {
	override fun forward(args:Array<Matrix<Double>>): Matrix<Double> {
		TODO()
	}

	override fun reverse(forward: Array<Matrix<Double>>, adjoint: Matrix<Double>, cachedOutput: Matrix<Double>?): Array<Matrix<Double>> {
		TODO()
	}
}

class ReshapeNode(input:Node, newRows:Int, newColumns:Int) : Node(newRows, newColumns, arrayOf(input)) {
	override fun forward(args:Array<Matrix<Double>>): Matrix<Double> {
		TODO()
	}

	override fun reverse(forward: Array<Matrix<Double>>, adjoint: Matrix<Double>, cachedOutput: Matrix<Double>?): Array<Matrix<Double>> {
		TODO()
	}
}