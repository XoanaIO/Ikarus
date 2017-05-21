package io.xoana.ikarus.nodes;

import koma.*;
import koma.matrix.*

abstract class Node<T : Number>(val rows:Int, val columns:Int, val inputs:Array<Node<T>>) {
	abstract fun forward(args:Array<Matrix<T>>): Matrix<T>;
	abstract fun reverse(forward: Array<Matrix<T>>, adjoint: Matrix<T>): Array<Matrix<T>>;
}

class InputNode<T:Number>(rows:Int, columns:Int) : Node<T>(rows, columns, arrayOf()) {
	override fun forward(args:Array<Matrix<T>>): Matrix<T> {
		return args[0]
	}

	override fun reverse(forward: Array<Matrix<T>>, adjoint: Matrix<T>): Array<Matrix<T>> {
		return arrayOf()
	}
}

class UnaryNode<T:Number>(input: Node<T>, val f: (T)->T, val df: (T)->T) : Node<T>(input.rows, input.columns, arrayOf(input)) {
	// Forward: z := f(x)
	// adj_x += (adj_z dot df(x))
	override fun forward(args:Array<Matrix<T>>): Matrix<T> {
		return args[0].map { f(it) }
	}

	override fun reverse(forward: Array<Matrix<T>>, adjoint: Matrix<T>): Array< Matrix<T> > {
		return forward.map{ fwd -> {
			fwd.map(df).elementTimes(adjoint)
		}.invoke() }.toTypedArray()
			//fwd.mapMat { df(it) } emul adjoint
	}
}
