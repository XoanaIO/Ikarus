package io.xoana.ikarus.nodes;

import koma.*;
import koma.matrix.*

import io.xoana.ikarus.*;

class InputNode(rows:Int, columns:Int) : Node(rows, columns, arrayOf()) {
	override fun forward(args:Array<Matrix<Double>>): Matrix<Double> {
		return args[0]
	}

	override fun reverse(forward: Array<Matrix<Double>>, adjoint: Matrix<Double>): Array<Matrix<Double>> {
		return arrayOf()
	}
}

open class UnaryNode(input: Node, val f: (Double)->Double, val df: (Double)->Double) : Node(input.rows, input.columns, arrayOf(input)) {
	// Forward: z := f(x)
	// adj_x += (adj_z dot df(x))
	override fun forward(args:Array<Matrix<Double>>): Matrix<Double> {
		return args[0].map { f(it) }
	}

	override fun reverse(forward: Array<Matrix<Double>>, adjoint: Matrix<Double>): Array< Matrix<Double> > {
		return forward.map{ fwd -> {
			fwd.map(df).elementTimes(adjoint)
		}.invoke() }.toTypedArray()
			//fwd.mapMat { df(it) } emul adjoint
	}
}

open class FastUnaryNode(input: Node, val f:(Matrix<Double>)->Matrix<Double>, val df:(Matrix<Double>)->Matrix<Double>) : Node(input.rows, input.columns, arrayOf(input)) {
	override fun forward(args:Array<Matrix<Double>>): Matrix<Double> {
		return f(args[0])
	}

	override fun reverse(forward: Array<Matrix<Double>>, adjoint: Matrix<Double>): Array<Matrix<Double>> {
		return arrayOf(df(forward[0]).elementTimes(adjoint))
	}
}

class TanhNode(input:Node) : UnaryNode(input, { x:Double -> x.tanh() }, { x:Double -> 1.0 - x.tanh().pow(2.0) });
//class TanhNode(input:Node) : FastUnaryNode(input, { (x:Matrix<Double>) -> tanh(x) }, { (x:Matrix<Double>) -> 1.0 - tanh(x)*tanh(x) });
