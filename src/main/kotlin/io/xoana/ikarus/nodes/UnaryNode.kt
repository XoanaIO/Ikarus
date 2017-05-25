package io.xoana.ikarus.nodes;

import koma.*;
import koma.matrix.*

import io.xoana.ikarus.*;

class InputNode(rows:Int, columns:Int) : Node(rows, columns, arrayOf()) {
	override fun forward(args:Array<Matrix<Double>>): Matrix<Double> {
		return args[0]
	}

	override fun reverse(forward: Array<Matrix<Double>>, adjoint: Matrix<Double>, cachedOutput: Matrix<Double>?): Array<Matrix<Double>> {
		return arrayOf()
	}
}

/*** UnaryNode
 * Unary node supports arbitrary operations on the elements of an input matrix.
 * It allows us (royal we) to quickly implement element-wise computations.
 * This is not as efficient as the fast unary operator because every element is transformed by the JVM instead of using
 * the native matrix computation library.  The upside is it's more general.  Anything can be computed.
 */
open class UnaryNode(input: Node, val f: (Double)->Double, val df: (Double)->Double) : Node(input.rows, input.columns, arrayOf(input)) {
	// Forward: z := f(x)
	// adj_x += (adj_z dot df(x))
	override fun forward(args:Array<Matrix<Double>>): Matrix<Double> {
		return args[0].map { f(it) }
	}

	override fun reverse(forward: Array<Matrix<Double>>, adjoint: Matrix<Double>, cachedOutput: Matrix<Double>?): Array< Matrix<Double> > {
		return forward.map{ fwd -> {
			fwd.map(df).elementTimes(adjoint)
		}.invoke() }.toTypedArray()
			//fwd.mapMat { df(it) } emul adjoint
	}
}

/*** FastUnaryNode
 * A faster version of the UnaryOp.  The operation must be expressable as a Matrix operation (i.e. have Koma support).
 */
open class FastUnaryNode(input: Node, val f:(Matrix<Double>)->Matrix<Double>, val df:(Matrix<Double>)->Matrix<Double>) : Node(input.rows, input.columns, arrayOf(input)) {
	override fun forward(args:Array<Matrix<Double>>): Matrix<Double> {
		return f(args[0])
	}

	override fun reverse(forward: Array<Matrix<Double>>, adjoint: Matrix<Double>, cachedOutput: Matrix<Double>?): Array<Matrix<Double>> {
		return arrayOf(df(forward[0]).elementTimes(adjoint))
	}
}

class AbsNode(input:Node) : FastUnaryNode(input, { x:Matrix<Double> -> abs(x) }, { x:Matrix<Double> -> sign(x) })
class ExpNode(input:Node) : FastUnaryNode(input, { x -> exp(x) }, { x -> exp(x) })
class InverseNode(input: Node) : UnaryNode(input, { x -> 1/x }, { x -> -1.0/(x*x) })
//class LogNode(input: Node) : FastUnaryNode(input, { x -> log(x) }, { x -> 1.0/x })
class LogNode(input: Node) : UnaryNode(input, { x -> x.log() }, { x -> 1.0/x })
class NegateNode(input: Node) : FastUnaryNode(input, {x -> -x}, {x -> fill(x.numRows(), x.numCols(), -1.0) })
class TanhNode(input:Node) : UnaryNode(input, { x:Double -> x.tanh() }, { x:Double -> 1.0 - x.tanh().pow(2.0) });
class SigmoidNode(input:Node) : UnaryNode(input, { x -> 1.0/(1.0+((-x).exp())) }, { x -> (-x).exp().div((1.0+(-x).exp()).pow(2.0)) })
class SinNode(input:Node) : FastUnaryNode(input, { x:Matrix<Double> -> sin(x) }, { x:Matrix<Double> -> cos(x) })

class PowerNode(input: Node, var c:Double) : FastUnaryNode(input, {x -> x.epow(c)}, {x -> x.times(c).epow(c-1.0) }) {
	override fun dataToString(): String = "$c"
	override fun stringToData(s:String) { this.c = s.toDouble() }
}

class ReLUNode(input: Node, var c:Double = 1e-6) : UnaryNode(input, {x -> Math.max(x, 0.0)}, {x -> if(x>0) { 1.0 } else { c } }) {
	override fun dataToString(): String = "$c"
	override fun stringToData(s:String) { this.c = s.toDouble() }
}

class AddConstantNode(input:Node, var c:Double) : FastUnaryNode(input, {x -> x+c}, {x:Matrix<Double> -> ones(x.numRows(), x.numCols())}) {
	override fun dataToString(): String = "$c"
	override fun stringToData(s:String) { this.c = s.toDouble() }
}