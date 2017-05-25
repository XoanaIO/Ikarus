package io.xoana.ikarus.nodes;

import koma.*;
import koma.matrix.*

abstract class Node(val rows:Int, val columns:Int, val inputs:Array<Node>) {
	var id=-1
	var name=""
	abstract fun forward(args:Array<Matrix<Double>>): Matrix<Double>;
	abstract fun reverse(forward: Array<Matrix<Double>>, adjoint: Matrix<Double>, cachedOutput: Matrix<Double>?=null): Array<Matrix<Double>>;
	open fun dataToString(): String = ""
	open fun stringToData(s:String) {} // Noop for abstract.

	companion object {
		@JvmStatic
		fun fromString(s: String, other: List<Node>): Node {
			return InputNode(-1, -1)
		}
	}
}

