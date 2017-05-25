package io.xoana.ikarus.nodes;

import koma.*;
import koma.matrix.*

abstract class Node(val rows:Int, val columns:Int, val inputs:Array<Node>) {
	abstract fun forward(args:Array<Matrix<Double>>): Matrix<Double>;
	abstract fun reverse(forward: Array<Matrix<Double>>, adjoint: Matrix<Double>): Array<Matrix<Double>>;
}

