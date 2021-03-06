package io.xoana.ikarus.nodes

import koma.matrix.*
import koma.*

/**
 * Created by jcatrambone on 5/24/17.
 */

class RepeatNode(input:Node, var hRepeats: Int, var vRepeats: Int) : Node(input.rows*vRepeats, input.columns*hRepeats, arrayOf(input)) {
	override fun forward(args:Array<Matrix<Double>>): Matrix<Double> {
		return fill(args[0].numRows()*vRepeats, args[0].numCols()*hRepeats, { i,j -> args[0][i%args[0].numRows(), j%args[0].numCols()]})
	}

	override fun reverse(forward: Array<Matrix<Double>>, adjoint: Matrix<Double>, cachedOutput: Matrix<Double>?): Array<Matrix<Double>> {
		val ret = zeros(forward[0].numRows(), forward[0].numCols())
		adjoint.forEachIndexed({ i, j, v -> ret[i%ret.numRows(), j%ret.numCols()] += v})
		return arrayOf(ret)
	}

	override fun dataToString(): String = "$hRepeats,$vRepeats"
	override fun stringToData(s:String) {
		val tokens = s.split(",")
		hRepeats = tokens[0].toInt()
		vRepeats = tokens[1].toInt()
		this.rows = inputs[0].rows*vRepeats
		this.columns = inputs[0].columns*hRepeats
	}
}

class HStackNode(left:Node, right:Node) : Node(left.rows, left.columns+right.columns, arrayOf(left, right)) {
	override fun forward(args:Array<Matrix<Double>>): Matrix<Double> {
		return hstack(*args)
	}

	override fun reverse(forward: Array<Matrix<Double>>, adjoint: Matrix<Double>, cachedOutput: Matrix<Double>?): Array<Matrix<Double>> {
		return arrayOf(
			adjoint[0..adjoint.numRows()-1,0..forward[0].numCols()-1],
			adjoint[0..adjoint.numRows()-1,forward[0].numCols()..adjoint.numCols()-1]
		)
	}
}

class VStackNode(left:Node, right:Node) : Node(left.rows+right.rows, left.columns, arrayOf(left, right)) {
	override fun forward(args:Array<Matrix<Double>>): Matrix<Double> {
		return vstack(*args)
	}

	override fun reverse(forward: Array<Matrix<Double>>, adjoint: Matrix<Double>, cachedOutput: Matrix<Double>?): Array<Matrix<Double>> {
		/*
		val res = mutableListOf<Matrix<Double>>()
		var columnsConsumed = 0 // We may be joining a variable number of columns.
		forward.forEachIndexed({i,v ->
			res.add(i, adjoint[0..adjoint.numRows()-1, 0..1])
		})
		return res.toTypedArray()
		*/
		return arrayOf(
			adjoint[0..forward[0].numRows()-1, 0..adjoint.numCols()-1],
			adjoint[forward[0].numRows()..adjoint.numRows()-1, 0..adjoint.numCols()-1]
		)
	}
}

class ReshapeNode(input:Node, newRows:Int, newColumns:Int) : Node(
		if(newRows == -1) { (input.rows*input.columns)/newColumns } else { newRows },
		if(newColumns == -1) { (input.rows*input.columns)/newRows } else { newColumns },
		arrayOf(input)
) {
	init {
		assert(rows != -1 || columns != -1)
		assert(rows*columns == input.rows*input.columns)
	}

	override fun forward(args:Array<Matrix<Double>>): Matrix<Double> {
		return create(args[0].getDoubleData(), rows, columns)
	}

	override fun reverse(forward: Array<Matrix<Double>>, adjoint: Matrix<Double>, cachedOutput: Matrix<Double>?): Array<Matrix<Double>> {
		return arrayOf(create(adjoint.getDoubleData(), forward[0].numRows(), forward[0].numCols()))
	}
}

class RowSumNode(input:Node) : Node(input.rows, 1, arrayOf(input)) {
	override fun forward(args:Array<Matrix<Double>>): Matrix<Double> {
		val ret = fill(args[0].numRows(), 1, { i, _ ->
			args[0].getRow(i).elementSum()
		})
		return ret
	}

	override fun reverse(forward: Array<Matrix<Double>>, adjoint: Matrix<Double>, cachedOutput: Matrix<Double>?): Array<Matrix<Double>> {
		return arrayOf(
			fill(forward[0].numRows(), forward[0].numCols(), { i,j ->
				adjoint[i,0]
			})
		)
	}
}

class CollapseSumNode(input:Node) : Node(1, 1, arrayOf(input)) {
	override fun forward(args:Array<Matrix<Double>>): Matrix<Double> {
		return mat[args[0].elementSum()]
	}

	override fun reverse(forward: Array<Matrix<Double>>, adjoint: Matrix<Double>, cachedOutput: Matrix<Double>?): Array<Matrix<Double>> {
		return arrayOf(fill(forward[0].numRows(), forward[0].numCols(), adjoint[0,0]))
	}
}