package io.xoana.ikarus.nodes

import io.xoana.ikarus.exp
import koma.matrix.*
import koma.matrix.Matrix
import koma.*

/**
 * Created by jcatrambone on 5/24/17.
 */
class VariableNode(rows:Int, columns:Int, weightScale:Double = 0.1) : Node(rows, columns, arrayOf()) {
	var data: Matrix<Double> = if(weightScale == 0.0) { zeros(rows, columns) } else { rand(rows,columns)*weightScale }

	override fun forward(args:Array<Matrix<Double>>): Matrix<Double> {
		return data
	}

	override fun reverse(forward: Array<Matrix<Double>>, adjoint: Matrix<Double>, cachedOutput: Matrix<Double>?): Array<Matrix<Double>> {
		return arrayOf()
	}

	override fun dataToString(): String = data.getDoubleData().joinToString()
	override fun stringToData(s:String) {
		this.data = create(s.split(',').map{ strVal -> strVal.toDouble() }.toDoubleArray(), this.rows, this.columns)
	}
}

class RandomNode(mean:Node, stddev:Node) : Node(mean.rows, mean.columns, arrayOf(mean, stddev)) {
	var lastOutput: Matrix<Double> = rand(rows,columns)

	override fun forward(args:Array<Matrix<Double>>): Matrix<Double> {
		lastOutput = args[0]+ rand(rows,columns).mapIndexed({ i,j,v -> args[1][i,j]*v.exp() })
		return lastOutput
	}

	override fun reverse(forward: Array<Matrix<Double>>, adjoint: Matrix<Double>, cachedOutput: Matrix<Double>?): Array<Matrix<Double>> {
		return arrayOf(adjoint, adjoint.mapIndexed({i,j,v -> v*forward[1][i,j].exp()}))
	}
}
