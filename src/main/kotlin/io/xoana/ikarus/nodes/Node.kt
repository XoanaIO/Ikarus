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

	override fun toString():String {
		// File format:
		// <class name>|id|name|#rows|#columns|<comma-separated input ids>|<extra data>\n
		val sb = StringBuilder()
		sb.append(this.javaClass.canonicalName)
		sb.append(SERALIZED_FIELD_SEPARATOR)
		sb.append(this.id)
		sb.append(SERALIZED_FIELD_SEPARATOR)
		sb.append(this.name)
		sb.append(SERALIZED_FIELD_SEPARATOR)
		sb.append(this.rows)
		sb.append(SERALIZED_FIELD_SEPARATOR)
		sb.append(this.columns)
		sb.append(SERALIZED_FIELD_SEPARATOR)
		sb.append(inputs.joinToString(separator=",", transform={ inp -> ""+inp.id }))
		sb.append(SERALIZED_FIELD_SEPARATOR)
		sb.append(this.dataToString())
		return sb.toString()
	}

	companion object {
		const val SERALIZED_FIELD_SEPARATOR = '|'

		@JvmStatic
		fun fromString(s: String, other: List<Node>): Node {
			val tokens = s.split(SERALIZED_FIELD_SEPARATOR)
			val className = tokens[0]
			val id:Int = tokens[1].toInt()
			val name = tokens[2]
			val rows = tokens[3].toInt()
			val cols = tokens[4].toInt()
			val inputs = tokens[5].split(',').map { inputIdToken -> other[inputIdToken.toInt()] }
			val data = tokens[6]
			val n = when(className) {
				"InputNode" -> InputNode(rows, cols)
				else -> {
					TODO()
				}
			}
			n.stringToData(data)
			return n
		}
	}
}

