package io.xoana.ikarus.nodes;

import koma.matrix.*

abstract class Node(var rows:Int, var columns:Int, var inputs:Array<Node>) {

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
			val inputs:Array<Node> = when(tokens[5]) {
				"" -> arrayOf()
				else -> tokens[5].split(',').map { inputIdToken -> other[inputIdToken.toInt()] }.toTypedArray()
			}
			val data = tokens[6]
			// TODO: We should really do this by reflection, even if it means including the Kotlin reflection lib.
			val n = when(className.substringAfterLast('.')) {
				// Binary ops
				"AddNode" -> AddNode(inputs[0], inputs[1])
				"MatrixMultiplyNode" -> MatrixMultiplyNode(inputs[0], inputs[1])
				"ElementMultiplyNode" -> ElementMultiplyNode(inputs[0], inputs[1])
				"SubtractNode" -> SubtractNode(inputs[0], inputs[1])
				// Special ops
				"VariableNode" -> VariableNode(rows, cols)
				"InputNode" -> InputNode(rows, cols)
				//::TanhNode.javaClass.canonicalName -> TanhNode(inputs[0])
				// Unary ops
				"AbsNode" -> AbsNode(inputs[0])
				"ExpNode" -> ExpNode(inputs[0])
				"InverseNode" -> InverseNode(inputs[0])
				"LogNode" -> LogNode(inputs[0])
				"NegateNode" -> NegateNode(inputs[0])
				"TanhNode" -> TanhNode(inputs[0])
				"SigmoidNode" -> SigmoidNode(inputs[0])
				"SinNode" -> SinNode(inputs[0])
				"PowerNode" -> PowerNode(inputs[0], 0.0)
				"ReLUNode" -> ReLUNode(inputs[0])
				"AddConstantNode" -> AddConstantNode(inputs[0], 0.0)
				// Reshape Nodes
				"RepeatNode" -> RepeatNode(inputs[0], rows, cols)
				"HStackNode" -> HStackNode(inputs[0], inputs[1])
				"VStackNode" -> VStackNode(inputs[0], inputs[1])
				"ReshapeNode" -> ReshapeNode(inputs[0], rows, cols)
				"RowSumNode" -> RowSumNode(inputs[0])
				"CollapseSumNode" -> CollapseSumNode(inputs[0])
				// Convolution Nodes
				"Convolution2DNode" -> Convolution2DNode(inputs[0], inputs[1], 1, 1) // The 1,1 will get fixed below.
				"Deconvolution2DNode" -> Deconvolution2DNode(inputs[0], inputs[1], 1, 1)
				// Everything else
				else -> {
					TODO()
				}
			}
			n.id = id
			n.name = name
			n.rows = rows
			n.columns = cols
			n.inputs = inputs
			n.stringToData(data)
			return n
		}
	}
}

