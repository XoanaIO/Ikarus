package io.xoana.ikarus.optimizers

import io.xoana.ikarus.Graph
import io.xoana.ikarus.nodes.Node
import io.xoana.ikarus.nodes.VariableNode
import koma.matrix.Matrix

/**
 * Created by jcatrambone on 5/25/17.
 */
class SGD(g: Graph, variables:Array<VariableNode>, var learningRate:Double):Optimizer(g, variables) {
	override fun minimize(loss: Node, inputFeed: Map<Node, Matrix<Double>>):Double {
		val fwd = graph.forward(inputFeed)
		val grads = graph.getGradient(inputFeed, fwd, loss)

		// Apply gradients.
		variables.forEach { v ->
			v.data -= grads[v.id]!!*learningRate
		}

		return fwd[loss.id].elementSum()
	}

}