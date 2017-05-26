package io.xoana.ikarus.optimizers

import koma.matrix.*

import io.xoana.ikarus.Graph
import io.xoana.ikarus.nodes.Node
import io.xoana.ikarus.nodes.VariableNode

/**
 * Created by jcatrambone on 5/25/17.
 */
abstract class Optimizer(val graph: Graph, val variables: Array<VariableNode>) {
	abstract fun minimize(loss: Node, inputFeed: Map<Node, Matrix<Double>>):Double
}