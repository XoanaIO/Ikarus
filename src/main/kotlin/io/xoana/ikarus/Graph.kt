package io.xoana.ikarus

import io.xoana.ikarus.nodes.*
import io.xoana.ikarus.nodes.Node;
import koma.*
import koma.matrix.*
import java.util.*

/**
 * Created by Joseph Catrambone on 2017/05/24.
 */
open class Graph {
	internal var nodes: MutableList<Node> = mutableListOf<Node>()

	fun addNode(n: Node): Node {
		// Make sure all the dependencies happen first.
		for (inp in n.inputs) {
			if (inp.id == -1) {
				this.addNode(inp)
			}
		}
		n.id = nodes.size
		nodes.add(n)
		return n // A pass-through.
	}

	open fun serializeToString(): String {
		val s = StringBuilder()
		for (n in nodes) {
			s.append(n.toString() + "\n")
		}
		return s.toString()
	}

	open fun restoreFromString(s: String) {
		var lines = s.split("[\\r\\n]+".toRegex()).dropLastWhile { it.isEmpty() }.toTypedArray() // Also removes empty lines.
		// Check for blank first line.
		if (lines[0] == "") {
			lines = Arrays.copyOfRange(lines, 1, lines.size) // Remove first line.
		}
		// Create new array and populate.
		nodes = ArrayList(lines.size) // Initial capacity = lines.length.
		for (i in lines.indices) {
			nodes.add(i, Node.fromString(lines[i], nodes))
		}
	}

	fun getOutput(inputs: Map<Node, DoubleArray>, node: Node): DoubleArray {
		// getOutput is different slightly from forward in that we don't care about unused paths.
		// For forward, we want to be sure _all_ different node values are populated in order.
		// getOutput spends a cycle or two figuring out which paths it can ignore (so we don't pay for training paths).
		val toProcess = Stack<Node>()
		val results = arrayOfNulls<Matrix<Double>>(node.id + 1)

		// Traverse our graph from the output to the inputs.
		toProcess.push(node)
		while (!toProcess.empty()) {
			val n = toProcess.pop()
			// Allocate an empty matrix for our results OR copy from input.
			if (n is InputNode) {
				results[n.id] = create(inputs[n]!!, n.rows, n.columns)
			} else {
				results[n.id] = zeros(n.rows, n.columns)
			}
			for (inp in n.inputs) {
				toProcess.push(inp)
			}
		}

		// Compute the values, skipping the dead nodes.
		for (i in results.indices) {
			val n = nodes[i]
			if (results[i] == null || n is InputNode) {
				continue
			}
			// Compile an array of values to be passed into the node.
			val forwardInputs = Array<Matrix<Double>>(n.inputs.size, { j -> results[n.inputs[j].id]!! })
			results[i] = nodes[i].forward(forwardInputs)
		}
		return results[node.id]!!.getDoubleData()

		/* // If we didn't care about path pruning, we could just do this:
		HashMap<Node, Matrix<Double>> remappedInputs = doubleMapToMatrixMap(inputs);
		return forward(remappedInputs)[node.id].data;
		*/
	}

	fun forward(datafeed: Map<Node, Matrix<Double>>): Array<Matrix<Double>> {
		val results = Array<Matrix<Double>>(nodes.size, { i -> zeros(nodes[i].rows, nodes[i].columns) })
		for (i in nodes.indices) {
			// Special case: inputs read from the input map.
			if (nodes[i] is InputNode) {
				if (datafeed[nodes[i]] == null) {
					System.err.println("Compute graph variable undefined on forward pass: Name '" + nodes[i].name + "'")
				} else {
					results[i] = datafeed[nodes[i]]!!
				}
			} else {
				// Compile an array of values to be passed into the node.
				val n = nodes[i]
				val forwardInputs = Array<Matrix<Double>>(n.inputs.size, { j -> results[n.inputs[j].id]!! })
				results[i] = nodes[i].forward(forwardInputs)
			}
		}
		return results
	}

	/***
	 * Calculate the gradient with respect to the given node.
	 * @param inputFeed A Hash Map of the input node -> matrix values.
	 * @param fwd The values from the forward pass if already computed.  If null, will compute them.
	 * @param node The value with respect to which we want the gradient.
	 * @return Returns an array of matrices wherein Matrix[node.id] corresponds to the node's gradient.
	 */
	fun getGradient(inputFeed: Map<Node, Matrix<Double>>, fwd: Array<Matrix<Double>>?, node: Node): Array<Matrix<Double>> {
		var fwd = fwd
		// If forward pass isn't calculated, do that.
		if (fwd == null) {
			fwd = forward(inputFeed)
		}

		// Populate our initial adjoints/gradients.
		// The output node id gets ones, everything else gets zeros.
		val grads = Array<Matrix<Double>>(nodes.size, { i -> zeros(nodes[i].rows, nodes[i].columns) })
		grads[node.id] = ones(node.rows, node.columns) // The output/target gets 1.0.

		// Starting from the out and propagating backwards, calculate the adjoints.
		for (i in node.id downTo 0) {
			// For all the inputs to this node, calculate their adjoints from this adjoint.
			val argInputs = Array<Matrix<Double>>(nodes[i].inputs.size, { j -> fwd!![nodes[i].inputs[j].id] })
			val nextAdjoints = nodes[i].reverse(argInputs, grads[i])
			for (j in 0..nodes[i].inputs.size - 1) {
				grads[nodes[i].inputs[j].id] += nextAdjoints[j]
			}
		}
		return grads
	}
}
