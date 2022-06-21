#!/usr/bin/env python

import itertools, random, string

# We disallow x->x edges because they seem to sometimes
# cause the SUT to reject the map as invalid.
ALLOW_SELFLOOPS = False

def random_word(minlength, maxlength):
	l = random.randint(minlength, maxlength)
	wordchars = [random.choice(string.ascii_letters) for i in range(l)]
	return ''.join(wordchars)

def random_weight(mindigits=1, maxdigits=10):
	assert maxdigits <= 10, "The SUT doesn't accept >10 digits in a cell."
	numdigits = random.randint(mindigits, maxdigits)
	if numdigits < 10:
		return ''.join(random.choice(string.digits) for i in range(numdigits))
	else: # enforce limit of Java MAX_INT
		return str(random.randint(1000000000, 2147483647))
	
def random_airports(howmany, minlength=1, maxlength=3):
	result = set()
	while len(result) < howmany:
		candidate = random_word(minlength, maxlength).upper()
		if candidate not in result:
			#print("Adding new candidate: " + candidate)
			result.add(candidate)
		else:
			#print("Skipping dup candidate: " + candidate)
			pass
	return list(result)

def random_graph(nodenames, mindens=0.0, maxdens=1.0, mindigits=1, maxdigits=10):
	'''Random map using given node names and parameters.'''
	nodes, edges = set(nodenames), set()
	edge_density = random.uniform(mindens, maxdens)
	for u, v in itertools.product(nodes, nodes):
		if u == v and not ALLOW_SELFLOOPS:
			continue
		if random.random() <= edge_density:
			weights = [random_weight(mindigits, maxdigits) for i in range(6)]
			edges.add((u, v) + tuple(weights))
	return nodes, edges

def random_map(n, mindens=0.0, maxdens=1.0, mindigits=1, maxdigits=10, minairportlen=1, maxairportlen=3):
	'''Random map within the given parameters.'''
	assert type(n) is int
	airportlist = random_airports(howmany=n, minlength=minairportlen, maxlength=maxairportlen)
	nodes, edges = random_graph(airportlist, mindens, maxdens, mindigits, maxdigits)
	return nodes, edges

def map2txt(nodes, edges):
	'''Convert our graph representation to the SUT's .txt format.'''
	numnodes, numedges = str(len(nodes)), str(len(edges))
	nodelines = '\n'.join(nodes)
	edgelines = '\n'.join([' '.join(map(str, e)) for e in edges])
	return '\n'.join([numnodes, nodelines, numedges, edgelines])

def ismapstronglyconnected(nodes, edges):
	'''This method can be used to determine the secret beforehand,
	   independently of the SUT. It requires the networkx package.'''
	import networkx
	g = networkx.DiGraph()
	for node in nodes:
		g.add_node(node)
	for edge in edges:
		g.add_edge(edge[0], edge[1]) # there is more stuff but we ignore it here
	return networkx.is_strongly_connected(g)


def random_graph_sc(nodenames, stronglyconnected=True, mindigits=1, maxdigits=10):
	assert len(nodenames) >= 3
	nodes, edges, finaledges = set(nodenames), set(), set()
	loner = nodenames[0]
	edger = nodenames[1]
	others = nodenames[2:]
	# edger starts a circle over all others and back to edger
	circle = [edger] + others
	for i in range(len(circle)):
		if i+1 < len(circle):
			edges.add((circle[i], circle[i+1]))
		else:
			edges.add((circle[i], circle[0]))
	# now add a random number of cross-edges between any except loner
	density = random.random()
	for u in circle:
		for v in circle:
			if random.random() > density:
				edges.add((u, v))
	# loner goes to edger
	edges.add((loner, edger))
	# edger goes to loner iff SC
	if stronglyconnected:
		edges.add((edger, loner))
	# final pass: add weights
	for edge in edges:
		weights = [random_weight(mindigits, maxdigits) for i in range(6)]
		finaledges.add(edge + tuple(weights))
	#print(ismapstronglyconnected(nodes, finaledges))
	return nodes, finaledges


def randomsc(n, stronglyconnected, minairportlen=1, maxairportlen=3, mindigits=6, maxdigits=10):
	airportlist = random_airports(howmany=n, minlength=minairportlen, maxlength=maxairportlen)
	nodes, edges = random_graph_sc(airportlist, stronglyconnected, mindigits, maxdigits)
	return nodes, edges


if __name__ == '__main__':
	#print("Example map with 7 cities, edge density between 0.3 and 0.5,")
	#print("cell digits between 3 and 5, and airport name length 1 to 3.")
	#nodes, edges = random_map(7, 0.3, 0.5, 3, 5, 1, 3)
	#print(map2txt(nodes, edges))
	print(map2txt(*randomsc(10, True)))
