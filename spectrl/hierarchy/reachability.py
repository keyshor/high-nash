class AbstractEdge:
    '''
    Class defining an abstract edge.
    Vertices are integers from 0 to |U|.

    Parameters:
        target: int (target vertex)
        predicate: state, resource -> float (predicate corresponding to target)
        constraints: list of constraints that needs to be maintained (one after the other)
    '''

    def __init__(self, target, predicate, constraints):
        self.target = target
        self.predicate = predicate
        self.constraints = constraints


class AbstractReachability:
    '''
    Class defining the abstract reachability problem.

    Parameters:
        abstract_graph: list of list of abstract edges (adjacency list).
        final_vertices: set of int (set of final vertices).

    Initial vertex is assumed to be 0.
    '''

    def __init__(self, abstract_graph, final_vertices):
        self.abstract_graph = abstract_graph
        self.final_vertices = final_vertices
        self.num_vertices = len(self.abstract_graph)

    def pretty_print(self):
        for i in range(self.num_vertices):
            targets = ''
            for edge in self.abstract_graph[i]:
                targets += ' ' + str(edge.target)
            print(str(i) + ' ->' + targets)


class HierarchicalPolicy:

    def __init__(self, abstract_policy, nn_policies, abstract_graph, sys_dim):
        self.abstract_policy = abstract_policy
        self.nn_policies = nn_policies
        self.abstract_graph = abstract_graph
        self.vertex = 0
        self.edge = self.abstract_graph[0][self.abstract_policy[0]]
        self.sys_dim = sys_dim

    def get_action(self, state):
        sys_state = state[:self.sys_dim]
        res_state = state[self.sys_dim:]
        if self.edge.predicate is not None:
            if self.edge.predicate(sys_state, res_state) > 0:
                self.vertex = self.edge.target
                self.edge = self.abstract_graph[self.vertex][self.abstract_policy[self.vertex]]
        return self.nn_policies[self.vertex][self.abstract_policy[self.vertex]].get_action(state)

    def reset(self):
        self.vertex = 0
        self.edge = self.abstract_graph[0][self.abstract_policy[0]]
