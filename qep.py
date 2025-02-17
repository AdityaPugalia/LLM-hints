import networkx as nx


class Graph:
    def __init__(self):
        self.G = nx.DiGraph()
        self.nodes = []
        self.edges = [[], []]
        self.node_counter = 0  # for ID generation

    def parseQep(self, qep):
        """
        Parse the query execution plan (QEP) and build the nodes and edges for the graph.
        """

        def traversePlan(plan_tree, parent_id=None):
            node_id = self.node_counter  # Use the counter as the node ID
            self.node_counter += 1
            node_label = plan_tree['Node Type']
            total_cost = plan_tree.get('Total Cost', -1)
            rows = plan_tree.get('Plan Rows', -1)
            rel_name = plan_tree.get('Relation Name', None)
            index_name = plan_tree.get('Index Name', None)
            node_filter = plan_tree.get('Filter', None)
            join_filter = plan_tree.get('Join Filter', None)
            hash_cond = plan_tree.get('Hash Cond', None)
            merge_cond = plan_tree.get('Merge Cond', None)
            index_cond = plan_tree.get('Index Cond', None)
            width = plan_tree.get('Plan Width', 0)
            parent_hash = False
            if parent_id != None:
                if self.nodes[parent_id]['label'] == 'Hash':
                    parent_hash = True

            node_info = {
                'parent id' : parent_id,
                'id': node_id,
                'label': node_label,
                'Cummulative Cost': total_cost,
                'Planned rows': rows,
                'Relation Name': rel_name,
                'Index Name': index_name,
                'Filter' : node_filter if node_filter else join_filter,
                'Join Condition' : hash_cond if hash_cond is not None else merge_cond,
                'Index Condition' : index_cond,
                'Width' : width,
                'Parent Hash': parent_hash
            }
            # print(f"Appending node: {node_info}")
            self.nodes.append(node_info)

            if parent_id is not None:
                # print(f"Appending edge: {node_id} -> {parent_id}")
                self.edges[0].append(node_id)
                self.edges[1].append(parent_id)

            if 'Plans' in plan_tree:
                for subplan in plan_tree['Plans']:
                    # Recursively traverse the plan tree
                    # print(f"Traversing subplan: {subplan['Node Type']}")
                    traversePlan(subplan, parent_id=node_id)

        # Start traversing from the root plan
        root_plan = qep[0]['Plan']
        # Traverse the plan tree
        traversePlan(root_plan)

    def printGraph(self):
        """
        Print the nodes and edges of the graph.
        """
        print(self.nodes)
        print(self.edges)

    def buildGraph(self, as_dict):
        """
        Build the graph from the nodes and edges.
        """

        for node in self.nodes:
            self.G.add_node(node['id'], label=node['label'], cost=node['cost'], rows=node['rows'], info=node['info'])
        for edge in self.edges:
            self.G.add_edge(edge[0], edge[1])
        if as_dict:
            return nx.to_dict_of_dicts(self.G)
        return self.G