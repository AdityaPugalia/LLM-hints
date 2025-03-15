# from ollama import chat, Client
from openai import OpenAI 
import openai
import config
import os
import json
import pandas as pd
from db import Database
from qep import Graph

class LLM:
    def __init__(self):
        self.client = OpenAI(api_key= config.OPENAI_API_KEY)
        self.index_prompt = self.load_indexes_from_file()
        self.chat_history = [
            {
                'role': 'system',
                'content': "".join([ "You are a Database query optimiser for postgresql.",
                            " We are using an extension for postgresql called pg_hint_plan which allows users to append hints to the query to suggest optimal physical query plan.",
                            " Your job is the produce these hints to be used for query optimisations. You will be provided 3 inputs: \n",
                            "1. The original SQL query\n",
                            "2. The list query nodes which provides information about all the nodes in the query plan tree. Each node has an id to identify it\n",
                            "3. the list of edges of the query plan tree in the format (a, b) where a is the id of the parent node and b is the id of the child node",
                            "Note: You cannot modify the query itself but only provide the hint. ",
                            "Note: The operator type might not be the most optimal one in the suggested query plan tree. Your task is to find the most optimal operator"
                ])
            },
            {
                'role': 'system',
                'content': "".join([
                    "For nodes with scan operators you must choose the best alternate scan method. You can choose one of the 6 hints below:\n",
                    "1. SeqScan(table_name)     #if sequential scan is the best option for the query.\n",
                    "2. TidScan(table_name)     #if TID Scan is the best option for the query\n"
                    "3. IndexScan(table_name index_column)      #if index scan is the best option.\n",
                    "4. IndexOnlyScan(table_name index_column)      #if index only scan is the best option.\n",
                    "5. BitmapScan(table_name index_column)       #if Bitmap scan is the best option.\n",
                    "6. no modification       #if the suggested scan in the input query plan tree is optimal.\n",
                    "table_name is the relation name on which scan is to be performed in the current node. you cannot use the name of a relation if it is not being scanned in the given node.\n",
                    "index_names contains comma seperated names of all columns on which the index is used.\n",
                    "For SeqScan, TidScan table name must be provided in the above format and for IndexScan, IndexOnlyScan, BitmapScan both relation name and index names must be provided in the abover format.\n",
                    "limit your hints for scan operators to the above choices. Do not provide anything else. You can only choose one of the above 6 hints for each scan node.",
                    "You do not always have to suggest a different operator for every scan operation. If you think the scan option is optimal select option 6: no modification"
                ])
            }, self.index_prompt,
            {
                'role': 'system',
                'content': "".join([
                    "For nodes with join operators you must choose the best alternate scan method. You can choose one of the 4 hints below:\n",
                    "1. NestLoop(table_names)     #if Nested loop join is the best plan according to you\n",
                    "2. HashJoin(table_names)     #if Hash join is the best plan according to you\n"
                    "3. MergeJoin(table_names)    #if Merge Join is the best plan according to you.\n",
                    "4. no modification           #If you think the current join opeator for the node is optimal and we do not need to use anything else.\n",
                    "table_names is the relation names on which join is to be performed. E.g. HashJoin(t1 t2) joins table t1 and t2 using hashjoin\n",
                    "limit your hints for join operators to the above choices. Do not provide anything else. You can only choose one of the above 4 hints for each scan node.",
                    "You do not always have to suggest a different operator for every join operation. If you think the scan option is optimal select option 6: no modification"
                ])
            },
            {
                'role' : 'system',
                'content' :"".join(['Give the hint for the query plan tree in the following format:\n',
                            'node 1: hint 1 ; node 2: hint 2 ; ... ; node n : hint n ;\n'])
            },
            {
                'role' : 'system',
                'content': 'Do not provide any justification for your hint. Simply state the hint'
            }
        ]
        self.demonstrations = []


    def askLLM(self, message : str, include_demonstration : bool = False):
        if include_demonstration:
            message = self.chat_history + self.demonstrations + [{'role': 'user', 'content': message}]
        else:
            message = self.chat_history + [{'role': 'user', 'content': message}]
        #print(message, '\n')
        response = self.client.chat.completions.create(model="gpt-4o", messages= message)
        return response.choices[0].message.content

    def getHints(self, query : str, qep : dict, edges = list, include_demonstration : bool = False):
        qep_str = json.dumps(qep)
        edges_str = json.dumps(edges)
        message = "".join((
                    f"provide hints for the following query"
                    f"The query is as follows: \n{query}\n"
                    f"The qep is as follows: \n{qep_str}\n"
                    f"The edges are as follows: \n{edges_str}\n"
                ))
        response = self.askLLM(message, include_demonstration)
        hints = self.parseResponse(response)
        return response, hints + query
    
    def get_n_hints(self, query : str, qep : dict, edges = list, n : int = 5):
        qep_str = json.dumps(qep)
        edges_str = json.dumps(edges)
        message = "".join((
                    f"Given the above instructions your role is to provide the top {n} different hint sets for the following query which could improve its execution time: \n"
                    f"The query is as follows: \n{query}\n"
                    f"The qep is as follows: \n{qep_str}\n"
                    f"The edges are as follows: \n{edges_str}\n"
                    f"Make sure that each of the hintsets are not exactly identical. the format to follow is : hintset1 | hintset2 | ... | hintsetn \n",
                    'If you cannot create n valid hints which can each likely improve the query execution time, then submit as many hints as you think would improve execution time, so if you cannot find a better alternative, stick to \'no modification\'',
                    'remember the rule that you can only suggest index-scans for colums which have indices.'
                ))
        response = self.askLLM(message)
        hintsets, hints = self.parse_n_responses(response)
        mod_queries = []
        for i in range (len(hints)):
            mod_queries.append(hints[i] + query)
        return hintsets, mod_queries
    
    def parseResponse(self, response: str):
        hints = response.split(';')
        final_hint = '/*+ ' 
        for hint in hints:
            hint = hint.strip()
            node_hint = hint.split(':')
            if (len(node_hint) < 2):
                continue
            node_hint[1] = node_hint[1].strip()
            if 'no modification' in node_hint[1]:
                continue
            else:
                final_hint += f"{node_hint[1]} "
        final_hint += '*/ '
        return final_hint
    
    def parse_n_responses(self, response: str):
        hintsets = response.split('|')
        hints = []
        for i in range(len(hintsets)):
            hints.append(self.parseResponse(hintsets[i]))
        return hintsets, hints
    
    def create_demonstrations(self, data: pd.DataFrame):
        self.demonstrations = []
        for i in range(data.shape[0]):
            content = "".join((
                    f"provide hints for the following query"
                    f"The query is as follows: \n{data['query'].iloc[i]}\n"
                    f"The qep is as follows: \n{data['qep_str'].iloc[i]}\n"
                    f"The edges are as follows: \n{data['edges_str'].iloc[i]}\n"
                ))
            response = data['hints'].iloc[i]
            self.demonstrations.extend([{'role' : 'user', 'content' : content}, {'role' : 'assisstant' , 'content' : response}])
    
    def load_indexes_from_file(self, file_path=config.DBASE_INDEX):
        """
        Reads the database index JSON file and returns the data as a dictionary.

        :param file_path: Path to the JSON file containing database indexes.
        :return: Dictionary containing index information.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found. Run get_indexes() first.")

        try:
            with open(file_path, 'r') as file:
                indexes = json.dumps(json.load(file))
                index_prompt = {
                    'role' : 'system',
                    'content' : "".join(['Use the following information on indices to guide you.\n',
                                         f'Here are the list of all indices for each table in the database: {indexes}\n'
                                         '**Suggest index-related scans (Index Scan, Index Only Scan, Bitmap Index Scan)** ONLY if the column has an index.\n'])
                }
                return index_prompt
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from {file_path}: {e}")

if __name__ == "__main__":
    model = LLM()
    print(model.chat_history)
    # command = 'SELECT COUNT(*) FROM customer INNER JOIN orders ON customer.c_custkey = orders.o_custkey WHERE o_orderdate > \'2015-01-01\';'
    # db = Database(user= config.USER, dbname= config.DBASE, password= config.PASSWORD)
    # db.connect()
    # qep, _, _, _, error = db.getQep(command)
    # if error:
    #     print(error)
    #     exit
    # g = Graph()
    # print('QEP: ', qep)
    # g.parseQep(qep)
    # # print(g.nodes, g.edges)
    # response, hintsets = model.get_n_hints(command, g.nodes, g.edges)
    # df = pd.DataFrame({'hints': hintsets})
    # for hint in hintsets:
    #     print(hint, '\n')
    # db.close()