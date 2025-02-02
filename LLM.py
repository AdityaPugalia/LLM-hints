from ollama import chat, Client
from openai import OpenAI 
import openai
import config
import os
import json

class LLM:
    def __init__(self):
        self.client = OpenAI(api_key= os.environ.get('OPENAI_API_KEY'))
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
                    "1. SeqScan[table_name]     #if sequential scan is the best option for the query.\n",
                    "2. TidScan[table_name]     #if TID Scan is the best option for the query\n"
                    "3. IndexScan[table_name[index_names]]      #if index scan is the best option.\n",
                    "4. IndexOnlyScan[table_name[index_names]]      #if index only scan is the best option.\n",
                    "5. BitmapScan[table_name[index_names]]       #if Bitmap scan is the best option.\n",
                    "6. no modification       #if the suggested scan in the input query plan tree is optimal.\n",
                    "table_name is the relation name on which scan is to be performed in the current node. you cannot use the name of a relation if it is not being scanned in the given node.\n",
                    "index_names contains comma seperated names of all columns on which the index is used.\n",
                    "For SeqScan, TidScan table name must be provided in the above format and for IndexScan, IndexOnlyScan, BitmapScan both relation name and index names must be provided in the abover format.\n",
                    "limit your hints for scan operators to the above choices. Do not provide anything else. You can only choose one of the above 6 hints for each scan node.",
                    "Do not choose the same scan operator as the one in the Query node. If you think the scan operator in the query node is optimal then output no modification"
                ])
            },
            {
                'role': 'system',
                'content': "".join([
                    "For nodes with join operators you must choose the best alternate scan method. You can choose one of the 4 hints below:\n",
                    "1. NestLoop[table_names]     #if Nested loop join is the best plan according to you\n",
                    "2. HashJoin[table_names]     #if Hash join is the best plan according to you\n"
                    "3. MergeJoin[table_names]    #if Merge Join is the best plan according to you.\n",
                    "4. no modification           #If you think the current join opeator for the node is optimal and we do not need to use anything else.\n",
                    "table_names is the relation names on which join is to be performed. E.g. HashJoin(t1 t2) joins table t1 and t2 using hashjoin\n",
                    "limit your hints for join operators to the above choices. Do not provide anything else. You can only choose one of the above 4 hints for each scan node.",
                    "Do not choose the same join operator as the one in the Query node. If you think the join operator in the query node is optimal then output no modification "
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

    def askLLM(self, message : str):
        message = self.chat_history + [{'role': 'user', 'content': message}]
        #print(message, '\n')
        response = self.client.chat.completions.create(model="gpt-4o", messages= message)
        return response.choices[0].message.content

    def getHints(self, query : str, qep : dict, edges = list):
        qep_str = json.dumps(qep)
        edges_str = json.dumps(edges)
        message = "".join((
                    f"provide hints for the following query"
                    f"The query is as follows: \n{query}\n"
                    f"The qep is as follows: \n{qep_str}\n"
                    f"The edges are as follows: \n{edges_str}\n"
                ))
        response = self.askLLM(message)
        hints = self.parseResponse(response)
        return response, hints + query
    
    def parseResponse(self, response: str):
        hints = response.split(';')
        final_hint = '/* ' 
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

if __name__ == "__main__":
    model = LLM()
    message = 'What is the Capital of India?'
    answer = model.ask_llm(message)
    print(answer)