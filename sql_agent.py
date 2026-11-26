

import os
import pathlib
import sqlite3

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents import create_agent


class SqlAgent:
    
    url = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
    local_path = pathlib.Path("Chinook.db")
            
    def __init__(self) -> None:
        
        self.open_api_key = os.environ["OPENAI_API_KEY"]
        self.model = init_chat_model("gpt-5.4-mini-2026-03-17")
        
        self.con = sqlite3.connect("Chinook.db")
        self.cursor = self.con.cursor()
        self.tools = []
    
    def select_names(self):
        
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in self.cursor.fetchall() if not row[0].startswith("sqllite")]
        
        print("Dialect: sqlite")
        print(f"Available tables: {tables}")
        
        self.cursor.execute("select * from artist limit 5;")
        print(f"Sample output: {self.cursor.fetchall()}")
        self.con.close()
    
    # tools 
    @tool
    def sql_db_list_tables(self) -> str:
        """ Input is an empty string, output is a comma-separated list of tables in the database"""
        
        try:
            self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in self.cursor.fetchall() if not row[0].startswith("sqlite_")]
            return ", ".join(tables)
        finally:
            self.con.close()
    
    
    @tool
    def sql_db_schema(self, table_names: str) -> str:
        """Input to this tool is a comma-separated list of tables, output is the schema and sample rows for those tables.
        Be sure that the tables actually exist by calling sql_db_list_tables first!
        Example Input: table1, table2, table3"""
        
        try:
            self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            valid_tables = {row[0] for row in self.cursor.fetchall() if not row[0].startswith("sqlite_")}
            results = []
            for table in table_names.split(","):
                table = table.strip()
                if table not in valid_tables:
                    results.append(f"Error: table_names {{{table!r}}} not found in database")
                    continue
                self.cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?;", (table,))
                schema_row = self.cursor.fetchone()
                if schema_row:
                    results.append(schema_row[0])
                    try:
                        quoted_table = '"' + table.replace('"', '""') + '"'
                        self.cursor.execute(f"SELECT * FROM {quoted_table} LIMIT 3;")
                        rows = self.cursor.fetchall()
                        if rows:
                            col_names = [description[0] for description in self.cursor.description]
                            results.append(
                                f"/*\n3 rows from {table} table:\n"
                                + "\t".join(col_names)
                                + "\n"
                                + "\n".join("\t".join(str(x) for x in row) for row in rows)
                                + "\n*/"
                            )
                    except Exception as e:
                        results.append(f"Error fetching sample rows: {e}")
            return "\n\n".join(results)
        finally:
            self.con.close()
     
     
    @tool        
    def sql_db_query(self, query: str) -> str:
        """Input to this tool is a detailed and correct SQL query, output is a result from the database.
        If the query is not correct, an error message will be returned.
        If an error is returned, rewrite the query, check the query, and try again.
        If you encounter an issue with Unknown column 'xxxx' in 'field list', use sql_db_schema to query the correct table fields."""
        
        try:
            self.cursor.execute(query)
            res = self.cursor.fetchall()
            return str(res)
        except Exception as e:
            return f"error: {e}"
        finally:
            self.con.close()
        
    @tool
    def sql_db_query_checker(self, query: str) -> str:
        """Use this tool to double check if your query is correct before executing it.
        Always use this tool before executing a query with sql_db_query!"""
        trigger_prompt = f"""{query}
        Double check the sqlite query above for common mistakes, including:
        - Using NOT IN with NULL values
        - Using UNION when UNION ALL should have been used
        - Using BETWEEN for exclusive ranges
        - Data type mismatch in predicates
        - Properly quoting identifiers
        - Using the correct number of arguments for functions
        - Casting to the correct data type
        - Using the proper columns for joins

        If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.

        Output the final SQL query only.

        SQL Query: """

        response = self.model.invoke(trigger_prompt)
        return response.text.strip()
            

    def print_all_tools(self):
        self.tools = [self.sql_db_list_tables, self.sql_db_schema, 
                 self.sql_db_query, self.sql_db_query_checker]
        for t in self.tools:
            print (f"{t.name}: {t.description}\n")
    
    
    def create_sql_agent(self):
        
        system_prompt = """
        You are an agent designed to interact with a SQL database.
        Given an input question, create a syntactically correct {dialect} query to run,
        then look at the results of the query and return the answer. Unless the user
        specifies a specific number of examples they wish to obtain, always limit your
        query to at most {top_k} results.

        You can order the results by a relevant column to return the most interesting
        examples in the database. Never query for all the columns from a specific table,
        only ask for the relevant columns given the question.

        You MUST double check your query before executing it. If you get an error while
        executing a query, rewrite the query and try again.

        DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
        database.

        To start you should ALWAYS look at the tables in the database to see what you
        can query. Do NOT skip this step.

        Then you should query the schema of the most relevant tables.
        """.format(
            dialect="sqlite",
            top_k=5,
        )
        
        sql_agent = create_agent(
            self.model, self.tools, system_prompt=system_prompt
        )
        
        return sql_agent
    
    def run_sql_agent(self):

        sql_agent = self.create_sql_agent()
        question = "Which genre on average has the longest tracks?"

        result = sql_agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            verbose=True
        )

        return result
        


sql_agent_example = SqlAgent()
sql_agent_example.run_sql_agent()