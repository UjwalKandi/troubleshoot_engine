import os
import sqlite3
import PyPDF2
import csv
import tkinter as tk
from tkinter import scrolledtext, filedialog
import requests
import json
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import numpy as np
from scipy.spatial.distance import cosine
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
import time
from requests.exceptions import RequestException

class LogDataAgent:
    def __init__(self, log_db: str):
        self.conn = sqlite3.connect(log_db)
        self.cursor = self.conn.cursor()
        self.create_table()
        self.check_and_update_schema()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def create_table(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS log_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                content TEXT,
                embedding BLOB
            )
        ''')
        self.conn.commit()

    def check_and_update_schema(self):
        self.cursor.execute("PRAGMA table_info(log_data)")
        columns = [column[1] for column in self.cursor.fetchall()]
        if 'embedding' not in columns:
            self.cursor.execute("ALTER TABLE log_data ADD COLUMN embedding BLOB")
            self.conn.commit()

    def add_entry(self, title: str, content: str):
        embedding = self.model.encode([content])[0]
        embedding_blob = embedding.tobytes()
        self.cursor.execute('INSERT INTO log_data (title, content, embedding) VALUES (?, ?, ?)', 
                            (title, content, embedding_blob))
        self.conn.commit()

    def get_all_entries(self):
        self.cursor.execute('SELECT id, title, content FROM log_data')
        return [{'id': row[0], 'title': row[1], 'content': row[2]} for row in self.cursor.fetchall()]

    def semantic_search(self, query: str, k: int = 5):
        query_vector = self.model.encode([query])[0]
        self.cursor.execute('SELECT id, title, content, embedding FROM log_data')
        results = []
        for row in self.cursor.fetchall():
            id, title, content, embedding_blob = row
            if embedding_blob is not None:
                embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                similarity = 1 - cosine(query_vector, embedding)
                results.append((similarity, {'id': id, 'title': title, 'content': content}))
            else:
                print(f"Warning: Entry '{title}' has no embedding. Skipping in semantic search.")
        results.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in results[:k]]

    def update_embeddings(self):
        self.cursor.execute('SELECT id, content FROM log_data WHERE embedding IS NULL')
        for id, content in self.cursor.fetchall():
            embedding = self.model.encode([content])[0]
            embedding_blob = embedding.tobytes()
            self.cursor.execute('UPDATE log_data SET embedding = ? WHERE id = ?', (embedding_blob, id))
        self.conn.commit()
            
    def extract_file_content(self, file_path):
        _, file_extension = os.path.splitext(file_path)
        if file_extension.lower() == '.pdf':
            return self.extract_pdf_content(file_path)
        elif file_extension.lower() == '.csv':
            return self.extract_csv_content(file_path)
        else:
            return "Unsupported file type"

    def extract_pdf_content(self, file_path):
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            return ' '.join(page.extract_text() for page in reader.pages)

    def extract_csv_content(self, file_path):
        try:
            df = pd.read_csv(file_path)
            return df.to_json(orient='split', date_format='iso')
        except Exception as e:
            return f"Error reading CSV: {str(e)}"

    def analyze_file(self, file_name):
        self.cursor.execute('SELECT content FROM log_data WHERE title = ?', (file_name,))
        result = self.cursor.fetchone()
        if result:
            content = result[0]
            file_extension = os.path.splitext(file_name)[1].lower()

            if file_extension == '.csv':
                return self.analyze_csv(file_name, content)
            elif file_extension == '.pdf':
                return {"content": content}
            else:
                return f"Unsupported file type: {file_extension}"
        else:
            return f"File not found: {file_name}"

    def analyze_csv(self, file_name, content):
        try:
            df = pd.read_json(StringIO(content), orient='split')
            
            if pd.api.types.is_datetime64_any_dtype(df.index):
                df.index = pd.to_datetime(df.index)
            else:
                df.index = range(len(df))
            
            stats = df.describe(include='all').to_dict()
            
            plt.figure(figsize=(12, 6))
            for column in df.select_dtypes(include=[np.number]).columns:
                plt.plot(df.index, df[column], label=column)
            plt.title(f"Time Series Plot: {file_name}")
            plt.xlabel("Index")
            plt.ylabel("Value")
            plt.legend()
            plot_path = f"{file_name}_plot.png"
            plt.savefig(plot_path)
            plt.close()
            
            return {
                "statistics": stats,
                "plot_path": plot_path,
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.to_dict()
            }
        except Exception as e:
            return f"Error analyzing CSV: {str(e)}"

    def add_file(self, file_path):
        file_name = os.path.basename(file_path)
        content = self.extract_file_content(file_path)
        if isinstance(content, str) and content.startswith("Error"):
            return content
        self.add_entry(file_name, content)
        return f"File added: {file_name}"

class QueryAgent:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"
        self.max_retries = 3
        self.timeout = 120

    def query_ollama(self, prompt: str) -> str:
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(self.api_url, json=data, timeout=self.timeout)
                if response.status_code == 200:
                    return response.json()['response']
                else:
                    return f"Error: HTTP {response.status_code}\nResponse: {response.text}"
            except requests.exceptions.Timeout:
                if attempt < self.max_retries - 1:
                    print(f"Request timed out. Retrying... (Attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(2)
                else:
                    return "Error: Request to Ollama API timed out after multiple attempts."
            except requests.exceptions.ConnectionError:
                return "Error: Unable to connect to Ollama API. Please ensure Ollama is running."
            except Exception as e:
                return f"Error: An unexpected error occurred: {str(e)}"

        return "Error: Failed to get a response from Ollama after multiple attempts."

class UIAgent:
    def __init__(self, master):
        self.master = master
        master.title("Agentic AI Log Troubleshooting System (AALTS)")

        self.text_area = scrolledtext.ScrolledText(master, wrap=tk.WORD, width=60, height=20)
        self.text_area.pack(padx=10, pady=10)

        self.entry = tk.Entry(master, width=50)
        self.entry.pack(padx=10, pady=5)

        self.submit_button = tk.Button(master, text="Submit Query")
        self.submit_button.pack(pady=5)

        self.file_entry = tk.Entry(master, width=50)
        self.file_entry.pack(padx=10, pady=5)
        self.file_entry.insert(0, "Enter log file path to analyze")

        self.analyze_button = tk.Button(master, text="Analyze Log File")
        self.analyze_button.pack(pady=5)

    def update_text_area(self, text: str):
        self.text_area.insert(tk.END, text + "\n")
        self.text_area.see(tk.END)

class AgenticAILogTroubleshootingSystem:
    def __init__(self, db_name: str, model_name: str):
        self.log_agent = LogDataAgent(db_name)
        self.log_agent.update_embeddings()
        self.query_agent = QueryAgent(model_name)
        self.root = tk.Tk()
        self.ui_agent = UIAgent(self.root)
        self.ui_agent.submit_button.config(command=self.process_query)
        self.ui_agent.analyze_button.config(command=self.analyze_file)

    def analyze_file(self):
        file_path = self.ui_agent.file_entry.get()
        if file_path:
            if os.path.exists(file_path):
                result = self.log_agent.add_file(file_path)
                self.ui_agent.update_text_area(result)
                
                file_name = os.path.basename(file_path)
                analysis_result = self.log_agent.analyze_file(file_name)
                
                if isinstance(analysis_result, dict):
                    if 'statistics' in analysis_result:
                        # CSV file
                        self.ui_agent.update_text_area(f"Log Analysis for {file_name}:")
                        self.ui_agent.update_text_area(f"Columns: {', '.join(analysis_result['columns'])}")
                        self.ui_agent.update_text_area("Data types:")
                        for col, dtype in analysis_result['dtypes'].items():
                            self.ui_agent.update_text_area(f"  {col}: {dtype}")
                        self.ui_agent.update_text_area(f"Statistics:\n{json.dumps(analysis_result['statistics'], indent=2)}")
                        self.ui_agent.update_text_area(f"Plot saved as: {analysis_result['plot_path']}")
                    else:
                        # PDF file
                        self.ui_agent.update_text_area(f"Log content extracted from {file_name}. You can now query this log data.")
                else:
                    self.ui_agent.update_text_area(f"Log analysis failed: {analysis_result}")
            else:
                self.ui_agent.update_text_area("File not found. Please enter a valid log file path.")
        else:
            self.ui_agent.update_text_area("Please enter a log file path to analyze.")

    def process_query(self):
        query = self.ui_agent.entry.get()
        self.ui_agent.update_text_area(f"User: {query}")

        search_results = self.log_agent.semantic_search(query, k=3)
        context = "\n".join([f"{result['title']}: {result['content']}" for result in search_results])
        prompt = f"Log Context:\n{context}\n\nTroubleshooting Query: {query}\n\nResponse:"
        response = self.query_agent.query_ollama(prompt)

        if response.startswith("Error:"):
            self.ui_agent.update_text_area(f"AALTS: Sorry, I encountered an error: {response}")
        else:
            self.ui_agent.update_text_area(f"AALTS: {response}")
            self.log_agent.add_entry(query, response)

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = AgenticAILogTroubleshootingSystem("log_data.db", "llama3")
    app.run()