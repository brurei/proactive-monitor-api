import csv
import json
import os
from datetime import datetime
import random

import requests
import tiktoken
import torch
from flasgger import Swagger, swag_from

from flask import Flask, render_template_string, request, jsonify, send_file, session, redirect, url_for
import math
from fpdf import FPDF
import pandas as pd
from matplotlib import pyplot as plt
#from sentence_transformers import SentenceTransformer

app = Flask(__name__)

class LogAI:
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self.api_key = os.getenv('OPENAI_API_KEY', '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')
        self.data = None
        self.contexts = []
        self.embeddings = []
       # self.model = SentenceTransformer('all-mpnet-base-v2')  # Usando modelo mais assertivo para embeddings
        self.max_tokens = 4096  # Máximo permitido pelo modelo (ajustado para contexto seguro)

    def load_logs(self):
        """Load and preprocess logs from the file with optimized performance."""
        # Carregar o arquivo em chunks para evitar alto consumo de memória
        chunks = pd.read_json(self.log_file_path, lines=True, chunksize=1000)

        processed_data = []
        for chunk in chunks:
            # Certificar-se de que os dados são strings antes de concatenar
            chunk['protocolo'] = chunk['protocolo'].astype(str)
            chunk['endpoint'] = chunk['endpoint'].astype(str)
            chunk['response_message'] = chunk['response_message'].astype(str)
            chunk['request_data'] = chunk['request_data'].apply(
                lambda x: ', '.join([f"{key}: {value}" for key, value in x.items()])
                if isinstance(x, dict) else ""
            )
            chunk['timestamps'] = chunk['timestamps'].apply(
                lambda x: ', '.join([f"{key}: {value}" for key, value in x.items()])
                if isinstance(x, dict) else ""
            )
            chunk['validation_result'] = chunk['validation_result'].apply(
                lambda x: x['status'] if isinstance(x, dict) and 'status' in x else ""
            )

            # Criar a coluna 'context' com as strings formatadas
            chunk['context'] = (
                    "Protocolo: " + chunk['protocolo'] +
                    "\nEndpoint: " + chunk['endpoint'] +
                    "\nMensagem: " + chunk['response_message'] +
                    "\nDados da requisição: " + chunk['request_data'] +
                    "\nTimestamps: " + chunk['timestamps'] +
                    "\nResultado da validação: " + chunk['validation_result']
            )
            processed_data.append(chunk[['context']])

        # Concatenar todos os chunks em um único DataFrame
        self.data = pd.concat(processed_data, ignore_index=True)
        self.contexts = self.data['context'].tolist()

    def generate_embeddings(self, batch_size=64):
        """Generate embeddings for the logs using Sentence Transformers in batches."""
        self.embeddings = []
        for i in range(0, len(self.contexts), batch_size):
            batch_contexts = self.contexts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch_contexts, convert_to_tensor=True)
            self.embeddings.append(batch_embeddings)

        # Concatenar todos os embeddings em um único tensor
        self.embeddings = torch.cat(self.embeddings)
    def calculate_token_count(self, text):
        """Calculate the number of tokens in a given text."""
        encoding = tiktoken.get_encoding("cl100k_base")  # Modelo compatível com GPT-4
        return len(encoding.encode(text))

    def split_contexts(self, contexts, max_context_tokens):
        """Split contexts into chunks that fit within the token limit."""
        chunks = []
        current_chunk = []
        current_tokens = 0

        for context in contexts:
            context_tokens = self.calculate_token_count(context)
            if current_tokens + context_tokens <= max_context_tokens:
                current_chunk.append(context)
                current_tokens += context_tokens
            else:
                chunks.append(current_chunk)
                current_chunk = [context]
                current_tokens = context_tokens

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def perguntar_openai(self, pergunta, contexto):
        """Ask OpenAI GPT-4 using the relevant contexts."""
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }

        examples = "Exemplos: Perguntas sobre protocolos, validações, ou endpoints."
        prompt = f"""
        Você é um assistente que responde perguntas com base nos logs fornecidos. Seja assertivo e detalhado.
        {examples}

        Aqui estão os logs relevantes (truncados para evitar excesso de tokens):
        {'\n'.join(contexto)}

        Pergunta: {pergunta}
        Resposta:"""

        data = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "Você é um especialista em análise de logs."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1000,
            "temperature": 0.3
        }

        response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data)

        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            raise Exception(f"Erro na requisição para o GPT-4: {response.text}")

    def process_full_logs(self, pergunta):
        """Process the entire log file by splitting it into manageable chunks."""
        max_context_tokens = self.max_tokens - 1000  # Reservar espaço para a pergunta e a resposta
        chunks = self.split_contexts(self.contexts, max_context_tokens)

        responses = []
        for chunk in chunks:
            try:
                response = self.perguntar_openai(pergunta, chunk)
                responses.append(response)
            except Exception as e:
                responses.append(f"Erro ao processar chunk: {e}")

        # Consolidar respostas
        consolidated_response = "\n\n".join(responses)
        return consolidated_response

#log_ai = LogAI('falhas_reembolso_errors2.json')
#log_ai.load_logs()
#log_ai.generate_embeddings()
# Credenciais padrão
DEFAULT_USERNAME = 'admin'
DEFAULT_PASSWORD = 'admin'
app.secret_key = 'sua_chave_secreta'  # Necessário para gerenciar a sessão
swagger = Swagger(app)
class SwaggerDocumentation:

    @staticmethod
    def summary_data_doc():
        return {
            'parameters': [
                {
                    'name': 'start_date',
                    'in': 'formData',
                    'type': 'string',
                    'required': True,
                    'description': 'Data inicial (YYYY-MM-DD)'
                },
                {
                    'name': 'end_date',
                    'in': 'formData',
                    'type': 'string',
                    'required': True,
                    'description': 'Data final (YYYY-MM-DD)'
                },
                {
                    'name': 'status',
                    'in': 'formData',
                    'type': 'string',
                    'required': False,
                    'description': 'Filtro de status (sucesso/falha)'
                }
            ],
            'responses': {
                200: {
                    'description': 'Resumo das solicitações',
                    'schema': {
                        'type': 'object',
                        'properties': {
                            'total_solicitacoes': {'type': 'integer'},
                            'total_sucessos': {'type': 'integer'},
                            'total_falhas': {'type': 'integer'},
                            'valor_total': {'type': 'number'},
                            'valor_total_falhas': {'type': 'number'}
                        }
                    }
                }
            }
        }

    @staticmethod
    def export_csv_doc():
        return {
            'parameters': [
                {
                    'name': 'start_date',
                    'in': 'formData',
                    'type': 'string',
                    'required': True,
                    'description': 'Data inicial (YYYY-MM-DD)'
                },
                {
                    'name': 'end_date',
                    'in': 'formData',
                    'type': 'string',
                    'required': True,
                    'description': 'Data final (YYYY-MM-DD)'
                },
                {
                    'name': 'status',
                    'in': 'formData',
                    'type': 'string',
                    'required': False,
                    'description': 'Filtro de status (sucesso/falha)'
                }
            ],
            'responses': {
                200: {
                    'description': 'Arquivo CSV gerado com sucesso',
                    'content': {
                        'text/csv': {
                            'schema': {
                                'type': 'string',
                                'format': 'binary'
                            }
                        }
                    }
                }
            }
        }
    @staticmethod
    def chart_data_doc():
        return {
            'responses': {
                200: {
                    'description': 'Dados para gráfico de pizza',
                    'schema': {
                        'type': 'object',
                        'properties': {
                            'labels': {
                                'type': 'array',
                                'items': {
                                    'type': 'string'
                                }
                            },
                            'data': {
                                'type': 'array',
                                'items': {
                                    'type': 'integer'
                                }
                            }
                        }
                    }
                }
            }
        }
    @staticmethod
    def load_log_data_doc():
        return {
            'responses': {
                200: {
                    'description': 'Lista de logs carregados',
                    'schema': {
                        'type': 'array',
                        'items': {
                            'type': 'object',
                            'properties': {
                                'protocolo': {'type': 'string'},
                                'nome_completo': {'type': 'string'},
                                'cpf_cnpj_prestador': {'type': 'string'},
                                'carteirinha': {'type': 'string'},
                                'email': {'type': 'string'},
                                'razao_social_prestador': {'type': 'string'},
                                'valor_apresentado': {'type': 'string'},
                                'data_atendimento': {'type': 'string'},
                                'created_at': {'type': 'string'},
                                'validated_at': {'type': 'string'},
                                'status': {'type': 'string'},
                                'status_operacao': {'type': 'string'},
                                'motivo_status_workflow': {'type': 'string'}
                            }
                        }
                    }
                }
            }
        }

    @staticmethod
    def search_doc():
        return {
            'parameters': [
                {
                    'name': 'query',
                    'in': 'query',
                    'type': 'string',
                    'required': True,
                    'description': 'Termo de busca (protocolo, nome ou CPF/CNPJ)'
                }
            ],
            'responses': {
                200: {
                    'description': 'Resultado da busca',
                    'schema': {
                        'type': 'array',
                        'items': {
                            'type': 'object',
                            'properties': {
                                'protocolo': {'type': 'string'},
                                'nome_completo': {'type': 'string'},
                                'cpf_cnpj_prestador': {'type': 'string'},
                                'carteirinha': {'type': 'string'},
                                'email': {'type': 'string'},
                                'razao_social_prestador': {'type': 'string'},
                                'valor_apresentado': {'type': 'string'},
                                'data_atendimento': {'type': 'string'},
                                'created_at': {'type': 'string'},
                                'validated_at': {'type': 'string'},
                                'status': {'type': 'string'},
                                'status_operacao': {'type': 'string'},
                                'motivo_status_workflow': {'type': 'string'}
                            }
                        }
                    }
                }
            }
        }

    @staticmethod
    def load_bar_chart_data_doc():
        return {
            'responses': {
                200: {
                    'description': 'Dados para gráfico de barras',
                    'schema': {
                        'type': 'object',
                        'properties': {
                            'status': {
                                'type': 'object',
                                'additionalProperties': {
                                    'type': 'integer'
                                }
                            }
                        }
                    }
                },
                500: {
                    'description': 'Erro ao carregar dados'
                }
            }
        }
    @staticmethod
    def generate_report_doc():
        return {
            'parameters': [
                {
                    'name': 'start_date',
                    'in': 'formData',
                    'type': 'string',
                    'required': True,
                    'description': 'Data inicial (YYYY-MM-DD)'
                },
                {
                    'name': 'end_date',
                    'in': 'formData',
                    'type': 'string',
                    'required': True,
                    'description': 'Data final (YYYY-MM-DD)'
                },
                {
                    'name': 'status',
                    'in': 'formData',
                    'type': 'string',
                    'required': False,
                    'description': 'Filtro de status (sucesso/falha)'
                }
            ],
            'responses': {
                200: {
                    'description': 'Relatório PDF gerado com sucesso',
                    'content': {
                        'application/pdf': {
                            'schema': {
                                'type': 'string',
                                'format': 'binary'
                            }
                        }
                    }
                }
            }
        }

@app.route('/summary_data', methods=['POST'])
@swag_from(SwaggerDocumentation.summary_data_doc())
def summary_data():
    start_date = request.form.get("start_date")
    end_date = request.form.get("end_date")
    status_filter = request.form.get("status")

    log_data = load_log_data()
    total_solicitacoes = len(log_data)
    total_sucessos = sum(1 for item in log_data if item["status"] == "sucesso")
    total_falhas = total_solicitacoes - total_sucessos
    valor_total = sum(float(item.get("valor_apresentado", 0)) for item in log_data)
    valor_total_falhas = sum(float(item.get("valor_apresentado", 0)) for item in log_data if item["status"] != "sucesso")

    return jsonify({
        'total_solicitacoes': total_solicitacoes,
        'total_sucessos': total_sucessos,
        'total_falhas': total_falhas,
        'valor_total': valor_total,
        'valor_total_falhas': valor_total_falhas
    })

@app.route('/load_logs', methods=['GET'])
@swag_from(SwaggerDocumentation.load_log_data_doc())
def load_logs():
    log_data = []
    try:
        with open('falhas_reembolso_errors.json', 'r') as file:
            for line in file:
                log_entry = json.loads(line)
                item = {
                    "protocolo": log_entry.get("protocolo", "Não disponível"),
                    "nome_completo": log_entry["request_data"].get("nome_completo", "Não disponível"),
                    "cpf_cnpj_prestador": log_entry["request_data"].get("cpf_cnpj_prestador", "Não disponível"),
                    "carteirinha": log_entry["request_data"].get("carteirinha", "Não disponível"),
                    "email": log_entry["request_data"].get("email", "Não disponível"),
                    "razao_social_prestador": log_entry["request_data"].get("razao_social_prestador", "Não disponível"),
                    "valor_apresentado": log_entry["request_data"].get("valor_apresentado", "Não disponível"),
                    "data_atendimento": log_entry["request_data"].get("data_atendimento", "Não disponível"),
                    "created_at": log_entry["timestamps"].get("created_at", "Não disponível"),
                    "validated_at": log_entry["timestamps"].get("validated_at", "Não disponível"),
                    "status": log_entry["validation_result"].get("status", "Não disponível")
                }
                log_data.append(item)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify(log_data)



@app.route('/login', methods=['GET', 'POST'])
def login():
    error_message = ""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Validação com usuário padrão
        if username == DEFAULT_USERNAME and password == DEFAULT_PASSWORD:
            session['logged_in'] = True
            return redirect(url_for('index'))
        else:
            error_message = "<p style='color: red; text-align: center;'>Credenciais inválidas. Tente novamente.</p>"

    return f"""
    <!DOCTYPE html>
    <html lang='en'>
    <head>
    
        <meta charset='UTF-8'>
        <meta name='viewport' content='width=device-width, initial-scale=1.0'>
        <title>Login</title>
        <style>
            body {{ font-family: Arial, sans-serif; background-color: #f4f4f4; margin: 0; padding: 0; }}
            .navbar {{ background-color: #002060; color: white; padding: 10px; text-align: center; }}
            .container {{ width: 400px; margin: 100px auto; background: white; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); padding: 30px; }}
            form {{ display: flex; flex-direction: column; align-items: center; gap: 15px; }}
            input {{ width: 100%; padding: 12px; border-radius: 5px; border: 1px solid #ddd; box-sizing: border-box; }}
            button {{ width: 100%; padding: 12px; background: linear-gradient(45deg, #007BFF, #00CFFF); color: white; border: none; border-radius: 5px; cursor: pointer; }}
            button:hover {{ background: linear-gradient(45deg, #0056b3, #007BFF); }}
        </style>
    </head>
    <body>
    
       <header>
        <img src="https://saudepetrobras.com.br/data/files/5F/D3/80/3D/FDFEB7108831CAB7004CF9C2/logo_desktop.svg" alt="Logo">
        <center><h2>GPROTREE - Gestor de Protocolos de Reembolso</h2></center>
    </header>
    
        <div class="container">
        
            <h2 style="text-align: center;">Login</h2>
            {error_message}
            <form method="post">
                <input type="text" name="username" placeholder="Usuário" required>
                <input type="password" name="password" placeholder="Senha" required>
                <button type="submit">Entrar</button>
            </form>
        </div>
    </body>
    </html>
    """

# Modifique as funções de busca e paginação para usar esses dados
@app.route('/')
def index():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    page = int(request.args.get('page', 1))
    per_page = 100
    start = (page - 1) * per_page
    end = start + per_page

    status_filter = request.args.get('status_filter', '')
    filtered_data = [item for item in log_data if status_filter in ('', item['status_operacao'])]

    total_pages = math.ceil(len(filtered_data) / per_page)
    paginated_data = filtered_data[start:end]

    return render_template_string(template, data=paginated_data, page=page, total_pages=total_pages, status_filter=status_filter, status_colors=status_colors)

@app.route('/search', methods=['GET'])
@swag_from(SwaggerDocumentation.search_doc())
def search():
    query = request.args.get('query', '').lower().strip()
    if not query:
        return jsonify([])  # Retorna vazio se a query estiver vazia

    filtered_data = [
        {
            "protocolo": item["protocolo"],
            "nome_completo": item.get("nome_completo", "Não disponível"),
            "cpf_cnpj_prestador": item.get("cpf_cnpj_prestador", "Não disponível"),
            "carteirinha": item.get("carteirinha", "Não disponível"),
            "email": item.get("email", "Não disponível"),
            "razao_social_prestador": item.get("razao_social_prestador", "Não disponível"),
            "valor_apresentado": item.get("valor_apresentado", "Não disponível"),
            "data_atendimento": item.get("data_atendimento", "Não disponível"),
            "created_at": item.get("created_at", "Não disponível"),
            "validated_at": item.get("validated_at", "Não disponível"),
            "status": item.get("status", "Não disponível"),
            "status_operacao": item.get("status_operacao", "Análise Não Iniciada"),
            "motivo_status_workflow": item.get("motivo_status_workflow", "Não disponível"),
        }
        for item in log_data
        if query in str(item["protocolo"]).lower() or query in item.get("nome_completo", "").lower() or query in item.get("cpf_cnpj_prestador", "").lower()
    ]
    return jsonify(filtered_data)

# Template HTML
template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <meta charset="UTF-8">
    <title>Gestor de Protocolos de Reembolso</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins&display=swap" rel="stylesheet">
   <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 0; background-color: #f4f4f9; }
        .chat-container { 
            max-width: 600px; 
            margin: 0; 
            background: #ffffff; 
            border-radius: 8px; 
            box-shadow: 0 0 10px rgba(0,0,0,0.1); 
            display: none; 
            flex-direction: column; 
            position: fixed; 
            bottom: 80px; 
            right: 20px; 
            width: 350px; 
            z-index: 1000; 
        }
        .chat-header { 
            background-color: #006c67; 
            color: #ffffff; 
            padding: 15px; 
            text-align: center; 
            font-size: 18px; 
            font-weight: bold; 
        }
        .chat-box { 
            padding: 15px; 
            flex: 1; 
            overflow-y: auto; 
            display: flex; 
            flex-direction: column; 
            gap: 10px; 
            max-height: 400px;
        }
        .chat-box .message { display: flex; align-items: flex-start; }
        .chat-box .message.user { justify-content: flex-end; }
        .chat-box .message.user .bubble { background-color: #006c67; color: #ffffff; }
        .chat-box .message.bot .bubble { background-color: #e8f1f2; color: #000000; }
        .chat-box .bubble { max-width: 70%; padding: 10px; border-radius: 15px; font-size: 14px; line-height: 1.5; }
        .input-container { display: flex; padding: 10px; background-color: #f0f0f0; border-top: 1px solid #ddd; }
        .input-container input { flex: 1; padding: 10px; border: 1px solid #ccc; border-radius: 20px; font-size: 14px; }
        .input-container button { margin-left: 10px; padding: 10px 20px; border: none; background-color: #006c67; color: white; border-radius: 20px; cursor: pointer; font-size: 14px; }
        .chat-toggle { 
            position: fixed; 
            bottom: 20px; 
            right: 20px; 
            width: 60px; 
            height: 60px; 
            background-color: #006c67; 
            border-radius: 50%; 
            display: flex; 
            justify-content: center; 
            align-items: center; 
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); 
            cursor: pointer; 
            z-index: 1000; 
        }
        .chat-toggle img { width: 30px; height: 30px; }
        .loading { color: #006c67; font-size: 14px; font-style: italic; margin-top: 10px; }
    </style>
   
    <style>
        .charts-container {
            display: flex;
            justify-content: space-around;
            align-items: flex-start;
            margin-top: 20px;
        }
        .chart-box {
            width: 400px;
            height: 300px;
        }
        .pizza-chart-box {
            width: 250px;
            height: 250px;
        }
    </style>
    <style>
    
        body {
            font-family: 'Poppins', sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f0f2f5;
        }
        header {
            background-color: #004f3d;
            padding: 20px;
            color: white;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        header img {
            height: 60px;
        }
        header h1 {
            font-size: 1.8rem;
        }
        .container {
            margin: 20px auto;
            max-width: 1000px;
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 6px 18px rgba(0, 0, 0, 0.1);
        }
        .search-bar input, .search-bar button, select, input[type="date"] {
            padding: 12px;
            border-radius: 8px;
            border: 1px solid #ccc;
            margin-right: 10px;
        }
        .search-bar2 input,select2 {
            padding: 12px;
            border-radius: 8px;
            border: 1px solid #ccc;
            margin-right: 10px;
            width:100%;
        }
        .search-bar button {
            background-color: #007BFF;
            color: white;
            border: none;
            cursor: pointer;
        }
        .search-bar button:hover {
            background-color: #0056b3;
        }
        
        .search-bar2 button {
            background-color: #007BFF;
            color: white;
            border: none;
            cursor: pointer;
            width: 100%
        }
        .search-bar2 button:hover {
            background-color: #0056b3;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        table th, table td {
            padding: 15px;
            border-bottom: 1px solid #ddd;
            text-align: left;
        }
        table th {
            background-color: #004f3d;
            color: white;
        }
        .status-success {
            color: green;
            font-weight: bold;
        }
        .status-failure {
            color: red;
            font-weight: bold;
        }
        button {
            padding: 10px 15px;
            background-color: #28a745;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
        }
        button:hover {
            background-color: #218838;
        }
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.5);
            justify-content: center;
            align-items: center;
        }
        .modal-content {
            background: white;
            padding: 25px;
            border-radius: 12px;
            max-width: 600px;
        }
        .close {
            float: right;
            font-size: 1.8rem;
            cursor: pointer;
        }
        #loading {
            display: none;
            text-align: center;
            color: #007BFF;
            font-weight: bold;
            margin-top: 10px;
        }
    </style>
</head>
    <script>
        function search() {
            const query = document.getElementById('search-query').value;
            fetch(`/search?query=${encodeURIComponent(query)}`)
                .then(response => response.json())
                .then(data => {
                    const tableBody = document.querySelector('table tbody');
                    tableBody.innerHTML = data.map(item => `
                        <tr>
                            <td>${item.protocolo}</td>
                            <td>${item.nome_completo}</td>
                            <td>${item.cpf_cnpj_prestador}</td>
                            <td>${item.valor_apresentado}</td>
                                                        <td class="${item.status === 'sucesso' ? 'status-success' : 'status-failure'}">${item.status}</td>

                            <td>${item.status_operacao}</td>
                                                        <td>${item.motivo_status_workflow}</td>

                            <td><button onclick="openModal('${item.protocolo}')">Ver Detalhes</button></td>
                        </tr>
                    `).join('');
                });
        }

        function openModal(protocolo) {
            const modal = document.getElementById('modal');
            const modalContent = document.getElementById('modal-content');

            fetch(`/search?query=${protocolo}`)
                .then(response => response.json())
                .then(data => {
                    if (data.length > 0) {
                        const item = data[0];
                        modalContent.innerHTML = `
                            <span class="close" onclick="closeModal()">&times;</span>
                            <h3>Detalhes do Protocolo ${item.protocolo}</h3>
                            <p><strong>Nome Completo:</strong> ${item.nome_completo}</p>
                            <p><strong>CPF/CNPJ:</strong> ${item.cpf_cnpj_prestador}</p>
                            <p><strong>Carteirinha:</strong> ${item.carteirinha}</p>
                            <p><strong>Email:</strong> ${item.email}</p>
                            <p><strong>Razão Social:</strong> ${item.razao_social_prestador || "Não disponível"}</p>
                            <p><strong>Valor Apresentado:</strong> ${item.valor_apresentado}</p>
                            <p><strong>Data Atendimento:</strong> ${item.data_atendimento}</p>
                            <p><strong>Data Hora da Solicitação:</strong> ${item.created_at}</p>
                            <p><strong>Data Hora da Validação:</strong> ${item.validated_at}</p>
                            <p><strong>Status Integração:</strong> <span style="color: ${item.status === 'sucesso' ? 'green' : 'red'};">${item.status}</span></p>
                                                    <p><strong>Status Operação:</strong> ${item.status_operacao || "Análise Não Iniciada"}</p>
                            <p><strong>Motivo:</strong> ${item.motivo_status_workflow}</p>

                        
                        `;
                        modal.style.display = 'flex';
                    }
                });
        }

        function closeModal() {
            const modal = document.getElementById('modal');
            modal.style.display = 'none';
        }
    </script>
</head>
<body>
 <header>
        <img src="https://saudepetrobras.com.br/data/files/5F/D3/80/3D/FDFEB7108831CAB7004CF9C2/logo_desktop.svg" alt="Logo">
        <h1>GPROTREE - Gestor de Protocolos de Reembolso</h1>
        <div class="chat-container" id="chat-container">
        <div class="chat-header">Fale com o ReemBotAi</div>
        <div id="chat" class="chat-box"></div>
        <div id="loading" class="loading" style="display: none;">Gerando resposta, por favor aguarde...</div>
        <div class="input-container">
            <input type="text" id="question" placeholder="Faça sua pergunta..." onkeypress="handleKeyPress(event)">
            <button onclick="askQuestion()">Enviar</button>
        </div>
    </div>

    <div class="chat-toggle" onclick="toggleChat()">
        <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRHaZ9bsbOJ4y5Q-WSbxj2hnRMhLUDwCl3G9w&s" alt="Chat">
    </div>

    <script>
        function toggleChat() {
            const chatContainer = document.getElementById('chat-container');
            chatContainer.style.display = chatContainer.style.display === 'flex' ? 'none' : 'flex';
        }

        async function askQuestion() {
            const questionInput = document.getElementById('question');
            const chatBox = document.getElementById('chat');
            const loadingIndicator = document.getElementById('loading');

            const question = questionInput.value.trim();
            if (!question) return;

            const userMessage = document.createElement('div');
            userMessage.className = 'message user';
            userMessage.innerHTML = `<div class='bubble'>${question}</div>`;
            chatBox.appendChild(userMessage);
            chatBox.scrollTop = chatBox.scrollHeight;

            questionInput.value = '';

            loadingIndicator.style.display = 'block';

            try {
                const response = await fetch('/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question })
                });

                const data = await response.json();
                const botMessage = document.createElement('div');
                botMessage.className = 'message bot';
                botMessage.innerHTML = `<div class='bubble'>${data.response || data.error}</div>`;
                chatBox.appendChild(botMessage);
                chatBox.scrollTop = chatBox.scrollHeight;
            } catch (error) {
                const errorMessage = document.createElement('div');
                errorMessage.className = 'message bot';
                errorMessage.innerHTML = `<div class='bubble'>Erro ao se comunicar com o servidor.</div>`;
                chatBox.appendChild(errorMessage);
                chatBox.scrollTop = chatBox.scrollHeight;
            } finally {
                loadingIndicator.style.display = 'none';
            }
        }

        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                askQuestion();
            }
        }
    </script>
    </header>
<h1>Estatísticas</h1>

<div class="charts-container">
        <div class="pizza-chart-box">
            <canvas id="pizzaChart"></canvas>
        </div>
        <div class="chart-box">
            <canvas id="radarChart" style="display: block; box-sizing: border-box; height: 400px; width: 400px;"></canvas>
        </div>
        <div class="chart-box">
            <canvas id="barChart"></canvas>
        </div>
    </div>
        <div class="charts-container">
                <div class="chart-box">

            <canvas id="visaoGeralChart"></canvas>
            </div>
                            <div class="chart-box">

            <canvas id="eficienciaChart"></canvas>
            </div>
                            
                            <div class="chart-box">

            <canvas id="financeiraChart"></canvas>
            </div>
                            
        </div>
                <div class="charts-container">
                <div class="chart-box">

            <canvas id="riscoChart"></canvas>
            </div>
                            <div class="chart-box">

            <canvas id="reprocessamentoChart"></canvas>
            </div>
            <div class="chart-box">

            <canvas id="qualidadeChart"></canvas>
            </div>
</div>
        <script>
            fetch('/dashboard_graficos_data').then(response => response.json()).then(data => {
                new Chart(document.getElementById('visaoGeralChart').getContext('2d'), data.visaoGeral);
                new Chart(document.getElementById('eficienciaChart').getContext('2d'), data.eficiencia);
                new Chart(document.getElementById('qualidadeChart').getContext('2d'), data.qualidade);
                new Chart(document.getElementById('financeiraChart').getContext('2d'), data.financeira);
                new Chart(document.getElementById('riscoChart').getContext('2d'), data.risco);
                new Chart(document.getElementById('reprocessamentoChart').getContext('2d'), data.reprocessamento);
            });
        </script>
    <script>
        fetch('/bar_chart_data')
            .then(response => response.json())
            .then(data => {
                const ctxBar = document.getElementById('barChart').getContext('2d');
                new Chart(ctxBar, {
                    type: 'bar',
                    data: {
                        labels: data.status_labels,
                        datasets: [{
                            label: 'Total por Tipo de Solicitação',
                            data: data.status_data,
                            backgroundColor: 'rgba(75, 192, 192, 0.2)',
                            borderColor: 'rgba(75, 192, 192, 1)',
                            borderWidth: 1
                        }]
                    },
                    options: {
                        scales: {
                            y: {
                                beginAtZero: true
                            }
                        }
                    }
                });
            });
    </script>
    <script>
        fetch('/chart_data_spider')
            .then(response => response.json())
            .then(data => {
                const ctxRadar = document.getElementById('radarChart').getContext('2d');
                new Chart(ctxRadar, {
                    type: 'radar',
                    data: {
                        labels: data.labels,
                        datasets: [{
                            label: 'Métricas de Desempenho',
                            data: data.data,
                            backgroundColor: 'rgba(54, 162, 235, 0.2)',
                            borderColor: 'rgba(54, 162, 235, 1)',
                            pointBackgroundColor: 'rgba(54, 162, 235, 1)'
                        }]
                    },
                    options: {
                        scale: {
                            ticks: {
                                beginAtZero: true
                            }
                        }
                    }
                });
            });
    </script>

   
<script>
        fetch('/chart_data')
            .then(response => response.json())
            .then(data => {
                const ctx = document.getElementById('pizzaChart').getContext('2d');
                new Chart(ctx, {
                    type: 'pie',
                    data: {
                        labels: data.labels,
                        datasets: [{
                            data: data.data,
                            backgroundColor: ['#28a745', '#dc3545', '#ffc107']
                        }]
                    }
                });
            });
    </script>
 
        <form method="POST" action="/generate_report">
            <label for="start_date">Data Inicial:</label>
            <input type="date" id="start_date" name="start_date" required>
            <label for="end_date">Data Final:</label>
            <input type="date" id="end_date" name="end_date" required>
            <label for="status">Status:</label>
            <select id="status" name="status">
                <option value="">Todos</option>
                <option value="sucesso">Sucesso</option>
                <option value="falha">Falha</option>
            </select>
            <button type="submit">Gerar Relatório PDF</button>
        </form>

        <form method="POST" action="/export_csv">
            <label for="start_date_excel">Data Inicial:</label>
            <input type="date" id="start_date_excel" name="start_date" required>
            <label for="end_date_excel">Data Final:</label>
            <input type="date" id="end_date_excel" name="end_date" required>
            <label for="status_excel">Status:</label>
            <select id="status_excel" name="status">
                <option value="">Todos</option>
                <option value="sucesso">Sucesso</option>
                <option value="falha">Falha</option>
            </select>
            <button type="submit">Gerar Relatório CSV</button>
        </form>
        </p>
 
        <div class="search-bar2">
            <input type="text" style="width: '100%'" id="search-query" placeholder="Digite ex: protocolo,nome,cpf, data antendimento, carteirinha">
            <button onclick="search()">Pesquisar</button>
        </div>
        <table>
        
            <thead>
                <tr>
                    <th>Protocolo</th>
                    <th>Nome</th>
                    <th>CPF/CNPJ</th>
                    <th>Valor</th>
                    <th>Status Integração:</th>
                    <th>Status Operação</th>
                    <th>Motivo Status Workflow</th>
                    <th>Ações</th>
                </tr>
            </thead>
            <tbody>
                {% for item in data %}
                <tr>
                    <td>{{ item.protocolo }}</td>
                    <td>{{ item.nome_completo }}</td>
                    <td>{{ item.cpf_cnpj_prestador }}</td>
                    <td>{{ item.valor_apresentado }}</td>
                    <td class="{{ 'status-success' if item.status == 'sucesso' else 'status-failure' }}">{{ item.status }}</td>
                   <td>{{ item.status_operacao }}</td>
                    <td>{{ item.motivo_status_workflow }}</td>
                    
                    <td><button onclick="openModal('{{ item.protocolo }}')">Ver Detalhes</button></td>
                </tr>
                {% endfor %}
            </tbody>
        </table>

        <div class="pagination">
            {% for i in range(1, total_pages + 1) %}
                <a href="/?page={{ i }}&status_filter={{ status_filter }}" class="{{ 'active' if i == page else '' }}">{{ i }}</a>
            {% endfor %}
        </div>
    </div>

    <div id="modal" class="modal">
        <div class="modal-content" id="modal-content"></div>
    </div>
     
    <script>
    <h4>Bem Vindo </h4>
            <a href="/logout"> Clique Aqui para Sair</a>
            </p>
            

        <form method="POST" action="/generate_report">
            <label for="start_date">Data Inicial:</label>
            <input type="date" id="start_date" name="start_date" required>
            <label for="end_date">Data Final:</label>
            <input type="date" id="end_date" name="end_date" required>
            <label for="status">Status:</label>
            <select id="status" name="status">
                <option value="">Todos</option>
                <option value="sucesso">Sucesso</option>
                <option value="falha">Falha</option>
            </select>
            <button type="submit">Gerar Relatório PDF</button>
        </form>

        <form method="POST" action="/export_csv">
            <label for="start_date_excel">Data Inicial:</label>
            <input type="date" id="start_date_excel" name="start_date" required>
            <label for="end_date_excel">Data Final:</label>
            <input type="date" id="end_date_excel" name="end_date" required>
            <label for="status_excel">Status:</label>
            <select id="status_excel" name="status">
                <option value="">Todos</option>
                <option value="sucesso">Sucesso</option>
                <option value="falha">Falha</option>
            </select>
            <button type="submit">Gerar Relatório CSV</button>
        </form>
        </p>
 
        <div class="search-bar2">
            <input type="text" style="width: '100%'" id="search-query" placeholder="Digite ex: protocolo,nome,cpf, data antendimento, carteirinha">
            <button onclick="search()">Pesquisar</button>
        </div>
        <table>
        
            <thead>
                <tr>
                    <th>Protocolo</th>
                    <th>Nome</th>
                    <th>CPF/CNPJ</th>
                    <th>Valor</th>
                    <th>Status Integração</th>
                    <th>Status Operação</th>
                    <th>Motivo Status Workflow</th>
                    <th>Ações</th>
                </tr>
            </thead>
            <tbody>
                {% for item in data %}
                <tr>
                    <td>{{ item.protocolo }}</td>
                    <td>{{ item.nome_completo }}</td>
                    <td>{{ item.cpf_cnpj_prestador }}</td>
                    <td>{{ item.valor_apresentado }}</td>
                    <td class="{{ 'status-success' if item.status == 'sucesso' else 'status-failure' }}">{{ item.status }}</td>
                   <td>{{ item.status_operacao }}</td>
                    <td>{{ item.motivo_status_workflow }}</td>
                    
                    <td><button onclick="openModal('{{ item.protocolo }}')">Ver Detalhes</button></td>
                </tr>
                {% endfor %}
            </tbody>
        </table>

        <div class="pagination">
            {% for i in range(1, total_pages + 1) %}
                <a href="/?page={{ i }}&status_filter={{ status_filter }}" class="{{ 'active' if i == page else '' }}">{{ i }}</a>
            {% endfor %}
        </div>
    </div>

    <div id="modal" class="modal">
        <div class="modal-content" id="modal-content"></div>
    </div>
    
</body>
</html>
"""

# Adicionar dropdown de filtro de status no template
filter_dropdown = """
<form method="GET" action="/">
    <label for="status_filter">Filtrar por Status da Operação:</label>
    <select id="status_filter" name="status_filter" onchange="this.form.submit()">
        <option value="">Todos</option>
        {% for status, color in status_colors.items() %}
            <option value="{{ status }}" {% if status == status_filter %}selected{% endif %}>{{ status }}</option>
        {% endfor %}
    </select>
</form>
"""

template = template.replace("<div class=\"search-bar2\">", filter_dropdown + "<div class=\"search-bar2\">")
# Ajustar exibição na tabela principal
template = template.replace("<td>{{ item.status_operacao || 'Análise Não Iniciada'}}</td>", "<td><span class='status-operacao' style='background-color: {{ item.status_operacao | status_color }};'>{{ item.status_operacao }}</span></td>")

# Adicionar estilo para colorir a célula
style = """
<style>
    .status-operacao {
        font-weight: bold;
        padding: 5px 10px;
        border-radius: 5px;
        color: white;
    }
</style>
"""

template = style + template.replace("<td>{{ item.status_operacao }}</td>", "<td><span class='status-operacao' style='background-color: {{ item.status_operacao | status_color }};'>{{ item.status_operacao }}</span></td>")


@app.route('/export_csv', methods=['POST'])
@swag_from(SwaggerDocumentation.export_csv_doc())
def export_csv():
    start_date = request.form.get("start_date")
    end_date = request.form.get("end_date")
    status_filter = request.form.get("status")

    def parse_date(date_str, format):
        try:
            return datetime.strptime(date_str, format).date()
        except (ValueError, TypeError):
            return None

    filtered_data = [
        item for item in log_data
        if (
            not start_date or (
                item.get("created_at") != "Não disponível" and
                parse_date(item["created_at"], "%d/%m/%Y %H:%M:%S") and
                parse_date(item["created_at"], "%d/%m/%Y %H:%M:%S") >= parse_date(start_date, "%Y-%m-%d")
            )
        )
        and (
            not end_date or (
                item.get("created_at") != "Não disponível" and
                parse_date(item["created_at"], "%d/%m/%Y %H:%M:%S") and
                parse_date(item["created_at"], "%d/%m/%Y %H:%M:%S") <= parse_date(end_date, "%Y-%m-%d")
            )
        )
        and (not status_filter or item["status"] == status_filter)
    ]

    csv_file_path = "relatorio_reembolsos.csv"
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        csvwriter.writerow([
            "Protocolo",
            "Nome Completo",
            "CPF/CNPJ",
            "Carteirinha",
            "Email",
            "Razão Social",
            "Valor Apresentado",
            "Data Atendimento",
            "Data Criação",
            "Data Validação",
            "Status",
            "Status Operacao",
            "Status Workflow"
        ])

        for item in filtered_data:
            csvwriter.writerow([
                item.get("protocolo", "Não disponível"),
                item.get("nome_completo", "Não disponível"),
                item.get("cpf_cnpj_prestador", "Não disponível"),
                item.get("carteirinha", "Não disponível"),
                item.get("email", "Não disponível"),
                item.get("razao_social_prestador", "Não disponível"),
                item.get("valor_apresentado", "Não disponível"),
                item.get("data_atendimento", "Não disponível"),
                item.get("created_at", "Não disponível"),
                item.get("validated_at", "Não disponível"),
                item.get("status", "Não disponível"),
                item.get("status_operacao", "Não disponível"),
                item.get("motivo_status_workflow", "Não disponível")
            ])

    return send_file(csv_file_path, as_attachment=True)


# Função para carregar os dados do log
def load_log_data():

    log_data = []
    try:
        with open('falhas_reembolso_errors.json', 'r') as file:
            for line in file:
                log_entry = json.loads(line)
                item = {
                    "protocolo": log_entry.get("protocolo", "Não disponível"),
                    "nome_completo": log_entry["request_data"].get("nome_completo", "Não disponível"),
                    "cpf_cnpj_prestador": log_entry["request_data"].get("cpf_cnpj_prestador", "Não disponível"),
                    "carteirinha": log_entry["request_data"].get("carteirinha", "Não disponível"),
                    "email": log_entry["request_data"].get("email", "Não disponível"),
                    "razao_social_prestador": log_entry["request_data"].get("razao_social_prestador", "Não disponível"),
                    "valor_apresentado": log_entry["request_data"].get("valor_apresentado", "Não disponível"),
                    "data_atendimento": log_entry["request_data"].get("data_atendimento", "Não disponível"),
                    "created_at": log_entry["timestamps"].get("created_at", "Não disponível"),
                    "validated_at": log_entry["timestamps"].get("validated_at", "Não disponível"),
                    "status": log_entry["validation_result"].get("status", "Não disponível"),
                    "status_operacao": log_entry["request_data"].get("status_operacao") or "Análise não iniciada",
                    "motivo_status_workflow": log_entry["request_data"].get("motivo_status_workflow", "Não disponível")
                }
                log_data.append(item)
    except Exception as e:
        print(f"Erro ao carregar arquivo de log: {e}")
    return log_data

# Adicionar cores aos status_operacao
status_colors = {
    "Cancelado": "#d9534f",        # Vermelho
    "Em análise": "#f0ad4e",        # Amarelo
    "Em análise - PEB - REINCI": "#f7e04a",  # Amarelo claro
    "Fechado para pagamento": "#5bc0de",  # Azul claro
    "Fechado pela análise": "#5bc0de",    # Azul claro
    "Finalizado": "#5cb85c",       # Verde
    "Negado": "#d9534f",           # Vermelho
    "Pago": "#428bca",             # Azul
    "Pendente Alçada": "#f0ad4e",   # Amarelo
    "Pendente Anexo": "#f0ad4e",     # Amarelo
    "Análise não iniciada": "#777"  # Cinza
}

# Função para retornar a cor
@app.template_filter('status_color')
def status_color_filter(status_operacao):
    return status_colors.get(status_operacao, "#777")  # Cinza padrão
class CustomPDF(FPDF):
    def header(self):
        # Logo
        csv_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logo-saude.jpg')

        if not os.path.exists(csv_file_path):
            download_image("https://i.ibb.co/B2R3714/logo-saude.jpg", csv_file_path)
        self.image(csv_file_path, 10, 8, 33)  # Ajuste o caminho da logo


        self.set_font("Arial", "B", 12)
        self.cell(0, 10, txt="Saúde Petrobras - Relatório Diário de Reembolsos", ln=True, align="C")
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Página {self.page_no()}", align="C")

    def set_table_header(self):
        self.set_fill_color(0, 102, 0)  # Verde escuro
        self.set_text_color(255, 255, 255)  # Branco
        self.set_font("Arial", "B", 8)
        headers = ["Protocolo", "Nome", "CPF/CNPJ", "Valor", "Status", "Status Operação", "Motivo Workflow"]
        widths = [30, 75, 35, 20, 25, 40, 50]
        for header, width in zip(headers, widths):
            self.cell(width, 8, header, 1, 0, 'C', True)
        self.ln()
def download_image(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as file:
            file.write(response.content)
@app.route('/chart_data')
@swag_from(SwaggerDocumentation.chart_data_doc())
def chart_data():
    log_data = load_log_data()
    total_sucesso = sum(1 for item in log_data if item["status"] == "sucesso")
    total_falha = sum(1 for item in log_data if item["status"] == "falha")
    total_outros = len(log_data) - (total_sucesso + total_falha)
    return jsonify({
        'labels': ['Sucesso', 'Falha', 'Outros'],
        'data': [total_sucesso, total_falha, total_outros]
    })
@app.route('/generate_report', methods=['POST'])
@swag_from(SwaggerDocumentation.generate_report_doc())
def generate_report():
        start_date = request.form.get("start_date")
        end_date = request.form.get("end_date")
        status_filter = request.form.get("status")

        log_data = load_log_data()
        pdf = CustomPDF(orientation='L', unit='mm', format='A4')
        pdf.add_page()
        pdf.set_font("Arial", size=10)

        total_solicitacoes = len(log_data)
        total_sucessos = sum(1 for item in log_data if item["status"] == "sucesso")
        total_falhas = total_solicitacoes - total_sucessos
        valor_total = sum(float(item.get("valor_apresentado", 0)) for item in log_data)
        valor_total_falhas = sum(
            float(item.get("valor_apresentado", 0)) for item in log_data if item["status"] != "sucesso")

        pdf.cell(0, 10, txt=f"Período: {start_date} a {end_date}", ln=True, align="C")
        pdf.ln(10)
        pdf.cell(0, 10, txt=f"Total de Solicitações: {total_solicitacoes}", ln=True)
        pdf.cell(0, 10, txt=f"Sucessos: {total_sucessos}", ln=True)
        pdf.cell(0, 10, txt=f"Falhas: {total_falhas} ({(total_falhas / total_solicitacoes * 100):.2f}%)", ln=True)
        pdf.cell(0, 10, txt=f"Valor Total Solicitado: R$ {valor_total:.2f}", ln=True)
        pdf.cell(0, 10, txt=f"Valor Total com Problemas: R$ {valor_total_falhas:.2f}", ln=True)
        pdf.ln(10)

        pdf.set_table_header()

        pdf.set_font("Arial", size=10)
        fill = False  # Alternar cores
        for item in log_data:
            pdf.set_fill_color(224, 235, 255) if fill else pdf.set_fill_color(255, 255, 255)
            pdf.set_text_color(0, 0, 0)

            pdf.cell(30, 8, str(item["protocolo"]), 1, 0, 'C', fill)
            pdf.cell(75, 8, item.get("nome_completo", "N/A"), 1, 0, 'L', fill)
            pdf.cell(35, 8, item.get("cpf_cnpj_prestador", "N/A"), 1, 0, 'C', fill)
            pdf.cell(20, 8, str(item.get("valor_apresentado", 0)), 1, 0, 'R', fill)
            pdf.cell(25, 8, item.get("status", "N/A"), 1, 0, 'C', fill)
            status_operacao = item.get("status_operacao") or "Análise não iniciada"
            pdf.cell(40, 8, status_operacao, 1, 0, 'C', fill)
            motivo_status_workflow = item.get("motivo_status_workflow") or "N/A"
            pdf.cell(50, 8, motivo_status_workflow, 1, 0, 'C', fill)
            pdf.ln()
            fill = not fill

        file_path = "relatorio_reembolsos_completo.pdf"
        pdf.output(file_path)

        return send_file(file_path, as_attachment=True)

# Rota para logout
@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

@app.route('/chart_data_spider')
def chart_data_spider():
    log_data = load_log_data_teste()
    total_sucesso = sum(1 for item in log_data if item['validation_result'].get("status") == "sucesso")
    total_falha = sum(1 for item in log_data if item['validation_result'].get("status") == "falha")
    total_outros = len(log_data) - (total_sucesso + total_falha)
    avg_approval_time = sum(item.get("tempo_aprovacao", 0) for item in log_data) / len(log_data)
    avg_value_requested = sum(float(item["request_data"].get("valor_apresentado", 0)) for item in log_data) / len(
        log_data)
    failure_rate = (total_falha / len(log_data)) * 100
    return jsonify({
        'labels': ['Tempo Médio de Aprovação', 'Taxa de Falha (%)', 'Valor Médio Solicitado'],
        'data': [avg_approval_time, failure_rate, avg_value_requested]
    })
log_data = load_log_data()
def load_log_data_teste():
    try:
        with open('falhas_reembolso_errors.json', 'r') as file:
            return [json.loads(line) for line in file]
    except Exception as e:
        print(f"Erro ao carregar dados reais: {e}")
        return []

@app.route('/bar_chart_data')
def bar_chart_data():
    status_counts = load_bar_chart_data()
    return jsonify({
        'status_labels': list(status_counts.keys()),
        'status_data': list(status_counts.values())
    })

@app.route('/load_bar_chart_data', methods=['GET'])
@swag_from(SwaggerDocumentation.load_bar_chart_data_doc())
def load_bar_chart_data():
    try:
        with open('falhas_reembolso_errors.json', 'r') as file:
            log_data = [json.loads(line) for line in file]
            status_counts = {}
            for item in log_data:
                status = item.get("request_data", {}).get("status_operacao", "Análise nãp Iniciada")
                status_counts[status] = status_counts.get(status, 0) + 1
            return status_counts
    except Exception as e:
        print(f"Erro ao carregar dados para o gráfico de barras: {e}")
        return {}

@app.route('/dashboard_graficos')
def gerar_graficos():
    df = load_log_data()

    fig, axs = plt.subplots(3, 2, figsize=(15, 15))

    # 1. Visão Geral de Solicitações
    status_counts = df['status'].value_counts()
    axs[0, 0].bar(status_counts.index, status_counts.values)
    axs[0, 0].set_title('Visão Geral de Solicitações')
    axs[0, 0].set_ylabel('Quantidade')

    # 2. Eficiência Operacional
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    df['validated_at'] = pd.to_datetime(df['validated_at'], errors='coerce')
    df['tempo_processamento'] = (df['validated_at'] - df['created_at']).dt.total_seconds() / 3600
    tempo_medio_status = df.groupby('status_operacao')['tempo_processamento'].mean()
    axs[0, 1].bar(tempo_medio_status.index, tempo_medio_status.values, color='orange')
    axs[0, 1].set_title('Eficiência Operacional - Tempo Médio')
    axs[0, 1].set_ylabel('Tempo (Horas)')
    axs[0, 1].tick_params(axis='x', rotation=45)

    # 3. Indicadores de Qualidade
    motivo_counts = df[df['status'] == 'falha']['motivo_status_workflow'].value_counts()
    axs[1, 0].pie(motivo_counts.values, labels=motivo_counts.index, autopct='%1.1f%%')
    axs[1, 0].set_title('Indicadores de Qualidade - Motivos de Reprovação')

    # 4. Análise Financeira
    valor_status = df.groupby('status')['valor_apresentado'].sum()
    axs[1, 1].bar(valor_status.index, valor_status.values, color='green')
    axs[1, 1].set_title('Análise Financeira - Valores Aprovados e Rejeitados')
    axs[1, 1].set_ylabel('Valor Total (R$)')

    # 5. Análise de Risco
    falhas_prestador = df[df['status'] == 'falha']['prestador'].value_counts().head(5)
    axs[2, 0].bar(falhas_prestador.index, falhas_prestador.values, color='red')
    axs[2, 0].set_title('Análise de Risco - Top 5 Prestadores com Falhas')
    axs[2, 0].set_ylabel('Quantidade de Falhas')

    # 6. Métricas de Reprocessamento
    taxa_reprocessamento = df[df['reprocessado'] == 'Sim']['status_reprocessamento'].value_counts()
    axs[2, 1].pie(taxa_reprocessamento.values, labels=taxa_reprocessamento.index, autopct='%1.1f%%', colors=['green', 'red'])
    axs[2, 1].set_title('Métricas de Reprocessamento - Taxa de Sucesso')

    plt.tight_layout()
    file_path = 'dashboard_graficos.png'
    plt.savefig(file_path)
    return send_file(file_path, mimetype='image/png')

@app.route('/dashboard_graficos_data')
def dashboard_graficos_data():
    df = load_log_data_dash()
    # Garantir que os dados estão corretos
    print(df[['protocolo', 'created_at', 'status']].head())

    # Gera a tendência de falhas e sucessos
    all_dates = pd.date_range(start=df['created_at'].min().date(), end=df['created_at'].max().date())
    sucessos_por_dia = df[df['status'] == 'sucesso'].groupby(df['created_at'].dt.date).size().reindex(all_dates,
                                                                                                      fill_value=0)
    falhas_por_dia = df[df['status'] == 'falha'].groupby(df['created_at'].dt.date).size().reindex(all_dates,
                                                                                fill_value=0)
    # Garantir que a coluna 'created_at' esteja como datetime
    df['created_at'] = pd.to_datetime(df['created_at'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
    # Agrupar por data e motivo de status do workflow
    motivo_trend = df.groupby([df['created_at'].dt.date, 'motivo_status_workflow']).size().unstack(fill_value=0)

    # Gerar cores únicas para cada motivo
    cores_motivos = {motivo: gerar_cor_hex() for motivo in motivo_trend.columns}

      # Preparar datasets para o gráfico de área
    datasets = []
    for motivo in motivo_trend.columns:
        cor = cores_motivos[motivo]
        datasets.append({
            'label': motivo,
            'data': motivo_trend[motivo].tolist(),
            'fill': True,
            'borderColor': cor,
            'backgroundColor': cor + '55',  # Cor com transparência
            'tension': 0.4  # Suaviza a curva do gráfico de área
        }) # Contagem de "Pago" e "Cancelado" convertendo para int
    status_counts = df['status_operacao'].value_counts()
    pago = int(status_counts.get('Pago', 0))
    cancelado = int(status_counts.get('Cancelado', 0))


    visao_geral = {
        'type': 'bar',
        'data': {
            'labels': df['status'].unique().tolist(),
            'datasets': [{'label': 'Visão Geral - Total por Status', 'data': df['status'].value_counts().tolist()}]
        }
    }

    eficiencia = {
        'type': 'bar',
        'data': {
            'labels': df['status_operacao'].unique().tolist(),
            'datasets': [{'label': 'Eficiência - Tempo Médio', 'data': df.groupby('status_operacao')['tempo_processamento'].mean().tolist()}]
        }
    }

    # Criação do gráfico de linha com tendência de falhas e sucessos
    qualidade = {
        'type': 'line',
        'data': {
            'labels': all_dates.strftime('%d/%m/%Y').tolist(),
            'datasets': [
                {
                    'label': 'Sucessos',
                    'data': sucessos_por_dia.tolist(),
                    'borderColor': 'green',
                    'backgroundColor': 'rgba(0, 128, 0, 0.1)',
                    'fill': True
                },
                {
                    'label': 'Falhas',
                    'data': falhas_por_dia.tolist(),
                    'borderColor': 'red',
                    'backgroundColor': 'rgba(255, 0, 0, 0.1)',
                    'fill': True
                }
            ]
        },
        'options': {
            'responsive': True,
            'scales': {
                'y': {
                    'beginAtZero': True,
                    'title': {
                        'display': True,
                        'text': 'Número de Solicitações'
                    }
                },
                'x': {
                    'title': {
                        'display': True,
                        'text': 'Data'
                    }
                }
            }
        }
    }

    financeira = {
        'type': 'doughnut',
        'data': {
            'labels': ['Pago', 'Cancelado'],
            'datasets': [{
                'data': [pago, cancelado],
                'backgroundColor': ['#28a745', '#dc3545']
            }]
        },
        'options': {
            'responsive': True,
            'plugins': {
                'legend': {
                    'position': 'top',
                },
                'title': {
                    'display': True,
                    'text': 'Distribuição de Protocolos - Pago vs Cancelado'
                }
            }
        }
    }

    risco = {
        'type': 'bar',
        'data': {
            'labels': df[df['status'] == 'falha']['prestador'].unique().tolist(),
            'datasets': [{'label': 'Risco - Falhas por Prestador', 'data': df[df['status'] == 'falha']['prestador'].value_counts().tolist()}]
        }
    }

    reprocessamento = {
         'type': 'line',
        'data': {
            'labels': motivo_trend.index.astype(str).tolist(),
            'datasets': datasets
        },
        'options': {
            'responsive': True,
            'plugins': {
                'legend': {
                    'position': 'top'
                },
                'title': {
                    'display': True,
                    'text': 'Evolução dos Motivos de Rejeição/Aprovação ao Longo do Tempo'
                }
            },
            'scales': {
                'y': {
                    'beginAtZero': True,
                    'title': {
                        'display': True,
                        'text': 'Quantidade de Ocorrências'
                    }
                },
                'x': {
                    'title': {
                        'display': True,
                        'text': 'Data'
                    }
                }
            }
        }
    }



    return jsonify({
        'visaoGeral': visao_geral,
        'eficiencia': eficiencia,
        'qualidade': qualidade,
        'financeira': financeira,
        'risco': risco,
        'reprocessamento': reprocessamento
    })

def gerar_cor_hex():
    """Gera uma cor aleatória em formato hexadecimal."""
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))


def parse_datetime(date_str):
    try:
        return datetime.strptime(date_str, "%d/%m/%Y %H:%M:%S")
    except (ValueError, TypeError):
        return None
def load_log_data_dash():
    log_data = []

    try:
        # Leitura em blocos para evitar sobrecarga de memória
        with open('falhas_reembolso_errors.json', 'r') as file:
            log_data = [
                {
                    "protocolo": str(entry.get("protocolo", "Não disponível")),
                    "status": entry.get("validation_result", {}).get("status", "Não disponível"),
                    "valor_apresentado": float(entry.get("request_data", {}).get("valor_apresentado", 0)),
                    "status_operacao": entry.get("request_data", {}).get("status_operacao", "Não disponível"),
                    "motivo_status_workflow": entry.get("request_data", {}).get("motivo_status_workflow", "Não disponível"),
                    "tempo_processamento": (
                        (datetime.strptime(entry.get("timestamps", {}).get("validated_at", ""), "%d/%m/%Y %H:%M:%S") -
                         datetime.strptime(entry.get("timestamps", {}).get("created_at", ""), "%d/%m/%Y %H:%M:%S")).total_seconds() / 3600
                        if entry.get("timestamps", {}).get("validated_at") and entry.get("timestamps", {}).get("created_at")
                        else 0
                    ),
                    "prestador": entry.get("request_data", {}).get("razao_social_prestador", "Não disponível"),
                    "created_at": entry.get("timestamps", {}).get("created_at", None)
                }
                for entry in (json.loads(line) for line in file)
            ]

    except Exception as e:
        print(f"Erro ao carregar arquivo de log: {e}")
        return pd.DataFrame()  # Retorna DataFrame vazio em caso de erro

    # Transformar em DataFrame
    df = pd.DataFrame(log_data)

    # Converter 'created_at' para datetime
    df['created_at'] = pd.to_datetime(df['created_at'], format="%d/%m/%Y %H:%M:%S", errors='coerce')

    # Calcular reprocessamentos em lote
    reprocessamento = df.groupby('protocolo').agg(
        reprocessado=('status', lambda x: 'Sim' if x.size > 1 else 'Não'),
        status_reprocessamento=('status', lambda x: 'Sucesso' if 'sucesso' in x.values else ('Falha' if (x == 'falha').sum() > 1 else 'Não Reprocessado'))
    )

    # Combinar resultados de reprocessamento com o DataFrame principal
    df = df.merge(reprocessamento, on='protocolo', how='left')

    return df


@app.route('/dashboard_graficos_tendencia')
def dashboard_graficos_tendencia():
    df = load_log_data()

    # Confere se os dados de data estão corretos
    print(df[['created_at', 'status']].head())

    # Garante que todas as datas sejam consideradas, mesmo sem registros
    all_dates = pd.date_range(start=df['created_at'].min().date(), end=df['created_at'].max().date())

    # Agrupa Sucessos e Falhas por data, preenchendo dias sem registros com 0
    sucessos_por_dia = df[df['status'] == 'sucesso'].groupby(df['created_at'].dt.date).size().reindex(all_dates, fill_value=0)
    falhas_por_dia = df[df['status'] == 'falha'].groupby(df['created_at'].dt.date).size().reindex(all_dates, fill_value=0)

    # Criação do gráfico de linha com tendência de falhas e sucessos
    tendencia_falhas_sucessos = {
        'type': 'line',
        'data': {
            'labels': all_dates.strftime('%d/%m/%Y').tolist(),
            'datasets': [
                {
                    'label': 'Sucessos',
                    'data': sucessos_por_dia.tolist(),
                    'borderColor': 'green',
                    'backgroundColor': 'rgba(0, 128, 0, 0.1)',
                    'fill': True
                },
                {
                    'label': 'Falhas',
                    'data': falhas_por_dia.tolist(),
                    'borderColor': 'red',
                    'backgroundColor': 'rgba(255, 0, 0, 0.1)',
                    'fill': True
                }
            ]
        },
        'options': {
            'responsive': True,
            'scales': {
                'y': {
                    'beginAtZero': True,
                    'title': {
                        'display': True,
                        'text': 'Número de Solicitações'
                    }
                },
                'x': {
                    'title': {
                        'display': True,
                        'text': 'Data'
                    }
                }
            }
        }
    }

    return jsonify({'tendenciaFalhasSucessos': tendencia_falhas_sucessos})

@app.route('/ask', methods=['POST'])
def ask():
    question = request.json.get('question', '')
    if not question:
        return jsonify({"error": "Pergunta não fornecida."}), 400

    try:
        # Processar os logs completos divididos em chunks
        #response = log_ai.process_full_logs(question)
        return ''
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
