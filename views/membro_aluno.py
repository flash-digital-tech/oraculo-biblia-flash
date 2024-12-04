import asyncio

import streamlit as st
from transformers import AutoTokenizer
import base64
import pandas as pd
import io
from fastapi import FastAPI
import stripe
from util import carregar_arquivos
import os
import glob
from forms.contact import cadastrar_cliente, agendar_reuniao 


import replicate
from langchain.llms import Replicate


import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from key_config import API_KEY_STRIPE, URL_BASE
from decouple import config


app = FastAPI()



# --- Verifica se o token da API está nos segredos ---
if 'REPLICATE_API_TOKEN':
    replicate_api = config('REPLICATE_API_TOKEN')
else:
    # Se a chave não está nos segredos, define um valor padrão ou continua sem o token
    replicate_api = None

# Essa parte será executada se você precisar do token em algum lugar do seu código
if replicate_api is None:
    # Se você quiser fazer algo específico quando não há token, você pode gerenciar isso aqui
    # Por exemplo, configurar uma lógica padrão ou deixar o aplicativo continuar sem mostrar nenhuma mensagem:
    st.warning('Um token de API é necessário para determinados recursos.', icon='⚠️')


#######################################################################################################################

def showMembroAluno():

    if "image" not in st.session_state:
        st.session_state.image = None
    
    def ler_arquivos_txt(pasta):
        """
        Lê todos os arquivos .txt na pasta especificada e retorna uma lista com o conteúdo de cada arquivo.

        Args:
            pasta (str): O caminho da pasta onde os arquivos .txt estão localizados.

        Returns:
            list: Uma lista contendo o conteúdo de cada arquivo .txt.
        """
        conteudos = []  # Lista para armazenar o conteúdo dos arquivos

        # Cria o caminho para buscar arquivos .txt na pasta especificada
        caminho_arquivos = os.path.join(pasta, '*.txt')

        # Usa glob para encontrar todos os arquivos .txt na pasta
        arquivos_txt = glob.glob(caminho_arquivos)

        # Lê o conteúdo de cada arquivo .txt encontrado
        for arquivo in arquivos_txt:
            with open(arquivo, 'r', encoding='utf-8') as f:
                conteudo = f.read()  # Lê o conteúdo do arquivo
                conteudos.append(conteudo)  # Adiciona o conteúdo à lista

        return conteudos  # Retorna a lista de conteúdos

    # Exemplo de uso da função
    pasta_conhecimento = './conhecimento'  # Caminho da pasta onde os arquivos .txt estão localizados
    conteudos_txt = ler_arquivos_txt(pasta_conhecimento)

    is_in_registration = False
    is_in_scheduling = False


    # Função para verificar se a pergunta está relacionada a cadastro
    def is_health_question(prompt):
        keywords = ["cadastrar", "inscrição", "quero me cadastrar", "gostaria de me registrar",
                    "desejo me cadastrar", "quero fazer o cadastro", "quero me registrar", "quero me increver",
                    "desejo me registrar", "desejo me inscrever","eu quero me cadastrar", "eu desejo me cadastrar",
                    "eu desejo me registrar", "eu desejo me inscrever", "eu quero me registrar", "eu desejo me registrar",
                    "eu quero me inscrever"]
        return any(keyword.lower() in prompt.lower() for keyword in keywords)

    #Função que analisa desejo de agendar uma reunião
    def is_schedule_meeting_question(prompt):
        keywords = [
            "agendar reunião", "quero agendar uma reunião", "gostaria de agendar uma reunião",
            "desejo agendar uma reunião", "quero marcar uma reunião", "gostaria de marcar uma reunião",
            "desejo marcar uma reunião", "posso agendar uma reunião", "posso marcar uma reunião",
            "Eu gostaria de agendar uma reuniao", "eu quero agendar", "eu quero agendar uma reunião,",
            "quero reunião"
        ]
        return any(keyword.lower() in prompt.lower() for keyword in keywords)

    # Atualizando o system_prompt
    system_prompt = f'''
    Você é o MESTRE BÍBLIA, um oráculo bíblico com mais de 20 anos de experiência no ensino de teologia na Faculdade de Teologia da Universidade Presbiteriana Mackenzie. Você possui um profundo conhecimento das Escrituras e é especialista em História da Igreja, Filosofia, Teologia Sistemática, Teologia Prática, Comunicação, Ensino, Liderança, Arqueologia Bíblica, Teologia e Arqueologia, Arqueologia da Terra Santa e Teologia e História Antiga.
    
    **Instruções**:
    
    1. **Limitação de Tema**:
       - Suas respostas devem se concentrar exclusivamente no estudo da semana ou do mês definido pela igreja. Por exemplo, se o tema do mês de Novembro é a carta de Filipenses, você não deve responder ou sugerir temas fora desse contexto.
    
    2. **Orientações e Consequências**:
       - Utilize seu poder como oráculo para prever as consequências das escolhas e ações dos indivíduos com base nos ensinamentos bíblicos. Explique, instrua e corrija quando necessário, sempre ancorando suas respostas nas Escrituras e nos princípios teológicos.
    
    3. **Abordagem Didática**:
       - Comunique-se de forma clara e eficaz, utilizando uma linguagem acessível, mas respeitosa ao contexto acadêmico e religioso. Mantenha um tom de liderança e inspiração, incentivando os fiéis a aprofundarem seu conhecimento e prática da fé.
    
    4. **Respostas Baseadas em Evidências**:
       - Quando apropriado, utilize evidências históricas e arqueológicas para apoiar suas respostas, especialmente em questões sobre a interseção entre teologia e arqueologia. Isso pode incluir referências a descobertas que corroboram narrativas bíblicas.
    
    5. **Atualização do Tema**:
       - Esteja preparado para atualizar o tema de estudo semanal ou mensal conforme necessário, mantendo sempre a relevância e a precisão das informações fornecidas.
    
    6. **Tema de Estudo do Mês**:
       - Você dará estudo aprofundado aos membros e alunos da Comunidade Cristâ Recomeçar sobre: FILIPENSES.
    
    7. - Se tiver algum documento ou estudo para ensinar você vai ler e orientar aqui: {conteudos_txt},se não
    tiver nenhum documento para estudo você continuará com o estudo do tópico 6.
    
    **Exemplo de Interação**:
    - "Olá, sou o MESTRE BÍBLIA. O tema de estudo deste mês é a carta de Filipenses. Como posso ajudá-lo a compreender melhor os ensinamentos desta carta e suas aplicações práticas em sua vida?"
    
    '''

    st.markdown(
        """
        <style>
        .highlight-creme {
            background: linear-gradient(90deg, #f5f5dc, gold);  /* Gradiente do creme para dourado */
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: bold;
        }
        .highlight-dourado {
            background: linear-gradient(90deg, gold, #f5f5dc);  /* Gradiente do dourado para creme */
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: bold;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Título da página
    st.markdown(
        f"<h1 class='title'>Estude com o <span class='highlight-creme'>MESTRE</span> <span class='highlight-dourado'>BÍBLIA</span></h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <style>
        .cover-glow {
            width: 100%;
            height: auto;
            padding: 3px;
            box-shadow: 
                0 0 5px #330000,
                0 0 10px #660000,
                0 0 15px #990000,
                0 0 20px #CC0000,
                0 0 25px #FF0000,
                0 0 30px #FF3333,
                0 0 35px #FF6666;
            position: relative;
            z-index: -1;
            border-radius: 30px;  /* Rounded corners */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Function to convert image to base64
    def img_to_base64(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()

    st.sidebar.markdown("---")

    # Load and display sidebar image with glowing effect
    img_path = "./src/img/mestre-biblia.png"
    img_base64 = img_to_base64(img_path)
    st.sidebar.markdown(
        f'<img src="data:image/png;base64,{img_base64}" class="cover-glow">',
        unsafe_allow_html=True,
    )


    # Inicializar o modelo da Replicate
    llm = Replicate(
        model="meta/meta-llama-3.1-405b-instruct",
        api_token=replicate_api
    )

    # Store LLM-generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{
            "role": "assistant", "content": '🌟 Bem-vindo ao Mestre Bíblia! Seu guia espiritual e oráculo bíblico, '
                    'pronto para ajudá-lo a compreender as Escrituras.'}]

    # Dicionário de ícones
    icons = {
        "assistant": "./src/img/mestre-biblia.png",  # Ícone padrão do assistente
        "user": "./src/img/perfil-usuario.png"            # Ícone padrão do usuário
    }
    
    # Caminho para a imagem padrão
    default_avatar_path = "./src/img/perfil-usuario.png"
    
    # Exibição das mensagens
    for message in st.session_state.messages:
        if message["role"] == "user":
            # Verifica se a imagem do usuário existe
            avatar_image = st.session_state.image if "image" in st.session_state and st.session_state.image else default_avatar_path
        else:
            avatar_image = icons["assistant"]  # Ícone padrão do assistente
    
        with st.chat_message(message["role"], avatar=avatar_image):
            st.image(avatar_image, width=50)
            st.write(message["content"])


    def clear_chat_history():
        st.session_state.messages = [{
            "role": "assistant", "content": 'Olá! Sou o MESTRE BÍBLIA, seu guia espiritual e oráculo bíblico, '
                    'pronto para ajudá-lo a compreender as Escrituras.'}]


    st.sidebar.button('LIMPAR CONVERSA', on_click=clear_chat_history, key='limpar_conversa')

    st.sidebar.markdown("Desenvolvido por [WILLIAM EUSTÁQUIO](https://www.instagram.com/flashdigital.tech/)")

    @st.cache_resource(show_spinner=False)
    def get_tokenizer():
        """Get a tokenizer to make sure we're not sending too much text
        text to the Model. Eventually we will replace this with ArcticTokenizer
        """
        return AutoTokenizer.from_pretrained("huggyllama/llama-7b")


    def get_num_tokens(prompt):
        """Get the number of tokens in a given prompt"""
        tokenizer = get_tokenizer()
        tokens = tokenizer.tokenize(prompt)
        return len(tokens)


    def check_safety(disable=False) -> bool:
        if disable:
            return True

        deployment = get_llamaguard_deployment()
        conversation_history = st.session_state.messages
        user_question = conversation_history[-1]  # pegar a última mensagem do usuário

        prediction = deployment.predictions.create(
            input=template)
        prediction.wait()
        output = prediction.output

        if output is not None and "unsafe" in output:
            return False
        else:
            return True

    # Function for generating Snowflake Arctic response
    def generate_arctic_response():

        prompt = []
        for dict_message in st.session_state.messages:
            if dict_message["role"] == "user":
                prompt.append("<|im_start|>user\n" + dict_message["content"] + "<|im_end|>")
            else:
                prompt.append("<|im_start|>assistant\n" + dict_message["content"] + "<|im_end|>")

        prompt.append("<|im_start|>assistant")
        prompt.append("")
        prompt_str = "\n".join(prompt)

        if is_health_question(prompt_str):
            cadastrar_cliente()


        if is_schedule_meeting_question(prompt_str):
            agendar_reuniao()

        for event in replicate.stream(
                "meta/meta-llama-3.1-405b-instruct",
                input={
                    "top_k": 0,
                    "top_p": 1,
                    "prompt": prompt_str,
                    "temperature": 0.1,
                    "system_prompt": system_prompt,
                    "length_penalty": 1,
                    "max_new_tokens": 8000,
                },
        ):
            yield str(event)


    def get_avatar_image():
        """Retorna a imagem do usuário ou a imagem padrão se não houver imagem cadastrada."""
        if st.session_state.image is not None:
            return st.session_state.image  # Retorna a imagem cadastrada
        else:
            return default_avatar_path  # Retorna a imagem padrão
    
    # User-provided prompt
    if prompt := st.chat_input(disabled=not replicate_api):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Chama a função para obter a imagem correta
        avatar_image = get_avatar_image()
        
        with st.chat_message("user", avatar=avatar_image):
            st.write(prompt)
    
    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant", avatar="./src/img/mestre-biblia.png"):
            response = generate_arctic_response()
            full_response = st.write_stream(response)
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)



