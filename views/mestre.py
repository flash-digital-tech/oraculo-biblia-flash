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
from decouple import config


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




################################################# ENVIO DE E-MAIL ####################################################
############################################# PARA CONFIRMAÇÃO DE DADOS ##############################################

# Função para enviar o e-mail
def enviar_email(destinatario, assunto, corpo):
    remetente = "mensagem@flashdigital.tech"  # Insira seu endereço de e-mail
    senha = "sua_senha"  # Insira sua senha de e-mail

    msg = MIMEMultipart()
    msg['From'] = remetente
    msg['To'] = destinatario
    msg['Subject'] = assunto
    msg.attach(MIMEText(corpo, 'plain'))

    try:
        server = smtplib.SMTP('mail.flashdigital.tech', 587)
        server.starttls()
        server.login(remetente, senha)
        server.sendmail(remetente, destinatario, msg.as_string())
        server.quit()
        st.success("E-mail enviado com sucesso!")
    except Exception as e:
        st.error(f"Erro ao enviar e-mail: {e}")

    # Enviando o e-mail ao pressionar o botão de confirmação
    if st.button("DADOS CONFIRMADO"):
        # Obter os dados salvos em st.session_state
        nome = st.session_state.user_data["name"]
        whatsapp = st.session_state.user_data["whatsapp"]
        email = st.session_state.user_data["email"]

        # Construindo o corpo do e-mail
        corpo_email = f"""
        Olá {nome},

        Segue a confirmação dos dados:
        - Nome: {nome}
        - WhatsApp: {whatsapp}
        - E-mail: {email}
        - Agendamento : {dias} e {turnos}

        Obrigado pela confirmação!
        """

        # Enviando o e-mail
        enviar_email(email, "Confirmação de dados", corpo_email)


#######################################################################################################################

def showMestre():

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

    processar_docs = carregar_arquivos()

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

    estudo = 'Filipenses'
    
    system_prompt = f'''
    Você é o MESTRE BÍBLIA, um oráculo bíblico com mais de 20 anos de experiência no ensino de teologia na Faculdade de Teologia da Universidade Presbiteriana Mackenzie. Você possui um profundo conhecimento das Escrituras e é especialista em História da Igreja, Filosofia, Teologia Sistemática, Teologia Prática, Comunicação, Ensino, Liderança, Arqueologia Bíblica, Teologia e Arqueologia, Arqueologia da Terra Santa e Teologia e História Antiga.
    
   **Instruções**:

1. **Limitação de Tema**:
   - Você responderá exclusivamente sobre o estudo da semana ou do mês definido pela igreja, de acordo com o tópico 6. Se o usuário fizer uma pergunta sobre algum personagem fora do tema ou outro livro, você não responderá e nem dará sugestão para estudar outros livros.
   - Se o usuário insistir com outro tema fora do estudo da semana ou do mês, não responda. Mesmo que ele cite outros nomes ou livros, você pedirá desculpas e dirá que foi programado para responder apenas sobre o estudo da semana ou do mês.
   - Se o usuário insistir pela terceira vez, diga a ele que está entrando em desobediência e que isso será relatado ao líder dele.

2. **Tema de Estudo do Mês**:
   - Você dará estudo aprofundado aos membros e alunos da Comunidade Cristã Recomeçar sobre: 
     {estudo}.

3. **Orientações e Consequências**:
   - Utilize seu poder como oráculo para prever as consequências das escolhas e ações dos indivíduos com base nos ensinamentos bíblicos. Explique, instrua e corrija quando necessário, sempre ancorando suas respostas nas Escrituras e nos princípios teológicos.

4. **Abordagem Didática**:
   - Comunique-se de forma clara e eficaz, utilizando uma linguagem acessível, mas respeitosa ao contexto acadêmico e religioso. Mantenha um tom de liderança e inspiração, incentivando os fiéis a aprofundarem seu conhecimento e prática da fé.

5. **Respostas Baseadas em Evidências**:
   - Quando apropriado, utilize evidências históricas e arqueológicas para apoiar suas respostas, especialmente em questões sobre a interseção entre teologia e arqueologia. Isso pode incluir referências a descobertas que corroboram narrativas bíblicas.

6. **Atualização do Tema**:
   - Esteja preparado para atualizar o tema de estudo conforme necessário, mantendo sempre a relevância e a precisão das informações fornecidas.

7. **Limite de Respostas**:
   - Você responderá de forma clara e objetiva, limitando sua resposta a no máximo 500 tokens. Certifique-se de que todas as informações essenciais sejam incluídas e utilize frases concisas para que a resposta não fique incompleta.

8. **Documentos de Estudo**:
   - Se tiver algum documento ou estudo para ensinar, você vai ler e orientar aqui: {conteudos_txt}. Se não tiver nenhum documento, você continuará com o estudo do tópico 6.

9. **Informações sobre a Comunidade Cristã Recomeçar**:
   - Se o aluno ou o membro quiser saber sobre a Comunidade Cristã Recomeçar, responda através das informações aqui: https://www.recomecar.com.br/home.

10. **Fac de Atendimento aos Membros e Alunos**:
   - **Quem somos?**: Nós somos uma Comunidade Cristã situada em Contagem - MG, que acredita no poder do Evangelho para transformar vidas. Nossa fé nos leva a buscar um relacionamento íntimo com o Pai de Amor, através da oração e da leitura das Escrituras Sagradas, bem como do serviço ao próximo. Estamos comprometidos em viver e compartilhar essa mensagem de esperança com a nossa comunidade e além dela.
   - **Qual a localização da igreja?**: https://maps.app.goo.gl/BP6CpR15zMzx1Z4n7
   - **Quem são os responsáveis pelo estudo bíblico?**: Pastor Cláudio e César.
    
    '''

    # Função para gerar o PDF
    def create_pdf(messages):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        for message in messages:
            role = message["role"].capitalize()
            content = message["content"]
            pdf.cell(200, 10, txt=f"{role}: {content}", ln=True)

        return pdf.output(dest='S').encode('latin1')


    # Função para gerar o Excel
    def create_excel(messages):
        df = pd.DataFrame(messages)
        buffer = io.BytesIO()
        df.to_excel(buffer, index=False)
        buffer.seek(0)
        return buffer.getvalue()


    # Set assistant icon to Snowflake logo
    icons = {"assistant": "./src/img/mestre-biblia.png", "user": "./src/img/perfil-usuario.png"}


    # Replicate Credentials
    with st.sidebar:
        st.markdown(
            """
            <h1 style='text-align: center;'>Doctor Med</h1>
            """,
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


        # Load and display sidebar image with glowing effect
        img_path = "./src/img/mestre-biblia.png"
        img_base64 = img_to_base64(img_path)
        st.sidebar.markdown(
            f'<img src="data:image/png;base64,{img_base64}" class="cover-glow">',
            unsafe_allow_html=True,
        )

        st.sidebar.markdown("---")

    # Inicializar o modelo da Replicate
    llm = Replicate(
        model="meta/meta-llama-3.1-405b-instruct",
        api_token=replicate_api
    )

    # Store LLM-generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{
            "role": "assistant", "content": 'Olá! Sou o MESTRE BÍBLIA, seu guia espiritual e oráculo bíblico, '
                    'pronto para ajudá-lo a compreender as Escrituras.'}]

    # Display or clear chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=icons[message["role"]]):
            st.write(message["content"])


    def clear_chat_history():
        st.session_state.messages = [{"role": "assistant", "content": 'Olá! Sou o MESTRE BÍBLIA, seu guia espiritual e oráculo bíblico, '
                    'pronto para ajudá-lo a compreender as Escrituras.'}]


    st.sidebar.button('LIMPAR CONVERSA', on_click=clear_chat_history)
    st.sidebar.caption(
        'Built by [Snowflake](https://snowflake.com/) to demonstrate [Snowflake Arctic](https://www.snowflake.com/blog/arctic-open-and-efficient-foundation-language-models-snowflake). App hosted on [Streamlit Community Cloud](https://streamlit.io/cloud). Model hosted by [Replicate](https://replicate.com/snowflake/snowflake-arctic-instruct).')
    st.sidebar.caption(
        'Build your own app powered by Arctic and [enter to win](https://arctic-streamlit-hackathon.devpost.com/) $10k in prizes.')

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
        if is_in_registration or is_in_scheduling:
            return "Por favor, complete o formulário de cadastro antes de continuar."

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


        # Verifica se o usuário deseja se cadastrar
        elif "quero ser parceiro" in prompt_str.lower() or "desejo criar uma conta de parceiro" in prompt_str.lower() or "conta de parceiro" in prompt_str.lower() or "quero me tornar parceiro" in prompt_str.lower() or "como faço para ser parceiro" in prompt_str.lower() or "quero ter uma conta de parceiro" in prompt_str.lower() or "quero ser um parceiro" in prompt_str.lower():
            st.write("Para se tornar um parceiro na ORÁCULO IA e começar a ter ganhos extraordinários clique no botão abaixo.")
            if st.button("QUERO SER PARCEIRO"):
                showSbconta()
                st.stop()


        elif get_num_tokens(prompt_str) >= 8000:  # padrão3072
            st.error(
                "Poxa, você já atingiu seu limite de demostração, mas pode ficar tranquilo. Clique no botão abaixo para "
                "pedir seu acesso.")
            st.button('PEDIR ACESSO', on_click=clear_chat_history, key="clear_chat_history")
            excel_bytes = create_excel(st.session_state.messages)
            pdf_bytes = create_pdf(st.session_state.messages)
            formato_arquivo = st.selectbox("Escolha como deseja baixar sua conversa:", ["PDF", "Excel"])
            if formato_arquivo == "PDF":
                st.download_button(
                    label="Baixar PDF",
                    data=pdf_bytes,
                    file_name="conversa.pdf",
                    mime="application/pdf",
                )
            else:
                st.download_button(
                    label="Baixar Excel",
                    data=excel_bytes,
                    file_name="conversa.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            st.stop()


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


    # User-provided prompt
    if prompt := st.chat_input(disabled=not replicate_api):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="./src/img/perfil-usuario.png"):
            st.write(prompt)


    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant", avatar="./src/img/mestre-biblia.png"):
            response = generate_arctic_response()
            full_response = st.write_stream(response)
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)



