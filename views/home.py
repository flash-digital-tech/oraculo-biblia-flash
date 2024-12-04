import streamlit as st
from streamlit_lottie import st_lottie
import requests
import json
from forms.contact import cadastrar_cliente  # Importando a função de cadastro


def showHome():
    # Adicionando Font Awesome para ícones e a nova fonte
    st.markdown(
        """
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@700&display=swap');
        .title {
            text-align: center;
            font-size: 50px;
            font-family: 'Poppins', sans-serif;
        }
        .highlight {
            color: #6A0DAD; /* Lilás escuro */
        }
        .subheader {
            text-align: center;
            font-size: 30px;
            font-family: 'Poppins', sans-serif;
        }
        .benefits {
            font-size: 20px;
            font-family: 'Poppins', sans-serif;
            margin: 20px 0;
        }
        .icon {
            background: linear-gradient(90deg, gold, #f5f5dc);  /* Gradiente do dourado para creme */
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 24px;
            margin-right: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Estilo CSS para misturar as cores creme e dourado
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
        f"<h1 class='title'>Bem-vindo ao <span class='highlight-creme'>ORÁCULO</span> <span class='highlight-dourado'>BÍBLIA</span></h1>",
        unsafe_allow_html=True
    )

    # Apresentação do MESTRE BÍBLIA
    st.write("O MESTRE BÍBLIA é sua inteligência espiritual, pronto para guiá-lo pelo estudo das Escrituras e ajudá-lo a entender as consequências de suas escolhas à luz da Palavra de Deus.")

    # Exibindo a imagem do MESTRE BÍBLIA
    st.image("./src/img/mestre-biblia.png", width=230)

    # --- BENEFÍCIOS DO ORÁCULO BÍBLIA ---
    st.subheader("Benefícios do Oráculo Bíblia", anchor=False)
    st.write(
        """
        <div class='benefits'>
            <i class="fas fa-book-open icon"></i> **Sabedoria das Escrituras:** Aprofunde seu conhecimento sobre a Bíblia e seus ensinamentos.
        </div>
        <div class='benefits'>
            <i class="fas fa-history icon"></i> **História da Igreja:** Explore a rica história da igreja, incluindo a Reforma Protestante.
        </div>
        <div class='benefits'>
            <i class="fas fa-brain icon"></i> **Filosofia e Ética:** Compreenda os fundamentos filosóficos que sustentam a fé cristã.
        </div>
        <div class='benefits'>
            <i class="fas fa-theater-masks icon"></i> **Teologia Prática:** Aprenda a aplicar os ensinamentos bíblicos em sua vida diária e no ministério.
        </div>
        <div class='benefits'>
            <i class="fas fa-users icon"></i> **Liderança Espiritual:** Desenvolva habilidades de liderança e inspire outros em sua jornada de fé.
        </div>
        <div class='benefits'>
            <i class="fas fa-chalkboard-teacher icon"></i> **Comunicação Clara:** Aprenda a transmitir a mensagem de forma eficaz e acessível.
        </div>
        <div class='benefits'>
            <i class="fas fa-lightbulb icon"></i> **Orientação Espiritual:** Receba insights e conselhos práticos para sua vida e escolhas.
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- CHAMADA À AÇÃO ---
    st.write("\n")
    st.markdown(
        f"<h2 class='subheader'>Preparado para aprofundar seu conhecimento espiritual?</h2>",
        unsafe_allow_html=True
    )
    st.write("Participe das aulas do MESTRE BÍBLIA e transforme sua compreensão da fé!")

    # --- BOTÃO DE INSCRIÇÃO ---
    if st.button("✉️ INSCREVA-SE AGORA"):
        # Chama a função de cadastro que contém o modal
        nome, whatsapp, email, endereco, username, password, image = cadastrar_cliente()  # Modifique a função para retornar os valores

        # Verifica se os campos foram preenchidos
        if nome and whatsapp and email and endereco and username and password and image:
            # Se todos os campos foram preenchidos, exibe a mensagem de sucesso
            st.success("Cadastro feito com sucesso!!!")
        else:
            # Se algum campo estiver vazio, exibe uma mensagem de erro
            st.error("Por favor, preencha todos os campos antes de confirmar o cadastro.")
