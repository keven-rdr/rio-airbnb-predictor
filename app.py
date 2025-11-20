import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Configuração da Página
st.set_page_config(page_title="Previsão Imobiliária DF", layout="wide")

st.title("🔮 Vidente Imobiliário — Distrito Federal")
st.markdown("Previsão de valor atual e **projeção de valorização futura**.")

# =============================
# 1. Carregar os arquivos (Atualizado para seu novo formato)
# =============================
try:
    model = joblib.load("modelo.pkl")
    scaler = joblib.load("scaler.pkl")
    colunas = joblib.load("colunas.pkl")
    # Garante que colunas estão limpas para comparação
    colunas_lower = [c.lower() for c in colunas]
except FileNotFoundError:
    st.error("Erro: Arquivos .pkl não encontrados. Certifique-se de que modelo.pkl, scaler.pkl e colunas.pkl estão na mesma pasta.")
    st.stop()

# =============================
# 2. Entradas do Usuário (Dados do Imóvel)
# =============================
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🏠 Características")
    area = st.number_input("Área (m²)", min_value=20, max_value=2000, value=50)
    quartos = st.number_input("Quartos", min_value=1, max_value=10, value=2)

with col2:
    st.subheader("📍 Localização e Tipo")
    # Sugestão: Adicione todos os bairros que existem no seu one-hot encoding original se possível
    bairros = ["ASA NORTE", "ASA SUL", "AGUAS CLARAS", "TAGUATINGA", "CEILÂNDIA", "LAGO NORTE", "LAGO SUL", "SUDOESTE", "GUARA", "PARK WAY"]
    tipos = ["APARTAMENTO", "CASA", "KITNET"]
    
    bairro = st.selectbox("Bairro", bairros)
    tipo = st.selectbox("Tipo do Imóvel", tipos)

with col3:
    st.subheader("📅 Máquina do Tempo")
    ano_atual = datetime.now().year
    ano_alvo = st.number_input("Prever valor para o Ano:", min_value=ano_atual, max_value=ano_atual+50, value=ano_atual+5)
    taxa_valorizacao = st.slider("Estimativa de Valorização Anual (%)", min_value=0, max_value=20, value=6, help="Média histórica de imóveis ou inflação + ganho real.")

# =============================
# 3. Lógica de Previsão
# =============================
if st.button("Calcular Futuro 🚀", use_container_width=True):

    # --- Passo A: Preparar dados para a IA (Igual ao Treino) ---
    entrada = pd.DataFrame(0, index=[0], columns=colunas)
    
    # Preencher numéricos
    if 'area' in entrada.columns: entrada['area'] = area
    if 'quartos' in entrada.columns: entrada['quartos'] = quartos
    
    # Preencher One-Hot (Bairro e Tipo)
    # Lógica: Procura a coluna que contém o nome do bairro escolhido
    col_bairro_alvo = f"bairro_{bairro.lower()}"
    col_tipo_alvo = f"tipo_{tipo.lower()}"
    
    # Varre as colunas do modelo para achar a correspondente (ex: bairro_asa norte)
    for col in colunas:
        if col.lower() == col_bairro_alvo:
            entrada[col] = 1
        if col.lower() == col_tipo_alvo:
            entrada[col] = 1

    # --- Passo B: Escalar e Prever HOJE ---
    entrada_scaled = scaler.transform(entrada)
    preco_hoje = model.predict(entrada_scaled)[0]
    
    # --- Passo C: Projetar o FUTURO (Matemática Financeira) ---
    # Fórmula: Valor Futuro = Valor Presente * (1 + taxa)^anos
    qtd_anos = ano_alvo - ano_atual
    taxa_decimal = taxa_valorizacao / 100
    preco_futuro = preco_hoje * ((1 + taxa_decimal) ** qtd_anos)
    lucro = preco_futuro - preco_hoje

    # =============================
    # 4. Apresentação dos Resultados
    # =============================
    st.divider()
    
    # Métricas lado a lado
    m1, m2, m3 = st.columns(3)
    m1.metric("Valor Hoje", f"R$ {preco_hoje:,.2f}")
    m2.metric(f"Valor em {ano_alvo}", f"R$ {preco_futuro:,.2f}", delta=f"+{qtd_anos} anos")
    m3.metric("Valorização Total", f"R$ {lucro:,.2f}", delta=f"{taxa_valorizacao}% a.a.")
    
    # --- Gráfico de Evolução ---
    st.subheader(f"📈 Evolução do Patrimônio ({ano_atual} a {ano_alvo})")
    
    # Criar dados para o gráfico
    lista_anos = list(range(ano_atual, ano_alvo + 1))
    lista_valores = []
    
    for i in range(len(lista_anos)):
        valor_ano = preco_hoje * ((1 + taxa_decimal) ** i)
        lista_valores.append(valor_ano)
    
    # Plotar com Matplotlib
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(lista_anos, lista_valores, marker='o', color='#00C851', linewidth=2)
    ax.fill_between(lista_anos, lista_valores, color='#00C851', alpha=0.1)
    
    # Formatação do gráfico
    ax.set_title(f"Crescimento do Investimento no Bairro {bairro}")
    ax.set_ylabel("Valor (R$)")
    ax.set_xlabel("Ano")
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Formatar eixo Y para não ficar notação científica (1e6)
    ax.ticklabel_format(style='plain', axis='y')
    
    st.pyplot(fig)

    # Tabela detalhada (Opcional)
    with st.expander("Ver tabela detalhada ano a ano"):
        df_evolucao = pd.DataFrame({"Ano": lista_anos, "Valor Projetado": lista_valores})
        st.dataframe(df_evolucao.style.format({"Valor Projetado": "R$ {:,.2f}"}))