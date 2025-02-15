import streamlit as st
import pandas as pd
import pickle
from scripts.preprocessing import preprocess_pipeline


@st.cache_resource
def load_model():
    with open("./models/xgboost_model.pkl", "rb") as f:
        return pickle.load(f)

def predict(df, model):
    df_features = df.select_dtypes(include=['number'])
    probabilities = model.model.predict_proba(df_features)  
    predictions = model.model.predict(df_features)  

    df_result = pd.DataFrame({
        "Nome do Aluno": df["Nome"],  
        "Previsão Atingiu PV": ["Sim" if pred == 1 else "Não" for pred in predictions],
        "Probabilidade (%)": [f"{prob[1] * 100:.2f}" for prob in probabilities]
    })
    return df_result

st.title("Predição de Alunos que Atingiram PV 🎓")

st.write("""
         O IPV (Indicador de Ponto de Virada) registra por meio de um questionário padronizado de nove perguntas, ponderadas em três classes de peso distintas, 
         a avaliação da mesma equipe de professores, pedagogos e psicopedagogos, sobre o desenvolvimento do estudante das aptidões necessárias para iniciar o uso da Educação 
         como um instrumento da transformação de sua vida.
         
         O objetivo do modelo proposto é classificar se um aluno atingiu ou não esse Ponto de Virada. O modelo utiliza dos mesmos dados/valores utilizados
         para fazer o cálculo do IPV, mas não utiliza dele para realizar a previsão. A tabela retornada contém:
         
         - Nome do Aluno: Neste caso, o nome anonimizado;
         - Previsão Atingiu PV: Classificação do modelo em 'Sim' ou 'Não';
         - Probabilidade (%): Probabilidade calculada pelo modelo para chegar à previsão mostrada. 
         """)

st.info("Exemplo de saída: ")

data = {
    "Nome do Aluno": [
        "Aluno-1", "Aluno-2", "Aluno-3", "Aluno-4", "Aluno-5"
    ],
    "Previsão Atingiu PV": [
        "Não", "Não", "Não", "Não", "Não"
    ],
    "Probabilidade (%)": [
        0.46, 0.05, 1.99, 0.05, 0.37
    ]
}

df = pd.DataFrame(data)
st.dataframe(df,use_container_width=True)


st.warning(""" O modelo apresentado é um MVP e foi construído com base na base de arquivos de 2022. Portanto, no momento ele trabalha
           apenas com arquivos no modelo do arquivo PEDE - DATATHON - PEDE2022.csv""")

st.write("Faça upload de um arquivo CSV com os dados dos alunos para obter a previsão.")


uploaded_file = st.file_uploader("Carregar CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if st.button("Gerar Previsão"):
        model = load_model()

        df = preprocess_pipeline(df)
        df_features = df.drop(columns=['Atingiu PV'], errors="ignore")

        df_result = predict(df_features, model)

        st.subheader("Resultados da Previsão")
        st.write(df_result)

        st.download_button(
            label="Baixar Previsões como CSV",
            data=df_result.to_csv(index=False),
            file_name="previsoes_alunos.csv",
            mime="text/csv"
        )
