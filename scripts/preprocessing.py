import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


def get_raw_data_2022(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=["Fase","Pedra 20","Pedra 21","Avaliador1","Avaliador2","Avaliador3","Avaliador4","Matem","Portug","Inglês","Fase ideal"])
    df = df.rename(columns={"Ano nasc": "Ano_Nascimento",
                        "Idade 22": "Idade",
                        "Pedra 22": "Pedra",
                        "INDE 22": "INDE",
                        "Nº Av": "Numero_Avaliacoes",
                        "Defas": "Defasagem"})
    df["Ano_Coleta_Dados"] = 2022
    return df

def convert_columns_to_numeric(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Converte colunas do tipo objeto para valores numéricos, substituindo vírgulas por pontos.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contendo os dados a serem convertidos.
    columns : list
        Lista de colunas a serem convertidas.

    Returns
    -------
    pd.DataFrame
        DataFrame atualizado com colunas convertidas para numérico.
    """
    df[columns] = df[columns].replace(",", ".", regex=True).apply(pd.to_numeric)
    return df

def encode_target_column(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Converte a coluna alvo de 'Sim' e 'Não' para 1 e 0.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contendo os dados.
    target_col : str
        Nome da coluna alvo.

    Returns
    -------
    pd.DataFrame
        DataFrame atualizado com a coluna alvo codificada.
    """
    df[target_col] = df[target_col].map({"Sim": 1, "Não": 0})
    return df

def converter_comentario(texto: str) -> int:
    """
    Atribui valores para colunas de comentários.
    - 'Destaque' -> 1
    - 'Melhorar' -> -1
    - Outros -> 0
    
    Parameters
    ----------
    texto : str
        Texto do comentário.

    Returns
    -------
    int
        Valor numérico correspondente ao tipo de comentário.
    """
    if texto.startswith("Destaque"):
        return 1
    elif texto.startswith("Melhorar"):
        return -1
    return 0

def encode_text_columns(df: pd.DataFrame, text_columns: list) -> pd.DataFrame:
    """
    Converte colunas de texto em valores numéricos e remove as colunas originais.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contendo os dados.
    text_columns : list
        Lista de colunas de texto a serem convertidas.

    Returns
    -------
    pd.DataFrame
        DataFrame atualizado com colunas numéricas.
    """
    for coluna in text_columns:
        df[coluna + "_num"] = df[coluna].fillna("").apply(converter_comentario)
        df.drop(columns=[coluna], inplace=True)
    return df

def one_hot_encode(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Aplica OneHotEncoding na coluna categórica informada e remove a original.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contendo os dados.
    column : str
        Nome da coluna categórica a ser transformada.

    Returns
    -------
    pd.DataFrame
        DataFrame atualizado com colunas one-hot encoded.
    """
    encoder = OneHotEncoder(sparse_output=False)
    encoded_array = encoder.fit_transform(df[[column]])
    df_encoded = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out([column]))
    df = pd.concat([df.drop(columns=[column]), df_encoded], axis=1)
    return df

def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renomeia as colunas do DataFrame para nomes mais descritivos.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contendo os dados.

    Returns
    -------
    pd.DataFrame
        DataFrame com colunas renomeadas.
    """
    rename_dict = {
        "Cf": "Ranking_Na_Fase",
        "Ct": "Ranking_Na_Turma",
        "Cg": "Ranking_Geral",
        "IAA": "Indicador_Auto_Avaliacao",
        "IPS": "Indicador_Psicossocial",
        "IEG": "Indicador_Engajamento",
        "IDA": "Indicador_Aprendizagem",
        "IPV": "Indicador_Ponto_Virada",
        "IAN": "Indicador_Adequacao_Nivel"
    }
    df.rename(columns=rename_dict, inplace=True)
    return df

def split_train_test(df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42):
    """
    Divide os dados em conjuntos de treino e teste.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contendo os dados.
    target_col : str
        Nome da coluna alvo.
    test_size : float, optional
        Proporção dos dados para teste, padrão é 0.2.
    random_state : int, optional
        Semente aleatória para reprodutibilidade, padrão é 42.

    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test
    """
    numeric_df = df.select_dtypes(include=['number'])
    X = numeric_df.drop(columns=[target_col])
    y = numeric_df[target_col]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

def preprocess_pipeline(df: pd.DataFrame) -> tuple:
    """
    Executa todas as etapas de pré-processamento no DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame bruto contendo os dados.

    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test após pré-processamento.
    """
    numeric_columns = ["Cg", "INDE", "IAA", "IAN", "IEG", "IPS", "IDA"]
    text_columns = ["Destaque IEG", "Destaque IDA", "Destaque IPV"]
    target_column = "Atingiu PV"
    categorical_column = "Instituição de ensino"

    df = get_raw_data_2022(df)
    df = convert_columns_to_numeric(df, numeric_columns)
    df = encode_target_column(df, target_column)
    df = encode_text_columns(df, text_columns)
    df = one_hot_encode(df, categorical_column)
    df = rename_columns(df)
    return df