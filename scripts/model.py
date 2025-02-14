import pickle
import xgboost as xgb
import pandas as pd

class XGBOOSTPredictor:
    """
    Classe para treinar e utilizar um modelo XGBoost para classificação.
    """
    def __init__(self):
        """
        Inicializa o modelo XGBoost com os hiperparâmetros especificados.
        """
        self.model = xgb.XGBClassifier(
            subsample=0.6,
            n_estimators=200,
            max_depth=7,
            learning_rate=0.2,
            objective="binary:logistic",
            eval_metric='logloss'
        )
    
    def fit(self, df_train: pd.DataFrame, target_column: str):
        """
        Treina o modelo XGBoost nos dados fornecidos.
        
        Parameters
        ----------
        df_train : pd.DataFrame
            DataFrame contendo os dados de treino.
        target_column : str
            Nome da coluna que contém os rótulos.
        """
        X_train = df_train.drop(columns=[target_column])
        y_train = df_train[target_column]
        self.model.fit(X_train, y_train)
        print("Modelo treinado com sucesso!")
    
    def predict(self, df_test: pd.DataFrame) -> pd.Series:
        """
        Realiza previsões utilizando o modelo treinado.
        
        Parameters
        ----------
        df_test : pd.DataFrame
            DataFrame contendo os dados de teste.
        
        Returns
        -------
        pd.Series
            Série com as previsões do modelo.
        """
        return pd.Series(self.model.predict(df_test))
    
    def save(self, file_path: str):
        """
        Salva o modelo treinado em um arquivo utilizando pickle.
        
        Parameters
        ----------
        file_path : str
            Caminho do arquivo onde o modelo será salvo.
        """
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Modelo salvo em {file_path}")
    
    @staticmethod
    def load(file_path: str) -> 'XGBOOSTPredictor':
        """
        Carrega um modelo XGBoost salvo anteriormente.
        
        Parameters
        ----------
        file_path : str
            Caminho do arquivo onde o modelo está salvo.
        
        Returns
        -------
        XGBOOSTPredictor
            Instância da classe XGBOOSTPredictor carregada.
        """
        with open(file_path, 'rb') as f:
            return pickle.load(f)