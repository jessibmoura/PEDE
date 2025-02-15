A Associação Passos Mágicos tem uma trajetória de 30 anos de atuação, trabalhando na transformação da vida de crianças e jovens de baixa renda os levando a melhores oportunidades de vida. A ONG oferece um programa de Educação de qualidade para crianças e jovens do município de Embu-Guaçu.

Este projeto tem por objetivo desenvolver uma proposta preditiva para prever o comportamento do estudante com base em algumas variáveis que podem ser cruciais para a identificação de seu desenvolvimento.

#### Uso do Modelo XGBoost para Prever a Variável 'Atingiu PV'

A previsão da variável "Atingiu PV" foi tratada como um problema de classificação binária, pois essa variável assume apenas dois valores possíveis:

- Sim (1): O aluno atingiu o Ponto de Virada (PV).
- Não (0): O aluno não atingiu o Ponto de Virada (PV).

Dado que a definição do PV está relacionada ao IPV (Indicador de Ponto de Virada), que mede o desenvolvimento das aptidões dos alunos com base em uma avaliação qualitativa e quantitativa, a previsão desse resultado pode ser útil para antecipar quais alunos podem precisar de apoio extra para atingir o PV.

#### Por que modelamos a previsão como um problema de classificação?

A decisão de tratar a previsão do PV como uma classificação binária se baseia na forma como a variável "Atingiu PV" é definida:

- A variável não representa um valor contínuo ou ordinal, mas sim uma decisão binária com base no IPV.
- O IPV pode mudar anualmente, mas a classificação se mantém: os alunos são categorizados como atingindo ou não o PV.
- Métodos de classificação são mais apropriados para prever probabilidades associadas à classe e podem ser interpretados para entender quais fatores contribuem mais para o atingimento do PV.

### Estrutura do projeto

- ``scripts\model.py``: Script python com a classe do modelo utilizado
- ``scripts\preprocessing.py``: Script python com todas as funções utilizadas para fazer o processamento dos dados
- ``pipeline.py``: Jupyter notebook para treinar e salvar o modelo de machine learning
- ``app.py``: Aplicação Streamlit com o deploy do modelo

### Dashboard

Você pode checar o projeto no link: [PEDE - Predição PV](https://fiap-tc4-petroleumpriceprediction.streamlit.app/), ou
rodando o seguinte comando:
```bash
streamlit run app.py
```