# Análise de Precificação de Imóveis

Este projeto realiza uma análise exploratória de dados (EDA) sobre a precificação de imóveis, usando um conjunto de dados CSV contendo informações sobre imóveis. O objetivo principal é examinar fatores que influenciam o preço dos imóveis, como a localização, a disponibilidade e as características dos imóveis.

## Tecnologias Utilizadas

- **Pandas**: Manipulação e análise de dados.
- **NumPy**: Operações matemáticas e estatísticas.
- **Matplotlib**: Criação de gráficos e visualizações.
- **Seaborn**: Visualizações estatísticas.
- **WordCloud**: Geração de nuvem de palavras a partir dos nomes dos imóveis.

## Dependências

Antes de rodar o projeto, instale as bibliotecas necessárias:

```bash
pip install pandas numpy matplotlib seaborn wordcloud
```

## Como Executar
Carregamento dos Dados: O arquivo teste_indicium_precificacao.csv é carregado como um DataFrame usando o pandas.read_csv().

Visualização Inicial dos Dados: A primeira visualização exibe as primeiras linhas do DataFrame e informações sobre as colunas, tipos de dados e valores ausentes.

Análise de Dados:

Exibição de resumo estatístico para variáveis numéricas.
Identificação de valores ausentes no conjunto de dados.
Visualizações:

Histograma da variável 'price': Distribuição dos preços dos imóveis.
Matriz de Correlação: Identificação de relações entre variáveis numéricas.
Boxplot por 'bairro_group': Comparação dos preços entre diferentes grupos de bairros.
Scatterplots de 'minimo_noites' e 'disponibilidade_365' vs. 'price': Análise de correlação entre essas variáveis e o preço dos imóveis.
Word Cloud: Visualização das palavras mais frequentes nos nomes dos imóveis.
Hipóteses de Negócio: São apresentadas três hipóteses baseadas nas análises preliminares:

Bairros centrais tendem a ter preços mais altos devido à demanda.
Maior disponibilidade pode otimizar preços para maximizar a ocupação.
Palavras-chave no nome podem indicar imóveis "premium", influenciando o preço.


## 👨‍💻 Contato

<p>
    <img 
      align=left 
      margin=10 
      width=80 
      src="https://avatars.githubusercontent.com/u/79885721?v=4"
    />
    <p>&nbsp&nbsp&nbspMurilo Bertella Ossanes<br>
    &nbsp&nbsp&nbsp
    <a href="https://github.com/muriloboss">
    GitHub</a>&nbsp;|&nbsp;
    <a href="https://www.linkedin.com/in/murilo-ossanes/">LinkedIn</a>
&nbsp;|&nbsp;
    <a href="https://www.instagram.com/murilo_boss/">
    Instagram</a>
&nbsp;|&nbsp;</p>
</p>
<br/><br/>
<p>
---
