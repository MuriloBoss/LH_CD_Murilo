import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)

df = pd.read_csv('teste_indicium_precificacao.csv')

print("Dados Iniciais:\n", df.head(), "\n")
print("Informações do DataFrame:\n")
df.info()
print("\nResumo Estatístico:\n", df.describe(), "\n")

missing_values = df.isnull().sum()
print("Valores Ausentes:\n", missing_values[missing_values > 0], "\n")

def plot_histogram(data, column, title, xlabel, ylabel, color='skyblue'):
    sns.histplot(data[column], bins=50, kde=True, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

plot_histogram(df, 'price', 'Distribuição dos Preços', 'Preço', 'Frequência')

sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title('Matriz de Correlação')
plt.show()

sns.boxplot(x='bairro_group', y='price', data=df, palette="Set2")
plt.title('Preços por Grupo de Bairro')
plt.xlabel('Grupo de Bairro')
plt.ylabel('Preço')
plt.xticks(rotation=45)
plt.show()

for col, label in [('minimo_noites', 'Número Mínimo de Noites'), ('disponibilidade_365', 'Disponibilidade (dias/ano)')]:
    sns.scatterplot(x=col, y='price', data=df, alpha=0.6)
    plt.title(f'Relação entre {label} e Preço')
    plt.xlabel(label)
    plt.ylabel('Preço')
    plt.show()

wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(' '.join(df['nome'].dropna()))
plt.figure(figsize=(15, 7.5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud dos Nomes dos Locais')
plt.show()

hipoteses = [
    "1. Bairros centrais tendem a ter preços mais altos devido à demanda.",
    "2. Maior disponibilidade pode otimizar preços para maximizar ocupação.",
    "3. Palavras-chave no nome podem indicar imóveis ""premium"", influenciando o preço."
]
print("Hipóteses de Negócio:\n", "\n".join(hipoteses), "\n")

