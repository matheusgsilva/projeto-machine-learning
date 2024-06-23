# Análise de Dados de Fraude com Técnicas de Balanceamento

Este projeto implementa várias técnicas de balanceamento de dados e treina modelos de aprendizado de máquina para detectar fraudes. O dataset utilizado é um arquivo ARFF.

## Estrutura do Projeto

- `main.py`: Arquivo principal do código que descompacta o dataset, treina os modelos e gera os resultados.
- `dataset.zip`: Arquivo zip contendo o dataset ARFF.
- `results/`: Pasta onde todos os resultados e gráficos gerados serão salvos.
- `.gitignore`: Arquivo para ignorar arquivos e pastas específicas no repositório Git.
- `README.md`: Este arquivo de documentação.

## Pré-requisitos

Antes de rodar o código, você precisa ter o Python 3.7 ou superior instalado em sua máquina. Você também precisará das seguintes bibliotecas Python:

- pandas
- scipy
- scikit-learn
- imbalanced-learn
- matplotlib
- seaborn
- numpy
- tensorflow

## Instalando Dependências

Você pode instalar todas as dependências necessárias executando os seguintes comandos:

```bash
pip install pandas scipy scikit-learn imbalanced-learn matplotlib seaborn numpy tensorflow
```

## Executar o Código

```bash
python main.py
```
