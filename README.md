# epi-sistema

Verificação de conformidade de EPI em ponto de controle de acesso, com visão computacional
e integração à automação. Projeto de mestrado (PECS/UEMA).

**Estado:** bases preparadas e divisões congeladas. Ainda não há modelos treinados.

- [Metodologia](docs/metodologia.md)
- [Plano por fases](docs/plano.md)

## Estrutura

| Pasta | Conteúdo |
|---|---|
| `notebooks/` | notebooks numerados, que apenas orquestram |
| `src/epi/` | código reutilizável |
| `configs/` | parâmetros |
| `tests/` | testes |
| `docs/` | metodologia, plano e registros de fase |
| `data/`, `runs/` | bases e resultados, **fora do git** |

## Google Colab

Abra `notebooks/00_ambiente.ipynb` no Colab, selecione GPU e execute tudo. O notebook clona
o repositório e instala o pacote; o PyTorch usado é o do Colab.

## Computador local (Windows, GPU NVIDIA)

Com Python 3.13:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
pip install -e ".[dev]"
nbstripout --install --attributes .gitattributes   # obrigatório antes de qualquer commit
pytest
jupyter nbconvert --to notebook --execute notebooks/00_ambiente.ipynb --output-dir runs/executados
```

## Preparação das bases

Abra `notebooks/01_dados.ipynb` no Google Colab (sem GPU; deixe ao menos 4 GB livres no
Drive) e execute tudo. A execução para de propósito para a revisão dos pares de imagens
parecidas; siga as instruções do notebook. Parâmetros em `configs/dados.yaml`.

As divisões congeladas estão em `configs/divisoes/`: nomes de arquivos, dimensões,
impressões digitais e contagens, sem imagens.

## Ponto de controle (em desenvolvimento)

```powershell
python -m epi.clp            # CLP simulado em 127.0.0.1:5020
jupyter nbconvert --to notebook --execute notebooks/05_ponto_de_controle.ipynb --output-dir runs/executados
```

Parâmetros em `configs/ponto_de_controle.yaml` (provisórios). Sem um modelo de EPI treinado,
a demonstração usa um detector roteirizado. O sistema é apoio ao controle de acesso e
**não é uma função de segurança certificada**.

## Regras

Imagens, vídeos, pesos e credenciais não entram no repositório. As saídas dos notebooks são
removidas no commit, pois podem conter imagens das bases.

## Licenças

Código sob [AGPL-3.0](LICENSE), por causa do Ultralytics. As bases de dados têm licenças
próprias e não são redistribuídas aqui.
