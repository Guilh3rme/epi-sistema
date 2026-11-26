# epi-sistema

Verificação de conformidade de EPI em ponto de controle de acesso, com visão computacional
e integração à automação. Projeto de mestrado (PECS/UEMA).

**Estado:** ambiente do projeto configurado. Ainda não há modelos treinados.

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

## Regras

Imagens, vídeos, pesos e credenciais não entram no repositório. As saídas dos notebooks são
removidas no commit, pois podem conter imagens das bases.

## Licenças

Código sob [AGPL-3.0](LICENSE), por causa do Ultralytics. As bases de dados têm licenças
próprias e não são redistribuídas aqui.
