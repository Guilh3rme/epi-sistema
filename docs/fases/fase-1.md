# Fase 1 — Repositório e ambiente

## Parte 1 — Abertura e detalhamento

- **Aberta em:** 25/11/2026
- **Lido antes de detalhar:** plano (status e decisões) e metodologia v1.1.

### Herdado das fases anteriores
Documentos e artigo vêm depois do sistema. Nenhum código existe. O
repositório local tem `git init`, sem commits.

### Objetivo e escopo
Deixar um projeto que roda igual no computador do autor e no Google Colab, pronto para a
preparação das bases. Fica fora: baixar SH17 ou Construction-PPE, treinar modelos do
projeto e qualquer lógica do ponto de controle.

### Respostas às perguntas da fase

**Estrutura**

    epi-sistema/
      notebooks/00_ambiente.ipynb   notebooks só orquestram
      src/epi/__init__.py           código reutilizável, instalável com pip install -e .
      src/epi/ambiente.py           detecção de Colab e registro da execução
      configs/                      parâmetros em YAML
      tests/test_ambiente.py        teste do registro da execução
      docs/                         metodologia, plano e fases
      pyproject.toml  LICENSE  README.md  .gitignore  .gitattributes

- Notebooks numerados (`00_`, `01_`...). A primeira célula detecta o Colab: nele, clona o
  repositório, instala o pacote e lê os segredos; no computador local, usa o `.venv`.
- **Saídas dos notebooks são removidas antes de cada commit** (`nbstripout` como filtro do
  git). Gráficos de predição embutem imagens das bases dentro do arquivo, o que levaria
  imagens, inclusive de pessoas, para o repositório.

**Dados e pesos:** fora do git, em `data/` e `runs/`. No Colab, as bases são baixadas para
o disco da máquina virtual; só pesos e resultados vão para o Google Drive.

**Versões**

| Item | Computador local | Colab |
|---|---|---|
| Python | 3.13 (`py -V:Astral/CPython3.13.18`) | o do Colab |
| PyTorch | 2.14, canal `cu130` (driver informa CUDA 13.5; o `cu132` traz o 2.15, mais novo) | o pré-instalado, apenas registrado |
| Ultralytics | 8.4.219 | 8.4.219 |

As versões são conferidas na instalação; qualquer ajuste é anotado no encerramento.

**Registro de execução:** uma função grava em JSON, por execução, Python, PyTorch,
Ultralytics, CUDA, GPU, semente e commit do código.

**Segredos:** credenciais do Kaggle (e do GitHub, se o repositório for privado) nos
segredos do Colab; localmente, em `~/.kaggle`. Nunca no repositório.

**Licença:** código sob AGPL-3.0, por causa do Ultralytics. O README avisa que as bases
têm licenças próprias.

### Entregas e critério de pronto
| Entrega | Onde fica | Como verificar |
|---|---|---|
| Estrutura e pacote | repositório | `pip install -e .` e teste passam |
| Ambiente local | `.venv` | `torch.cuda.is_available()` verdadeiro e nome da GPU impresso |
| `00_ambiente.ipynb` local | `notebooks/` | executa do início ao fim: registro, predição com `yolo26n` em imagem de exemplo e 1 época no `coco8` (no Windows, `workers=0`) |
| O mesmo no Colab | Google Colab | **o autor executa com GPU** e envia a saída da célula de ambiente |

A fase só encerra depois da execução no Colab.

### Passos
1. Estrutura, `pyproject.toml`, licença e filtro dos notebooks.
2. `.venv` e instalação; conferir a GPU.
3. `ambiente.py`, seu teste e o notebook `00`.
4. Executar o notebook localmente.
5. Repositório no GitHub e primeiro commit, conforme a autorização do autor.
6. Execução no Colab pelo autor; ajustes e encerramento.

### Riscos e plano B
- PyTorch/torchvision sem pacote compatível: usar o canal `cu126`.
- Colab com versão de PyTorch incompatível com o Ultralytics: registrar e ajustar só no Colab.
- Sem repositório no GitHub, o Colab recebe o projeto por um `.zip` no Drive.

### Aprovação
- **Aprovado pelo autor em:** 25/11/2026

## Parte 2 — Encerramento

- **Encerrada em:**

### O que foi feito
Estrutura do repositório, pacote `epi`, notebook `00_ambiente`, testes, licença e
publicação no GitHub. Falta a execução no Colab para encerrar a fase.

### O que mudou em relação ao detalhamento (e por quê)
- Repositório público: o Colab clona sem token do GitHub.
- Segredos do Kaggle e uso do Google Drive passam para a F2 e a F3; o notebook `00`
  grava tudo em `runs/`.
- Criada `configurar_ultralytics()`: sem ela, o Ultralytics salvava bases fora do projeto.

### Decisões tomadas
| Decisão | Motivo | Permanente? (copiar para o plano) |
|---|---|---|

### Resultados
Execução local em 25/11/2026: Python 3.13.18, PyTorch 2.14.1+cu130, torchvision 0.29.1,
Ultralytics 8.4.219, RTX 4060 Ti. Dois testes passam; o notebook `00` executa do início ao
fim, também a partir de um clone limpo do repositório (usando o ambiente local).

### Pendências
- Executar `00_ambiente.ipynb` no Google Colab com GPU e registrar a saída.

### O que a próxima fase precisa saber
- Chamar `configurar_ultralytics()` antes de importar `ultralytics`; bases ficam em `data/`.
- Os notebooks são gravados sem saídas (`nbstripout`).

### Atualizações feitas
- [ ] Status no plano
- [ ] Decisões acumuladas no plano
- [ ] Metodologia (somente se houve mudança aprovada de método)
