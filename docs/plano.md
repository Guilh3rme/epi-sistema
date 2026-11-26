# Plano faseado (v1.1)

Segue a [metodologia v1.1](metodologia.md), que é a fonte de verdade. Este plano é
**macro**: tarefas, arquivos, parâmetros e entregáveis de cada fase são definidos só quando
a fase começa, em `docs/fases/fase-N.md` (modelo em [fases/_modelo.md](fases/_modelo.md)).

## Protocolo de passagem entre fases

**Ao abrir uma fase**
1. Ler este plano (status e decisões acumuladas) e a metodologia.
2. Ler o **encerramento da fase anterior** (e de qualquer fase da qual esta dependa).
3. Criar `docs/fases/fase-N.md` a partir do modelo e preencher a abertura: escopo,
   respostas às perguntas da fase, entregas, critério de pronto, riscos.
4. **Obter aprovação do autor** do detalhamento. Só então desenvolver.

**Ao encerrar uma fase**
1. Preencher o encerramento: o que foi feito, o que mudou em relação ao detalhamento,
   decisões e motivos, resultados, pendências, o que a próxima fase precisa saber.
2. Copiar as decisões permanentes para "Decisões acumuladas" e atualizar o status abaixo.

**Mudança de escopo ou de método** (bases, divisão, métricas, cenários, tamanho do teste,
participante, uso do teste) atualiza este plano **e** a metodologia, com nova versão, e
exige aprovação antes de ser aplicada. Fases em paralelo registram encerramento
separadamente; quem depende delas lê os dois.

## Status das fases

A prioridade é o sistema (F1 a F6). Documentos e artigo vêm depois dele.

| Fase | Nome | Depende de | Status |
|---|---|---|---|
| F0 | Documento do projeto | sistema | depois do sistema |
| F1 | Repositório e ambiente | metodologia v1.1 | em desenvolvimento |
| F2 | Preparação das bases | F1 | não iniciada |
| F3 | Treino E1 e E2 | F2 | não iniciada |
| F4 | Avaliação e teste entre bases | F3 | não iniciada |
| F5 | Ponto de controle e CLP simulado | F1; modelo de F3 para integrar | não iniciada |
| F6 | Gravações de calibração e de teste | F4, F5 | não iniciada |
| F7 | Consolidação dos resultados | F6 | não iniciada |
| F8 | Extensão: comparações e intervalos de confiança | F7 | não iniciada |
| F9 | Dissertação | F7, F8 | não iniciada |
| F10 | Artigo | F8 | não iniciada |

Mínimo viável: F1 a F7.

## Fases

### F0 — Documento do projeto
- **Objetivo:** apresentar o projeto no modelo do programa, a partir da metodologia e dos
  resultados do sistema.
- **Passa adiante quando:** documento gerado e revisado pelo autor.
- **Perguntas do detalhamento:** qual modelo e formato? Quais resultados entram? Quem
  revisa?

### F1 — Repositório e ambiente
- **Objetivo:** estrutura do repositório com notebooks, ambiente no Colab e ambiente local
  de demonstração, mantendo imagens, pesos e credenciais fora do git.
- **Passa adiante quando:** um notebook mínimo roda no Colab com GPU e o ambiente local
  roda inferência na RTX 4060 Ti.
- **Perguntas:** estrutura de pastas e convenção de notebooks? Onde ficam dados e pesos?
  Versões de Ultralytics e PyTorch? Qual Python local? Como registrar sementes, versões e
  hardware? Licença do código?

### F2 — Preparação das bases
- **Objetivo:** SH17 e Construction-PPE prontas para E1 e E2: cópia redimensionada do
  SH17, mapeamento de classes, divisões treino/desenvolvimento/testes fixadas antes de
  qualquer aumento, busca de duplicatas dentro e entre as bases.
- **Passa adiante quando:** divisões congeladas e registradas, contagens conferidas
  (inclusive as do `val_files.txt`), relatório de duplicatas resolvido.
- **Perguntas:** resolução da cópia do SH17? Onde guardar dados no Colab? Credenciais do
  Kaggle? Fração e critério da divisão de desenvolvimento? Limiar do *hash* perceptual e o
  que fazer com duplicatas? Tratamento das classes não comuns e das `no_*`? Como registrar
  a proporção de imagens geradas na Construction-PPE? Pedir aos autores do SH17 as
  marcações extras?

### F3 — Treino E1 e E2
- **Objetivo:** treinar YOLO26n em cada base, nas mesmas condições, com parada antecipada
  e sementes registradas.
- **Passa adiante quando:** pesos de E1 e E2 salvos fora do git, curvas e configuração
  registradas, nenhuma decisão tomada com os conjuntos de teste.
- **Perguntas:** hiperparâmetros e aumentos comuns? Épocas e paciência? Orçamento de GPU do
  Colab e plano se acabar? Onde salvar os pesos? Três sementes são viáveis?

### F4 — Avaliação e teste entre bases
- **Objetivo:** responder à PQ1: métricas por classe e por tamanho de objeto no teste
  interno e no teste entre bases, executados uma vez.
- **Passa adiante quando:** tabelas de E1 e E2 registradas, separando classes principais
  (pessoa, capacete, colete, luvas) das exploratórias (calçado, óculos).
- **Perguntas:** limiares de confiança e IoU fixados em desenvolvimento? Como definir
  faixas de tamanho? Formato das tabelas?

### F5 — Ponto de controle e CLP simulado (paralela a F2–F4)
- **Objetivo:** fluxo câmera → pessoa e pose → detecção → associação (A e B) → votação em N
  quadros → regra da área → decisão → Modbus TCP para CLP simulado e registro do evento,
  sem armazenar imagens em operação.
- **Passa adiante quando:** o fluxo roda localmente de ponta a ponta com um modelo de F3,
  produz as três decisões e mede tempo até a decisão, latência até o CLP e FPS.
- **Perguntas:** modelo de pose? Definição das regiões do corpo na associação B? Regras de
  `INDETERMINADO`? Simulador de CLP e mapa de registradores? Crachá/RFID simulado entra no
  mínimo viável? Como medir latência? Pode começar com modelo pré-treinado enquanto F3 não
  termina?

### F6 — Gravações de calibração e de teste
- **Objetivo:** responder à PQ2 e à PQ3: calibração (~30 passagens) em um dia para escolher
  o modelo do sistema, limiares e N, e para ajustar as associações A e B; congelar tudo;
  teste (100 passagens) em outro dia, executado uma única vez com **A e B**. Vídeos brutos
  ficam no computador do autor, fora do git.
- **Depende também de:** definição, com o orientador, sobre o Comitê de Ética e das metas
  de TLI e de indeterminados, antes da calibração; EPIs de segurança disponíveis.
- **Passa adiante quando:** configuração congelada e registrada antes do teste; matriz de
  passagens preenchida com TLI, TNC e taxas de indeterminados.
- **Perguntas:** roteiro, ordem e distribuição de luz e distância? Como marcar o início e o
  fim de cada passagem e a situação real?

### F7 — Consolidação dos resultados
- **Objetivo:** consolidar os resultados de F4 e F6, com vídeo da demonstração local.
- **Passa adiante quando:** tabelas e demonstração revisadas pelo autor.
- **Perguntas:** que resultados e formato? Alguma recomendação muda o método?

### F8 — Extensão: comparações e intervalos de confiança
- **Objetivo:** YOLOv8n e YOLO26s, resolução 640 contra uma maior, três sementes no modelo
  final se possível, e intervalos de Clopper-Pearson.
- **Passa adiante quando:** tabelas comparativas fechadas.
- **Perguntas:** quais comparações e em que ordem? Comparações sobre as gravações de teste
  são só avaliação (nunca escolha do modelo)? Precisa de nova sessão de teste?

### F9 — Dissertação
- **Objetivo:** escrever a dissertação.
- **Passa adiante quando:** versão revisada pelo autor.
- **Perguntas:** qual modelo? Qual o formato do produto do mestrado profissional?

### F10 — Artigo
- **Objetivo:** reescrever o artigo com os novos resultados.
- **Passa adiante quando:** versão revisada pelo autor.
- **Perguntas:** qual veículo e recorte de resultados? Como as licenças das bases afetam o
  que se publica?

## Riscos técnicos

1. **Volume:** o SH17 (14,2 GB) e as cotas de GPU do Colab afetam F2 e F3.
2. **Ambiente local:** usar CPython 3.13.
3. **Gravações:** dependem das definições de ética e de metas antes da calibração.

## Decisões acumuladas

| Data | Fase | Decisão | Motivo |
|---|---|---|---|
| 25/11/2026 | — | Plano macro com fases F0–F10 | Pedido do autor |
| 25/11/2026 | F0 | Documentos e artigo ficam para depois do sistema | Prioridade do autor |
| 25/11/2026 | F6 | Associações A e B congeladas na calibração e avaliadas no mesmo teste único | Metodologia v1.1 |
| 25/11/2026 | F6 | Vídeos brutos de pesquisa ficam locais e fora do git; "sem armazenar imagens" vale para o sistema em operação | Metodologia v1.1 |
| 25/11/2026 | F1 | Repositório público no GitHub, código sob AGPL-3.0 | Escolha do autor; licença do Ultralytics |
