# Fase 2 — Preparação das bases

## Parte 1 — Abertura e detalhamento

- **Aberta em:** 25/11/2026
- **Lido antes de detalhar:** plano (status e decisões), metodologia v1.1 e encerramento da
  fase-1.md.

### Herdado das fases anteriores
Ambiente validado no computador local e no Colab (Ultralytics 8.4.219); chamar
`configurar_ultralytics()` antes de importar `ultralytics`; bases em `data/`, fora do git;
notebooks sem saídas; o Colab clona o repositório público. Credenciais do Kaggle e Google
Drive ficaram para depois da F1. Pendências: nada.

### Objetivo e escopo
Deixar SH17 e Construction-PPE prontas para E1 e E2: download conferido, cópia
redimensionada, classes mapeadas, duplicatas resolvidas e divisões congeladas antes de
qualquer aumento. Fica fora: treino, métricas, escolha de limiares, aumento de dados,
mistura das bases no treino e correção manual de rótulos.

### Fatos conferidos em 25/11/2026

| Item | Constatado | Fonte |
|---|---|---|
| SH17 no Kaggle | versão 1, CC BY-NC-SA 4.0; 14.237.350.242 bytes descompactados; zip de 14.096.291.832 bytes | [1] |
| Conteúdo | 8.099 arquivos em cada uma de `images/`, `labels/` (YOLO), `voc_labels/` e `meta-data/` (dados do Pexels), mais as duas listas | [1] |
| `train_files.txt` / `val_files.txt` | 6.479 e 1.620 nomes de imagem, sem caminho, com CRLF; sem sobreposição; a união são as 8.099 imagens | [1] |
| Download | arquivo completo (só os primeiros bytes e o índice) e arquivos avulsos responderam sem credenciais, fora do Colab; avulsos usam `%2F` no caminho e podem vir em zip | [1] |
| Ids do SH17 | `sh17.yaml` confere com o VOC em 60 arquivos (664 caixas, todas pareadas); o id 6, `foot`, em um arquivo à parte | [1][2] |
| Instâncias no artigo | pessoa 13.802; capacete 927; óculos 1.945; luvas 2.790; calçado 4.560; colete 530 (a recontar) | [3] |
| Imagens do SH17 | lado maior de 3.500 a 8.192 px e média de 1,9 MB em amostra de 40; nenhuma com rotação EXIF | [1] |
| Construction-PPE | zip de 647.666.037 bytes; 800 `train` e 200 `val`, sem teste; 14.863 caixas (11.902 e 2.961) segundo o manifesto; CC BY 4.0 | [4][5] |
| Origem das imagens | a página declara imagens coletadas e geradas e anotação manual e automática, sem proporção e sem campo de origem no manifesto | [4][5] |

A numeração de 1 a 17 do README do SH17 **não** é a dos ids; valem os do `sh17.yaml` [2].
O artigo e o repositório falam em *train/test* (`test_files.txt`), mas a base publica
`val_files.txt` [1][2][3]. O arquivo atual da Construction-PPE é de 03/11/2026 [5]; as
versões serão fixadas por impressão digital:

- zip da Construction-PPE, SHA-256: `e5817aa9c73f33e525d76ca6bc6af4e4eac9dcea9bcfef1cfd1a71c09fbc1232`
- zip do SH17, MD5 informado pelo servidor: `2e9eb8708de4d1ca84246ed230ca66f8`
- `train_files.txt`, SHA-256: `6599473388c5b2ca68e909a9ab557a36bd829ab562f4852486caacc7ed3aaad9`
- `val_files.txt`, SHA-256: `b8c7936d6842e8db04423aeb10426240dddb00c28efb90a084c3eae3d0ce6121`

### Respostas às perguntas da fase
**Resolução da cópia (decisão):** lado maior de **1280 px**, sem ampliar, JPEG com
qualidade 95 (PNG continua PNG), orientação EXIF aplicada e nomes mantidos. Estimativa para
o SH17, pela amostra de 40 imagens: 2,7 GB (em 640 seriam 0,8 GB, mas a comparação com
resolução maior na F8 exigiria refazer tudo). A mesma regra vale para a Construction-PPE.
Rótulos YOLO são normalizados e não mudam com o redimensionamento; as dimensões originais
ficam registradas para as faixas de tamanho da F4.

**Onde guardar os dados no Colab:** no disco da máquina virtual, em `data/`, sem GPU. O zip
do SH17 é baixado por HTTP direto do endereço de download [1], lido sem extrair e depois
apagado: pico de cerca de 17 GB, contra 28 GB mais a cópia com `kagglehub`, que extrai antes
de apagar o zip [6]. Com a Construction-PPE e o pacote para o Drive, perto de 22 GB; o
notebook exige 25 GB livres. Disco da VM **a confirmar** (o Colab não publica limites fixos
[7]). **(decisão)** Salvar no Drive um único `.tar` com as bases preparadas (cerca de 3–4 GB,
a confirmar), para a F3 não repetir o download; o FAQ recomenda arquivos agrupados [7]. Isso
amplia a regra da F1 (só pesos e resultados no Drive).

**Credenciais do Kaggle:** não são necessárias, a confirmar a partir do Colab. Se o acesso
anônimo falhar, o plano B é o `kagglehub` com o segredo `KAGGLE_API_TOKEN` criado pelo autor
no Colab [6], o que eleva o pico de disco para cerca de 28 GB ou mais; usar o token no
download direto é **a confirmar**. O valor não é escrito no notebook nem no repositório.

**Divisão de desenvolvimento (decisão):** **15%** do treino de cada base (antes de remover
duplicatas: cerca de 972 de 6.479 no SH17 e 120 de 800 na Construction-PPE). Sorteio com
semente 0, depois da remoção de duplicatas; grupos de quase-duplicatas ficam inteiros de um
lado; estratificação pela classe do projeto mais rara presente em cada grupo e, na
Construction-PPE, também pela família do nome do arquivo. Testes não mudam:
`val_files.txt` (1.620), `val` da Construction-PPE (200) e a Construction-PPE completa.

**Limiar do *hash* perceptual e duplicatas (decisão):** arquivos idênticos por SHA-256 e
pHash de 64 bits sobre a cópia preparada, em todos os pares das 9.099 imagens. Distância de
Hamming até 4: duplicata; de 5 a 10: revisão visual do autor; acima de 10: distinta. O
histograma é inspecionado e o limiar final registrado antes de dividir. Implementação com
`cv2.dct`, do OpenCV já instalado, sem dependência nova.

**(decisão)** A imagem sai das listas de treino/desenvolvimento do experimento cujo teste
(interno ou entre bases) contém o par dela, antes de separar o desenvolvimento. Outros pares
são mantidos, agrupados e registrados. Remoções e desenvolvimento são **exclusões nas listas
de cada experimento**: as pastas das bases nunca são podadas, nenhum teste perde imagem e o
teste entre bases de E1 continua com as 1.000, incluindo as retiradas do treino de E2.

**Classes não comuns e `no_*` (decisão):** rótulos convertidos para ids únicos do projeto.

| Id | Classe | SH17 (id nativo) | Construction-PPE (id nativo) |
|---|---|---|---|
| 0 | pessoa | `person` (0) | `Person` (5) |
| 1 | capacete | `helmet` (10) | `helmet` (0) |
| 2 | óculos | `glasses` (8) | `goggles` (4) |
| 3 | luvas | `gloves` (9) | `gloves` (1) |
| 4 | calçado | `shoes` (14) | `boots` (3) |
| 5 | colete | `safety-vest` (16) | `vest` (2) |
| 6–10 | `no_helmet`, `no_goggle`, `no_gloves`, `no_boots`, `no_vest` | não há | 6, 7, 8, 9, 11 |

- E1 treina **somente as 6 classes** (0–5); as outras 11 do SH17 são descartadas. Perde-se a
  comparação direta com os resultados publicados, obtidos com 17 classes [2].
- E2 treina 11 classes (0–10). `forklift` não é classe do projeto: caixas descartadas.
- Imagens que ficarem sem caixas são mantidas como fundo.
- Para E1 testar na Construction-PPE há uma segunda vista dela só com os ids 0–5.
- O teste de E2 no SH17 declara os 11 nomes, com os ids 6–10 sem rótulos.

**Proporção de imagens geradas:** registrar as contagens por família de nome: 707
`ultralytics-ppe-nov2026-*` (566 `train`, 141 `val`) e 293 `source-*` (234 e 59). As 707
têm cinco formatos fixos em quantidades quase iguais, indício de geração; **a confirmar**,
pois a fonte não identifica a origem. O autor inspeciona uma amostra de 50 por família e o
resultado é declarado como estimativa, não como metadado da base.

**Marcações extras do SH17 (decisão):** o artigo descreve atributos de uso (`on`/`off`) e de
tipo (segurança ou não, para óculos e calçado) em uma versão estendida [3]; não há pasta nem
campo para eles na base pública ou no repositório [1][2]. Recomendação: o autor pede por
*issue* ou pelo contato do README, sem bloquear a F2. Se chegarem, usá-las será mudança de
método.

### Entregas e critério de pronto
| Entrega | Onde fica | Como verificar: assistente | Só o autor |
|---|---|---|---|
| Preparação (download, cópia, mapa, duplicatas, divisão) | `src/epi/dados.py`, `tests/test_dados.py` | testes com dados sintéticos: mapa de ids, CRLF, redimensionamento, pHash, regras de remoção e divisão repetível | — |
| Notebook | `notebooks/01_dados.ipynb` | execução local com a Construction-PPE e uma amostra do SH17 | executar completo no Colab e enviar as saídas |
| Parâmetros e impressões digitais | `configs/dados.yaml` | conferidos contra as fontes | aprovar o limiar final |
| Divisões congeladas (só nomes) | `configs/divisoes/*.txt` | sem sobreposição; somas e SHA-256; testes completos | trazer os arquivos gerados no Colab |
| YAMLs de E1, E2 e testes entre bases | `configs/` | `val:` aponta para desenvolvimento e `test:` para o teste; o `sh17.yaml` original, com `val:` no teste, não é usado; `model.val` em poucas imagens de cada vista de teste, com um modelo qualquer de mesmo número de classes | — |
| Relatório de contagens e manifesto (nome, dimensões, hashes) | `configs/divisoes/` | 6.479, 1.620, 800, 200; 8.099 rótulos; classes do SH17 contra [3] e total de caixas contra [5] | conferir |
| Relatório de duplicatas | `configs/divisoes/duplicatas.csv` | nenhum par pendente entre treino/desenvolvimento e teste | revisar visualmente os pares de 5 a 10 |
| Bases preparadas | `data/` e Drive, fora do git | contagens e leitura de todas as imagens | montar o Drive e conferir o espaço livre |

**Critério de pronto:** divisões congeladas e registradas e nenhum par de duplicatas sem
decisão. As contagens da origem (6.479, 1.620, 800 e 200) são conferidas antes das remoções,
e divergência interrompe a fase; as contagens após as remoções são registradas.

### Passos
1. `configs/dados.yaml`, `dados.py` e testes.
2. Notebook `01`; execução local com a Construction-PPE e a amostra do SH17.
3. Commit e envio ao GitHub, com autorização do autor (o Colab usa o código publicado).
4. Colab: conferir disco, baixar, validar impressões digitais, preparar as duas bases (com
   conferência de dimensões contra o VOC no SH17) e contar.
5. Calcular os *hashes*; autor revisa os pares; fixar o limiar e as remoções.
6. Gerar e congelar as divisões; salvar o pacote no Drive.
7. Registrar listas e relatórios no repositório; encerrar a fase.

### Riscos e plano B
- Disco insuficiente: ler as imagens do SH17 remotamente, uma a uma ou por trechos do zip.
- Sessão do Colab cai: retomar o download parcial; se a VM for apagada, recomeçar.
- Kaggle recusa acesso anônimo: token (ver acima). Último recurso: script de download pelo
  Pexels [2], registrando falhas e diferenças.
- Pares demais para revisar: reduzir a faixa de revisão e registrar a mudança.
- pHash não detecta recortes nem fotos da mesma sessão (possível entre treino e `val` do
  SH17): contar fotógrafos em comum no `meta-data`, sem publicar nomes, e declarar a
  limitação, sem alterar o teste.

### Fontes
1. https://www.kaggle.com/datasets/mugheesahmad/sh17-dataset-for-ppe-detection (API em
   `/api/v1/datasets/{view,list,download}/mugheesahmad/sh17-dataset-for-ppe-detection`)
2. https://github.com/ahmadmughees/SH17dataset
3. https://arxiv.org/html/2407.04590
4. https://docs.ultralytics.com/datasets/detect/construction-ppe
5. https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip
   (digest: https://api.github.com/repos/ultralytics/assets/releases/tags/v0.0.0)
6. https://github.com/Kaggle/kagglehub (arquivo `src/kagglehub/http_resolver.py`)
7. https://research.google.com/colaboratory/faq.html

### Aprovação
- **Aprovado pelo autor em:** 25/11/2026, com as recomendações marcadas como decisão e a
  preparação no Google Colab.

## Parte 2 — Encerramento
- **Encerrada em:** —
### O que foi feito
Nada.
### O que mudou em relação ao detalhamento (e por quê)
Nada.
### Decisões tomadas
| Decisão | Motivo | Permanente? (copiar para o plano) |
|---|---|---|
### Resultados
Nada.
### Pendências
Nada.
### O que a próxima fase precisa saber
Nada.
### Atualizações feitas
- [ ] Status no plano
- [ ] Decisões acumuladas no plano
- [ ] Metodologia (somente se houve mudança aprovada de método)
