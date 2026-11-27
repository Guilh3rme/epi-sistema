# Fase 4 — Avaliação e teste entre bases

## Parte 1 — Abertura e detalhamento

- **Aberta em:** 27/11/2026
- **Lido antes de detalhar:** plano v1.3, metodologia v1.3 e encerramentos de
  [fase-2.md](fase-2.md) e [fase-3.md](fase-3.md). Comportamento da validação conferido no
  código do Ultralytics 8.4.219 instalado.

### Herdado das fases anteriores
- Pesos de E1 e E2 (SHA-256 na [fase-3.md](fase-3.md)) em `runs/pesos/` e no Drive.
- Divisões congeladas; testes ainda não usados. Pacote das bases no Drive em
  `epi-sistema/f2/bases_preparadas.tar`; o computador local não tem a cópia completa.
- Vistas de teste: `configs/base_e1.yaml`, `base_e1_cppe.yaml`, `base_e2.yaml` e
  `base_e2_sh17.yaml`, na chave `test`.
- Teste interno de E1 provavelmente otimista (92% das imagens de fotógrafos do treino).

### Objetivo e escopo
Responder à PQ1: métricas por classe e por tamanho de objeto de E1 e E2, no teste interno e
no teste entre bases, executadas uma única vez.

**Fica fora:** novo treino; ajuste de qualquer parâmetro depois de ver um resultado de
teste; escolha do modelo do sistema e dos limiares do ponto de controle (F6); outros
modelos, resoluções, sementes e intervalos de confiança (F8).

| Execução | Peso | Imagens |
|---|---|---|
| E1, interno | E1 | `val_files.txt` do SH17 (1.620) |
| E1, entre bases | E1 | Construction-PPE completa (1.000) |
| E2, interno | E2 | `val` da Construction-PPE (200) |
| E2, entre bases | E2 | `val_files.txt` do SH17 (1.620) |

### Respostas às perguntas da fase

**Limiar de confiança (decisão):** o mAP não depende de limiar. Já P, R e F1 do Ultralytics
são medidos na confiança de F1 máximo **do próprio conjunto avaliado**, o que no teste
seria escolher o limiar no teste. Por isso:

- um limiar por experimento, escolhido **no desenvolvimento**: o de maior F1 médio de
  pessoa, capacete, colete e luvas (curva suavizada como no Ultralytics);
- o mesmo limiar vale para o teste interno e o entre bases daquele experimento;
- no teste, P, R e F1 são lidos das curvas no ponto desse limiar;
- os dois valores são gravados em `configs/avaliacao.yaml` e publicados **antes** do teste;
- o limiar serve só para relatar a detecção: não define `conf_min` nem `conf_ok` do ponto
  de controle, calibrados na F6.

**Limiares de IoU (decisão):** pareamento com a marcação em 0,5 para P, R, F1 e mAP@0,5, e
de 0,5 a 0,95 para mAP@0,5:0,95. Supressão de caixas repetidas em 0,7, o padrão da
validação, sem busca.

**Demais argumentos (decisão):** o teste usa a mesma chamada da validação no desenvolvimento
da F3 (`treino.validar`, com um parâmetro de divisão): `imgsz` 640 e `batch` 16, com o
padrão da versão no restante (confiança mínima 0,001, até 300 detecções por imagem, `nms` e
precisão). A única adição é `save_json`, que grava as predições.

**Faixas de tamanho (decisão):** por área relativa, medida como a área da caixa depois de a
imagem ter o lado maior levado a 640 px, a entrada do modelo. Cortes do COCO: **pequeno**
abaixo de 32² px, **médio** de 32² a 96², **grande** acima de 96². Não dependem da resolução
nativa e ficam fixados antes de qualquer resultado.

O cálculo por tamanho do próprio Ultralytics usa pixels da cópia preparada, cujo lado maior
varia (até 1280, sem ampliar), e por isso não é usado. As predições salvas na execução e as
caixas marcadas são reescaladas **imagem a imagem** para lado maior 640, o campo `area` da
marcação é recalculado nessa escala e a avaliação é feita com `faster-coco-eval`, com até
300 detecções por imagem (o padrão do COCO é 100). **Dependência nova:**
`faster-coco-eval==1.8.1` no `pyproject.toml`, com pacote pronto para Windows (Python 3.13)
e Linux.

Por faixa não há P nem F1: um falso positivo não corresponde a uma caixa marcada, e
atribuir-lhe uma faixa seria uma regra à parte. Mostram-se R no limiar fixado e AP na regra
do COCO, em que caixas marcadas de outra faixa são ignoradas, assim como detecções sem par
fora da faixa. O AP por tamanho usa a interpolação do COCO e não é somável nem diretamente
comparável com o AP geral do Ultralytics; as tabelas dizem isso.

**Formato das tabelas (decisão):** CSV e Markdown em `experimentos/f4/`.

| Tabela | Linhas | Colunas |
|---|---|---|
| Por classe | uma por execução e classe | imagens, caixas, P, R, F1, mAP@0,5, mAP@0,5:0,95 |
| Por tamanho | execução, classe e faixa | caixas, R, AP@0,5, AP@0,5:0,95 |

- Resultado principal: pessoa, capacete, colete e luvas, com a média simples das quatro.
- Calçado e óculos em bloco separado, **exploratórios**, sem entrar na média.
- Classes `no_*`: só no teste interno de E2, em bloco separado. A linha "all" do
  Ultralytics não é usada, pois mistura classes diferentes conforme a execução.
- Faixa sem caixas aparece como "—"; a contagem de caixas acompanha cada valor.
- E1 e E2 só são comparados no mesmo conjunto: `val` do SH17 (E1 interno, E2 entre bases).
  O E2 não treinou no SH17 e não tem a sobreposição de fotógrafos que afeta o E1.

**Execução única (decisão):**
1. Todo o código é testado no desenvolvimento, sem abrir a divisão `test`.
2. Limiares e código são publicados; o notebook recusa código modificado.
3. Antes do teste, o desenvolvimento é reproduzido na mesma GPU e o mAP@0,5 e o
   mAP@0,5:0,95 (geral e por classe) são comparados com os da F3 (tolerância de 0,005);
   divergência interrompe. A diferença de P e R é só registrada (ver abaixo).
4. Cada execução grava predições, curvas e um marcador por último. Com o marcador, não
   roda de novo. Uma queda antes do marcador permite repetir, com registro.
5. Erro só na montagem das tabelas é corrigido a partir das predições salvas, sem nova
   inferência, e registrado.

**Ajuste durante o desenvolvimento (27/11/2026):** a tolerância valia para todas as
métricas. No desenvolvimento do E2 repetido no computador local (RTX 4060 Ti, cópia das
imagens preparada localmente), o mAP diferiu no máximo 0,0012 e a precisão de `no_helmet`,
0,012: P e R são lidos no ponto de F1 máximo, que se desloca com variações mínimas. A
tolerância passou a valer só para o mAP. Nenhum teste foi lido.

**Onde rodar (decisão):** Colab com Tesla T4, restaurando o pacote do Drive.

### Entregas e critério de pronto
| Entrega | Onde fica | Como verificar |
|---|---|---|
| Avaliação, faixas e tabelas | `src/epi/avaliacao.py`, `tests/test_avaliacao.py` | testes com caixas sintéticas: limiar, cortes de faixa, faixa vazia, médias, marcador |
| Limiares e argumentos | `configs/avaliacao.yaml` | commit anterior ao teste |
| Notebook | `notebooks/04_avaliacao.ipynb` | executa local no desenvolvimento e no Colab completo |
| Predições e curvas | Drive `epi-sistema/f4/` e cópia local, fora do git | SHA-256 no registro |
| Registro e tabelas | `experimentos/f4/` | commit, `codigo_modificado: false`, GPU, versões, SHA-256 dos pesos e das listas, limiares |

**Critério de pronto:** as quatro execuções concluídas uma vez, tabelas publicadas com as
classes principais separadas das exploratórias e revisadas pelo autor.

### Passos
1. `avaliacao.py`, testes e dependência fixada.
2. Notebook `04`; execução local em modo curto (`data/preparado_curto`, sem valor de
   resultado). Commit e envio, com autorização do autor.
3. Colab, só desenvolvimento: restaurar, conferir pesos e listas, reproduzir as métricas da
   F3 e escolher os limiares.
4. Gravar os limiares em `configs/avaliacao.yaml`; commit e envio.
5. Colab: rodar o teste.
6. Baixar os resultados, publicar tabelas e encerrar a fase.

### Riscos e plano B
- Sessão do Colab cai: repetir só as execuções sem marcador.
- Colab sem GPU disponível: esperar; rodar na máquina local exige baixar o pacote e é
  mudança a registrar.
- Poucas caixas em alguma faixa (colete, capacete): mostrar a contagem e não concluir sobre
  ela.
- `faster-coco-eval` falha ao instalar ou importar: parar e propor alternativa antes do
  teste.
- A metodologia pede P, R, F1 e AP "por classe e por tamanho"; por faixa saem só R e AP.
  Aprovar este detalhamento inclui registrar essa precisão na metodologia (v1.4).

### Observação para a F6
O ponto de controle usa o padrão de predição da versão, com supressão de caixas repetidas
em 0,5, diferente do 0,7 da avaliação. Definir o valor do sistema na calibração.

### Aprovação
- **Aprovado pelo usuário em:** 27/11/2026, com a dependência `faster-coco-eval` e commit e
  push a cada etapa testada.

## Parte 2 — Encerramento

Fase não encerrada.
