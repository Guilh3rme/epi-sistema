# Fase 3 — Treino E1 e E2

## Parte 1 — Abertura e detalhamento

- **Aberta em:** 26/11/2026
- **Lido antes de detalhar:** plano v1.3, metodologia v1.3 e encerramentos de
  [fase-1.md](fase-1.md) e [fase-2.md](fase-2.md). Comportamento do treino conferido no
  código do Ultralytics 8.4.219 instalado.

### Herdado das fases anteriores
- Divisões congeladas em `configs/divisoes/`; bases preparadas no Drive em
  `epi-sistema/f2/bases_preparadas.tar`.
- Treino: E1 com 5.453 imagens (desenvolvimento 962), E2 com 677 (120).
- Poucas caixas de colete (384) e capacete (673) no treino de E1.
- No SH17, 92% das imagens de `val` são de fotógrafos presentes no treino. O
  desenvolvimento de E1 sai do treino do SH17, então suas métricas também tendem a ser
  otimistas.
- Colab usado pela extensão do VS Code, com GPU T4 na F1 e 2 CPUs na F2.

### Objetivo e escopo
Treinar o YOLO26n em cada base (E1: SH17; E2: Construction-PPE) com a mesma configuração,
parada antecipada pelo desenvolvimento e semente registrada.

**Fica fora:** qualquer avaliação nos testes (F4); YOLOv8n, YOLO26s, resolução maior e três
sementes (F8); busca de hiperparâmetros; reamostragem de classes; integração ao ponto de
controle (F5); escolha do modelo do sistema (F6).

### Respostas às perguntas da fase

**Hiperparâmetros e aumentos (decisão):** um único `configs/treino.yaml` para E1 e E2.

| Parâmetro | Valor | Motivo |
|---|---|---|
| peso inicial | `yolo26n.pt` (COCO), SHA-256 `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef` | conferido antes de treinar |
| `imgsz` | 640 | metodologia |
| `batch` | 16, fixo | o ajuste automático depende da GPU |
| `optimizer`, `lr0` | AdamW, 0,001 | ver abaixo |
| `momentum`, `warmup_bias_lr` | 0,9 e 0,0 | os mesmos do modo automático |
| `seed`, `deterministic` | 0, sim | reprodutibilidade |
| `cache` | `disk` | o cache em memória não é determinístico |
| aumentos e demais | padrão da versão 8.4.219 | iguais nos dois, sem busca |

O otimizador não fica em `auto`: ele escolhe pelo número de iterações e de classes. Com 100
épocas, daria AdamW com `lr0` 0,001 em E1 (6 classes) e 0,00067 em E2 (11 classes), e
trocaria para MuSGD em E1 acima de 116 épocas. Fixar AdamW com 0,001 iguala as condições.

**Épocas e paciência (decisão):** 100 épocas e paciência 30, nos dois. O padrão de
paciência (100) não pararia antes do fim. A paciência é folgada porque o desenvolvimento de
E2 tem só 120 imagens. O melhor peso é o de maior mAP@0,5:0,95 no desenvolvimento. Se a
parada ocorrer antes da época 91, a fase final sem *mosaic* (`close_mosaic` = 10) não
acontece; isso será registrado.

**Orçamento de GPU (a medir):** não há estimativa confiável; com 2 CPUs, a leitura das
imagens deve limitar o treino. O tempo é medido nas primeiras épocas do treino real, que
não são descartadas. E2, menor, roda primeiro e valida o processo. **Plano B:** a RTX
4060 Ti local, com o mesmo pacote de bases. E1 e E2 usam o mesmo tipo de GPU, e um
experimento não troca de máquina no meio.

**Onde salvar os pesos (decisão):** o treino grava no disco da VM e copia para
`MyDrive/epi-sistema/f3/<exp>_yolo26n_s0/` a cada 5 épocas e no fim (`last.pt`, `best.pt`,
`results.csv`, `args.yaml`). Evita acumular versões no Drive, que tinha 6,7 GB livres.
Perde-se no máximo 4 épocas numa queda. Ao reexecutar o notebook:
- com marcador de conclusão: o experimento é pulado;
- só com `last.pt`: o treino é retomado;
- sem nenhum: o treino começa.

O marcador é gravado por último, depois das impressões digitais. Um treino retomado não é
idêntico, época a época, a um contínuo; as retomadas são registradas.

**Três sementes:** decisão na F8, com o tempo medido aqui.

### Proteção dos testes
- Treino somente com `configs/base_e1.yaml` e `base_e2.yaml`; o notebook confere que `val:`
  aponta para a lista de desenvolvimento e nunca avalia a divisão `test`.
- A validação final do Ultralytics usa o `best.pt` no desenvolvimento. Registram-se apenas
  métricas por classe do desenvolvimento, que não comparam E1 com E2 (conjuntos diferentes).
- Antes de treinar: impressões digitais das listas e número de imagens lidas igual ao das
  listas.

### Privacidade
O Ultralytics vem com o envio de estatísticas de uso ligado (`sync`): modelo, nome do YAML
de dados, CPU e GPU, épocas e métrica, para o Google Analytics, sem imagens. As execuções
anteriores provavelmente enviaram esses dados. **(decisão)** Desligar em
`configurar_ultralytics()`, com teste.

### Entregas e critério de pronto
| Entrega | Onde fica | Como verificar |
|---|---|---|
| Envio de estatísticas desligado | `src/epi/ambiente.py`, `tests/test_ambiente.py` | teste lê a configuração |
| Configuração de treino | `configs/treino.yaml` | um arquivo para os dois experimentos |
| Treino, retomada, cópia e registro | `src/epi/treino.py`, `tests/test_treino.py` | testes com modelo simulado; execução local curta, interrompida e retomada |
| Notebook | `notebooks/03_treino.ipynb` | executa local (modo curto) e no Colab |
| Pesos | Drive e cópia em `runs/pesos/`, fora do git | SHA-256 registrado |
| Registro | `experimentos/f3/<exp>/`: `resumo.json` e `results.csv` | commit, GPU, versões, semente, pesos, divisões, época do melhor peso, métricas de desenvolvimento; sem caminhos locais |

**Critério de pronto:** E1 e E2 concluídos com a mesma configuração e o mesmo tipo de GPU,
`best.pt` salvo e conferido, registros publicados e nenhuma avaliação nos testes.

### Passos
1. Desligar o envio de estatísticas; `treino.yaml`, `treino.py` e testes.
2. Notebook `03`; execução local curta com interrupção e retomada.
3. Commit e envio, com autorização do autor.
4. Colab: restaurar o pacote, conferir as listas e o peso inicial, treinar E2.
5. Medir o tempo por época, confirmar o plano e treinar E1.
6. Copiar os pesos, publicar os registros e encerrar a fase.

### Riscos e plano B
- Sessão ou cota do Colab cai: retomar do Drive; se a cota acabar, esperar ou usar o plano B.
  O computador precisa ficar ligado enquanto o VS Code mantém a sessão.
- Treino lento demais: medir, e então decidir entre esperar e usar a máquina local.
- Drive cheio: conferir o espaço antes de cada cópia.
- Parada antecipada ruidosa em E2: paciência 30; resultado declarado como é.
- Colete e capacete raros em E1: sem reamostragem; declarar a limitação.

### Aprovação
- **Aprovado pelo usuário em:** 26/11/2026, com commit e push a cada etapa testada.

## Parte 2 — Encerramento

- **Encerrada em:** 27/11/2026

### O que foi feito
Envio de estatísticas desligado, `treino.yaml`, módulo `treino`, testes e notebook `03`,
verificados no computador local (RTX 4060 Ti): interrupção no início da época 2, retomada a
partir da cópia em outra pasta, experimento concluído pulado, registro refeito sem treinar e
notebook completo em modo curto (números sem valor de resultado). `pytest`: 100 testes passam.

E2 e E1 treinados no Colab (Tesla T4, commit `148e05a`, sem código modificado), uma sessão
cada, sem retomada. Pesos e registros baixados; o SHA-256 dos dois `best.pt` confere com o
registrado no treino, os dois carregam localmente com as classes esperadas e a impressão do
relatório das divisões é a das divisões congeladas. Nenhum conjunto de teste foi usado.

### O que mudou em relação ao detalhamento (e por quê)
- O YAML do treino é gerado a partir de `base_<exp>.yaml` **sem a chave `test`**.
- Um treino caído depois da última época é tratado como encerrado (não há o que retomar).
- A troca de GPU é recusada na retomada; um recomeço do zero é permitido.
- O notebook para antes do E1 até a confirmação (`CONFIRMADO_E1`).
- Os registros ficam em `experimentos/f3/<exp>_yolo26n_s0/` (nome da execução, e não só
  `<exp>`), para caber outros modelos e sementes.

### Decisões tomadas
| Decisão | Motivo | Permanente? (copiar para o plano) |
|---|---|---|
| E1 e E2 treinados no Colab, em Tesla T4 (escolha do autor, após ver o tempo do E2) | cerca de 2,2 min por época no E1; o plano B local não foi necessário | sim: comparações e sementes da F8 usam o mesmo tipo de GPU |

### Resultados
| | E1 (SH17) | E2 (Construction-PPE) |
|---|---|---|
| imagens de treino / desenvolvimento | 5.453 / 962 | 677 / 120 |
| épocas executadas | 100 | 100 |
| melhor época | 78 | 73 |
| tempo de treino (aprox.) | 3,7 h | 20 min |

A parada antecipada não atuou: o limite de 100 épocas chegou antes de 30 épocas sem melhora.
A fase final sem *mosaic* (épocas 91 a 100) ocorreu nos dois, mas o melhor peso é anterior a
ela em ambos.

**Métricas no desenvolvimento** (`best.pt`). Não são resultado de teste, não comparam E1 com
E2 (conjuntos diferentes; a linha geral de E2 inclui as classes `no_*`) e não escolhem o
modelo do sistema.

| Classe | E1 P | E1 R | E1 mAP50 | E1 mAP50-95 | E2 P | E2 R | E2 mAP50 | E2 mAP50-95 |
|---|---|---|---|---|---|---|---|---|
| geral | 0,773 | 0,568 | 0,628 | 0,413 | 0,898 | 0,654 | 0,728 | 0,525 |
| pessoa | 0,908 | 0,876 | 0,917 | 0,737 | 0,961 | 0,846 | 0,911 | 0,731 |
| capacete | 0,763 | 0,635 | 0,722 | 0,499 | 0,927 | 0,693 | 0,760 | 0,583 |
| oculos | 0,818 | 0,570 | 0,628 | 0,343 | 0,961 | 0,964 | 0,981 | 0,670 |
| luvas | 0,702 | 0,413 | 0,485 | 0,285 | 0,965 | 0,707 | 0,812 | 0,626 |
| calcado | 0,769 | 0,499 | 0,578 | 0,318 | 0,900 | 0,665 | 0,749 | 0,558 |
| colete | 0,675 | 0,417 | 0,441 | 0,296 | 0,950 | 0,736 | 0,803 | 0,619 |
| no_helmet | — | — | — | — | 0,880 | 0,447 | 0,550 | 0,337 |
| no_goggle | — | — | — | — | 0,698 | 0,226 | 0,342 | 0,103 |
| no_gloves | — | — | — | — | 0,931 | 0,588 | 0,655 | 0,472 |
| no_boots | — | — | — | — | 0,826 | 0,714 | 0,766 | 0,595 |
| no_vest | — | — | — | — | 0,879 | 0,606 | 0,684 | 0,484 |

| Exp. | Peso (cópia local, fora do git) | SHA-256 do `best.pt` |
|---|---|---|
| E1 | `runs/pesos/e1_yolo26n_s0.pt` | `be63f8282c23856dd99c015998262b055f41462410fdae164a68820ad94203e4` |
| E2 | `runs/pesos/e2_yolo26n_s0.pt` | `3699e24794547147570842171f4b2a8f4137085071249ad89d846d77ad949d8b` |

Registros completos: [E1](../../experimentos/f3/e1_yolo26n_s0/resumo.json) e
[E2](../../experimentos/f3/e2_yolo26n_s0/resumo.json), com a curva em `results.csv`.

### Limitações
- No desenvolvimento de E1, `colete` e `luvas` têm as menores métricas; no de E2, as classes
  de ausência `no_goggle` e `no_helmet`.
- O desenvolvimento de E2 tem só 120 imagens e a base tem forte indício de imagens geradas;
  o de E1 sai do treino do SH17, com fotógrafos em comum. As métricas podem ser otimistas.
- Uma semente por experimento.

### Pendências
- Pergunta aberta ao autor: incluir na metodologia a limitação dos fotógrafos em comum no
  SH17 (registrada na F2).

### O que a próxima fase precisa saber
- Pesos: `runs/pesos/` no computador local e `MyDrive/epi-sistema/f3/` no Drive; o pacote das
  bases continua em `MyDrive/epi-sistema/f2/` (manter os dois).
- Os testes ainda não foram usados; a F4 os executa uma única vez.
- Para a F5 (nível 3): o modelo treinado usa os nomes do projeto (`pessoa`, `capacete`...),
  mas o mapa `classes` de `configs/ponto_de_controle.yaml` só traduz nomes em inglês, e
  `detectar()` descarta o que não está no mapa. Incluir esses nomes antes de integrar.

### Atualizações feitas
- [x] Status no plano
- [x] Decisões acumuladas no plano
- [ ] Metodologia (somente se houve mudança aprovada de método)
