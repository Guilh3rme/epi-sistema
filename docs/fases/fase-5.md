# Fase 5 — Ponto de controle e CLP simulado

## Parte 1 — Abertura e detalhamento

- **Aberta em:** 25/11/2026
- **Lido antes de detalhar:** plano v1.1, metodologia v1.1 e encerramento de
  [fase-1.md](fase-1.md).

### Herdado das fases anteriores
Pacote `epi` e `.venv` (Python 3.13, PyTorch 2.14.1+cu130, Ultralytics 8.4.219, RTX 4060 Ti);
`configurar_ultralytics()` antes de importar `ultralytics`; notebooks só orquestram, sem saídas
no git; `data/` e `runs/` fora do git. A F5 roda em paralelo à F2, ainda sem modelo do projeto.

### Objetivo e escopo
Fluxo local sobre vídeo gravado: pessoa e pose → detecção de EPI → associações A e B →
votação em N quadros → regra da área → decisão → Modbus TCP para CLP simulado e registro do
evento, sem gravar imagens, medindo tempo até a decisão, latência até o CLP e FPS.

**Fica fora:** treinar modelos (F3); escolher modelo, limiares, N ou regiões (F6); câmera ao
vivo; crachá/RFID; classes `no_*`; rastreamento de várias pessoas; interface gráfica; CLP
físico; supervisão de comunicação (*heartbeat*).

### Respostas às perguntas da fase

**Modelo de pose:** `yolo26n-pose.pt`, pré-treinado em COCO, 17 pontos
([documentação](https://docs.ultralytics.com/tasks/pose/), corrente, não fixada na 8.4.219):
nariz 0, olhos 1–2, orelhas 3–4, ombros 5–6, cotovelos 7–8, punhos 9–10, quadris 11–12,
joelhos 13–14, tornozelos 15–16. O pacote instalado lista o arquivo e expõe `keypoints.conf`
(N, 17). Download e execução local: a confirmar.

**Associações:** nas duas, o EPI é associado quando o **centro** da sua caixa cai na área:
a caixa da pessoa em A (sem pose) ou a região do corpo em B. O COCO não tem mãos nem pés,
que são estimados; os tamanhos são parâmetros.

| EPI | Pontos exigidos em B | Região em B |
|---|---|---|
| capacete | um de 0–4 e os ombros | do nível dos olhos ou orelhas para cima; escala pelos ombros |
| óculos | um olho | faixa em torno dos olhos visíveis, mesma escala |
| colete | 5, 6, 11, 12 | retângulo de ombros a quadris, com margem |
| luvas | cotovelo e punho, por lado | quadrado além do punho, prolongando o antebraço |
| calçado | joelho e tornozelo, por lado | quadrado do tornozelo para baixo, proporcional à canela |

Um ponto é visível se sua confiança alcança `conf_ponto` e ele está dentro do quadro.

**Estados por quadro** (`PRESENTE`, `AUSENTE` ou `INCERTO`):
- **Capacete, óculos e colete:** item associado com confiança ≥ `conf_ok` → presente; senão,
  item associado só entre `conf_min` e `conf_ok`, ou área não visível → incerto; nos demais
  casos → ausente.
- **Luvas e calçado exigem os dois lados** (metodologia v1.2), com uma detecção distinta
  para cada lado; a mesma detecção nunca vale pelos dois.
  - Em B, cada lado recebe um estado pela regra acima. Algum lado ausente → ausente; senão,
    algum incerto → incerto; senão, presente.
  - Em A, que não conhece lados: duas ou mais detecções com confiança ≥ `conf_ok` →
    presente; senão, duas ou mais com confiança ≥ `conf_min` → incerto; senão, ausente.
- Em A, a área é visível sempre que há pessoa válida.
- Se as bases anotarem um par como uma única caixa (a conferir na F2), o item será dado como
  ausente; o que persistir é resultado.
- Todo o quadro fica incerto com mais de uma pessoa, sem pessoa (dentro da passagem) ou com
  pessoa abaixo de `conf_ok`; em B, também sem pose.

**Passagem, votação e regras de `INDETERMINADO`**
- A passagem abre após `k_entrada` quadros seguidos com pessoa e fecha após `k_saida` sem.
  Usa a caixa `pessoa` do detector de EPI, igual para A e B.
- **Portão de entrada, igual para A e B:** um quadro só entra na janela se a caixa da
  pessoa estiver inteira no quadro, a pelo menos `margem_borda` de todas as bordas, com
  altura de ao menos `altura_min` da imagem. Evita decidir com a pessoa ainda entrando.
  `max_quadros` continua contando todos os quadros da passagem.
- Na janela dos últimos N quadros que passaram pelo portão, um EPI fica presente ou ausente com ao menos a fração
  `maioria` (> 0,5); senão, indefinido.
- Com a janela cheia: algum exigido ausente → `BLOQUEADO`; todos presentes → `LIBERADO`; senão,
  aguarda. A decisão é única por passagem, e a janela reinicia na seguinte.
- Sem decisão quando a passagem fecha, o vídeo termina ou passam `max_quadros`; ou falha de
  câmera (vídeo não abre ou leitura interrompida) → `INDETERMINADO`.
- A regra da área é a lista de EPIs exigidos, em configuração.

**Simulador de CLP** (forma final registrada na Parte 2)**:** servidor Modbus TCP em processo separado, com
[`pymodbus` 3.16.2](https://pypi.org/pypi/pymodbus/json) (Python ≥ 3.10, classificador 3.13),
não instalado. Na API atual, o
[exemplo oficial de servidor](https://github.com/pymodbus-dev/pymodbus/blob/v3.16.2/examples/server_async.py)
usa `SimDevice` e `SimData` com `StartAsyncTcpServer`; o cliente usa
[`write_registers(..., device_id=1)`](https://github.com/pymodbus-dev/pymodbus/blob/v3.16.2/pymodbus/client/mixin.py),
que [substituiu `slave=` na 3.10](https://github.com/pymodbus-dev/pymodbus/blob/v3.16.2/API_changes.rst).
Escuta só em `127.0.0.1`, porta 5020.

| Registrador | Conteúdo |
|---|---|
| 0 | decisão: 0 nenhuma, 1 `LIBERADO`, 2 `BLOQUEADO`, 3 `INDETERMINADO` |
| 1 | número sequencial da decisão, para o CLP reconhecer uma nova |
| 2 | bits dos EPIs ausentes: 0 capacete, 1 óculos, 2 luvas, 3 calçado, 4 colete |

Os três são escritos em uma operação. O registrador 0 volta a 0 na partida, no início de cada
passagem e no fim do programa; `LIBERADO` também ao fim da passagem, para não ficar retido.

**Crachá/RFID:** fora do mínimo viável; a metodologia só o prevê "quando necessária".

**Medições** (com hardware e versões de `ambiente.registrar`; as da F5 são provisórias)
- Latência: ida e volta de `write_registers` por `perf_counter`, na mesma máquina. FPS:
  quadros / tempo do laço, sem aquecimento. Custo de A: detector; de B: detector + pose.
- Tempo até a decisão: quadros desde o primeiro com pessoa / FPS do vídeo + processamento do
  último. Só vale se o fluxo for ao menos tão rápido quanto o vídeo; senão, declarar.

**Pré-treinado enquanto a F3 não termina?** Sim, mas o COCO não detecta os cinco EPIs. Testes:
1. caixas e pontos montados à mão, sem modelo;
2. pose real e **detector roteirizado**: a pessoa vem do `yolo26n.pt`, e um YAML define por
   faixa de quadros quais EPIs "existem", a confiança e a posição relativa à caixa da pessoa.
   Produz as três decisões e um item fora do lugar (presente em A, ausente em B). O vídeo,
   montado em `runs/`, repete o recorte do **homem ao centro** de `bus.jpg` (uma pessoa, com
   punhos e tornozelos aparentes; resposta da pose a confirmar), com trechos vazios;
3. um modelo de E1 ou E2, quando existir. **A F5 só encerra depois deste nível**; usá-lo na
   integração não o escolhe como modelo do sistema.

### Desenho dos módulos (`src/epi/`)

    # associacao.py (puro, sem torch); Deteccao(classe, conf, caixa)
    selecionar_pessoa(dets, par) -> (ha_alguem, pessoa_valida | None)
    associar_a(pessoa, dets, par) -> {epi: Estado}
    regioes(pontos, forma, par) -> {epi: [caixa | None por lado]}   # pontos: 17 x (x, y, conf)
    associar_b(pessoa, pontos, dets, forma, par) -> {epi: Estado}

    # decisao.py (puro)
    votar(janela, n, maioria) -> {epi: Estado}                      # INCERTO = indefinido
    decidir(votos, exigidos) -> Decisao | None                      # None: aguardar
    Passagem(par).atualizar(ha_alguem, estados) -> [Evento]         # decisão e fim
    Passagem(par).encerrar(motivo) -> [Evento]                      # fim do vídeo ou falha

    # deteccao.py (Ultralytics, save=False)
    detectar(modelo, quadro, mapa_classes) -> [Deteccao]
    estimar_pose(modelo, quadro, pessoa) -> pontos | None
    detector_roteirizado(roteiro) -> função(indice, pessoas) -> [Deteccao]

    # clp.py: simulador (python -m epi.clp, imprime escritas) e enviar(cliente, decisao, seq, ausentes)
    # fluxo.py: executar(video, config) -> resumo

- `configs/ponto_de_controle.yaml`: EPIs exigidos, mapa de classes, associação ativa e
  parâmetros. A e B rodam na mesma passada, cada uma com sua `Passagem`; só a ativa vai ao CLP.
- Valores **provisórios**: N = 15, `maioria` 0,6, `conf_min` 0,25, `conf_ok` 0,5,
  `conf_ponto` 0,5, `k_entrada` 3, `k_saida` 15, `max_quadros` 150, `margem_borda` 1% do
  quadro, `altura_min` 50% da imagem. Todos são congelados na F6.
- Eventos em `runs/eventos/*.jsonl`: horário, passagem, associação, decisão, motivo, estados,
  tempos, resultado da escrita, modelos, configuração e commit. Nenhuma imagem.
- Exibição opcional na tela, com caixas e regiões, sem gravar.

### Proteção contra ajuste indevido
- A F5 não grava nem lê gravações de calibração ou de teste, nem os conjuntos de teste das
  bases. Usa dados sintéticos, imagens de exemplo e um **vídeo de fumaça**.
- O vídeo de fumaça serve para achar defeitos de funcionamento: não escolhe modelo, N, limiar
  ou região, não gera resultado reportado e não entra na calibração nem no teste.
- Os parâmetros são ajustados e congelados na F6, de onde vêm os resultados reportados.

### Entregas e critério de pronto
| Entrega | Onde fica | Como verificar |
|---|---|---|
| Associação, votação e decisão | `src/epi/`, `tests/` | assistente: `pytest` cobre A e B, item na mão, parte não visível, duas pessoas, pessoa conforme entrando aos poucos (não pode ser `BLOQUEADO` antes do portão), janela, reinício, tempo esgotado, falha e as três decisões |
| CLP simulado e cliente | `src/epi/clp.py`, `tests/` | assistente: escreve, relê e confere a zeragem; servidor parado gera falha registrada |
| Fluxo e configuração | `src/epi/`, `configs/` | assistente: roteiros produzem as três decisões; nenhuma imagem é salva pelo fluxo |
| Notebook | `notebooks/05_ponto_de_controle.ipynb` | assistente: executa localmente do início ao fim |
| Vídeo de fumaça | `data/fumaca/` | **só o autor grava:** cerca de 1 min, só ele, câmera fixa, corpo inteiro; entra, para e sai 3 vezes, uma com os pés fora do quadro |
| Demonstração provisória | tela e `runs/` | assistente executa; **o autor confere** regiões e registradores na tela |
| Integração com modelo da F3 | `runs/` | assistente: ponta a ponta, três medições e hardware registrados |

**Pronto quando:** o fluxo roda de ponta a ponta com um modelo da F3, produz as três decisões
(se o modelo não produzir alguma nos vídeos permitidos, demonstrá-la por roteiro e declarar) e
registra tempo até a decisão, latência até o CLP e FPS.

### Passos
1. `associacao.py` e `decisao.py` com testes.
2. Fixar `pymodbus==3.16.2` no `pyproject.toml`, instalar; `clp.py` e teste.
3. Baixar o modelo de pose; `deteccao.py`, `fluxo.py`, roteiros e configuração.
4. Notebook `05`, execução local com roteiros e atualização do README.
5. O autor grava o vídeo de fumaça; corrigir defeitos sem ajustar parâmetros.
6. Integrar um modelo da F3, medir e encerrar. Se a F3 atrasar, a fase fica aberta aqui.

### Riscos e plano B
- Pose falha em punhos, tornozelos ou olhos cobertos pelos EPIs: B tende a `INDETERMINADO`.
  Observar na calibração; plano B: `yolo26s-pose.pt`.
- Regiões estimadas erram de lado, perdem a caixa única de um par (anotação a conferir na F2)
  ou aceitam item não usado (luva na mão, calçado no chão junto aos pés): a posição não separa
  esses casos; ajustar na calibração, e o que persistir é resultado.
- `pymodbus`, a confirmar na instalação: execução do servidor, importação de
  `pymodbus.simulator` sem o extra `simulator` e se o aviso "não usar" da
  [documentação do simulador](https://pymodbus.readthedocs.io/en/stable/source/simulator.html)
  atinge `SimDevice`. Alternativa: `ModbusDeviceContext`, o armazenamento anterior.
- Processo morto à força deixa o último valor no CLP (limitação declarada; o sistema nunca é o
  único intertravamento). Detecção de leitura interrompida no OpenCV: a confirmar.

### Aprovação
- **Aprovado pelo autor em:** 25/11/2026, exigindo os dois lados para luvas e calçado.
- **Vídeo de fumaça:** pendente; o autor vai gravar depois. Não bloqueia o desenvolvimento
  com dados sintéticos e roteiros.

## Parte 2 — Encerramento

- **Encerrada em:** dd/mm/aaaa

### O que foi feito
Fase **em andamento**. Módulos `associacao`, `decisao`, `deteccao`, `clp` e `fluxo`, testes,
configuração, roteiro de demonstração e notebook `05`, até o nível 2 (pose real e detector
roteirizado).

### O que mudou em relação ao detalhamento (e por quê)
- **Segurança, após revisão:** só libera em um quadro aprovado pelo portão e com uma única
  pessoa; um `LIBERADO` ativo é zerado quando aparece uma segunda pessoa.
- **Segurança, após revisão:** caixas da mesma classe com IoU acima de `iou_duplicata`
  (0,5, provisório) contam como um só objeto, para uma detecção repetida não formar um par.
- Servidor criado com `ModbusTcpServer`, `SimDevice` e `SimData` (pymodbus 3.16.2).
- YAML lido com o leitor do Ultralytics, que não aceita âncoras; PyYAML não está instalado.
- Óculos também exigem os ombros, usados como escala. Associação ativa provisória: B.
- Falha fora de uma passagem gera `INDETERMINADO` com passagem 0.

### Decisões tomadas
| Decisão | Motivo | Permanente? (copiar para o plano) |
|---|---|---|
| Liberar só com uma pessoa e dentro do portão; zerar com segunda pessoa | falha nunca libera | sim |
| Caixas duplicadas contam como um objeto | evitar par falso | sim |

### Resultados
Nível 2, RTX 4060 Ti, provisórios, antes das duas correções de segurança: 35,3 quadros/s no
laço e 27,8 com pessoa (vídeo a 30, logo o tempo até a decisão ainda não vale); A 17,1 ms e
B 35,5 ms por quadro; latência local até o CLP com mediana de 0,43 ms. Depois das correções,
75 testes passam e o notebook executa.

### Pendências
- O autor gravar o vídeo de fumaça (`data/fumaca/fumaca.mp4`) e conferir regiões e
  registradores com `mostrar: true`.
- Nível 3: integrar um modelo da F3 e medir novamente.

### O que a próxima fase precisa saber
- Limitações: o `LIBERADO` fica retido enquanto a passagem estiver aberta; processo morto
  retém o último valor no CLP; regiões de B encolhem com a pessoa de lado; calçado começa no
  tornozelo (bota alta pode aparecer como ausente); a leitura interrompida depende da
  contagem de quadros do vídeo.

### Atualizações feitas
- [ ] Status no plano
- [ ] Decisões acumuladas no plano
- [ ] Metodologia (somente se houve mudança aprovada de método)
