# Metodologia do projeto (v1.4)

**Aprovada pelo aluno em 25/11/2026.** v1.1: as duas associações entram no mesmo teste e os
vídeos de pesquisa ficam locais. v1.2: luvas e calçado exigem os dois lados. v1.3
(26/11/2026): liberação só com uma única pessoa e detecções repetidas contadas uma vez.
v1.4 (27/11/2026): por tamanho de objeto são reportados somente revocação e AP.
A definir com o orientador: dispensa do Comitê de Ética nas gravações só com o autor e metas
de liberação indevida e de indeterminados.

Estrutura alinhada aos itens do Art. 22, §6º do Regimento do PECS.

## Identificação

- Aluno: Guilherme Ferreira dos Reis
- Orientador: Prof. José Pinheiro de Moura
- Programa: PECS/UEMA (mestrado profissional), linha Telemática e Automação

## Título (proposta)

Verificação de conformidade de EPI em ponto de controle de acesso com visão
computacional e integração à automação industrial

## Justificativa

- A fiscalização manual de EPI é intermitente e não registra evidências de forma sistemática.
- Grande parte dos estudos para na **detecção**: encontrar um capacete na imagem não prova
  que ele está sendo usado pela pessoa avaliada.
- A revisão de Vukicevic et al. (2024) aponta a necessidade de tratar em conjunto
  identificação, pose, detecção e rastreamento, e indica que soluções de **ponto de
  verificação** parecem mais próximas do mercado que o monitoramento contínuo.
- Este projeto evolui um detector desenvolvido anteriormente, acrescentando teste
  independente, medição de velocidade e decisão por trabalhador.

## Fundamentação e trabalhos de referência

| Trabalho | O que este projeto aproveita |
|---|---|
| Vukicevic et al. (2024) | Taxonomia (detecção x classificação por região; checkpoint x contínuo), privacidade, falta de dados públicos |
| Chen e Demachi (2020) | Associação EPI-pessoa pela distância a pontos-chave do corpo |
| Ahmad e Rahimi (2025), SH17 | Base pública de referência, com resultados publicados por classe |
| Ultralytics (2026), Construction-PPE | Base pequena com as cinco classes e classes de ausência |
| Peffers et al. (2007) | Design Science Research como método |
| Kapoor e Narayanan (2023) | Prevenção de vazamento de dados |
| ISO/IEC TR 5469:2024 | Limites do uso de IA em funções de segurança |

## Objetivos

**Geral:** desenvolver e avaliar um sistema que verifica, em um ponto de controle de acesso,
se o trabalhador está **usando** os EPIs exigidos e sinaliza a decisão a um controlador
industrial.

**Específicos:**
1. Treinar e comparar detectores compactos de um estágio com protocolo sem vazamento.
2. Comparar duas formas de associar o EPI à pessoa: caixa da pessoa e regiões por pose.
3. Implementar a regra de conformidade com votação temporal e estado indeterminado.
4. Integrar a decisão a um CLP simulado.
5. Avaliar o sistema por detecção, por decisão e por tempo de resposta.

## Perguntas de pesquisa

- **PQ1:** Qual o desempenho por classe de detectores compactos para as cinco classes, em
  teste interno e em base diferente da de treino?
- **PQ2:** A associação por pose reduz liberações indevidas em relação à caixa da pessoa?
  A que custo de latência?
- **PQ3:** Qual a taxa de decisões corretas e o tempo de decisão em gravações roteirizadas
  no ponto de controle?

## Metodologia

### Método de pesquisa

Pesquisa aplicada, conduzida por *Design Science Research* (Peffers et al., 2007):
problema, objetivos da solução, projeto e desenvolvimento, demonstração, avaliação e
comunicação. A parte de aprendizado de máquina segue ciclos de dados, treino e avaliação.

### Cenário

Adotado: **ponto de controle** antes da área de trabalho. Uma pessoa por vez, câmera
fixa, distância e iluminação controladas, EPIs exigidos conhecidos. A identidade, quando
necessária, vem de crachá/RFID simulado; não há reconhecimento facial.

Descartado por ora: monitoramento contínuo por CFTV. Exige rastreamento de várias pessoas,
oclusão e distâncias variadas; fica como trabalho futuro.

### Arquitetura

    câmera -> pessoa e pose -> detecção de EPI -> associação EPI/região do corpo
           -> votação em N quadros -> regra da área -> decisão -> CLP simulado e registro

- **Pose:** modelo pré-treinado, sem treino próprio (cabeça, olhos, mãos, tronco, pés).
- **Associação A (base):** EPI contido na caixa da pessoa.
- **Associação B:** EPI na região do corpo correspondente.
- **Itens em par:** luvas e calçado só contam como presentes com os dois lados, cada um com
  uma detecção própria.
- **Detecções repetidas:** caixas da mesma classe sobrepostas acima de um limiar de IoU
  contam como um só objeto, para que uma detecção repetida não forme um par. O limiar é
  provisório até ser congelado na calibração, nunca escolhido no teste.
- **Votação temporal:** estado de cada EPI definido em uma janela de N quadros.
- **Decisão:** `LIBERADO`, `BLOQUEADO` ou `INDETERMINADO` (corpo não visível, baixa
  confiança, falha de câmera). Indeterminado **não libera** e pede conferência humana.
- **Liberação:** `LIBERADO` só é emitido em um quadro que passou pelo portão de entrada
  (pessoa inteira na imagem) e tem uma única pessoa detectada. Se aparecer uma segunda
  pessoa com a liberação ativa, o sinal é retirado (a saída volta a nenhuma decisão) e não
  é restabelecido na mesma passagem.
- **Saída:** Modbus TCP para CLP simulado.

### Escopo de segurança

O sistema é apoio ao controle administrativo. A saída é um alarme ou um sinal permissivo
para a automação convencional, **nunca o único intertravamento** nem uma função de
segurança certificada. A ISO/IEC TR 5469:2024 é um relatório técnico sobre IA e segurança
funcional, não uma certificação.

### Dados

| Base | A favor | Contra |
|---|---|---|
| **SH17** | Fotos do Pexels; 8.099 imagens; resultados publicados | CC BY-NC-SA 4.0; 14,2 GB; `shoes` mistura calçado de segurança, tênis e outros; óculos não são só de segurança; poucos capacetes (927) e coletes (530) |
| **Construction-PPE** | CC BY 4.0; cinco classes, pessoa e ausências | 1.000 imagens; sem teste; mistura imagens geradas e anotação automática (amostra inspecionada com aparência majoritariamente sintética) |

O artigo do SH17 cita marcações extras (calçado ou óculos de segurança, usado ou não),
mas elas **não apareceram** nas 60 anotações públicas conferidas. Pode-se pedi-las aos autores.

Decisão: **treinar em cada base e testar na outra**, com as classes comuns.

| Experimento | Treino e desenvolvimento | Teste interno | Teste entre bases |
|---|---|---|---|
| E1 | parte de treino do SH17, com fração separada para desenvolvimento | `val_files.txt` do SH17 (1.620, a confirmar no download) | Construction-PPE completa (1.000) |
| E2 | `train` da Construction-PPE (800), com fração separada para desenvolvimento | `val` da Construction-PPE (200) | `val_files.txt` do SH17 (1.620) |

| Classe do projeto | SH17 | Construction-PPE |
|---|---|---|
| pessoa | `person` | `Person` |
| capacete | `helmet` | `helmet` |
| óculos | `glasses` | `goggles` |
| luvas | `gloves` | `gloves` |
| calçado | `shoes` | `boots` |
| colete | `safety-vest` | `vest` |

`shoes`/`boots` e `glasses`/`goggles` não têm a mesma definição nas duas bases. Por isso,
o resultado principal do teste entre bases considera **pessoa, capacete, colete e luvas**;
calçado e óculos são reportados como exploratórios. A presença de imagens geradas na
Construction-PPE será declarada.

As classes `no_*` da Construction-PPE (ausência visível) são mantidas no treino de E2 e
podem ser usadas pela regra de decisão para distinguir `BLOQUEADO` de `INDETERMINADO`. A
comparação de detecção entre bases usa somente as classes positivas comuns.

**Limitação esperada de E1:** no SH17, `shoes` inclui tênis e chinelos, `glasses` inclui
óculos de grau, e as marcações de uso não estão nos rótulos públicos conferidos. As classes
indicam presença do item, não uso nem EPI certificado; o uso é inferido pela posição no
corpo. Assim, espera-se que E1 libere o cenário de calçado ou óculos comuns, o que será
reportado como resultado.

O SH17 tem 14,2 GB, com imagens de até 8K, e exigirá uma cópia redimensionada para o Colab.
Antes de E1 e E2, a busca de duplicatas é feita também **entre as duas bases**.

O modelo do sistema será escolhido nas **gravações de calibração**, nunca pelos resultados
de teste. As gravações de teste avaliam o sistema, com EPIs de segurança reais.

As bases não serão misturadas no treino, pois classes não anotadas em uma delas virariam
falsos negativos. Enquanto não houver dado específico, as classes serão chamadas de
**óculos** e **calçado**, e não de óculos e botas de segurança.

### Controle de vazamento

- Divisão definida antes de qualquer aumento de dados; aumento somente durante o treino.
- Busca de duplicatas (*hash* perceptual) entre treino, desenvolvimento e testes.
- Gravações separadas por sessão: calibração em um dia, teste em outro.
- Limiares e N definidos só em desenvolvimento e calibração; teste executado uma vez.
- As associações A e B são congeladas na calibração e rodam sobre as mesmas gravações de
  teste, o que responde à PQ2 com comparação pareada.
- Mesmo processamento no treino e na inferência (sem filtro mediano ou equalização).

### Modelos e treinamento

- Primeiro modelo, usado em E1 e E2: YOLO26n.
- Comparação posterior, nas mesmas condições: YOLOv8n (continuidade com o trabalho
  anterior) e YOLO26s; resolução 640 contra uma maior, por causa de objetos pequenos.
- Mesma divisão, épocas com parada antecipada e sementes registradas; três sementes no
  modelo final, se o tempo permitir.
- Treinamento no **Google Colab**. A RTX 4060 Ti local fica para demonstração, medição de
  latência e reserva.

### Gravações do sistema

Cenários: todos os EPIs; cada EPI ausente, um por vez; **EPI presente mas não usado**
(na mão, no chão); calçado e óculos comuns. Variar iluminação e distância e repetir cada
combinação. Participante: **somente o autor**. A dispensa do Comitê de Ética nesse caso é
definida com o orientador antes das gravações; incluir outras pessoas exige submissão ao
CEP e TCLE.

### Avaliação

| Nível | Métricas |
|---|---|
| Detecção | P, R, F1, mAP@0,5 e mAP@0,5:0,95 por classe; R e AP por tamanho de objeto |
| Decisão | tabela abaixo, taxa de liberação indevida, taxa de negação a conformes, taxa de indeterminados |
| Sistema | tempo até a decisão, latência até o CLP, FPS, hardware informado |

P, R e F1 usam um limiar de confiança fixado no desenvolvimento. Por tamanho não há P nem
F1, pois uma detecção sem caixa marcada correspondente não tem faixa definida.

Uma **passagem** começa quando a pessoa entra no campo de visão, recebe uma decisão e
termina quando ela sai. Quadros de uma mesma tomada não são passagens distintas, e a janela
de votação é reiniciada a cada passagem. Cada passagem entra em uma célula:

| Situação real | Liberado | Bloqueado | Indeterminado |
|---|---|---|---|
| Conforme (C passagens) | CL | CB | CI |
| Não conforme (NC passagens) | NL | NB | NI |

As taxas são dadas por:

- liberação indevida (crítica): TLI = NL / NC;
- negação a conformes: TNC = (CB + CI) / C, pois nos dois casos a pessoa fica retida;
- indeterminados, em cada linha: CI / C e NI / NC.

A TLI nunca será apresentada sem as demais: um sistema que responde sempre `INDETERMINADO`
teria TLI zero e nenhuma utilidade. As metas de TLI e o teto de indeterminados serão
**definidos com o orientador antes do teste**.

**Tamanho do teste (proposta):** 30 passagens conformes e 10 em cada um dos 7 cenários não
conformes (cada um dos cinco EPIs ausente, EPI presente mas não usado, calçado ou óculos
comuns), ou seja, 100 passagens, distribuídas entre as condições de luz e distância. A
calibração usa cerca de 30 passagens em outro dia. Os intervalos de confiança de 95% serão
calculados pelo método de Clopper-Pearson, que admite contagem zero. Zero liberação
indevida em 70 passagens não conformes independentes ainda admite uma taxa de até cerca de
3/70 (4,3%); com 10 por cenário, os resultados por cenário são apenas indicativos.

### Ética e privacidade

Processamento local, sem reconhecimento facial. Em operação, o sistema não armazena imagens
por padrão e registra apenas o evento de decisão, em atenção à LGPD. Os vídeos brutos de
calibração e teste, que mostram somente o autor, são dados de pesquisa mantidos no
computador dele, fora do repositório.

### Limitações previstas

Um ponto de controle não garante o uso do EPI ao longo do turno; um único participante
limita a generalização, e repetições da mesma pessoa não são observações totalmente
independentes; as bases públicas não cobrem especificamente EPIs de segurança.

## Aplicabilidade

Portarias e acessos a áreas de risco em indústrias e terminais, como apoio à equipe de
segurança do trabalho, com integração a catraca, sinaleiro ou supervisório via CLP.

Para uso do produto: um modelo treinado no SH17 (CC BY-NC-SA 4.0) deve ser tratado como
restrito a uso não comercial; o treinado somente na Construction-PPE (CC BY 4.0) não tem
essa restrição de dados. Em ambos valem as condições do Ultralytics (AGPL-3.0).

## Etapas

As fases, suas dependências e critérios de passagem estão no [plano](plano.md).

## Evolução em relação ao trabalho anterior

Passam a fazer parte do método a decisão por trabalhador, a pose, a integração à
automação, os testes independentes e as métricas de decisão. O aumento de dados ocorre
somente no treino e as licenças das bases são registradas.

## Referências

- AHMAD, H. M.; RAHIMI, A. SH17: a dataset for human safety and personal protective equipment detection in manufacturing industry. *Journal of Safety Science and Resilience*, v. 6, n. 2, p. 175-185, 2025. https://doi.org/10.1016/j.jnlssr.2024.09.002
- CHEN, S.; DEMACHI, K. A vision-based approach for ensuring proper use of personal protective equipment (PPE) in decommissioning of Fukushima Daiichi nuclear power station. *Applied Sciences*, v. 10, 5129, 2020.
- ISO/IEC TR 5469:2024. *Artificial intelligence: functional safety and AI systems*.
- KAPOOR, S.; NARAYANAN, A. Leakage and the reproducibility crisis in machine-learning-based science. *Patterns*, v. 4, n. 9, 100804, 2023.
- PEFFERS, K. et al. A design science research methodology for information systems research. *Journal of Management Information Systems*, v. 24, n. 3, p. 45-77, 2007.
- ULTRALYTICS. Construction-PPE dataset. 2026. https://docs.ultralytics.com/datasets/detect/construction-ppe
- VUKICEVIC, A. M. et al. A systematic review of computer vision-based personal protective equipment compliance in industry practice: advancements, challenges and future directions. *Artificial Intelligence Review*, v. 57, 319, 2024. https://doi.org/10.1007/s10462-024-10978-x

Licença do código: o Ultralytics é AGPL-3.0, o que importa se o sistema for distribuído.
