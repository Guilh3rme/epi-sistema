"""Vídeo -> pessoa e pose -> EPI -> associações A e B -> decisão -> CLP e registro.

O fluxo não grava imagens: só eventos em JSONL. `montar_video` gera o vídeo de teste.
"""

import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

import cv2
import numpy as np

from epi import clp
from epi.ambiente import RAIZ, configurar_ultralytics, registrar
from epi.associacao import (
    EPIS, Deteccao, associar_a, associar_b, no_portao, regioes, selecionar_pessoa,
)
from epi.decisao import FIM_VIDEO, LIBERADO, Passagem
from epi.deteccao import detectar, detector_roteirizado, estimar_pose

ASSOCIACOES = ("A", "B")


def carregar_yaml(caminho) -> dict:
    """Usa o leitor de YAML do Ultralytics, que não aceita âncoras; PyYAML não é dependência."""
    if "ultralytics" not in sys.modules:
        configurar_ultralytics()
    from ultralytics.utils import YAML

    return YAML.load(RAIZ / caminho)


def _validar(config):
    exigidos, p = config["exigidos"], config["parametros"]
    if not exigidos or set(exigidos) - set(EPIS):
        raise ValueError(f"`exigidos` deve listar ao menos um EPI entre {EPIS}.")
    if config["associacao_ativa"] not in ASSOCIACOES:
        raise ValueError("`associacao_ativa` deve ser A ou B.")
    if not 0.5 < p["maioria"] <= 1:
        raise ValueError("`maioria` deve estar em (0,5; 1].")
    if min(p["n"], p["k_entrada"], p["k_saida"], p["max_quadros"]) < 1:
        raise ValueError("`n`, `k_entrada`, `k_saida` e `max_quadros` devem ser positivos.")
    if not 0 <= p["conf_min"] <= p["conf_ok"] <= 1:
        raise ValueError("Exige-se 0 <= conf_min <= conf_ok <= 1.")


def _carregar_modelos(config):
    """(detector(indice, quadro), pose(quadro, pessoa)) com os modelos da configuração."""
    configurar_ultralytics()
    from ultralytics import YOLO

    m, p = config["modelos"], config["parametros"]
    modelo = YOLO(RAIZ / m["detector"])
    modelo_pose = YOLO(RAIZ / m["pose"])
    dispositivo = config.get("dispositivo")
    itens = detector_roteirizado(carregar_yaml(m["roteiro"])) if m.get("roteiro") else None

    def detector(indice, quadro):
        dets = detectar(modelo, quadro, config["classes"], p["conf_min"], dispositivo)
        if itens is None:
            return dets
        pessoas = [d for d in dets if d.classe == "pessoa"]
        return pessoas + itens(indice, pessoas)

    def pose(quadro, pessoa):
        return estimar_pose(modelo_pose, quadro, pessoa, p["iou_pose"], dispositivo)

    return detector, pose


class _Saida:
    """Escritas no CLP; zeragem que falha fica pendente e é repetida."""

    def __init__(self, porta, registrar_evento):
        self.cliente = clp.cliente(porta)
        self.registrar_evento = registrar_evento
        self.sequencia = 0
        self.ultima = None
        self.pendente = False
        self.tentativa = 0.0
        self.escritas = []

    def _escrever(self, decisao, ausentes=()):
        self.tentativa = time.monotonic()
        escrita = clp.enviar(self.cliente, decisao, self.sequencia, ausentes)
        self.escritas.append(escrita)
        if escrita["ok"]:
            self.pendente = False
        return escrita

    def zerar(self, motivo):
        escrita = self._escrever(None)
        if not escrita["ok"]:
            self.pendente = True
        self.registrar_evento(tipo="zeragem", motivo=motivo, escrita=escrita)
        return escrita["ok"]

    def decidir(self, decisao):
        self.sequencia += 1
        self.ultima = decisao.resultado
        escrita = self._escrever(decisao.resultado, decisao.ausentes)
        if not escrita["ok"] and decisao.resultado == LIBERADO:
            self.zerar("falha_escrita")  # a escrita pode ter chegado mesmo sem resposta
        return escrita

    def repetir_zeragem(self):
        # shortcut: uma tentativa por segundo, bloqueando o laço por até 0,5 s; usar outra
        # thread se o CLP fora do ar precisar não afetar o FPS.
        if self.pendente and time.monotonic() - self.tentativa >= 1:
            self.zerar("nova_tentativa")

    def encerrar(self):
        try:
            for _ in range(3):
                if self.zerar("fim_programa"):
                    break
            if self.pendente:
                print("ATENÇÃO: não foi possível zerar o CLP; o registrador pode reter a decisão.")
            return clp.ler(self.cliente)
        finally:
            self.cliente.close()


def desenhar(quadro, dets, areas, linhas):
    """Cópia do quadro com detecções (verde), regiões de B (amarelo) e texto."""
    tela = quadro.copy()
    for d in dets:
        x1, y1, x2, y2 = map(int, d.caixa)
        cv2.rectangle(tela, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(tela, f"{d.classe} {d.conf:.2f}", (x1, y1 - 2), 0, 0.35, (0, 255, 0))
    for epi, lados in areas.items():
        for caixa in filter(None, lados):
            x1, y1, x2, y2 = map(int, caixa)
            cv2.rectangle(tela, (x1, y1), (x2, y2), (0, 255, 255), 1)
            cv2.putText(tela, epi, (x1, y2 + 10), 0, 0.35, (0, 255, 255))
    for i, linha in enumerate(linhas):
        cv2.putText(tela, linha, (5, 15 + 15 * i), 0, 0.4, (0, 0, 0), 3)
        cv2.putText(tela, linha, (5, 15 + 15 * i), 0, 0.4, (255, 255, 255), 1)
    return tela


def executar(video=None, config="configs/ponto_de_controle.yaml", detector=None, pose=None) -> dict:
    """Processa um vídeo e devolve o resumo; `detector` e `pose` substituem os modelos.

    Falha de câmera, modelo ou processamento aparece em `resumo["falha"]`.
    """
    if not isinstance(config, dict):
        config = carregar_yaml(config)
    _validar(config)
    video = str(video or config["video"])
    par = {**config["parametros"], "exigidos": config["exigidos"]}
    ativa = config["associacao_ativa"]

    pasta = RAIZ / config["eventos"]
    ambiente = {}
    arquivo = pasta / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.jsonl"
    pasta.mkdir(parents=True, exist_ok=True)

    def registrar_evento(**campos):
        linha = {
            "horario": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            **campos,
            "video": video,
            "modelos": config["modelos"],
            "config": config,
            "commit": ambiente.get("commit"),
        }
        with arquivo.open("a", encoding="utf-8") as saida_eventos:
            saida_eventos.write(json.dumps(linha, ensure_ascii=False) + "\n")

    passagens = {a: Passagem(par) for a in ASSOCIACOES}
    decisoes, custos = [], {a: [] for a in ASSOCIACOES}
    custo = dict.fromkeys(ASSOCIACOES, 0.0)
    situacao = {a: "aguardando" for a in ASSOCIACOES}
    indice, fps_video, inicio, exibicao = 0, 0.0, None, 0.0
    iteracoes = []  # segundos por quadro com pessoa: leitura até CLP e registro

    def fps_laco():
        tempo = 0 if inicio is None else time.perf_counter() - inicio - exibicao
        return indice / tempo if tempo > 0 else 0.0

    def acompanha_video():
        """Julgado pelos quadros com pessoa: os vazios são baratos e inflariam a média."""
        return bool(iteracoes) and 0 < fps_video <= 1 / mean(iteracoes)

    def processar(associacao, eventos):
        na_saida = associacao == ativa
        for ev in eventos:
            if ev.tipo == "inicio":
                situacao[associacao] = "aguardando"
                if na_saida:
                    saida.zerar("inicio_passagem")
            elif ev.tipo == "fim":
                if na_saida and saida.ultima == LIBERADO:
                    saida.zerar("fim_passagem")
            else:
                escrita = saida.decidir(ev.decisao) if na_saida else None
                tempo = None
                if ev.quadros and fps_video > 0:
                    tempo = ev.quadros / fps_video + custo[associacao]
                registro = {
                    "passagem": ev.passagem,
                    "associacao": associacao,
                    "ativa": na_saida,
                    "decisao": ev.decisao.resultado,
                    "motivo": ev.decisao.motivo,
                    "ausentes": list(ev.decisao.ausentes),
                    "estados": ev.votos,
                    "tempos": {
                        "quadro": indice,
                        "quadros_desde_pessoa": ev.quadros,
                        "processamento_ultimo_ms": custo[associacao] * 1000,
                        "ate_decisao_s": tempo,
                        "fluxo_acompanha_video": acompanha_video(),
                    },
                    "sequencia": saida.sequencia if na_saida else None,
                    "escrita": escrita,
                }
                decisoes.append(registro)
                situacao[associacao] = ev.decisao.resultado
                registrar_evento(tipo="decisao", **registro)

    saida = _Saida(config["clp"]["porta"], registrar_evento)
    captura = None
    motivo, falha = "interrompido", "interrompido"
    try:
        motivo = "falha_processamento"
        # Dentro do `try`: se falhar, o CLP ainda é zerado.
        ambiente.update(registrar(pasta.parent / "ambiente", semente=0))
        saida.zerar("partida")
        motivo = "falha_camera"
        captura = cv2.VideoCapture(str(RAIZ / video))
        if not captura.isOpened():
            raise OSError(f"o vídeo não abre: {video}")
        fps_video = captura.get(cv2.CAP_PROP_FPS)
        total = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))
        forma = (
            int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(captura.get(cv2.CAP_PROP_FRAME_WIDTH)),
            3,
        )

        motivo = "falha_modelo"
        if detector is None or pose is None:
            carregado = _carregar_modelos(config)
            detector, pose = detector or carregado[0], pose or carregado[1]
        vazio = np.zeros(forma, np.uint8)  # aquecimento, fora das medições
        detector(0, vazio)
        pose(vazio, Deteccao("pessoa", 1.0, (0, 0, forma[1], forma[0])))

        motivo = "falha_processamento"
        inicio = time.perf_counter()
        while True:
            antes_da_leitura = time.perf_counter()
            ok, quadro = captura.read()
            if not ok:
                # shortcut: confia na contagem de quadros do contêiner; sem ela, ou se
                # faltarem quadros, o fim é tratado como leitura interrompida.
                if indice == 0 or total <= 0 or indice < total:
                    motivo = "falha_camera"
                    raise OSError(f"leitura interrompida no quadro {indice} de {total}")
                motivo, falha = FIM_VIDEO, None
                break
            t0 = time.perf_counter()
            dets = detector(indice, quadro)
            t1 = time.perf_counter()
            ha_alguem, pessoa = selecionar_pessoa(dets, par)
            pontos = None if pessoa is None else pose(quadro, pessoa)
            t2 = time.perf_counter()
            estados = {
                "A": associar_a(pessoa, dets, par),
                "B": associar_b(pessoa, pontos, dets, quadro.shape, par),
            }
            custo.update(A=t1 - t0, B=t2 - t0)
            if pessoa is not None:
                for a in ASSOCIACOES:
                    custos[a].append(custo[a] * 1000)
            passou = no_portao(dets, quadro.shape, par)
            varias = sum(d.classe == "pessoa" and d.conf >= par["conf_min"] for d in dets) > 1
            for a in ASSOCIACOES:
                processar(a, passagens[a].atualizar(
                    ha_alguem, estados[a] if passou else None, varias
                ))
            # Uma segunda pessoa não pode herdar a liberação da primeira.
            if varias and saida.ultima == LIBERADO:
                saida.zerar("segunda_pessoa")
                saida.ultima = None
            saida.repetir_zeragem()
            indice += 1
            if pessoa is not None:
                iteracoes.append(time.perf_counter() - antes_da_leitura)

            if config.get("mostrar"):
                antes = time.perf_counter()
                areas = {} if pontos is None else regioes(pontos, quadro.shape, par)
                linhas = [f"{a}: {situacao[a]}" + (" (ativa)" if a == ativa else "")
                          for a in ASSOCIACOES]
                linhas.append(f"CLP: {clp.ler(saida.cliente)}  portao: {'sim' if passou else 'nao'}")
                cv2.imshow("ponto de controle (q encerra)", desenhar(quadro, dets, areas, linhas))
                resto = 1 / fps_video - (time.perf_counter() - t0) if fps_video > 0 else 0
                tecla = cv2.waitKey(max(1, int(resto * 1000)))
                exibicao += time.perf_counter() - antes
                if tecla & 0xFF == ord("q"):
                    motivo = falha = "interrompido"
                    break
    except Exception as erro:  # qualquer falha vira INDETERMINADO, nunca libera
        falha = f"{motivo}: {type(erro).__name__}: {erro}"
        print("FALHA:", falha)
    finally:
        fps = fps_laco()
        try:
            for a in ASSOCIACOES:
                processar(a, passagens[a].encerrar(motivo))
        finally:
            registradores = saida.encerrar()
            if captura is not None:
                captura.release()
            if config.get("mostrar"):
                cv2.destroyAllWindows()

    latencias = [e["latencia_ms"] for e in saida.escritas if e["ok"]]
    return {
        "falha": falha,
        "eventos": str(arquivo),
        "associacao_ativa": ativa,
        "decisoes": decisoes,
        "quadros": indice,
        "fps_video": fps_video,
        "fps_laco": fps,
        "fps_com_pessoa": 1 / mean(iteracoes) if iteracoes else None,
        "fluxo_acompanha_video": acompanha_video(),
        "custo_medio_ms_com_pessoa": {a: mean(c) if c else None for a, c in custos.items()},
        "latencia_clp_ms": {
            "escritas": len(latencias),
            "mediana": median(latencias) if latencias else None,
            "maxima": max(latencias, default=None),
        },
        "falhas_clp": sum(not e["ok"] for e in saida.escritas),
        "registradores_finais": registradores,
        "ambiente": ambiente,
    }


def montar_video(roteiro, imagem, destino):
    """Vídeo de teste: o recorte de `imagem` sobre fundo cinza, nas faixas do roteiro.

    `deslocamento` [de, até] move o recorte na horizontal, em pixels a partir do centro.
    """
    foto = cv2.imread(str(imagem))
    if foto is None:
        raise OSError(f"imagem não lida: {imagem}")
    x1, y1, x2, y2 = roteiro["recorte"]
    recorte = foto[y1:y2, x1:x2]
    largura, altura = roteiro["quadro"]
    h, w = recorte.shape[:2]
    topo = (altura - h) // 2
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    video = cv2.VideoWriter(
        str(destino), cv2.VideoWriter_fourcc(*"mp4v"), roteiro["fps"], (largura, altura)
    )
    if not video.isOpened():
        raise OSError(f"vídeo não criado: {destino}")
    for faixa in roteiro["faixas"]:
        for dx in np.linspace(*faixa.get("deslocamento", [0, 0]), faixa["quadros"]):
            tela = np.full((altura, largura, 3), 114, np.uint8)
            if faixa.get("pessoa", True):
                esquerda = (largura - w) // 2 + int(dx)
                de, ate = max(esquerda, 0), min(esquerda + w, largura)
                if ate > de:
                    tela[topo:topo + h, de:ate] = recorte[:, de - esquerda:ate - esquerda]
            video.write(tela)
    video.release()
    return destino
