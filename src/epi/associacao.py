"""Associação EPI-pessoa: A (caixa da pessoa) e B (região do corpo). Módulo puro."""

from dataclasses import dataclass
from enum import Enum
from itertools import permutations
from math import hypot

# A ordem define o bit de cada EPI no registrador 2 do CLP.
EPIS = ("capacete", "oculos", "luvas", "calcado", "colete")
PARES = ("luvas", "calcado")

# Índices do COCO (17 pontos); nos pares, o primeiro lado é o esquerdo da pessoa.
CABECA, OLHOS, ORELHAS = (0, 1, 2, 3, 4), (1, 2), (3, 4)
OMBROS, TRONCO = (5, 6), (5, 6, 11, 12)
BRACOS = ((7, 9), (8, 10))  # cotovelo, punho
PERNAS = ((13, 15), (14, 16))  # joelho, tornozelo


class Estado(str, Enum):
    PRESENTE = "PRESENTE"
    AUSENTE = "AUSENTE"
    INCERTO = "INCERTO"


_ORDEM = {Estado.AUSENTE: 0, Estado.INCERTO: 1, Estado.PRESENTE: 2}


@dataclass(frozen=True)
class Deteccao:
    classe: str
    conf: float
    caixa: tuple  # x1, y1, x2, y2 em pixels


def _pessoas(dets, par):
    return [d for d in dets if d.classe == "pessoa" and d.conf >= par["conf_min"]]


def selecionar_pessoa(dets, par):
    """(há alguém, pessoa válida): válida só se for a única e alcançar `conf_ok`."""
    pessoas = _pessoas(dets, par)
    unica = len(pessoas) == 1 and pessoas[0].conf >= par["conf_ok"]
    return bool(pessoas), pessoas[0] if unica else None


def no_portao(dets, forma, par) -> bool:
    """Portão de entrada: caixa de pessoa inteira no quadro, afastada das bordas e alta.

    Sem pessoa não há caixa. Com várias, basta uma passar: o quadro entra na janela como
    incerto e atrasa a decisão.
    """
    altura, largura = forma[:2]
    mx, my = par["margem_borda"] * largura, par["margem_borda"] * altura
    return any(
        x1 >= mx and y1 >= my and x2 <= largura - mx and y2 <= altura - my
        and y2 - y1 >= par["altura_min"] * altura
        for x1, y1, x2, y2 in (d.caixa for d in _pessoas(dets, par))
    )


def iou(a, b) -> float:
    largura = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    altura = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = largura * altura
    uniao = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / uniao if uniao > 0 else 0.0


def _itens(dets, classe, par):
    """Detecções da classe sem repetições: duas caixas do mesmo objeto não formam um par."""
    candidatas = (d for d in dets if d.classe == classe and d.conf >= par["conf_min"])
    mantidas = []
    for d in sorted(candidatas, key=lambda d: d.conf, reverse=True):
        if all(iou(d.caixa, m.caixa) <= par["iou_duplicata"] for m in mantidas):
            mantidas.append(d)
    return mantidas


def _centro_em(det, caixa) -> bool:
    x = (det.caixa[0] + det.caixa[2]) / 2
    y = (det.caixa[1] + det.caixa[3]) / 2
    return caixa[0] <= x <= caixa[2] and caixa[1] <= y <= caixa[3]


def _estado_item(det, par) -> Estado:
    if det is None:
        return Estado.AUSENTE
    return Estado.PRESENTE if det.conf >= par["conf_ok"] else Estado.INCERTO


def _incertos():
    return {epi: Estado.INCERTO for epi in EPIS}


def associar_a(pessoa, dets, par):
    if pessoa is None:
        return _incertos()
    estados = {}
    for epi in EPIS:
        confs = [d.conf for d in _itens(dets, epi, par) if _centro_em(d, pessoa.caixa)]
        exigidas = 2 if epi in PARES else 1
        if sum(c >= par["conf_ok"] for c in confs) >= exigidas:
            estados[epi] = Estado.PRESENTE
        elif len(confs) >= exigidas:
            estados[epi] = Estado.INCERTO
        else:
            estados[epi] = Estado.AUSENTE
    return estados


def regioes(pontos, forma, par):
    """{epi: [caixa ou None por lado]}; None quando faltam pontos visíveis."""
    altura, largura = forma[:2]
    vistos = {
        i: (x, y) for i, (x, y, c) in enumerate(pontos)
        if c >= par["conf_ponto"] and 0 <= x < largura and 0 <= y < altura
    }
    r = {epi: [None, None] if epi in PARES else [None] for epi in EPIS}

    # Cabeça: a escala é a largura dos ombros, exigida também para os óculos.
    if all(i in vistos for i in OMBROS):
        escala = hypot(vistos[5][0] - vistos[6][0], vistos[5][1] - vistos[6][1])
        cabeca = [vistos[i] for i in CABECA if i in vistos]
        olhos = [vistos[i] for i in OLHOS if i in vistos]
        if cabeca:
            referencia = olhos or [vistos[i] for i in ORELHAS if i in vistos] or cabeca
            y = sum(p[1] for p in referencia) / len(referencia)
            x = sum(p[0] for p in cabeca) / len(cabeca)
            meia = par["capacete_largura"] * escala / 2
            r["capacete"] = [(x - meia, y - par["capacete_altura"] * escala, x + meia, y)]
        if olhos:
            xs = [p[0] for p in olhos]
            y = sum(p[1] for p in olhos) / len(olhos)
            mx, my = par["oculos_margem"] * escala, par["oculos_meia_altura"] * escala
            r["oculos"] = [(min(xs) - mx, y - my, max(xs) + mx, y + my)]

    if all(i in vistos for i in TRONCO):
        xs, ys = zip(*(vistos[i] for i in TRONCO))
        mx = par["colete_margem"] * (max(xs) - min(xs))
        my = par["colete_margem"] * (max(ys) - min(ys))
        r["colete"] = [(min(xs) - mx, min(ys) - my, max(xs) + mx, max(ys) + my)]

    for lado, (cotovelo, punho) in enumerate(BRACOS):
        if cotovelo in vistos and punho in vistos:
            (ex, ey), (px, py) = vistos[cotovelo], vistos[punho]
            antebraco = hypot(px - ex, py - ey)
            if antebraco > 0:
                x = px + (px - ex) * par["luva_avanco"]
                y = py + (py - ey) * par["luva_avanco"]
                meio = par["luva_lado"] * antebraco / 2
                r["luvas"][lado] = (x - meio, y - meio, x + meio, y + meio)

    for lado, (joelho, tornozelo) in enumerate(PERNAS):
        if joelho in vistos and tornozelo in vistos:
            (jx, jy), (x, y) = vistos[joelho], vistos[tornozelo]
            tam = par["calcado_lado"] * hypot(x - jx, y - jy)
            if tam > 0:
                r["calcado"][lado] = (x - tam / 2, y, x + tam / 2, y + tam)
    return r


def _combinar(estados) -> Estado:
    """Algum lado ausente: ausente; senão, algum incerto: incerto; senão, presente."""
    return min(estados, key=_ORDEM.__getitem__)


def _estado_b(lados, dets, par) -> Estado:
    if len(lados) == 1:
        if lados[0] is None:
            return Estado.INCERTO
        itens = [d for d in dets if _centro_em(d, lados[0])]
        return _estado_item(max(itens, key=lambda d: d.conf, default=None), par)
    # Testa todas as distribuições de detecções distintas entre os dois lados (None: nenhuma),
    # porque dar a melhor ao primeiro lado pode deixar o segundo sem a única que lhe serve.
    opcoes = [*range(len(dets)), None, None]
    resultados = []
    for escolha in permutations(opcoes, 2):
        estados = []
        for caixa, i in zip(lados, escolha):
            if i is not None and (caixa is None or not _centro_em(dets[i], caixa)):
                break
            if caixa is None:
                estados.append(Estado.INCERTO)
            else:
                estados.append(_estado_item(None if i is None else dets[i], par))
        else:
            resultados.append(_combinar(estados))
    return max(resultados, key=_ORDEM.__getitem__)


def associar_b(pessoa, pontos, dets, forma, par):
    if pessoa is None or pontos is None:
        return _incertos()
    areas = regioes(pontos, forma, par)
    return {epi: _estado_b(areas[epi], _itens(dets, epi, par), par) for epi in EPIS}
