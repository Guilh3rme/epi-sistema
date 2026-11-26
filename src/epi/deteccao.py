"""Inferência com Ultralytics, sem gravar imagens, e detector roteirizado.

Não importa `ultralytics`: os modelos chegam carregados (ver `fluxo`).
"""

from epi.associacao import Deteccao, iou


def detectar(modelo, quadro, mapa_classes, conf=0.25, dispositivo=None):
    """Detecções cujas classes estão em `mapa_classes` (nome no modelo -> nome no projeto)."""
    r = modelo.predict(quadro, conf=conf, device=dispositivo, save=False, verbose=False)[0]
    caixas = zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())
    return [
        Deteccao(mapa_classes[r.names[int(c)]], p, tuple(caixa))
        for caixa, c, p in caixas
        if r.names[int(c)] in mapa_classes
    ]


def estimar_pose(modelo, quadro, pessoa, iou_min=0.5, dispositivo=None):
    """17 pontos (x, y, conf) da pose que corresponde à caixa da pessoa, ou None."""
    if pessoa is None:
        return None
    r = modelo.predict(quadro, device=dispositivo, save=False, verbose=False)[0]
    if r.keypoints is None or r.keypoints.conf is None:
        return None
    ious = [iou(pessoa.caixa, caixa) for caixa in r.boxes.xyxy.tolist()]
    if not ious or max(ious) < iou_min:
        return None
    return r.keypoints.data[ious.index(max(ious))].tolist()


def detector_roteirizado(roteiro):
    """Função(indice, pessoas) que devolve os EPIs que o roteiro diz existirem no quadro.

    Em `itens`, cada nome tem `classe`, `conf` e `caixa` em frações da caixa da pessoa;
    cada faixa de quadros lista os nomes presentes.
    """
    faixas, inicio = [], 0
    for faixa in roteiro["faixas"]:
        itens = [roteiro["itens"][nome] for nome in faixa.get("itens", [])]
        faixas.append((range(inicio, inicio + faixa["quadros"]), itens))
        inicio += faixa["quadros"]

    def itens_do_quadro(indice, pessoas):
        if not pessoas:
            return []
        x1, y1, x2, y2 = max(pessoas, key=lambda p: p.conf).caixa
        w, h = x2 - x1, y2 - y1
        return [
            Deteccao(i["classe"], i["conf"], (
                x1 + i["caixa"][0] * w, y1 + i["caixa"][1] * h,
                x1 + i["caixa"][2] * w, y1 + i["caixa"][3] * h,
            ))
            for intervalo, itens in faixas if indice in intervalo
            for i in itens
        ]

    return itens_do_quadro
