import pytest

from epi.associacao import EPIS, Deteccao, Estado, associar_a, no_portao, selecionar_pessoa
from epi.decisao import (
    BLOQUEADO, FIM_VIDEO, INDETERMINADO, LIBERADO, Passagem, decidir, votar,
)

P, A, I = Estado.PRESENTE, Estado.AUSENTE, Estado.INCERTO
PAR = {
    "exigidos": list(EPIS), "n": 5, "maioria": 0.6,
    "k_entrada": 3, "k_saida": 4, "max_quadros": 20,
}


def quadro(estado=P, **outros):
    return {**dict.fromkeys(EPIS, estado), **outros}


def rodar(passagem, quadros):
    """Cada item: estados (pessoa no portão), "fora" (pessoa fora do portão) ou None (vazio)."""
    eventos = []
    for q in quadros:
        eventos += passagem.atualizar(q is not None, q if isinstance(q, dict) else None)
    return eventos


def resultados(eventos):
    return [e.decisao.resultado for e in eventos if e.tipo == "decisao"]


def test_votar_maioria_e_indefinido():
    janela = [quadro(P)] * 3 + [quadro(A)] * 2
    assert votar(janela, 5, 0.6)["capacete"] == P
    assert votar(janela, 5, 0.8)["capacete"] == I
    assert votar([quadro(A)] * 3, 5, 0.6)["capacete"] == A
    assert votar([quadro(A)] * 5 + [quadro(P)] * 5, 5, 0.6)["capacete"] == P  # só os últimos N


def test_decidir():
    assert decidir(quadro(P), EPIS).resultado == LIBERADO
    bloqueio = decidir(quadro(P, luvas=A, oculos=I), EPIS)
    assert (bloqueio.resultado, bloqueio.ausentes) == (BLOQUEADO, ("luvas",))
    assert decidir(quadro(P, oculos=I), EPIS) is None
    assert decidir(quadro(P, oculos=A), ["capacete"]).resultado == LIBERADO  # regra da área
    with pytest.raises(ValueError):
        decidir(quadro(A), [])


@pytest.mark.parametrize("estado, esperado", [(P, LIBERADO), (A, BLOQUEADO)])
def test_decisao_unica_por_passagem(estado, esperado):
    passagem = Passagem(PAR)
    eventos = rodar(passagem, [quadro(estado)] * 5)
    assert [e.tipo for e in eventos] == ["inicio", "decisao"]
    assert resultados(eventos) == [esperado] and eventos[-1].quadros == 5
    depois = rodar(passagem, [quadro(A)] * 40 + [None] * 4)
    assert [e.tipo for e in depois] == ["fim"]


def test_janela_incompleta_nao_decide():
    assert resultados(rodar(Passagem(PAR), [quadro(P)] * 4)) == []


def test_abertura_exige_quadros_seguidos():
    passagem = Passagem(PAR)
    assert rodar(passagem, [quadro(), quadro(), None, quadro(), quadro()]) == []
    assert [e.tipo for e in rodar(passagem, [quadro()])] == ["inicio"]


def test_falha_curta_de_deteccao_nao_fecha_a_passagem():
    eventos = rodar(Passagem(PAR), [quadro()] * 3 + [None] * 3 + [quadro()] * 2)
    assert [e.tipo for e in eventos] == ["inicio", "decisao"]


def test_reinicio_da_janela_na_passagem_seguinte():
    passagem = Passagem(PAR)
    primeira = rodar(passagem, [quadro(P)] * 5 + [None] * 4)
    segunda = rodar(passagem, [quadro(A)] * 4)
    assert resultados(primeira) == [LIBERADO] and resultados(segunda) == []
    terceira = rodar(passagem, [quadro(A)])
    assert resultados(terceira) == [BLOQUEADO] and terceira[-1].passagem == 2


def test_passagem_fecha_sem_decisao():
    eventos = rodar(Passagem(PAR), [quadro(I)] * 6 + [None] * 4)
    assert [e.tipo for e in eventos] == ["inicio", "decisao", "fim"]
    assert eventos[1].decisao.motivo == "passagem_fechada"
    assert eventos[1].decisao.resultado == INDETERMINADO


def test_tempo_esgotado_conta_quadros_fora_do_portao():
    eventos = rodar(Passagem(PAR), ["fora"] * 20)
    assert resultados(eventos) == [INDETERMINADO]
    assert eventos[-1].decisao.motivo == "tempo_esgotado" and eventos[-1].quadros == 20


@pytest.mark.parametrize("motivo", [FIM_VIDEO, "falha_camera"])
def test_encerrar_passagem_aberta(motivo):
    passagem = Passagem(PAR)
    rodar(passagem, [quadro()] * 4)
    eventos = passagem.encerrar(motivo)
    assert [e.tipo for e in eventos] == ["decisao", "fim"]
    assert resultados(eventos) == [INDETERMINADO]


def test_falha_fora_de_passagem_gera_indeterminado_e_fim_do_video_nao():
    assert Passagem(PAR).encerrar(FIM_VIDEO) == []
    eventos = Passagem(PAR).encerrar("falha_camera")
    assert resultados(eventos) == [INDETERMINADO] and eventos[0].passagem == 0


def test_falha_apos_decisao_nao_repete():
    passagem = Passagem(PAR)
    rodar(passagem, [quadro()] * 5)
    assert [e.tipo for e in passagem.encerrar("falha_camera")] == ["fim"]


def test_pessoa_conforme_entrando_aos_poucos_nao_e_bloqueada():
    """Com a caixa na borda o item fora do quadro some; o portão não deixa isso votar."""
    par = {**PAR, "conf_min": 0.25, "conf_ok": 0.5, "margem_borda": 0.01, "altura_min": 0.5, "iou_duplicata": 0.5}
    forma = (600, 400, 3)

    def dets(x1, com_calcado):
        pessoa = Deteccao("pessoa", 0.9, (x1, 40, x1 + 200, 580))
        itens = [
            Deteccao(epi, 0.9, (x1 + 60 + 40 * lado, 300, x1 + 80 + 40 * lado, 320))
            for epi in EPIS
            for lado in range(2 if epi in ("luvas", "calcado") else 1)
            if com_calcado or epi != "calcado"
        ]
        return [pessoa, *itens]

    def processar(passagem, d, usar_portao=True):
        ha, pessoa = selecionar_pessoa(d, par)
        passou = no_portao(d, forma, par) or not usar_portao
        return passagem.atualizar(ha, associar_a(pessoa, d, par) if passou else None)

    sequencia = [dets(0, False)] * 8 + [dets(100, True)] * 5
    passagem = Passagem(par)
    assert resultados([e for d in sequencia for e in processar(passagem, d)]) == [LIBERADO]
    # Sem o portão, a mesma sequência bloquearia.
    passagem = Passagem(par)
    sem_portao = [e for d in sequencia for e in processar(passagem, d, False)]
    assert resultados(sem_portao) == [BLOQUEADO]


def test_duas_pessoas_nunca_liberam():
    par = {**PAR, "conf_min": 0.25, "conf_ok": 0.5, "margem_borda": 0.01, "altura_min": 0.5, "iou_duplicata": 0.5}
    d = [Deteccao("pessoa", 0.9, (50, 40, 200, 580)), Deteccao("pessoa", 0.9, (210, 40, 360, 580))]
    passagem = Passagem(par)
    eventos = []
    for _ in range(25):
        ha, pessoa = selecionar_pessoa(d, par)
        assert no_portao(d, (600, 400, 3), par)
        eventos += passagem.atualizar(ha, associar_a(pessoa, d, par))
    assert resultados(eventos) == [INDETERMINADO]
