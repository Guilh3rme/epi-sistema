"""Votação temporal, regra da área e ciclo da passagem. Módulo puro."""

from collections import deque
from dataclasses import dataclass

from epi.associacao import EPIS, Estado

LIBERADO, BLOQUEADO, INDETERMINADO = "LIBERADO", "BLOQUEADO", "INDETERMINADO"
FIM_VIDEO = "fim_video"  # qualquer outro motivo de `encerrar` é falha


@dataclass(frozen=True)
class Decisao:
    resultado: str
    motivo: str
    ausentes: tuple = ()


@dataclass(frozen=True)
class Evento:
    tipo: str  # "inicio", "decisao" ou "fim"
    passagem: int  # 0: falha fora de qualquer passagem
    quadros: int = 0  # desde o primeiro com pessoa
    decisao: Decisao | None = None
    votos: dict | None = None


def votar(janela, n, maioria):
    """Estado de cada EPI nos últimos `n` quadros; INCERTO significa indefinido."""
    ultimos = list(janela)[-n:]
    votos = {}
    for epi in EPIS:
        votos[epi] = Estado.INCERTO
        for estado in (Estado.PRESENTE, Estado.AUSENTE):
            if sum(q[epi] == estado for q in ultimos) / n >= maioria:
                votos[epi] = estado
    return votos


def decidir(votos, exigidos) -> Decisao | None:
    """Aplica a regra da área; None significa aguardar."""
    if not exigidos:
        raise ValueError("A lista de EPIs exigidos não pode ser vazia.")
    ausentes = tuple(e for e in exigidos if votos[e] == Estado.AUSENTE)
    if ausentes:
        return Decisao(BLOQUEADO, "epi_ausente", ausentes)
    if all(votos[e] == Estado.PRESENTE for e in exigidos):
        return Decisao(LIBERADO, "todos_presentes")
    return None


class Passagem:
    def __init__(self, par):
        self.par = par
        self.numero = 0
        self._reiniciar()

    def _reiniciar(self):
        self.aberta = self.decidida = False
        self.quadros = self.sem_pessoa = 0
        self.janela = deque(maxlen=self.par["n"])

    def _votos(self):
        return votar(self.janela, self.par["n"], self.par["maioria"])

    def _evento(self, tipo, decisao=None):
        votos = None
        if decisao is not None:
            self.decidida = True
            votos = self._votos()
        return Evento(tipo, self.numero if self.aberta else 0, self.quadros, decisao, votos)

    def atualizar(self, ha_alguem, estados, varias_pessoas=False):
        """Processa um quadro; `estados=None` quando ele não passou pelo portão.

        Só libera em um quadro que passou pelo portão e tem uma única pessoa.
        """
        if not self.aberta and not ha_alguem:
            self._reiniciar()  # a abertura exige quadros seguidos com pessoa
            return []
        self.quadros += 1
        self.sem_pessoa = 0 if ha_alguem else self.sem_pessoa + 1
        if estados is not None and not self.decidida:
            self.janela.append(estados)

        eventos = []
        if not self.aberta:
            if self.quadros < self.par["k_entrada"]:
                return []
            self.aberta = True
            self.numero += 1
            eventos.append(self._evento("inicio"))
        if not self.decidida:
            decisao = None
            if len(self.janela) == self.par["n"]:
                decisao = decidir(self._votos(), self.par["exigidos"])
            sem_condicao = estados is None or varias_pessoas
            if decisao is not None and decisao.resultado == LIBERADO and sem_condicao:
                decisao = None
            if decisao is None and self.quadros >= self.par["max_quadros"]:
                decisao = Decisao(INDETERMINADO, "tempo_esgotado")
            if decisao is not None:
                eventos.append(self._evento("decisao", decisao))
        if self.sem_pessoa >= self.par["k_saida"]:
            eventos += self.encerrar("passagem_fechada", falha=False)
        return eventos

    def encerrar(self, motivo, falha=None):
        """Fecha a passagem (saída, fim do vídeo ou falha); sem decisão, é INDETERMINADO.

        Uma falha fora de passagem também gera INDETERMINADO.
        """
        if falha is None:
            falha = motivo != FIM_VIDEO
        eventos = []
        if not self.decidida and (self.aberta or falha):
            eventos.append(self._evento("decisao", Decisao(INDETERMINADO, motivo)))
        if self.aberta:
            eventos.append(self._evento("fim"))
        self._reiniciar()
        return eventos
