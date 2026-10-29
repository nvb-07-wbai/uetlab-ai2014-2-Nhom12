import chess
import pytest

from student.search import MATE_SCORE, Searcher
from student.time_manager import TimeManager


def test_quiescence_restores_board_if_recursive_search_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    board = chess.Board("4k3/8/8/8/8/8/p7/R3K3 w - - 0 1")
    original_fen = board.fen()
    original_stack_size = len(board.move_stack)

    searcher = Searcher()
    original_quiescence = searcher._quiescence
    recursive_calls = 0

    def raise_on_recursive_call(*args: object, **kwargs: object) -> int:
        nonlocal recursive_calls
        recursive_calls += 1

        if recursive_calls == 1:
            raise RuntimeError("simulated recursive-search failure")

        return original_quiescence(*args, **kwargs)

    monkeypatch.setattr(searcher, "_quiescence", raise_on_recursive_call)

    with pytest.raises(RuntimeError, match="simulated recursive-search failure"):
        original_quiescence(
            board,
            -MATE_SCORE,
            MATE_SCORE,
            0,
            TimeManager(1_000),
            0,
        )

    assert board.fen() == original_fen
    assert len(board.move_stack) == original_stack_size
