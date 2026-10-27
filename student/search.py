"""
search.py – Thuật toán tìm kiếm nâng cao cho ChessLab StudentBot.

Kiến thức nền tảng từ môn học AI:
1. Thuật toán Đối kháng Minimax / Negamax:
   - Giả định đối thủ luôn đưa ra phản ứng tối ưu.
   - Thể hiện nguyên lý quy tắc đối ngẫu (Zero-Sum Game): Điểm Max của bên này là -Min của bên kia.
2. Cắt tỉa Alpha-Beta Pruning:
   - Duy trì khoảng [alpha, beta] để loại bỏ các nhánh cây không ảnh hưởng đến giá trị Minimax gốc.
   - Giúp giảm độ phức tạp từ O(b^d) xuống tiệm cận O(b^(d/2)) khi sắp xếp nước đi lý tưởng.
3. Đào sâu lặp (Iterative Deepening Search - IDS):
   - Kết hợp ưu điểm tiết kiệm bộ nhớ của DFS với khả năng kiểm soát độ sâu và thời gian của BFS.
   - Cung cấp nước đi dự phòng tin cậy khi gặp giới hạn thời gian (Time-bounded search).
4. Heuristic Sắp xếp Nước đi (Move Ordering):
   - Tối đa hóa số lượng cắt nhánh (Beta Cutoffs) sớm:
     + Nước tốt nhất từ Transposition Table (Hash Move)
     + Phong cấp thành Hậu
     + Nước bắt quân có lợi qua Static Exchange Evaluation (SEE >= 0) kết hợp MVV-LVA
     + Killer Moves Heuristic (nước đi từng gây cắt tỉa ở cùng độ sâu)
     + History Heuristic (tần suất gây cắt tỉa trong toàn bộ quá trình tìm kiếm)
     + Nước bắt quân bất lợi (SEE < 0)
5. Quiescence Search (Tìm kiếm tĩnh lặng):
   - Giải quyết triệt để hiệu ứng chân trời (Horizon Effect) bằng cách chỉ đào sâu các chuỗi bắt quân và thoát chiếu.
   - Delta Pruning loại bỏ các nhánh bắt quân không có khả năng lật ngược tình thế.
6. Kỹ thuật nâng cao:
   - Principal Variation Search (PVS / Zero-Window Search): Tìm kiếm với cửa sổ rỗng [alpha, alpha+1] cho các nước đi phía sau để tăng tốc cắt tỉa.
   - Aspiration Windows: Thu hẹp cửa sổ tìm kiếm ở gốc quanh điểm số ước lượng của độ sâu trước.
   - Null Move Pruning (NMP): Thử nước rỗng để kiểm tra ưu thế vượt trội (tránh Zugzwang).
   - Late Move Reductions (LMR): Giảm độ sâu cho các nước yên lặng ít triển vọng; không giảm nước chiếu.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import chess

from .evaluation import evaluate
from .time_manager import TimeManager

# ---------------------------------------------------------------------------
# Hằng số
# ---------------------------------------------------------------------------
MATE_SCORE = 30_000          # Điểm chiếu hết
MATE_THRESHOLD = 29_000      # Ngưỡng nhận diện điểm chiếu hết
DRAW_SCORE = 0
MAX_DEPTH = 64
MAX_QSEARCH_DEPTH = 8
TT_SIZE = 1 << 18            # 262,144 mục

# Loại mục trong Transposition Table
TT_EXACT = 0
TT_LOWER = 1   # fail-high (score >= beta)
TT_UPPER = 2   # fail-low  (score <= alpha)

# Giá trị quân cờ dùng cho MVV-LVA và Static Exchange Evaluation (SEE)
PIECE_VALS: dict[chess.PieceType | None, int] = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  1000,
    chess.KING:   20000,
    None:         0,
}

_MVV_LVA_SCORES: dict[chess.PieceType | None, int] = {
    chess.PAWN:   1,
    chess.KNIGHT: 2,
    chess.BISHOP: 3,
    chess.ROOK:   4,
    chess.QUEEN:  5,
    chess.KING:   6,
    None:         0,
}


def score_to_tt(score: int, ply: int) -> int:
    """Chuẩn hóa điểm trước khi lưu vào TT (đặc biệt là mate score)."""
    if score >= MATE_THRESHOLD:
        return score + ply
    elif score <= -MATE_THRESHOLD:
        return score - ply
    return score


def score_from_tt(score: int, ply: int) -> int:
    """Khôi phục điểm khi đọc từ TT."""
    if score >= MATE_THRESHOLD:
        return score - ply
    elif score <= -MATE_THRESHOLD:
        return score + ply
    return score


def see(board: chess.Board, move: chess.Move) -> int:
    """
    Static Exchange Evaluation (SEE):
    Ước lượng sự chênh lệch vật chất thuần sau chuỗi đổi quân tại ô đích (move.to_square).
    Trả về giá trị theo centipawn:
    >= 0: Nước bắt quân có lợi hoặc đổi ngang giá trị.
    < 0: Nước bắt quân bị lỗ (ví dụ: lấy Hậu ăn Tốt bị bảo vệ).
    """
    to_sq = move.to_square
    from_sq = move.from_square

    victim = board.piece_type_at(to_sq)
    if victim is None:
        if board.is_en_passant(move):
            victim = chess.PAWN
        else:
            return 0

    attacker = board.piece_type_at(from_sq)
    if attacker is None:
        return 0

    v_val = PIECE_VALS[victim]
    a_val = PIECE_VALS[attacker]

    # Nếu quân đi ăn có giá trị <= quân bị ăn (ví dụ Tốt ăn Hậu, Xe ăn Hậu, Tượng ăn Mã...):
    # Luôn là nước bắt quân có lợi hoặc ít nhất là đổi quân ngang
    if a_val <= v_val:
        return v_val - a_val

    # Nếu quân lớn ăn quân nhỏ (ví dụ Hậu ăn Tốt):
    # Kiểm tra xem ô đích có được đối phương bảo vệ không
    opp = not board.turn
    if not board.is_attacked_by(opp, to_sq):
        return v_val  # Quân treo tự do!

    # Ô đích được bảo vệ: quân tấn công sẽ bị ăn lại và lỗ vật chất
    return v_val - a_val


# ---------------------------------------------------------------------------
# Transposition Table
# ---------------------------------------------------------------------------

class TTEntry(NamedTuple):
    key: Any
    depth: int
    score: int
    flag: int
    best_move: chess.Move | None
    age: int


class TranspositionTable:
    """Bảng băm lưu trữ trạng thái tìm kiếm với kích thước cố định."""

    def __init__(self, size: int = TT_SIZE) -> None:
        self.size = size
        self._table: dict[int, TTEntry] = {}

    def probe(self, key: Any) -> TTEntry | None:
        idx = hash(key) & (self.size - 1)
        entry = self._table.get(idx)
        if entry is not None and entry.key == key:
            return entry
        return None

    def store(
        self,
        key: Any,
        depth: int,
        score: int,
        flag: int,
        best_move: chess.Move | None,
        age: int,
    ) -> None:
        idx = hash(key) & (self.size - 1)
        existing = self._table.get(idx)
        if (
            existing is None
            or existing.key == key
            or existing.age != age
            or depth >= existing.depth
        ):
            self._table[idx] = TTEntry(key, depth, score, flag, best_move, age)

    def clear(self) -> None:
        self._table.clear()


# ---------------------------------------------------------------------------
# Searcher
# ---------------------------------------------------------------------------

class Searcher:
    """
    Bộ máy tìm kiếm cờ vua Negamax + Alpha-Beta, PVS, LMR, NMP và Aspiration Windows.
    """

    def __init__(self) -> None:
        self.tt = TranspositionTable()
        self.killer_moves: list[list[chess.Move | None]] = [
            [None, None] for _ in range(MAX_DEPTH + 16)
        ]
        self.history: dict[tuple[chess.Color, int, int], int] = {}
        self._timeout: bool = False
        self._nodes: int = 0
        self._age: int = 0

    def _decay_history(self) -> None:
        """Làm suy giảm điểm history giữa các lượt đi để tránh tích lũy dữ liệu cũ."""
        for k in list(self.history.keys()):
            val = self.history[k] // 2
            if val > 0:
                self.history[k] = val
            else:
                del self.history[k]

    def _store_killer(self, move: chess.Move, ply: int) -> None:
        if ply < len(self.killer_moves):
            if self.killer_moves[ply][0] != move:
                self.killer_moves[ply][1] = self.killer_moves[ply][0]
                self.killer_moves[ply][0] = move

    def _update_history(self, move: chess.Move, color: chess.Color, depth: int) -> None:
        key = (color, move.from_square, move.to_square)
        self.history[key] = self.history.get(key, 0) + depth * depth

    def _order_moves(
        self,
        board: chess.Board,
        moves: list[chess.Move],
        ply: int,
        tt_move: chess.Move | None,
    ) -> list[chess.Move]:
        """
        Sắp xếp nước đi heuristic:
        1. TT Move (+2,000,000)
        2. Queen Promotion (+1,200,000)
        3. Good Captures (SEE >= 0) kết hợp MVV-LVA (+1,000,000)
        4. Killer Move 1 (+700,000)
        5. Killer Move 2 (+650,000)
        6. History Heuristic (+1 .. +500,000)
        7. Bad Captures (SEE < 0) (+50,000 + SEE)
        8. Các nước phong cấp khác (+40,000)
        9. Quiet moves
        """
        killers = self.killer_moves[ply] if ply < len(self.killer_moves) else [None, None]
        turn = board.turn

        def score_move(move: chess.Move) -> int:
            if move == tt_move:
                return 2_000_000

            if move.promotion == chess.QUEEN:
                return 1_200_000

            is_capture = board.is_capture(move)
            if is_capture:
                victim = board.piece_type_at(move.to_square)
                attacker = board.piece_type_at(move.from_square)
                mvv_lva = _MVV_LVA_SCORES[victim] * 10 - _MVV_LVA_SCORES[attacker]
                see_val = see(board, move)

                if see_val >= 0:
                    return 1_000_000 + mvv_lva * 100 + see_val
                else:
                    return 50_000 + see_val

            if move == killers[0]:
                return 700_000
            if move == killers[1]:
                return 650_000

            if move.promotion is not None:
                return 40_000

            key = (turn, move.from_square, move.to_square)
            return min(self.history.get(key, 0), 500_000)

        moves.sort(key=score_move, reverse=True)
        return moves

    # ------------------------------------------------------------------
    # Quiescence Search (Tìm kiếm tĩnh lặng)
    # ------------------------------------------------------------------

    def _quiescence(
        self,
        board: chess.Board,
        alpha: int,
        beta: int,
        ply: int,
        tm: TimeManager,
        qdepth: int = 0,
    ) -> int:
        """
        Quiescence Search: Chỉ đào sâu các nước bắt quân có lợi và thoát chiếu
        để khắc phục triệt để hiện tượng chân trời (Horizon Effect).
        """
        self._nodes += 1

        # Kiểm tra timeout mỗi 512 nodes
        if (self._nodes & 511) == 0 and tm.should_stop():
            self._timeout = True
            return 0

        if self._timeout:
            return 0

        # Kiểm tra hòa lặp và 50 nước
        if board.is_repetition(2) or board.halfmove_clock >= 100:
            return DRAW_SCORE

        in_check = board.is_check()

        # Stand-pat: ước lượng giá trị thế cờ hiện tại nếu không đi nước nào
        if not in_check:
            stand_pat = evaluate(board)
            if stand_pat >= beta:
                return beta
            if stand_pat > alpha:
                alpha = stand_pat

            # Delta Pruning: nếu vị trí quá kém, dù ăn được Hậu (+1000)
            # vẫn không thể vượt qua alpha thì cắt tỉa luôn nhánh này
            if stand_pat + 1150 < alpha:
                return alpha

        if qdepth >= MAX_QSEARCH_DEPTH and not in_check:
            return alpha

        # Lựa chọn nước đi
        if in_check:
            moves = list(board.legal_moves)
            if not moves:
                return -(MATE_SCORE - ply)
        else:
            # Chỉ xét bắt quân có lợi (SEE >= 0) và phong cấp Hậu
            moves = [
                m for m in board.legal_moves
                if m.promotion == chess.QUEEN or (board.is_capture(m) and see(board, m) >= 0)
            ]

        if not moves:
            return alpha

        # Sắp xếp các nước bắt quân trong Quiescence theo MVV-LVA
        def q_order(m: chess.Move) -> int:
            if m.promotion == chess.QUEEN:
                return 1000
            v = board.piece_type_at(m.to_square)
            a = board.piece_type_at(m.from_square)
            return _MVV_LVA_SCORES[v] * 10 - _MVV_LVA_SCORES[a]

        moves.sort(key=q_order, reverse=True)

        for move in moves:
            board.push(move)
            try:
                score = -self._quiescence(board, -beta, -alpha, ply + 1, tm, qdepth + 1)
            finally:
                board.pop()
            if self._timeout:
                return 0

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha

    # ------------------------------------------------------------------
    # Negamax với Alpha-Beta Pruning & PVS
    # ------------------------------------------------------------------

    def _negamax(
        self,
        board: chess.Board,
        depth: int,
        alpha: int,
        beta: int,
        ply: int,
        tm: TimeManager,
        allow_nmp: bool = True,
    ) -> int:
        """
        Negamax với Alpha-Beta, Principal Variation Search (PVS), NMP và LMR.
        """
        self._nodes += 1

        if (self._nodes & 511) == 0 and tm.should_stop():
            self._timeout = True
            return 0

        if self._timeout:
            return 0

        # Kiểm tra hòa lặp và thiếu quân
        if ply > 0:
            if board.is_repetition(2) or board.halfmove_clock >= 100:
                return DRAW_SCORE
            if board.is_insufficient_material():
                return DRAW_SCORE

        # --- Tra cứu Transposition Table ---
        tt_key = board._transposition_key()
        tt_entry = self.tt.probe(tt_key)
        tt_move: chess.Move | None = None

        if tt_entry is not None:
            tt_move = tt_entry.best_move
            if tt_entry.depth >= depth:
                tt_score = score_from_tt(tt_entry.score, ply)
                if tt_entry.flag == TT_EXACT:
                    return tt_score
                elif tt_entry.flag == TT_LOWER and tt_score >= beta:
                    return tt_score
                elif tt_entry.flag == TT_UPPER and tt_score <= alpha:
                    return tt_score

        # Khi hết độ sâu: chuyển sang Quiescence Search
        if depth <= 0:
            return self._quiescence(board, alpha, beta, ply, tm, 0)

        in_check = board.is_check()

        # --- Null Move Pruning (NMP) ---
        if allow_nmp and not in_check and depth >= 3 and ply > 0:
            has_pieces = any(
                board.pieces(pt, board.turn)
                for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
            )
            if has_pieces:
                r = 2 if depth <= 4 else 3
                board.push(chess.Move.null())
                try:
                    null_score = -self._negamax(
                        board, depth - 1 - r, -beta, -beta + 1, ply + 1, tm, False
    )
                finally:
                    board.pop()

                if self._timeout:
                    return 0

                if null_score >= beta:
                    return beta

        # Sinh và sắp xếp nước đi
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            if in_check:
                return -(MATE_SCORE - ply)
            return DRAW_SCORE

        moves = self._order_moves(board, legal_moves, ply, tt_move)

        best_score = -MATE_SCORE - 1
        best_move: chess.Move | None = None
        original_alpha = alpha
        moves_searched = 0

        for move in moves:
            is_capture = board.is_capture(move)
            gives_check = board.gives_check(move)

            board.push(move)
            try:
                moves_searched += 1

                # --- Late Move Reductions (LMR) ---
                # Giảm 1 độ sâu cho các nước đi yên lặng xếp sau; không giảm nếu là nước chiếu
                do_lmr = (
                    depth >= 3
                    and moves_searched >= 4
                    and not in_check
                    and not is_capture
                    and move.promotion is None
                    and not gives_check
                    and move != tt_move
                )

                # --- Principal Variation Search (PVS) ---
                if moves_searched == 1:
                    # Nước đầu tiên (PV move): Tìm kiếm với cửa sổ đầy đủ [alpha, beta]
                    score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1, tm, True)
                else:
                    # Các nước sau: Tìm kiếm với cửa sổ rỗng [-alpha - 1, -alpha]
                    r = 1 if do_lmr else 0
                    score = -self._negamax(board, depth - 1 - r, -alpha - 1, -alpha, ply + 1, tm, True)

                    # Nếu LMR fail-high: tìm kiếm lại đầy đủ không giảm độ sâu
                    if do_lmr and score > alpha:
                        score = -self._negamax(board, depth - 1, -alpha - 1, -alpha, ply + 1, tm, True)

                    # Nếu cửa sổ rỗng fail-high: tìm kiếm lại với cửa sổ đầy đủ [alpha, beta]
                    if score > alpha and score < beta:
                        score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1, tm, True)

            finally:
                board.pop()

            if self._timeout:
                return 0

            if score > best_score:
                best_score = score
                best_move = move

            if score > alpha:
                alpha = score

            if alpha >= beta:
                # Cắt tỉa Beta Cutoff
                if not is_capture:
                    self._store_killer(move, ply)
                    self._update_history(move, board.turn, depth)
                break

        # Lưu kết quả vào Transposition Table
        if not self._timeout:
            if best_score <= original_alpha:
                flag = TT_UPPER
            elif best_score >= beta:
                flag = TT_LOWER
            else:
                flag = TT_EXACT
            tt_score = score_to_tt(best_score, ply)
            self.tt.store(tt_key, depth, tt_score, flag, best_move, self._age)

        return best_score

    # ------------------------------------------------------------------
    # Iterative Deepening với Aspiration Windows
    # ------------------------------------------------------------------

    def search(
        self,
        board: chess.Board,
        tm: TimeManager,
    ) -> chess.Move:
        """
        Tìm kiếm lặp tăng dần độ sâu (Iterative Deepening Search).
        Kết hợp Aspiration Windows để thu hẹp cửa sổ [alpha, beta] ở gốc.
        """
        legal = list(board.legal_moves)
        if not legal:
            raise ValueError("No legal moves available")

        if len(legal) == 1:
            return legal[0]

        best_move = legal[0]
        self._timeout = False
        self._nodes = 0
        self._age += 1
        self._decay_history()
        tm.start()

        prev_score = 0
        aspiration_delta = 40

        for depth in range(1, MAX_DEPTH + 1):
            if tm.should_stop():
                break

            if depth > 1 and not tm.can_start_next_depth():
                break

            self._timeout = False
            candidate: chess.Move | None = None
            best_score_this_depth = -MATE_SCORE - 1

            # Thiết lập Aspiration Window ở depth >= 4
            if depth >= 4 and abs(prev_score) < MATE_THRESHOLD:
                alpha = max(-MATE_SCORE - 1, prev_score - aspiration_delta)
                beta = min(MATE_SCORE + 1, prev_score + aspiration_delta)
            else:
                alpha = -MATE_SCORE - 1
                beta = MATE_SCORE + 1

            while True:
                candidate = None
                best_score_this_depth = -MATE_SCORE - 1

                tt_entry = self.tt.probe(board._transposition_key())
                root_tt_move = tt_entry.best_move if tt_entry is not None else None

                moves = self._order_moves(board, list(board.legal_moves), ply=0, tt_move=root_tt_move)

                moves_searched = 0
                for move in moves:
                    board.push(move)
                    try:
                        moves_searched += 1

                        if moves_searched == 1:
                            score = -self._negamax(board, depth - 1, -beta, -alpha, 1, tm, True)
                        else:
                            score = -self._negamax(board, depth - 1, -alpha - 1, -alpha, 1, tm, True)
                            if score > alpha and score < beta:
                                score = -self._negamax(board, depth - 1, -beta, -alpha, 1, tm, True)

                    finally:
                        board.pop()

                    if self._timeout:
                        break

                    if score > best_score_this_depth:
                        best_score_this_depth = score
                        candidate = move

                    if score > alpha:
                        alpha = score

                if self._timeout:
                    break

                # Kiểm tra Aspiration Window fail-low hoặc fail-high
                if depth >= 4 and abs(best_score_this_depth) < MATE_THRESHOLD:
                    if best_score_this_depth <= prev_score - aspiration_delta:
                        # Fail-low: mở rộng alpha về vô cùng
                        alpha = -MATE_SCORE - 1
                        continue
                    elif best_score_this_depth >= prev_score + aspiration_delta:
                        # Fail-high: mở rộng beta về vô cùng
                        beta = MATE_SCORE + 1
                        continue

                break

            if not self._timeout and candidate is not None:
                best_move = candidate
                prev_score = best_score_this_depth

                self.tt.store(
                    board._transposition_key(),
                    depth,
                    score_to_tt(best_score_this_depth, 0),
                    TT_EXACT,
                    best_move,
                    self._age,
                )

                if best_score_this_depth >= MATE_THRESHOLD:
                    break

        return best_move

