# Cell 1 — metrics + STM tensor (non-white-centric)
import torch, chess

BB_CENTER4   = chess.BB_D4 | chess.BB_E4 | chess.BB_D5 | chess.BB_E5
BB_CENTER16  = (chess.BB_C3 | chess.BB_D3 | chess.BB_E3 | chess.BB_F3 |
                chess.BB_C4 | chess.BB_D4 | chess.BB_E4 | chess.BB_F4 |
                chess.BB_C5 | chess.BB_D5 | chess.BB_E5 | chess.BB_F5 |
                chess.BB_C6 | chess.BB_D6 | chess.BB_E6 | chess.BB_F6)
BB_ALL = (1 << 64) - 1
MAT_VAL = {"P":100,"N":310,"B":330,"R":500,"Q":900}

def pawn_adv_steps(sq, color):
    r = chess.square_rank(sq)
    return (6 - r) if color == chess.WHITE else (r - 1)

def m_material(b):
    s=0
    for p,k in [(chess.PAWN,"P"),(chess.KNIGHT,"N"),(chess.BISHOP,"B"),
                (chess.ROOK,"R"),(chess.QUEEN,"Q")]:
        s += MAT_VAL[k]*(len(b.pieces(p, True)) - len(b.pieces(p, False)))
    return float(s)

def m_pawn_adv(b):
    s=0
    for c in (chess.WHITE,chess.BLACK):
        for sq in b.pieces(chess.PAWN,c):
            s += (12*pawn_adv_steps(sq,c))*(1 if c==chess.WHITE else -1)
    return float(s)

def _center_bonus_for_piece(b, piece_type, c4=12, c16=6):
    s=0
    for c in (chess.WHITE,chess.BLACK):
        for sq in b.pieces(piece_type,c):
            bonus=0
            if chess.BB_SQUARES[sq] & BB_CENTER4: bonus += c4
            elif chess.BB_SQUARES[sq] & BB_CENTER16: bonus += c16
            s += bonus if c==chess.WHITE else -bonus
    return float(s)

def m_knight_center(b): return _center_bonus_for_piece(b, chess.KNIGHT, 25, 12)
def m_bishop_center(b): return _center_bonus_for_piece(b, chess.BISHOP, 15, 8)
def m_rook_center(b):   return _center_bonus_for_piece(b, chess.ROOK,   10, 6)
def m_queen_center(b):  return _center_bonus_for_piece(b, chess.QUEEN,  12, 6)
def m_king_center(b):   return _center_bonus_for_piece(b, chess.KING,    6, 3)

def m_central_ctrl(b):
    s=0
    for c in (chess.WHITE,chess.BLACK):
        ctrl=0
        for sq in chess.SquareSet(b.occupied_co[c]):
            ctrl += len(b.attacks(sq) & BB_CENTER16)
        s += 2*ctrl if c==chess.WHITE else -2*ctrl
    return float(s)

def m_mobility_total(b):
    s=0
    for c in (chess.WHITE,chess.BLACK):
        own=b.occupied_co[c]
        mob=0
        for sq in chess.SquareSet(own):
            mob += len(b.attacks(sq) & ((~own) & BB_ALL))
        s += mob if c==chess.WHITE else -mob
    return float(s)

def m_coordination(b):
    s=0
    for c in (chess.WHITE,chess.BLACK):
        own=b.occupied_co[c]
        defended=0
        for sq in chess.SquareSet(own):
            defended += len(b.attacks(sq) & own)
        s += 2*defended if c==chess.WHITE else -2*defended
    return float(s)

def m_space(b):
    WHITE_HALF = chess.BB_RANK_1 | chess.BB_RANK_2 | chess.BB_RANK_3 | chess.BB_RANK_4
    BLACK_HALF = chess.BB_RANK_5 | chess.BB_RANK_6 | chess.BB_RANK_7 | chess.BB_RANK_8
    s=0
    for c in (chess.WHITE,chess.BLACK):
        half = BLACK_HALF if c==chess.WHITE else WHITE_HALF
        ctrl=0
        for sq in chess.SquareSet(b.occupied_co[c]):
            ctrl += len(b.attacks(sq) & half)
        s += ctrl if c==chess.WHITE else -ctrl
    return float(s)

def m_development(b):
    def side(col):
        back = chess.BB_RANK_1 if col==chess.WHITE else chess.BB_RANK_8
        minors_on_back=0
        for pt in (chess.KNIGHT,chess.BISHOP):
            minors_on_back += len(chess.SquareSet(b.pieces(pt,col) & back))
        rooks=b.pieces(chess.ROOK,col)
        rooks_connected = 1 if (len(rooks)>=2 and not b.pieces(chess.BISHOP,col)
                                and not b.pieces(chess.KNIGHT,col) and not b.pieces(chess.QUEEN,col)) else 0
        return -(minors_on_back*20) + rooks_connected*15
    return float(side(chess.WHITE)-side(chess.BLACK))

def m_castling_ready(b):
    def side(col):
        bonus=0
        if b.has_kingside_castling_rights(col) or b.has_queenside_castling_rights(col):
            bonus += 10
        k_sq = b.king(col)
        if k_sq is not None:
            r=chess.square_rank(k_sq); f=chess.square_file(k_sq)
            if (col==chess.WHITE and r==0 and f in (6,2)) or (col==chess.BLACK and r==7 and f in (6,2)):
                bonus += 25
        return bonus
    return float(side(chess.WHITE)-side(chess.BLACK))

METRIC_FUNCS = [
    m_material, m_pawn_adv,
    m_knight_center, m_bishop_center, m_rook_center, m_queen_center, m_king_center,
    m_central_ctrl, m_mobility_total, m_coordination, m_space,
    m_development, m_castling_ready,
]
N_METRICS = len(METRIC_FUNCS)

def game_phase(b: chess.Board) -> float:
    w=0
    w += len(b.pieces(chess.KNIGHT, True)) + len(b.pieces(chess.KNIGHT, False))
    w += len(b.pieces(chess.BISHOP, True)) + len(b.pieces(chess.BISHOP, False))
    w += 2*(len(b.pieces(chess.ROOK, True)) + len(b.pieces(chess.ROOK, False)))
    w += 4*(len(b.pieces(chess.QUEEN, True)) + len(b.pieces(chess.QUEEN, False)))
    return min(1.0, max(0.0, w/24.0))

# ⚠️ STM features: compute white-centric metrics, then flip if Black to move
def board_to_metric_tensor(b: chess.Board) -> torch.Tensor:
    vals = [f(b) for f in METRIC_FUNCS]  # centipawns, white-centric
    phase = game_phase(b)
    IDX_PADV, IDX_SPACE, IDX_CTRL = 1, 10, 7
    vals[IDX_PADV]  *= (0.3 + 1.0*(1.0 - phase))
    vals[IDX_SPACE] *= (0.6 + 0.6*(1.0 - phase))
    vals[IDX_CTRL]  *= 1.1
    x = torch.tensor(vals, dtype=torch.float32) / 1000.0
    if b.turn == chess.BLACK:
        x = -x
    return x  # cp/1000, STM (positive = good for side to move)


# Cell 2 — model + robust checkpoint loading
import torch.nn as nn

class LinearMetricModel(nn.Module):
    def __init__(self, n=N_METRICS, use_bias=True):
        super().__init__()
        self.use_bias = use_bias
        self.w = nn.Parameter(torch.zeros(n))
        if use_bias:
            self.b = nn.Parameter(torch.zeros(1))
    def forward(self, X):
        out = X @ self.w
        if self.use_bias:
            out = out + self.b
        return out  # cp/1000 (STM)

device = "cuda" if torch.cuda.is_available() else "cpu"
#device ="cpu"

ckpt_path = "/home/dan/code/python_projects/lichessBot/quick_04403200_acc.pt"

state = torch.load(ckpt_path, map_location="cpu")
sd = state["model"]

has_bias = "b" in sd
model = LinearMetricModel(n=N_METRICS, use_bias=has_bias).to(device)
if has_bias:
    model.load_state_dict(sd, strict=True)
else:
    model.load_state_dict(sd, strict=False)

model.eval()

NORM_MEAN = state.get("norm_mean", None)
NORM_STD  = state.get("norm_std", None)
if NORM_MEAN is not None and NORM_STD is not None:
    NORM_MEAN = NORM_MEAN.to(device)
    NORM_STD  = torch.clamp(NORM_STD, min=1e-8).to(device)


# Cell 3 — alpha-beta search using STM model
import time
from typing import List, Tuple, Optional
from chess.polyglot import zobrist_hash

USE_QSEARCH = True
FUTILITY_MARGIN_CP = 150.0
MATE_SCORE = 1_000_000
CONTEMPT_CP = 20
PRINT_PV_EACH_DEPTH = True
PRINT_FINAL_SUMMARY = True

PIECE_VAL = {None:0, chess.PAWN:100, chess.KNIGHT:300, chess.BISHOP:300, chess.ROOK:500, chess.QUEEN:900, chess.KING:10000}

def mvv_lva(board: chess.Board, move: chess.Move) -> int:
    v = PIECE_VAL.get(board.piece_type_at(move.to_square), 0)
    a = PIECE_VAL.get(board.piece_type_at(move.from_square), 0)
    return 16*v - a

def ordered_legal_moves(board: chess.Board, tt_move=None, ply=0):
    caps, quiets = [], []
    for mv in board.legal_moves:
        if tt_move and mv == tt_move: continue
        (caps if board.is_capture(mv) else quiets).append(mv)
    caps.sort(key=lambda m: mvv_lva(board, m), reverse=True)
    return ([tt_move] if tt_move else []) + caps + quiets

def eval_stm_cp(board: chess.Board, model, device="cpu") -> float:
    key = zobrist_hash(board)
    val = EVAL_CACHE.get(key)
    if val is None:
        x = board_to_metric_tensor(board).unsqueeze(0).to(device)
        if NORM_MEAN is not None and NORM_STD is not None:
            x = (x - NORM_MEAN) / NORM_STD
        with torch.no_grad():
            val = model(x).item() * 1000.0
        EVAL_CACHE[key] = val
    return val

def terminal_score_stm(board: chess.Board, ply_from_root: int) -> Optional[float]:
    if board.is_checkmate(): return -(MATE_SCORE - ply_from_root)
    if (board.is_stalemate() or board.is_insufficient_material()
        or board.is_seventyfive_moves() or board.is_fivefold_repetition()):
        return 0.0
    return None

def qsearch(board: chess.Board, model, device="cpu", alpha=-1e9, beta=1e9, ply_from_root=0):
    stand = eval_stm_cp(board, model, device)
    if stand >= beta: return beta
    if stand > alpha: alpha = stand
    for mv in board.legal_moves:
        if not board.is_capture(mv): continue
        board.push(mv)
        score = -qsearch(board, model, device, -beta, -alpha, ply_from_root+1)
        board.pop()
        if score >= beta: return beta
        if score > alpha: alpha = score
    return alpha

# --- time control helper ---
def time_up(t0: float, time_limit_s: float) -> bool:
    return time_limit_s > 0 and (time.time() - t0) >= time_limit_s

EVAL_CACHE = {}
TT = {}
_nodes = 0

def negamax(board: chess.Board, depth: int, model, device="cpu",
            alpha=-1e9, beta=1e9, ply_from_root=0,
            t0: float=0.0, time_limit_s: float=0.0) -> Tuple[float, List[chess.Move]]:
    global _nodes
    _nodes += 1

    ts = terminal_score_stm(board, ply_from_root)
    if ts is not None:
        return ts, []

    if depth == 0:
        return (qsearch(board, model, device, alpha, beta, ply_from_root) if USE_QSEARCH
                else eval_stm_cp(board, model, device)), []

    key = zobrist_hash(board)
    tt_hit = TT.get(key)
    tt_move = None
    if tt_hit:
        tt_depth, tt_score, _, tt_mv_uci = tt_hit
        if tt_mv_uci:
            tt_move = chess.Move.from_uci(tt_mv_uci)
        if tt_depth >= depth:
            return tt_score, [tt_move] if tt_move else []

    best_score = -1e9
    best_move = None
    best_pv: List[chess.Move] = []

    for mv in ordered_legal_moves(board, tt_move, ply_from_root):
        board.push(mv)
        child_score, child_pv = negamax(board, depth-1, model, device,
                                        -beta, -alpha, ply_from_root+1,
                                        t0, time_limit_s)
        score = -child_score
        board.pop()

        if score > best_score:
            best_score, best_move, best_pv = score, mv, [mv] + child_pv

        if best_score > alpha: alpha = best_score
        if alpha >= beta: break
        if time_up(t0, time_limit_s): break

    TT[key] = (depth, best_score, 0, best_move.uci() if best_move else None)
    return best_score, best_pv

def search_root(board: chess.Board, model, depth_full: int, device="cpu", time_limit_s: float=0.0):
    global TT, EVAL_CACHE, _nodes
    TT, EVAL_CACHE, _nodes = {}, {}, 0
    t0 = time.time()

    best_move = None; best_eval = -1e9; best_pv = []
    for d in range(1, depth_full + 1):
        score, pv = negamax(board, d, model, device, -1e9, 1e9, 0,
                            t0=t0, time_limit_s=time_limit_s)
        if pv: best_move, best_eval, best_pv = pv[0], score, pv
        if PRINT_PV_EACH_DEPTH:
            elapsed = time.time() - t0
            print(f"[depth {d}] eval={score:+.1f} cp  nodes={_nodes:,}  time={elapsed:.2f}s")
        if time_up(t0, time_limit_s): break

    if PRINT_FINAL_SUMMARY and best_pv:
        elapsed = time.time() - t0
        print(f"[final] eval={best_eval:+.1f} cp  nodes={_nodes:,}  time={elapsed:.2f}s  best={' '.join(m.uci() for m in best_pv)}")
    return best_move, best_eval, best_pv

def best_move(board: chess.Board, model, depth=2, device="cpu", time_limit_s: float=0.0):
    mv, sc, pv = search_root(board, model, depth_full=depth, device=device, time_limit_s=time_limit_s)
    return mv, sc
