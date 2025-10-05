
# pip install requests python-chess

import os
import json
import time
import threading
import requests
import chess
from engine import best_move, model, device
LICHESS_TOKEN = LICHESS_TOKEN
BASE = "https://lichess.org"
HDRS = {
    "Authorization": f"Bearer {LICHESS_TOKEN}",
    "Accept": "application/x-ndjson",
    "User-Agent": "simple-bot/0.1 (+python-requests)"
}
FIRST_ENGINE_MOVE = True
#move_count = 0
# ---------------- ENGINE HOOK ----------------
def choose_move(board, time_left_ms=None, inc_ms=None, moves_played=None):
    multiplication = 1
    #mv, score = best_move(board, model, depth=64, device=device, time_limit_s=2000 / 1000.0)
    #return mv
    #global move_count
    #move count += 1
    #if move_count >= 20:
        #multiplication = 2
    #if move_count >= 40:
        #multiplication = 3
    # Expect BOTH time_left_ms and inc_ms in MILLISECONDS.
    # Policy: spend 100% increment + 3% of current clock. Emergency <5s → ~0.2s.
    if time_left_ms is None: time_left_ms = 0
    if inc_ms is None:       inc_ms = 0

    if time_left_ms <= 5000:
        budget_ms = max(1, min(200, time_left_ms - 1))                  # ~0.20s
    else:
        budget_ms = int(inc_ms*0.65 + multiplication * 0.02 * time_left_ms)                   # 100% inc + 3% clock
        budget_ms = max(1, min(budget_ms, max(1, time_left_ms - 1)))    # clamp to available

    mv, score = best_move(board, model, depth=64, device=device, time_limit_s=budget_ms / 1000.0)
    return mv
# ------------------------------------------------

def post(url, data=None):
    # Lichess wants form-encoded bodies on POSTs
    return requests.post(url, headers={"Authorization": f"Bearer {LICHESS_TOKEN}"}, data=data or {})

def accept_challenge(ch_id):
    r = post(f"{BASE}/api/challenge/{ch_id}/accept")
    if r.status_code // 100 != 2:
        print(f"[challenge] accept failed {ch_id}: {r.status_code} {r.text}")
    else:
        print(f"[challenge] accepted {ch_id}")

def decline_challenge(ch_id, reason="generic"):
    r = post(f"{BASE}/api/challenge/{ch_id}/decline", data={"reason": reason})
    print(f"[challenge] declined {ch_id} ({reason}): {r.status_code}")

def make_move(game_id, uci):
    r = post(f"{BASE}/api/bot/game/{game_id}/move/{uci}")
    if r.status_code // 100 != 2:
        print(f"[move] fail {game_id} {uci}: {r.status_code} {r.text}")
        return False
    return True

def resign(game_id):
    r = post(f"{BASE}/api/bot/game/{game_id}/resign")
    print(f"[resign] {game_id}: {r.status_code}")

def parse_ndjson_stream(resp):
    for line in resp.iter_lines(decode_unicode=True):
        if not line:  # keep-alive heartbeat
            continue
        try:
            yield json.loads(line)
        except Exception:
            pass

def game_loop(game_id):
    """Stream a single game and reply with moves."""
    url = f"{BASE}/api/bot/game/stream/{game_id}"
    print(f"[game {game_id}] start stream")
    with requests.get(url, headers=HDRS, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        board = chess.Board()
        my_color = None  # 'white' or 'black'

        # server-provided clock snapshot (ms)
        my_time_ms = 0
        my_inc_ms  = 0

        for ev in parse_ndjson_stream(resp):
            t = ev.get("type")

            if t == "gameFull":
                # initial state
                white_id = ev["white"].get("id")
                black_id = ev["black"].get("id")

                me = ev.get("you", {}).get("id") or who_am_i()
                my_color = "white" if white_id == me else "black"

                # initial FEN if provided
                initial_fen = ev.get("initialFen")
                if initial_fen and initial_fen != "startpos":
                    board = chess.Board(fen=initial_fen)

                # play existing moves
                s = ev.get("state", {})  # includes moves + clocks
                moves = s.get("moves", "")
                for u in moves.split():
                    board.push_uci(u)

                # read initial time & increment (ms) directly from stream
                wtime, btime = s.get("wtime"), s.get("btime")
                winc,  binc  = s.get("winc"),  s.get("binc")
                if my_color == "white":
                    my_time_ms, my_inc_ms = (wtime or 0), (winc or 0)
                else:
                    my_time_ms, my_inc_ms = (btime or 0), (binc or 0)

                _maybe_play_move(game_id, board, my_color, my_time_ms, my_inc_ms)

            elif t == "gameState":
                # rebuild board from moves (lichess streams startpos here)
                moves = ev.get("moves", "")
                board = chess.Board()
                for u in moves.split():
                    board.push_uci(u)

                # updated clock & increment from server (ms)
                wtime, btime = ev.get("wtime"), ev.get("btime")
                winc,  binc  = ev.get("winc"),  ev.get("binc")
                if my_color == "white":
                    my_time_ms, my_inc_ms = (wtime or 0), (winc or 0)
                else:
                    my_time_ms, my_inc_ms = (btime or 0), (binc or 0)

                _maybe_play_move(game_id, board, my_color, my_time_ms, my_inc_ms)

            elif t == "chatLine":
                print(f"[chat {game_id}] {ev.get('username')}: {ev.get('text')}")

            elif t == "gameFinish":
                print(f"[game {game_id}] finished.")
                break

def _maybe_play_move(game_id, board: chess.Board, my_color: str, my_time_ms: int, my_inc_ms: int):
    if my_color is None:
        return

    want_to_move = (board.turn == chess.WHITE and my_color == "white") or \
                   (board.turn == chess.BLACK and my_color == "black")
    if not want_to_move or board.is_game_over():
        return

    # Call your engine with the *server* values (ms)
    mv = choose_move(board, my_time_ms or 0, my_inc_ms or 0, board.fullmove_number)
    if mv is None:
        print(f"[game {game_id}] no legal move; resigning")
        resign(game_id)
        return

    uci = mv.uci()
    ok = make_move(game_id, uci)
    print(f"[game {game_id}] played {uci}  ok={ok}  | server_time={my_time_ms}ms inc={my_inc_ms}ms")

def who_am_i():
    r = requests.get(f"{BASE}/api/account", headers={"Authorization": f"Bearer {LICHESS_TOKEN}"})
    if r.status_code // 100 == 2:
        return r.json().get("id")
    return None

def event_loop():
    """Main loop: accept challenges and spawn a thread for each game."""
    print("[bot] connecting to event stream…")
    with requests.get(f"{BASE}/api/stream/event", headers=HDRS, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        for ev in parse_ndjson_stream(resp):
            t = ev.get("type")
            if t == "challenge":
                ch = ev.get("challenge", {})
                ch_id = ch.get("id")
                variant_ok = (ch.get("variant", {}).get("key") == "standard")
                speed_ok = True  # accept all speeds; tweak if you want
                if variant_ok and speed_ok:
                    accept_challenge(ch_id)
                else:
                    decline_challenge(ch_id, reason="variant")
            elif t == "gameStart":
                g = ev.get("game", {})
                gid = g.get("id")
                th = threading.Thread(target=game_loop, args=(gid,), daemon=True)
                th.start()
                print(f"[bot] game thread started: {gid}")

def main():
    print("[bot] hello from", who_am_i())
    backoff = 1.0
    while True:
        try:
            event_loop()
            backoff = 1.0
        except requests.exceptions.RequestException as e:
            print("[bot] network error:", e)
        except Exception as e:
            print("[bot] error:", e)
        time.sleep(backoff)
        backoff = min(30.0, backoff * 2)

if __name__ == "__main__":
    main()
