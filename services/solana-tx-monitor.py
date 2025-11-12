#
# Copyright (c) 2025 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>
#

import websocket, json, pymongo, os, requests, time, datetime

SOLANA_WS_URL = "wss://api.mainnet-beta.solana.com"
SOLANA_RPC_URL = "https://api.mainnet-beta.solana.com"
ACCOUNT_ADDRESS = "918Y2TZvy386gXLWxGM9sBVutviT77xJriCDQsZeheEF"
USDT_TOKEN_ADDRESS = "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"

solana_collection_tx = pymongo.MongoClient(os.getenv('MONGODB_IST_MEDIA'))["1_media_demo"]["solana_tx"]
solana_collection_tmp = pymongo.MongoClient(os.getenv('MONGODB_IST_MEDIA'))["1_media_demo"]["solana_tmp"]

def get_tx_details(signature):
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getTransaction",
        "params": [ signature, { "encoding" : "jsonParsed" } ]
    }
    response = requests.post(SOLANA_RPC_URL, json=payload)
    if response.status_code == 200:
        tx = { "signature" : signature }
        data = response.json()
        if not data["result"]:
            return { "status" : 100 } # TX data not yet available
        try:
            signer_pubkey = next(
                acc["pubkey"]
                for acc in data["result"]["transaction"]["message"]["accountKeys"]
                if acc["signer"]
            )
            tx["signer"] = signer_pubkey

            sol_transfer = next(
                instr for instr in data["result"]["transaction"]["message"]["instructions"]
                if isinstance(instr, dict) and isinstance(instr.get("parsed"), dict) and instr["parsed"].get("type") == "transfer"
            )
            transfer_info = sol_transfer["parsed"]["info"]
            sender = transfer_info["source"]
            receiver = transfer_info["destination"]
            lamports = int(transfer_info["lamports"])
            sol_amount = lamports / 1_000_000_000

            tx["amount"] = sol_amount
            tx["sender"] = sender
            tx["receiver"] = receiver

            block_time = data["result"].get("blockTime")
            if block_time:
                tx["date"] = datetime.datetime.utcfromtimestamp(block_time)

            memo_instr = next(
                instr for instr in data["result"]["transaction"]["message"]["instructions"]
                if isinstance(instr, dict) and instr.get("program") == "spl-memo"
            )
            memo_text = memo_instr["parsed"] if isinstance(memo_instr["parsed"], str) else None

            if memo_text:
                tx["memo"] = memo_text

            return tx


        except StopIteration:
            return { "status" : 500 }
        except KeyError as e:
            return { "status" : 500 }
    else:
        return { "status" : response.status_code, "message" : response.text }

def clean_memo(memo_text):
    # If the memo is formatted like 'Memo (len 4): "ABCD"', extract just 'ABCD'
    if 'Memo (len' in memo_text and '"' in memo_text:
        start = memo_text.find('"') + 1
        end = memo_text.rfind('"')
        if start < end:
            return memo_text[start:end]
    # Otherwise, return the text as-is after stripping
    return memo_text.strip()

def on_message(ws, message):
    try:
        data = json.loads(message)
        if "method" in data and data["method"] == "logsNotification":
            logs = data["params"]["result"]["value"]["logs"]
            signature = data["params"]["result"]["value"]["signature"]

            # Look for memo data in logs with more flexibility
            memo = None
            for log in logs:
                if "Program MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr invoke" in log:
                    # Memo program invoked; look for memo in subsequent logs
                    continue
                if "Program log:" in log:
                    # Extract potential memo text from any "Program log:" entry
                    potential_memo = log.replace("Program log:", "").strip()
                    if "Memo (len" in potential_memo and '"' in potential_memo:
                        # Matches format like 'Memo (len 4): "bjjl"'
                        memo = potential_memo
                        break
                    elif potential_memo:  # Any non-empty log entry as a fallback
                        memo = potential_memo
                        # Don't break here; keep looking for a more specific memo format

            if memo:
                cleaned_memo = clean_memo(memo)
                print(f"Signature: {signature}")
                print(f"Memo: {cleaned_memo}")
                tx_tmp = { "signature" : signature, "memo" : cleaned_memo }
                solana_collection_tmp.insert_one(tx_tmp)

                tx = {}
                while True:
                    time.sleep(15)
                    tx = get_tx_details(signature)
                    if "status" in tx and tx["status"] == 100:
                        print("tx details not yet available -- waiting...")
                        continue
                    else:
                        break
                print(tx)

                solana_collection_tmp.delete_one(tx_tmp)
                solana_collection_tx.insert_one(tx)

            else:
                print(f"Signature: {signature}")
                print("No memo found in logs")

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except Exception as e:
        print(f"Unexpected error in on_message: {e}")

def on_error(ws, error):
    print(f"WebSocket error: {error}")

def on_close(ws, close_status_code, close_msg):
    print(f"WebSocket connection closed. Status: {close_status_code}, Message: {close_msg}")

def on_open(ws):
    """Subscribes to transaction logs mentioning the account."""
    subscription = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "logsSubscribe",
        "params": [
            {"mentions": [ACCOUNT_ADDRESS]},
            {"commitment": "confirmed"}
        ]
    }

    try:
        ws.send(json.dumps(subscription))
        print("Subscribed to transaction logs")
    except Exception as e:
        print(f"Error in subscription: {e}")

if __name__ == "__main__":
    try:
        ws = websocket.WebSocketApp(
            SOLANA_WS_URL,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )

        ws.run_forever(
            ping_interval=30,
            ping_timeout=10,
            reconnect=True
        )
    except KeyboardInterrupt:
        print("Program terminated by user")
    except Exception as e:
        print(f"WebSocket initialization error: {e}")
