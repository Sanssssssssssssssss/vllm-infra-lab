import argparse
import json
import sys
import urllib.error
import urllib.request


def request_json(url: str, payload: dict, api_key: str, timeout: int) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Test a vLLM OpenAI-compatible chat endpoint.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--message", default="Give me a short hello from the local model.")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--disable-thinking", action="store_true")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": args.message},
        ],
        "temperature": 0.2,
        "max_tokens": args.max_tokens,
    }

    if args.disable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    try:
        result = request_json(url, payload, args.api_key, args.timeout)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP {exc.code}: {detail}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
