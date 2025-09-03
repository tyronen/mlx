# --- Lazy model cache (pod stays warm; model loads once) ---


# --- Actions ---


# --- Router & handler ---
ACTIONS = {
    "predict": do_predict,
    "submit": do_submit,
    "list": do_list,
}


def handler(event):
    """
    Expected input:
    {
      "input": {
        "action": "predict" | "submit" | "list",
        "payload": { ... }   # per action
      }
    }
    """
    try:
        inp = event.get("input") or {}
        action = inp.get("action")
        payload = inp.get("payload") or {}
        fn = ACTIONS.get(action)
        if not fn:
            return {
                "ok": False,
                "error": f"unknown action '{action}'",
                "actions": list(ACTIONS.keys()),
            }
        return fn(payload)
    except Exception as e:
        return {"ok": False, "error": f"unhandled server error: {e}"}


runpod.serverless.start({"handler": handler})
