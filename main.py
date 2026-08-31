"""
main.py - a runnable REPL demo of the whole loop.

    python main.py --domain general --live
    python main.py --domain study --live --backend adk

--live routes generation through a real Gemini-backed LLM instead of the
built-in stub. --backend controls which one: 'auto' (default) prefers
llm/adk_backend.py (Google's Agent Development Kit -- pip install
google-adk), falling back to llm/gemini.py's raw google-genai SDK call
if ADK isn't installed, then the stub if no key is set either way.

Commands inside the REPL:
    :profile          show the Internal State ("what I know about you")
    :feed             show the live 3-source unified-database ticker
    :metrics          show session-over-session adaptation metrics
    :up / :down       give explicit feedback on the last response
    :fact k=v|label|value|weight   feed a structured fact
    :consolidate      run the Consolidator now
    :forget <key>     delete a stored fact (privacy control)
    :newsession       start a new session (for metrics comparison)
    :image <path>     stage an image, attached to your next message
                      (bare ":image" clears a staged one); requires
                      --live with a vision-capable model to actually
                      be seen -- the stub just acknowledges it's there
    :quit

Your profile/facts are saved to companion.db (created next to this file)
and persist across runs -- delete that file to start fresh.
"""

from __future__ import annotations

import argparse
import mimetypes
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass  # python-dotenv not installed -- fine if the key is set another way

from agent import Companion  # noqa: E402
from llm.stub import StubBackend  # noqa: E402

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "companion.db")


def _build_domain(name: str):
    if name == "general":
        from domains.general import GENERAL_DOMAIN
        return GENERAL_DOMAIN
    if name == "study":
        from domains.study import STUDY_DOMAIN
        return STUDY_DOMAIN
    raise SystemExit(f"unknown domain: {name}")


def _build_llm(backend_choice: str, instruction: str = ""):
    """backend_choice: 'auto' (prefer ADK, fall back to raw Gemini SDK, then
    stub), or an explicit 'adk' | 'gemini' | 'stub'. `instruction` is the
    domain's system_prompt/purpose -- passed to ADKBackend so it becomes
    the Agent's persistent instruction rather than being re-stated in
    every turn's prompt text."""
    if backend_choice == "stub":
        return StubBackend()

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("No GEMINI_API_KEY/GOOGLE_API_KEY found -- using the local StubBackend.")
        return StubBackend()

    if backend_choice in ("auto", "adk"):
        try:
            from llm.adk_backend import ADKBackend, ADKBackendError
        except ImportError:
            if backend_choice == "adk":
                print("google-adk not installed (pip install google-adk) -- falling back to GeminiBackend.")
        else:
            try:
                backend = ADKBackend(instruction=instruction)
                print(f"Gemini key detected -- initializing ADKBackend (Google Agent Development Kit, model={backend.model}).")
                return backend
            except ADKBackendError as exc:
                print(f"ADKBackend failed to initialize ({exc}) -- falling back to GeminiBackend.")

    try:
        from llm.gemini import GeminiBackend, GeminiBackendError
    except ImportError:
        print("google-genai not installed (pip install google-genai) -- falling back to StubBackend.")
        return StubBackend()
    try:
        backend = GeminiBackend()
        print(f"Gemini key detected -- initializing GeminiBackend (model={backend.model}).")
        return backend
    except GeminiBackendError as exc:
        print(f"GeminiBackend failed to initialize ({exc}) -- falling back to StubBackend.")
        return StubBackend()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="general", choices=["general", "study"])
    parser.add_argument("--user", default="demo-user")
    parser.add_argument("--live", action="store_true", help="use a real Gemini-backed LLM if a key is available")
    parser.add_argument(
        "--backend", default="auto", choices=["auto", "adk", "gemini", "stub"],
        help="which LLM backend --live should use: 'auto' prefers ADK (Google Agent Development Kit), "
             "falls back to the raw google-genai SDK, then the stub",
    )
    parser.add_argument("--fresh", action="store_true", help="start with an empty in-memory profile instead of companion.db")
    args = parser.parse_args()

    domain = _build_domain(args.domain)
    llm = _build_llm(args.backend, instruction=domain.system_prompt or domain.purpose) if args.live else StubBackend()
    db_path = ":memory:" if args.fresh else DB_PATH
    companion = Companion(domain=domain, llm=llm, db_path=db_path)
    companion.new_session(args.user)

    print(f"\n[Companion] domain={domain.name} user={args.user} storage={db_path}")
    print("Type a message, or a : command (:profile, :feed, :metrics, :image <path>, :quit)\n")

    pending_image: dict = {"bytes": None, "mime": None, "path": None}

    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue

        try:
            if line == ":quit":
                break
            elif line == ":profile":
                for f in companion.profile(args.user):
                    print(f"  [{f['status']}] {f['label']}  (confidence {f['confidence']})")
            elif line == ":feed":
                for e in companion.live_feed(args.user):
                    print(f"  {e['source']:>12} | {e['event_type']:<10} | {e['bits']}")
            elif line == ":metrics":
                for s in companion.adaptation_metrics(args.user):
                    print(f"  {s}")
            elif line == ":up":
                companion.give_feedback(args.user, "up")
                print("  (noted: helpful)")
            elif line == ":down":
                companion.give_feedback(args.user, "down")
                print("  (noted: not helpful)")
            elif line.startswith(":fact "):
                _, rest = line.split(" ", 1)
                key, label, value, weight = rest.split("|")
                companion.give_feedback(args.user, "correction", {
                    "key": key, "label": label, "value": value, "confidence": float(weight),
                })
                print("  (fact recorded, run :consolidate to fold it into memory)")
            elif line == ":consolidate":
                companion.consolidate(args.user)
                print("  (consolidated)")
            elif line.startswith(":forget "):
                key = line.split(" ", 1)[1]
                ok = companion.forget(args.user, key)
                print("  forgotten." if ok else "  no such fact.")
            elif line == ":newsession":
                companion.new_session(args.user)
                print("  (new session started)")
            elif line == ":image":
                pending_image = {"bytes": None, "mime": None, "path": None}
                print("  (image cleared)")
            elif line.startswith(":image "):
                path = line.split(" ", 1)[1].strip()
                mime, _ = mimetypes.guess_type(path)
                if not mime or not mime.startswith("image/"):
                    print(f"  [error] '{path}' doesn't look like an image file (guessed type: {mime})")
                else:
                    with open(path, "rb") as fh:
                        pending_image = {"bytes": fh.read(), "mime": mime, "path": path}
                    print(f"  (image staged: {path} -- it'll be attached to your next message)")
            else:
                result = companion.turn(
                    args.user, line,
                    image_bytes=pending_image["bytes"], image_mime=pending_image["mime"],
                )
                if pending_image["bytes"]:
                    print(f"  (sent with image: {pending_image['path']})")
                    pending_image = {"bytes": None, "mime": None, "path": None}
                tag = "ASK" if result.asked_clarifying else ("PROFILE" if result.used_profile else "GENERIC")
                print(f"companion [{tag}, confidence={result.confidence:.2f}]> {result.response}")
                if result.pending_confirmations:
                    for pc in result.pending_confirmations:
                        print(f"  (pending confirmation on '{pc['key']}': {pc['label']})")
        except ValueError:
            print("  usage: :fact key|label|value|evidence_weight")
        except Exception as exc:  # noqa: BLE001 -- last line of defense for the REPL
            print(f"  [error] {type(exc).__name__}: {exc}")
            print("  (that turn failed, but your session and profile are still intact -- try again)")

    print("\nsession ended.")


if __name__ == "__main__":
    main()
