"""A local OpenAI-compatible server as a role transport (f_compat + the role manager's generated options). Offline:
the real HTTP in here goes to stand-in servers on the loopback."""
import contextlib
import json
import shutil
import socket
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "brain"))
sys.path.insert(0, str(ROOT / "tests"))
from _llm_stub import LlmStub, unused_url                           # noqa: E402
import f_compat                                                     # noqa: E402
import f_memory                                                     # noqa: E402
from f_compat import (COMPAT, CompatClient, ThinkFilter, compat_base_url, compat_role_config,   # noqa: E402
                      discover, load_compat_config, local_address, save_compat_config, strip_thinking)
from f_memory import MemoryRuntime                                   # noqa: E402
from f_openrouter import describe_error                              # noqa: E402
from f_roles import NATIVE_KEYS, ROLES, install_roles                # noqa: E402
from f_usage import UsageMeter                                       # noqa: E402

BASE = "http://127.0.0.1:1234/v1"
SERVER = {"base_url": BASE, "models": [{"name": "local-model"}]}


class Response:
    def __init__(self, lines=None, data=None, status=200):
        self.status_code, self._lines, self._data, self.encoding = status, lines or [], data, None
    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("http " + str(self.status_code))
    def iter_lines(self, decode_unicode=True):
        return iter(self._lines)
    def json(self):
        return self._data
    def close(self):
        pass


class HTTP:
    def __init__(self):
        self.responses, self.calls = [], []
    def post(self, url, **kwargs):
        self.calls.append((url, json.loads(json.dumps(kwargs.get("json"))), kwargs))
        return self.responses.pop(0)
    def get(self, url, **kwargs):
        self.calls.append((url, None, kwargs))
        return self.responses.pop(0)


def sse(*chunks, usage=None, finish="stop"):
    rows = [{"model": "/models/x.gguf", "choices": [{"delta": {"content": c}, "finish_reason": None}]} for c in chunks]
    rows.append({"model": "/models/x.gguf", "choices": [{"delta": {}, "finish_reason": finish}]})
    if usage:
        rows.append({"choices": [], "usage": usage})
    return ["data: " + json.dumps(r) for r in rows] + ["data: [DONE]"]


def run(role, *chunks):
    think = ThinkFilter(role)
    out = "".join(think.feed(c) for c in chunks)
    return out + think.feed("", final=True)


def resolves_to(*addresses):
    """socket.getaddrinfo as if the name stood for these addresses."""
    return patch.object(f_compat.socket, "getaddrinfo", lambda host, port, **kw: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port)) for address in addresses])


class AddressTests(unittest.TestCase):
    def test_addresses(self):
        self.assertEqual(compat_base_url(""), BASE)
        self.assertEqual(compat_base_url("http://box:8080"), "http://box:8080/v1")
        self.assertEqual(compat_base_url("http://box:8080/api/v1/"), "http://box:8080/api/v1")
        self.assertEqual(compat_base_url("http://box:1234/v1/chat/completions"), "http://box:1234/v1")
        for bad in ("ftp://box", "http://user:pw@box/v1", "http://box/v1?x=1", "http://box:notaport/v1", "box:1234"):
            with self.assertRaises(ValueError):
                compat_base_url(bad)

    def test_this_machine_and_private_networks_are_accepted(self):
        for url in ("http://127.0.0.1:8000/v1", "http://localhost:8000/v1", "http://192.168.1.20:8000/v1",
                    "http://10.0.0.5:8080/v1", "http://100.64.0.7:8000/v1", "http://[::1]:8000/v1"):
            self.assertEqual(compat_base_url(url), url)
        for url in ("http://127.0.0.1:8000/v1", "http://192.168.1.20:8000/v1"):
            self.assertTrue(local_address(url))
        with resolves_to("192.168.1.20"):
            self.assertEqual(local_address("http://box:8000/v1"), ["192.168.1.20"])

    def test_a_public_address_and_https_are_refused(self):
        for url in ("http://8.8.8.8:8000/v1", "http://[2001:4860:4860::8888]/v1"):
            with self.assertRaisesRegex(ValueError, "is a public address"):
                compat_base_url(url)
        for url in ("https://api.example.com/v1", "https://127.0.0.1:8000/v1"):
            with self.assertRaisesRegex(ValueError, "plain http"):
                compat_base_url(url)

    def test_a_name_is_judged_by_what_it_stands_for(self):
        self.assertEqual(compat_base_url("http://models.example.com/v1"), "http://models.example.com/v1")
        with resolves_to("192.168.1.20", "8.8.4.4"):            # one public address among them is enough
            with self.assertRaisesRegex(ValueError, r"models\.example\.com is a public address \(8\.8\.4\.4\)"):
                local_address("http://models.example.com/v1")
        with patch.object(f_compat.socket, "getaddrinfo", side_effect=OSError("no such host")):
            with self.assertRaisesRegex(ValueError, "does not resolve"):
                local_address("http://nowhere.invalid/v1")


class ThinkingTests(unittest.TestCase):
    def test_a_normal_answer_streams_at_once(self):
        think = ThinkFilter("deep")
        self.assertEqual(think.feed('[casual_happy]"He'), '[casual_happy]"He')
        self.assertEqual(think.feed('llo."'), 'llo."')

    def test_a_leading_block_is_removed_even_split_across_chunks(self):
        self.assertEqual(run("deep", "<th", "ink>plan the", " reply</th", 'ink>\n', '\n[casual_happy]"Hi"'), '[casual_happy]"Hi"')

    def test_reasoning_whose_opening_tag_was_in_the_prompt_is_never_released(self):
        think = ThinkFilter("surface")
        self.assertEqual(think.feed("Okay, the user said hi. "), "")
        self.assertEqual(think.feed("I should reply.</think>\nstatus: ok"), "status: ok")

    def test_an_odd_answer_with_no_tags_still_arrives_whole_at_the_end(self):
        self.assertEqual(run("deep", "just some ", "plain text"), "just some plain text")

    def test_thinking_that_never_ended_says_nothing(self):
        self.assertEqual(run("deep", "<think>still going"), "")

    def test_whole_replies(self):
        self.assertEqual(strip_thinking("<think>x</think>\nanswer"), "answer")
        self.assertEqual(strip_thinking("x</think>answer"), "answer")
        self.assertEqual(strip_thinking("<think>never closed"), "")
        quoted = '{"facts": [{"quote": "she typed </think> here"}]}'
        self.assertEqual(strip_thinking(quoted), quoted)


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.http, self.handles, self.receipts = HTTP(), [], []
        f_compat._QUIRKS.clear()
        self.client = self.make(BASE)

    def make(self, base_url):
        return CompatClient(self.http, lambda request: self.handles.append(request) or "h1",
                            lambda handle, receipt: self.receipts.append((handle, receipt)), base_url=base_url)

    def test_stream_settles_complete_despite_another_model_name(self):
        cfg = compat_role_config(SERVER, "local-model", "deep")
        self.http.responses = [Response(lines=sse('[casual_happy]"Hel', 'lo."', usage={"prompt_tokens": 80, "completion_tokens": 9}))]
        text = "".join(self.client.stream(cfg, [{"role": "user", "content": "u"}], metadata={"phase": "deep"}))
        self.assertEqual(text, '[casual_happy]"Hello."')
        url, body, kwargs = self.http.calls[0]
        self.assertEqual(url, BASE + "/chat/completions")
        self.assertNotIn("Authorization", kwargs["headers"])
        self.assertFalse(kwargs["allow_redirects"])
        self.assertEqual(body["stream_options"], {"include_usage": True})
        self.assertNotIn("response_format", body)
        receipt = self.receipts[0][1]
        self.assertEqual((receipt["status"], receipt["cost_usd"], receipt["actual_model"]), ("complete", 0.0, "/models/x.gguf"))
        self.assertEqual(receipt["usage"]["tokens"]["total_input_tokens"], 80)
        json.dumps(receipt)                                  # the archive must be able to write it

    def test_a_server_without_usage_still_completes(self):
        cfg = compat_role_config(SERVER, "local-model", "surface")
        self.http.responses = [Response(lines=sse("status: ok"))]
        self.assertEqual("".join(self.client.stream(cfg, [{"role": "user", "content": "u"}])), "status: ok")
        self.assertEqual(self.receipts[0][1]["status"], "complete")

    def test_thinking_stays_out_of_the_text_and_the_receipt(self):
        cfg = compat_role_config(SERVER, "local-model", "deep")
        self.http.responses = [Response(lines=sse("<think>secret</think>", '[casual_happy]"Hi"'))]
        self.assertEqual("".join(self.client.stream(cfg, [{"role": "user", "content": "u"}])), '[casual_happy]"Hi"')
        self.assertNotIn("secret", self.receipts[0][1]["content"])

    def test_answer_budget_used_up_by_thinking_is_marked(self):
        cfg = compat_role_config(SERVER, "local-model", "deep")
        self.http.responses = [Response(lines=sse("<think>long", finish="length"))]
        with contextlib.redirect_stdout(None):
            self.assertEqual("".join(self.client.stream(cfg, [{"role": "user", "content": "u"}])), "")
        self.assertTrue(self.receipts[0][1]["reasoning_unfinished"])

    def test_refused_parameter_is_adapted_and_remembered(self):
        cfg = compat_role_config(SERVER, "local-model", "memory_write")
        data = {"choices": [{"message": {"content": "<think>x</think>{\"facts\": []}"}, "finish_reason": "stop"}]}
        self.http.responses = [Response(status=400, data={"error": {"message": "Use 'max_completion_tokens' instead."}}), Response(data=data)]
        self.assertEqual(self.client.complete(cfg, [{"role": "user", "content": "u"}])["text"], '{"facts": []}')
        self.assertIn("max_tokens", self.http.calls[0][1])
        self.assertIn("max_completion_tokens", self.http.calls[1][1])
        self.assertNotIn("max_tokens", self.http.calls[1][1])
        self.http.responses = [Response(data=data)]
        self.client.complete(cfg, [{"role": "user", "content": "u"}])
        self.assertEqual(len(self.http.calls), 3)            # no second refusal: the server's rule was remembered

    def test_other_errors_explain_themselves(self):
        cfg = compat_role_config(SERVER, "local-model", "surface")
        self.http.responses = [Response(status=400, data={"error": {"message": "context too long"}})]
        with self.assertRaises(Exception) as caught:
            list(self.client.stream(cfg, [{"role": "user", "content": "u"}]))
        self.assertIn("HTTP 400: context too long", str(caught.exception))
        self.assertEqual(describe_error(caught.exception), "Your model server: HTTP 400: context too long")

    def test_a_server_that_wants_a_key_tells_the_person_so_in_her_chat(self):
        cfg = compat_role_config(SERVER, "local-model", "surface")
        self.http.responses = [Response(status=401, data={"error": {"message": "Unauthorized"}})]
        with self.assertRaises(Exception) as caught:
            list(self.client.stream(cfg, [{"role": "user", "content": "u"}]))
        self.assertEqual(describe_error(caught.exception),
                         "Your model server: HTTP 401: it asks for an API key, and none is ever sent: start it without one")

    def test_no_request_ever_carries_a_key(self):
        cfg = compat_role_config(SERVER, "local-model", "surface")
        self.http.responses = [Response(lines=sse("status: ok")), Response(data={"data": [{"id": "m"}]})]
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-from-the-environment", "OPENAI_COMPAT_API_KEY": "sk-other"}):
            list(self.client.stream(cfg, [{"role": "user", "content": "u"}]))
            discover(self.http, BASE)
        for _, _, kwargs in self.http.calls:
            self.assertNotIn("authorization", {name.lower() for name in kwargs["headers"]})

    def test_a_name_that_has_come_to_stand_for_a_public_address_is_not_called(self):
        client = self.make("http://models.example.com/v1")
        cfg = compat_role_config(SERVER, "local-model", "surface")
        with resolves_to("192.168.1.20"):
            self.http.responses = [Response(lines=sse("status: ok"))]
            self.assertEqual("".join(client.stream(cfg, [{"role": "user", "content": "u"}])), "status: ok")
        with resolves_to("8.8.4.4"):
            with self.assertRaisesRegex(Exception, "is a public address"):
                list(client.stream(cfg, [{"role": "user", "content": "u"}]))
        self.assertEqual(len(self.http.calls), 1)            # the second turn never left this machine
        self.assertNotEqual(self.receipts[1][1]["status"], "complete")

    def test_a_redirect_is_an_error(self):
        cfg = compat_role_config(SERVER, "local-model", "surface")
        self.http.responses = [Response(status=302, data={})]
        with self.assertRaises(Exception):
            list(self.client.stream(cfg, [{"role": "user", "content": "u"}]))


class DiscoverTests(unittest.TestCase):
    def test_discover_keeps_chat_models(self):
        http = HTTP()
        http.responses = [Response(data={"data": [{"id": "z-model"}, {"id": "text-embedding-3"}, {"id": "a-model"}]})]
        self.assertEqual(discover(http, BASE), [{"name": "a-model"}, {"name": "z-model"}])
        self.assertEqual(http.calls[0][2]["headers"], {})

    def test_a_server_that_wants_a_key_is_told_apart_from_one_that_is_down(self):
        http = HTTP()
        http.responses = [Response(status=401, data={})]
        with self.assertRaisesRegex(Exception, "asks for an API key, and this transport sends none"):
            discover(http, BASE)

    def test_a_public_address_is_not_asked(self):
        http = HTTP()
        with resolves_to("8.8.4.4"):
            with self.assertRaisesRegex(Exception, "only talks to local servers"):
                discover(http, "http://models.example.com/v1")
        self.assertEqual(http.calls, [])

    def test_config_defaults_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config").mkdir()
            self.assertEqual((load_compat_config(root)["base_url"], load_compat_config(root)["models"]), (BASE, []))
            save_compat_config(root, dict(SERVER, detected_at="t"))
            self.assertEqual(load_compat_config(root)["models"], [{"name": "local-model"}])
            save_compat_config(root, dict(SERVER, base_url="https://api.example.com/v1"))
            self.assertEqual(load_compat_config(root)["base_url"], BASE)      # a hand-written address that is not local


class RouteTests(unittest.TestCase):
    """The page's detect route: only a local address is asked or saved, and there is no key to give it."""
    def setUp(self):
        from flask import Flask, jsonify, request
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        (self.root / "brain").mkdir()
        (self.root / "config").mkdir()
        self.http = HTTP()
        app = Flask("test")
        with contextlib.redirect_stdout(None):
            f_compat.install_compat({"app": app, "request": request, "jsonify": jsonify, "http_requests": self.http,
                                     "__file__": str(self.root / "brain" / "webui.py")})
        self.web = app.test_client()

    def detect(self, base, **extra):
        self.http.responses = [Response(data={"data": [{"id": "m1"}]})]
        return self.web.post("/api/build_f/compat/detect", json=dict(extra, base_url=base),
                             headers={"X-Build-F-Action": "compat-detect"})

    def test_a_detect_saves_the_address_and_its_models_and_nothing_else(self):
        reply = self.detect(BASE, api_key="sk-secret")           # a key in the request is not a field: it is dropped
        self.assertEqual(reply.status_code, 200)
        self.assertEqual(sorted(reply.get_json()), ["base_url", "detected_at", "error", "in_use", "models", "ok", "reachable"])
        saved = json.loads((self.root / "config" / f_compat.CONFIG_NAME).read_text(encoding="utf-8"))
        self.assertEqual((saved["base_url"], saved["models"]), (BASE, [{"name": "m1"}]))
        self.assertEqual(sorted(saved), ["base_url", "detected_at", "models"])
        self.assertFalse((self.root / "brain" / ".env").exists())
        self.assertEqual(self.http.calls[-1][2]["headers"], {})

    def test_a_public_address_is_refused_before_anything_is_asked_or_saved(self):
        for base in ("http://8.8.8.8:8000/v1", "https://api.example.com/v1"):
            reply = self.detect(base)
            self.assertEqual(reply.status_code, 400, base)
        with resolves_to("8.8.4.4"):
            reply = self.detect("http://models.example.com/v1")
        self.assertEqual(reply.status_code, 400)
        self.assertIn("only talks to local servers", reply.get_json()["error"])
        self.assertEqual(self.http.calls, [])
        self.assertFalse((self.root / "config" / f_compat.CONFIG_NAME).exists())

    def test_a_detect_keeps_hand_written_entries_and_refreshes_the_rest(self):
        save_compat_config(self.root, {"base_url": BASE, "models": [
            {"name": "gone"},
            {"name": "think", "served_name": "m1", "base_url": "http://box:8610/v1", "chat_template_kwargs": {"enable_thinking": True}}]})
        self.detect(BASE)
        models = load_compat_config(self.root)["models"]
        self.assertEqual([m["name"] for m in models], ["m1", "think"])
        self.assertEqual(models[1], {"name": "think", "served_name": "m1", "base_url": "http://box:8610/v1",
                                     "chat_template_kwargs": {"enable_thinking": True}})

    def test_only_the_page_may_detect(self):
        self.assertEqual(self.web.post("/api/build_f/compat/detect", json={}, headers={
            "X-Build-F-Action": "compat-detect", "Origin": "http://elsewhere"}).status_code, 403)
        self.assertEqual(self.web.post("/api/build_f/compat/detect", json={"base_url": BASE}).status_code, 403)

    def test_a_server_that_does_not_answer_is_a_503_with_the_reason(self):
        self.http.responses = [Response(status=401, data={})]
        reply = self.web.post("/api/build_f/compat/detect", json={"base_url": BASE}, headers={"X-Build-F-Action": "compat-detect"})
        self.assertEqual(reply.status_code, 503)
        self.assertIn("asks for an API key, and this transport sends none", reply.get_json()["error"])
        self.http.responses = []                                 # the fake raises: nothing is listening
        reply = self.web.post("/api/build_f/compat/detect", json={"base_url": BASE}, headers={"X-Build-F-Action": "compat-detect"})
        self.assertEqual((reply.status_code, reply.get_json()["error"]),
                         (503, f"Nothing answers at {BASE} — is the model server running?"))
        status = self.web.get("/api/build_f/compat").get_json()
        self.assertEqual((status["reachable"], status["error"]), (False, None))


class RoleManagerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "config").mkdir()
        shutil.copyfile(ROOT / "config/openrouter_models.json", self.root / "config/openrouter_models.json")
        document = {"version": 1, "options": {}, "selections": {}, "fallback_preset": "cloud", "presets": [
            {"name": "cloud", "provider": "openrouter", "selections": {role: "or:" + role for role in ROLES}}]}
        for role in ROLES:
            document["options"][role] = [
                {"key": "native:" + NATIVE_KEYS[role], "native_key": NATIVE_KEYS[role], "name": "Native"},
                {"key": "or:" + role, "model_id": "deepseek/deepseek-v4-flash-0731", "name": "Reviewed " + role}]
            document["selections"][role] = "or:" + role
        (self.root / "config/llm_roles.json").write_text(json.dumps(document), encoding="utf-8")
        save_compat_config(self.root, dict(SERVER, detected_at="t"))
        self.memory = MemoryRuntime(self.root / "memory_f/live", lambda events, chunk: {"native": True},
                                    lambda texts, task: [[1.] + [0.] * 767 for t in texts])
        self.meter = UsageMeter(self.root / "memory_f/usage.sqlite", {"models": {}}, budget_usd=1)
        self.g = {"AVAILABLE_MODELS": {key: {"name": key, "provider": "native", "model": key} for key in NATIVE_KEYS.values()},
                  "_ws_active_lock": threading.RLock(), "_ws_active": {}, "_thinking_inflight": 0,
                  "http_requests": HTTP(), "OPENROUTER_KEY": "fake-key",
                  "_pick_stream": lambda p, c, **kw: iter(["native"]),
                  "_openai_messages": lambda p, **kw: [{"role": "user", "content": p["user_content"]}]}
        @contextlib.contextmanager
        def scope(phase, turn_id="", **kwargs):
            yield
        self.manager = install_roles(self.g, self.root, self.memory, self.meter, scope, threading.local(), {})

    def tearDown(self):
        self.memory.close()
        self.tmp.cleanup()

    def test_a_turn_reaches_the_server_and_the_ledger(self):
        http = self.g["http_requests"]
        http.responses = [Response(lines=sse("status: ok", usage={"prompt_tokens": 12, "completion_tokens": 3}))]
        parts = {"user_content": "u", "_f_phase": "surface"}
        text = "".join(self.g["_pick_stream"](parts, self.g["AVAILABLE_MODELS"]["surface:compat:local-model"]))
        self.assertEqual(text, "status: ok")
        self.assertEqual(http.calls[-1][0], BASE + "/chat/completions")
        self.assertEqual(self.meter.summary()["active_requests"], 0)
        self.assertTrue(list((self.root / "memory_f/compat_receipts").glob("*.json")))

    def test_selection_and_a_vanished_model(self):
        result = self.manager.set({role: f"{role}:compat:local-model" for role in ROLES})
        self.assertEqual(result["transport"], COMPAT)
        save_compat_config(self.root, {"base_url": BASE, "models": [{"name": "other"}], "detected_at": "t2"})
        result = self.manager.reload_compat()
        self.assertNotIn("deep:compat:local-model", self.manager.options["deep"])
        self.assertIn("deep:compat:other", self.manager.options["deep"])
        self.assertEqual(result["selections"]["deep"], "or:deep")


class PerEntryTests(unittest.TestCase):
    TWO = {"base_url": BASE, "models": [
        {"name": "fast"},
        {"name": "big-on", "served_name": "big", "base_url": "http://box:8610", "chat_template_kwargs": {"enable_thinking": True},
         "thinking_tokens": 2048},
        {"name": "big-off", "served_name": "big", "base_url": "http://box:8610/v1", "chat_template_kwargs": {"enable_thinking": False}}]}

    def loaded(self, doc=None):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config").mkdir()
            save_compat_config(root, doc or self.TWO)
            return load_compat_config(root)

    def body(self, compat, name):
        return f_compat.make_compat_body(compat_role_config(compat, name, "deep"), [{"role": "user", "content": "u"}])

    def test_two_entries_reach_two_addresses_as_their_served_names(self):
        compat = self.loaded()
        self.assertEqual(compat_role_config(compat, "fast", "deep")["base_url"], BASE)
        self.assertEqual(compat_role_config(compat, "big-off", "deep")["base_url"], "http://box:8610/v1")
        self.assertEqual(self.body(compat, "big-on")["model"], "big")
        self.assertEqual(self.body(compat, "fast")["model"], "fast")

    def test_a_plain_config_is_one_server_and_its_models(self):
        old = {"base_url": BASE, "detected_at": "t", "models": [{"name": "local-model"}]}
        self.assertEqual(self.loaded(old), old)
        cfg = compat_role_config(old, "local-model", "reflection")
        self.assertEqual((cfg["api_model"], cfg["base_url"]), ("local-model", BASE))
        self.assertNotIn("chat_template_kwargs", cfg)
        self.assertEqual(sorted(self.body(old, "local-model")), ["max_tokens", "messages", "model", "stream", "stream_options"])

    def test_template_options_go_only_with_their_own_entry(self):
        compat = self.loaded()
        self.assertEqual(self.body(compat, "big-on")["chat_template_kwargs"], {"enable_thinking": True})
        self.assertEqual(self.body(compat, "big-off")["chat_template_kwargs"], {"enable_thinking": False})
        self.assertNotIn("chat_template_kwargs", self.body(compat, "fast"))

    def test_only_the_thinking_entry_gets_more_budget(self):
        compat = self.loaded()
        answer = f_compat.ANSWER_TOKENS["reflection"]
        self.assertEqual(compat_role_config(compat, "big-on", "reflection")["max_tokens"], answer + 2048)
        self.assertEqual(compat_role_config(compat, "big-off", "reflection")["max_tokens"], answer)

    def test_a_malformed_entry_is_not_offered(self):
        doc = dict(self.TWO, models=[{"name": "ok"}, {"name": "bad-url", "base_url": "ftp://x"},
                                     {"name": "bad-kwargs", "chat_template_kwargs": "on"},
                                     {"name": "empty", "base_url": ""}])
        with contextlib.redirect_stdout(None):
            self.assertEqual([m["name"] for m in self.loaded(doc)["models"]], ["ok"])

    def test_a_saved_config_keeps_every_field(self):
        models = self.loaded()["models"]
        self.assertEqual(models[1], {"name": "big-on", "served_name": "big", "base_url": "http://box:8610/v1",
                                     "chat_template_kwargs": {"enable_thinking": True}, "thinking_tokens": 2048})
        self.assertEqual(models[0], {"name": "fast"})


class TwoServersEndToEnd(unittest.TestCase):
    """The real transport over real HTTP: each role's request reaches its own server with its own options."""
    def setUp(self):
        from flask import Flask, jsonify, request
        self.fast = LlmStub(["fast"], "status: from fast")
        self.big = LlmStub(["big"], "from big")
        self.addCleanup(self.fast.close)
        self.addCleanup(self.big.close)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        (self.root / "config").mkdir()
        shutil.copyfile(ROOT / "config/openrouter_models.json", self.root / "config/openrouter_models.json")
        document = {"version": 1, "options": {}, "selections": {}, "fallback_preset": "cloud", "presets": [
            {"name": "cloud", "provider": "openrouter", "selections": {role: "or:" + role for role in ROLES}}]}
        for role in ROLES:
            document["options"][role] = [
                {"key": "native:" + NATIVE_KEYS[role], "native_key": NATIVE_KEYS[role], "name": "Native"},
                {"key": "or:" + role, "model_id": "deepseek/deepseek-v4-flash-0731", "name": "Reviewed " + role}]
            document["selections"][role] = "or:" + role
        (self.root / "config/llm_roles.json").write_text(json.dumps(document), encoding="utf-8")
        self.config()
        self.memory = MemoryRuntime(self.root / "memory_f/live", lambda events, chunk: {"native": True},
                                    lambda texts, task: [[1.] + [0.] * 767 for t in texts])
        self.addCleanup(self.memory.close)
        self.meter = UsageMeter(self.root / "memory_f/usage.sqlite", {"models": {}}, budget_usd=1)
        self.g = {"AVAILABLE_MODELS": {key: {"name": key, "provider": "native", "model": key} for key in NATIVE_KEYS.values()},
                  "_ws_active_lock": threading.RLock(), "_ws_active": {}, "_thinking_inflight": 0,
                  "http_requests": requests, "OPENROUTER_KEY": "fake-key", "parse_response": lambda raw: [],
                  "_pick_stream": lambda p, c, **kw: iter(["native"]),
                  "_openai_messages": lambda p, **kw: [{"role": "user", "content": p["user_content"]}],
                  "app": Flask("test"), "request": request, "jsonify": jsonify, "__file__": str(self.root / "brain" / "webui.py")}
        @contextlib.contextmanager
        def scope(phase, turn_id="", **kwargs):
            yield
        self.manager = install_roles(self.g, self.root, self.memory, self.meter, scope, threading.local(), {})
        with contextlib.redirect_stdout(None):
            f_compat.install_compat(self.g)
        self.web = self.g["app"].test_client()
        self.manager.set({"surface": "surface:compat:fast", "deep": "deep:compat:big-off",
                          "reflection": "reflection:compat:big-on", "memory_write": "memory_write:compat:big-on"})

    def config(self, *, big_url=None):
        save_compat_config(self.root, {"base_url": self.fast.url, "detected_at": "t", "models": [
            {"name": "fast"},
            {"name": "big-on", "served_name": "big", "base_url": big_url or self.big.url,
             "chat_template_kwargs": {"enable_thinking": True}, "thinking_tokens": 2048},
            {"name": "big-off", "served_name": "big", "base_url": big_url or self.big.url,
             "chat_template_kwargs": {"enable_thinking": False}}]})

    def turn(self, role, model):
        parts = {"user_content": "u", "_f_phase": role}
        return "".join(self.g["_pick_stream"](parts, self.g["AVAILABLE_MODELS"][f"{role}:compat:{model}"]))

    def test_each_role_reaches_its_own_server_with_its_own_options(self):
        self.assertEqual(self.turn("surface", "fast"), "status: from fast")
        self.assertEqual(self.turn("deep", "big-off"), "from big")
        result = self.g["F_ROLE_COMPLETE"]("reflection", [{"role": "user", "content": "u"}])
        self.assertEqual(result["text"], "from big")
        fast, (deep, reflection) = self.fast.chats(), self.big.chats()
        self.assertEqual(len(fast), 1)
        self.assertEqual(fast[0]["path"], "/v1/chat/completions")
        self.assertEqual(fast[0]["body"]["model"], "fast")
        self.assertNotIn("chat_template_kwargs", fast[0]["body"])
        self.assertEqual((deep["body"]["model"], deep["body"]["chat_template_kwargs"]), ("big", {"enable_thinking": False}))
        self.assertEqual((reflection["body"]["model"], reflection["body"]["chat_template_kwargs"]), ("big", {"enable_thinking": True}))
        self.assertEqual(reflection["body"]["max_tokens"], f_compat.ANSWER_TOKENS["reflection"] + 2048)
        self.assertEqual(self.meter.summary()["active_requests"], 0)

    def test_no_server_is_ever_sent_a_key(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-from-the-environment", "OPENAI_COMPAT_API_KEY": "sk-other"}):
            self.turn("surface", "fast")
            self.turn("deep", "big-off")
            self.g["F_ROLE_COMPLETE"]("reflection", [{"role": "user", "content": "u"}])
            self.web.get("/api/build_f/compat")
        seen = self.fast.requests + self.big.requests
        self.assertGreaterEqual(len(seen), 5)
        self.assertEqual({r["auth"] for r in seen}, {None})

    def test_these_calls_cost_nothing_and_the_ledger_says_so(self):
        self.turn("surface", "fast")
        receipt = json.loads(next((self.root / "memory_f/compat_receipts").glob("*.json")).read_text(encoding="utf-8"))
        self.assertEqual((receipt["status"], receipt["cost_usd"], receipt["usage"]["cost_source"]), ("complete", 0.0, "local"))
        self.assertEqual(self.meter.summary()["active_requests"], 0)

    def test_a_server_that_is_down_fails_only_its_own_role(self):
        self.config(big_url=unused_url())
        self.manager.reload_compat()
        with self.assertRaises(Exception) as caught:
            self.turn("deep", "big-off")
        down = self.manager.compat["models"][1]["base_url"]
        self.assertEqual(describe_error(caught.exception), f"Your model server: nothing answers at {down} — is it running?")
        self.assertRegex(type(caught.exception).__name__ + ": ", f_memory.TRANSIENT_ERROR)    # memory jobs try again later
        self.assertEqual(self.turn("surface", "fast"), "status: from fast")
        self.assertEqual(self.meter.summary()["active_requests"], 0)

    def test_a_switch_while_she_is_answering_waits_for_unlock(self):
        use = lambda body: self.web.post("/api/build_f/compat/use", json=body, headers={"X-Build-F-Action": "compat-use"})
        self.g["_ws_active"]["t1"] = {}
        reply = use({"model": "fast"})
        self.assertEqual((reply.status_code, reply.get_json()["code"], reply.get_json()["busy"]),
                         (409, "model_busy", ["she is answering"]))
        self.assertEqual(self.manager.selections["deep"], "deep:compat:big-off")
        reply = use({"model": "fast", "force": True})                 # what the page sends after Unlock
        self.assertEqual((reply.status_code, reply.get_json()["in_use"]["deep"]), (200, "fast"))
        self.assertEqual(use({"model": "fast", "force": "yes"}).status_code, 400)

    def test_the_status_route_reports_each_server_the_roles_use(self):
        reply = self.web.get("/api/build_f/compat").get_json()
        self.assertTrue(reply["reachable"])
        self.assertEqual({s["base_url"]: (s["reachable"], sorted(s["roles"])) for s in reply["servers"]},
                         {self.fast.url: (True, ["surface"]), self.big.url: (True, ["deep", "memory_write", "reflection"])})
        self.config(big_url=unused_url())
        self.manager.reload_compat()
        rows = {s["base_url"]: s["reachable"] for s in self.web.get("/api/build_f/compat").get_json()["servers"]}
        self.assertEqual(sorted(rows.values()), [False, True])
        self.assertTrue(rows[self.fast.url])


if __name__ == "__main__":
    unittest.main()
