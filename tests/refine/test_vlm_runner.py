import asyncio
import http.server
import json
import threading

import pytest

from dnt.refine.config import VLMConfig
from dnt.refine.vlm import VLMAnswer, VLMTransientError, parse_answer
from dnt.refine.vlm.cache import AnswerCache
from dnt.refine.vlm.fake import FakeBackend
from dnt.refine.vlm.runner import Question, VLMRunner

OPTS = ["same_individual", "different", "unsure"]
CREATED = []


@pytest.fixture(autouse=True)
def _close_runners():
    yield
    for r in CREATED:
        r.close()
    CREATED.clear()


def reply(answer="different", conf=0.9):
    return json.dumps({"answer": answer, "confidence": conf, "reason": "r"})


def q(tag="LINK:link-r0-000001", image=b"img"):
    return Question(tag, image, "PROMPT", list(OPTS))


class Sleeps:
    def __init__(self):
        self.waits = []

    async def __call__(self, s):
        self.waits.append(s)


def runner(script, cache=None, sleep=None, backend=None, **cfg):
    backend = backend or FakeBackend(script)
    r = VLMRunner(
        VLMConfig(backend="openai_compat", model="m", **cfg), backend, cache, sleep=sleep or Sleeps()
    )
    CREATED.append(r)
    return r, backend


def test_one_vote_uses_the_models_confidence_and_temperature_zero():
    r, b = runner({"*": reply("different", 0.83)})
    (v,) = r.ask_many([q()])
    assert (v.answer, v.confidence, v.error, v.cached) == ("different", 0.83, None, False)
    assert v.reason == "r" and v.votes == {"different": 1}
    assert b.calls[0]["temperature"] == 0.0 and r.calls == 1 and r.cache_hits == 0


def test_majority_vote_confidence_and_temperature():
    r, b = runner(
        {"*": [reply("different"), reply("same_individual"), reply("different")]},
        votes=3,
        vote_temperature=0.7,
    )
    (v,) = r.ask_many([q()])
    assert v.answer == "different" and v.confidence == pytest.approx(2 / 3)
    assert v.votes == {"different": 2, "same_individual": 1}
    assert [c["temperature"] for c in b.calls] == [0.7, 0.7, 0.7] and r.calls == 3


def test_a_tie_has_no_answer():
    r, _ = runner({"*": [reply("different"), reply("same_individual")]}, votes=2)
    (v,) = r.ask_many([q()])
    assert v.answer is None and v.error == "tie"
    assert v.votes == {"different": 1, "same_individual": 1}
    assert r.failures == 1


def test_cache_hits_are_free_and_votes_are_cached_separately(tmp_path):
    cache = AnswerCache(tmp_path)
    r, _ = runner(
        {"*": [reply("different"), reply("same_individual"), reply("different")]},
        cache=cache,
        votes=3,
    )
    (v1,) = r.ask_many([q()])
    assert r.calls == 3 and len(list(tmp_path.rglob("*.json"))) == 3
    r2, b2 = runner({}, cache=cache, votes=3)  # an empty script would raise if asked
    (v2,) = r2.ask_many([q()])
    assert b2.calls == [] and r2.calls == 0 and r2.cache_hits == 3
    assert v2.answer == v1.answer and v2.cached is True and v1.cached is False


def test_a_cached_entry_with_a_bad_confidence_or_option_is_not_a_hit(tmp_path):
    cache = AnswerCache(tmp_path)
    key = AnswerCache.key(b"img", "PROMPT", OPTS, "fake", "fake-1", 0.0, 0)
    path = tmp_path / key[:2] / f"{key}.json"
    path.parent.mkdir(parents=True)
    for entry in (
        {"answer": "different", "confidence": float("nan"), "reason": "", "raw": ""},
        {"answer": "maybe", "confidence": 0.99, "reason": "", "raw": ""},
    ):
        path.write_text(json.dumps(entry))
        r, b = runner({"*": reply("same_individual", 0.8)}, cache=cache)
        (v,) = r.ask_many([q()])
        assert v.answer == "same_individual" and v.cached is False and len(b.calls) == 1
        assert r.cache_hits == 0  # the fresh answer then overwrote the bad entry


def test_invalid_output_is_retried_once_with_a_reminder_and_is_one_call():
    r, b = runner({"*": ["not json at all", reply("same_individual", 0.8)]})
    (v,) = r.ask_many([q()])
    assert v.answer == "same_individual" and r.calls == 1 and r.retries == 1
    assert b.calls[0]["prompt"] == "PROMPT"
    assert b.calls[1]["prompt"].startswith("PROMPT") and "not valid" in b.calls[1]["prompt"]
    assert all(o in b.calls[1]["prompt"] for o in OPTS)


def test_repeated_invalid_output_ends_the_question_and_is_not_cached(tmp_path):
    r, _ = runner({"*": ["nope"]}, cache=AnswerCache(tmp_path))
    (v,) = r.ask_many([q()])
    assert v.answer is None and v.error.startswith("invalid output")
    assert r.calls == 1 and r.retries == 1 and r.failures == 1
    assert list(tmp_path.rglob("*.json")) == []


def test_transient_errors_back_off_and_recover():
    sl = Sleeps()
    r, _ = runner({"*": [VLMTransientError("429"), VLMTransientError("503"), reply()]}, sleep=sl)
    (v,) = r.ask_many([q()])
    assert v.answer == "different" and sl.waits == [2.0, 4.0]
    assert r.calls == 1 and r.retries == 2


def test_three_transient_failures_give_up():
    sl = Sleeps()
    r, _ = runner({"*": [VLMTransientError("503")]}, sleep=sl)
    (v,) = r.ask_many([q()])
    assert v.answer is None and v.error.startswith("transient") and sl.waits == [2.0, 4.0]
    assert r.failures == 1 and r.calls == 1 and r.retries == 2


def test_a_timeout_is_retried():
    class Slow:
        name, model = "slow", "m"
        n = 0

        async def ask(self, image, prompt, options, temperature, *, tag=""):
            Slow.n += 1
            if Slow.n == 1:
                await asyncio.sleep(1.0)
            return parse_answer(reply(), options)

    r, _ = runner(None, backend=Slow(), timeout_s=0.05)
    (v,) = r.ask_many([q()])
    assert v.answer == "different" and Slow.n == 2


def test_other_exceptions_do_not_retry_and_never_leak_keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-very-secret")
    r, _ = runner({"*": [RuntimeError("auth failed for sk-very-secret")]})
    (v,) = r.ask_many([q()])
    assert v.answer is None and v.error.startswith("RuntimeError") and r.calls == 1
    assert r.retries == 0
    assert "sk-very-secret" not in v.error and "***" in v.error
    monkeypatch.setenv("MY_KEY", "tok-123")
    r2, _ = runner({"*": [RuntimeError("bad tok-123")]}, api_key_env="MY_KEY")
    (v2,) = r2.ask_many([q()])
    assert "tok-123" not in v2.error


def test_a_backend_returning_a_bad_answer_object_is_treated_as_invalid_output():
    bad = VLMAnswer("different", float("nan"), "", "")
    r, _ = runner({"*": [bad]})
    (v,) = r.ask_many([q()])
    assert v.answer is None and v.error.startswith("invalid output") and r.retries == 1
    for answer in (VLMAnswer("maybe", 0.9, "", ""), VLMAnswer("different", 1.5, "", ""),
                   VLMAnswer("different", -0.1, "", ""), VLMAnswer("different", True, "", ""),
                   VLMAnswer("different", float("inf"), "", "")):
        r2, b2 = runner({"*": [answer]})
        (v2,) = r2.ask_many([q()])
        assert v2.answer is None and v2.error.startswith("invalid output")
        assert len(b2.calls) == 2  # the one retry was made, then it gave up
    r3, _ = runner({"*": [VLMAnswer("different", 0.9, "", ""), ]})
    assert r3.ask_many([q()])[0].answer == "different"
    # a bad object followed by a good one: the good one wins
    r4, _ = runner({"*": [bad, VLMAnswer("different", 0.9, "r", "")]})
    assert r4.ask_many([q()])[0].answer == "different"


def test_three_invalid_votes_never_add_up_to_a_majority():
    bad = VLMAnswer("different", float("nan"), "", "")
    r, _ = runner({"*": [bad]}, votes=3, vote_temperature=0.7)
    (v,) = r.ask_many([q()])
    assert v.answer is None and v.error.startswith("invalid output") and v.confidence == 0.0


def test_the_budget_is_spent_in_question_order():
    r, b = runner({"*": [reply()]}, max_calls=3)
    vs = r.ask_many([q(f"LINK:link-r0-{i:06d}") for i in range(5)])
    assert [v.error for v in vs] == [None, None, None, "budget", "budget"]
    assert r.calls == 3 and r.budget_skipped == 2 and r.failures == 0 and len(b.calls) == 3


def test_a_question_gets_all_its_votes_or_none_even_with_concurrency():
    # three votes need three calls: the first question takes the whole budget, the second
    # gets nothing (before: two calls to the first and one to the second, and neither finished)
    r, b = runner({"*": [reply("different")]}, votes=3, max_calls=3, max_concurrency=2)
    v1, v2 = r.ask_many([q("LINK:a"), q("LINK:b")])
    assert v1.answer == "different" and v1.votes == {"different": 3}
    assert v2.error == "budget" and v2.votes == {}
    assert r.calls == 3 and r.budget_skipped == 1 and len(b.calls) == 3


def test_a_cheaper_later_question_still_uses_what_is_left(tmp_path):
    cache = AnswerCache(tmp_path)
    k0 = AnswerCache.key(b"img3", "PROMPT", OPTS, "fake", "fake-1", 0.7, 0)
    cache.put(k0, VLMAnswer("different", 0.9, "r", "raw"))  # q3 has one of its two votes cached
    r, _ = runner({"*": [reply("different")]}, cache=cache, votes=2, max_calls=3, max_concurrency=2)
    qs = [q("LINK:a", b"img1"), q("LINK:b", b"img2"), q("LINK:c", b"img3")]
    v1, v2, v3 = r.ask_many(qs)  # needs: 2, 2, 1; budget 3
    assert v1.error is None and v2.error == "budget" and v3.error is None
    assert r.calls == 3 and r.cache_hits == 1


def test_retries_draw_only_from_what_admission_left_over():
    sl = Sleeps()
    script = {"LINK:a": [VLMTransientError("503"), reply()], "LINK:b": [reply()]}
    # budget 3: a and b are admitted (2 votes), one unit is left over for a's retry
    r, b = runner(script, sleep=sl, max_calls=3, max_concurrency=2)
    v1, v2 = r.ask_many([q("LINK:a"), q("LINK:b")])
    assert v1.error is None and v2.error is None
    assert r.calls == 2 and r.retries == 1 and sl.waits == [2.0] and len(b.calls) == 3
    # budget 2: nothing is left over, so a's transient failure leaves a pending ("budget") and
    # b, whose vote was reserved, is still answered
    sl2 = Sleeps()
    r2, b2 = runner(script, sleep=sl2, max_calls=2, max_concurrency=2)
    w1, w2 = r2.ask_many([q("LINK:a"), q("LINK:b")])
    assert w1.error == "budget" and w1.answer is None and w2.error is None
    assert r2.calls == 2 and r2.retries == 0 and len(b2.calls) == 2 and sl2.waits == []
    assert r2.budget_skipped == 1 and r2.failures == 0


def test_max_calls_is_a_hard_limit_on_backend_invocations():
    # the review's case: transient, transient, invalid, transient, transient, then a valid reply
    # would be six invocations; with max_calls=1 only the first may be made
    T = VLMTransientError
    script = {"*": [T("1"), T("2"), "not json", T("3"), T("4"), reply()]}
    r, b = runner(script, max_calls=1)
    (v,) = r.ask_many([q()])
    assert v.answer is None and v.error == "budget"
    assert len(b.calls) == 1 and r.calls == 1 and r.retries == 0
    for cap in (2, 3, 4, 5):
        r, b = runner(script, max_calls=cap)
        r.ask_many([q()])
        assert len(b.calls) <= cap and r.calls + r.retries == len(b.calls)


def test_the_limit_holds_across_stage_calls_and_with_many_questions():
    T = VLMTransientError
    r, b = runner({"*": [T("x"), reply(), reply()]}, max_calls=5, max_concurrency=3)
    for batch in range(3):
        r.ask_many([q(f"LINK:{batch}-{i}") for i in range(4)])
    assert len(b.calls) <= 5 and r.calls + r.retries == len(b.calls)


def test_cache_hits_do_not_consume_the_budget(tmp_path):
    cache = AnswerCache(tmp_path)
    r1, _ = runner({"*": [reply()]}, cache=cache, max_calls=1)
    r1.ask_many([q()])
    r2, b2 = runner({"*": [reply()]}, cache=cache, max_calls=0)
    (v,) = r2.ask_many([q()])
    assert v.answer == "different" and b2.calls == [] and r2.budget_skipped == 0


def test_concurrency_is_bounded_and_order_is_preserved():
    class Probe:
        name, model = "probe", "m"
        live = peak = 0

        async def ask(self, image, prompt, options, temperature, *, tag=""):
            Probe.live += 1
            Probe.peak = max(Probe.peak, Probe.live)
            await asyncio.sleep(0.01)
            Probe.live -= 1
            return VLMAnswer(options[0], 1.0, tag, "")

    r, _ = runner(None, backend=Probe(), max_concurrency=2)
    vs = r.ask_many([q(f"LINK:{i}") for i in range(7)])
    assert Probe.peak == 2 and [v.reason for v in vs] == [f"LINK:{i}" for i in range(7)]


def test_it_works_inside_a_running_event_loop():
    r, _ = runner({"*": reply()})

    async def main():
        return r.ask_many([q()])  # a synchronous call from a coroutine (Jupyter)

    (v,) = asyncio.run(main())
    assert v.answer == "different"


def test_an_empty_batch_is_fine_and_starts_no_thread():
    r, _ = runner({})
    assert r.ask_many([]) == [] and r._thread is None


# ---- one live loop for the runner's whole life (review: pooled connections are loop-bound) ----


def test_every_call_runs_on_the_same_live_loop_and_close_stops_it():
    seen = []

    class Loops:
        name, model = "loops", "m"

        async def ask(self, image, prompt, options, temperature, *, tag=""):
            seen.append(asyncio.get_running_loop())
            return VLMAnswer(options[0], 1.0, "", "")

    r, _ = runner(None, backend=Loops())
    r.ask_many([q("LINK:a")])
    r.ask_many([q("LINK:b")])  # a second stage
    assert len(seen) == 2 and seen[0] is seen[1] and not seen[0].is_closed()
    thread = r._thread
    r.close()
    assert not thread.is_alive() and seen[0].is_closed()
    with pytest.raises(RuntimeError, match="closed"):
        r.ask_many([q()])
    r.close()  # idempotent


def test_close_closes_the_backend_once_on_its_own_loop():
    closes = []

    class Closing:
        name, model = "closing", "m"

        async def ask(self, image, prompt, options, temperature, *, tag=""):
            return VLMAnswer(options[0], 1.0, "", "")

        async def aclose(self):
            closes.append(asyncio.get_running_loop())

    r, _ = runner(None, backend=Closing())
    r.ask_many([q()])
    loop = r._loop
    r.close()
    r.close()
    assert closes == [loop]
    quiet, fake = runner({})
    quiet.close()  # never used: nothing to close, no thread
    assert fake.closed == 0 and quiet._thread is None


class _Quiet(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # keep-alive, like a real API server

    def do_GET(self):
        body = b"ok"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def local_server():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Quiet)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/"
    server.shutdown()
    server.server_close()


def make_http_backend(url):
    import httpx

    class HttpBackend:
        name, model = "http", "m"

        def __init__(self):
            self._client = None

        async def ask(self, image, prompt, options, temperature, *, tag=""):
            if self._client is None:  # created on the first loop, like AsyncOpenAI
                self._client = httpx.AsyncClient()
            resp = await self._client.get(url)  # reuses a pooled keep-alive connection
            assert resp.status_code == 200
            return VLMAnswer(options[0], 1.0, "", "")

        async def aclose(self):
            if self._client is not None:
                await self._client.aclose()

    return HttpBackend()


def test_pooled_http_connections_survive_successive_stage_calls(local_server):
    pytest.importorskip("httpx")
    r, _ = runner(None, backend=make_http_backend(local_server))
    for stage in ("switch", "screen", "link-1", "link-2"):  # one ask_many per stage call
        (v,) = r.ask_many([q(f"LINK:{stage}")])
        assert v.error is None, (stage, v.error)  # a per-call loop fails with "Event loop is closed"


def test_pooled_http_connections_also_work_from_a_notebook_loop(local_server):
    pytest.importorskip("httpx")
    r, _ = runner(None, backend=make_http_backend(local_server))

    async def main():
        first = r.ask_many([q("LINK:a")])
        second = r.ask_many([q("LINK:b")])
        return first, second

    first, second = asyncio.run(main())
    assert first[0].error is None and second[0].error is None
