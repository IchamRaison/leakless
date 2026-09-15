import numpy as np
import pytest

from pipe.temporal_language import facts, model_sample, CHANNELS


def history(scores):
    result=np.zeros((len(scores),10)); result[:,9]=scores
    return result


def test_temporal_boundaries_order_and_padding():
    assert facts(history([.9]*30))["pattern"] == "brief"
    assert facts(history([.9]*31))["pattern"] == "persistent"
    assert facts(history([.9]*10+[.1]*5))["pattern"] == "ended"
    assert facts(history([.9]*5+[.1]*5+[.9]*5))["pattern"] == "intermittent"
    x=history([.1]*33+[.9]*31)
    assert facts(x)["pattern"] == "persistent"
    assert facts(x[::-1])["pattern"] == "ended"
    sample=model_sample(history([.9]*31),np.zeros(10),np.ones(10))
    assert tuple(sample["time_series"].shape)==(10,64)
    assert sample["time_series"][:,:33].count_nonzero()==0
    assert "31 observed" in sample["pre_prompt"]
    assert "absolute percentile 75" == CHANNELS[4]
    for value in (np.zeros((65,10)),np.zeros((5,9)),np.full((5,10),np.nan),history([2])):
        with pytest.raises(ValueError): facts(value)


def test_background_description_is_bounded_and_explicit_on_failure(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Lock
    from pipe.api.temporal import schedule_language
    from pipe.temporal_tracking import Tracker
    policy={"open_threshold":.8,"close_threshold":.4,"open_seconds":3,"close_seconds":5,
            "max_gap_seconds":2,"language_after_seconds":30}
    tracker=Tracker(tmp_path/"jobs.db",policy,"test")
    session=tracker.create(now=100)["session_id"]
    for i in range(31):
        tracker.consume(session,i,101+i,101+i,"a"*64,.9,[1.]*9)
    class Narrator:
        def describe(self,history):
            assert np.asarray(history).shape==(31,10)
            assert facts(history)["pattern"]=="persistent"
            return {"description":"Persistance observée.","description_source":"test_model","fallback_used":False}
    with ThreadPoolExecutor(max_workers=1) as executor:
        value={"tracker":tracker,"narrator":Narrator(),"executor":executor,"language_lock":Lock(),"language_future":None}
        schedule_language(value); value["language_future"].result(timeout=5)
        ready=tracker.snapshot(session,131)["previews"][0]
        assert ready["status"]=="ready" and ready["description_source"]=="test_model"
        assert not ready["sent"] and ready["recipient"]=="Nevil"
        for i in range(31,36): tracker.consume(session,i,101+i,101+i,"a"*64,.1,[1.]*9)
        class FailedNarrator:
            def describe(self,history): raise RuntimeError("test failure")
        value["narrator"]=FailedNarrator()
        schedule_language(value); value["language_future"].result(timeout=5)
        ending=tracker.snapshot(session,136)["previews"][0]
        assert ending["status"]=="ready" and ending["description_source"]=="template_fallback"
        assert ending["temporal_description"]["fallback_reason"]=="RuntimeError"
        assert tracker.pending_language() is None
