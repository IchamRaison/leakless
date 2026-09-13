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
