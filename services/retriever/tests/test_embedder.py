"""Unit tests for embedder.load_model / embed_query.

The SentenceTransformer class is stubbed: no 80 MB download, no HuggingFace,
no network. The parity test loads the *ingestion* embedder by file path
(importing it by name would collide with this service's own `embedder`
module) and proves both paths drive `encode` the same way.
"""

import importlib.util
import sys
from pathlib import Path

import embedder as retriever_embedder
import numpy as np
import pytest


def _load_ingestion_embedder():
    repo_root = Path(__file__).resolve().parents[3]
    path = repo_root / "services" / "ingestion" / "embedder.py"
    spec = importlib.util.spec_from_file_location(
        "ingestion_embedder_under_test", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeModel:
    """Stands in for SentenceTransformer and records encode calls."""

    instances = []

    def __init__(self, *args, **kwargs):
        self.encode_calls = []
        FakeModel.instances.append(self)

    def get_sentence_embedding_dimension(self):
        return 384

    def encode(self, texts, **kwargs):
        self.encode_calls.append((texts, kwargs))
        # Mirror SentenceTransformer shapes: a single string encodes to a 1-D
        # vector, a batch to a 2-D array. embed_query relies on the 1-D shape
        # (.tolist() must yield 384 floats, not one row).
        if isinstance(texts, str):
            return np.array([float(sum(ord(c) for c in texts) % 100) / 100.0] * 384)
        return np.array(
            [[float(sum(ord(c) for c in t) % 100) / 100.0] * 384 for t in texts]
        )


class WrongDimModel(FakeModel):
    def get_sentence_embedding_dimension(self):
        return 768


@pytest.fixture(autouse=True)
def _forget_instances():
    FakeModel.instances.clear()
    yield
    FakeModel.instances.clear()


def test_load_model_returns_model_on_matching_dim(monkeypatch):
    monkeypatch.setattr(retriever_embedder, "SentenceTransformer", FakeModel)

    assert isinstance(retriever_embedder.load_model("anything"), FakeModel)


def test_load_model_rejects_wrong_dimension_at_startup(monkeypatch):
    monkeypatch.setattr(retriever_embedder, "SentenceTransformer", WrongDimModel)

    with pytest.raises(ValueError, match="768.*384|384.*768"):
        retriever_embedder.load_model("wrong-dim-model")


def test_embed_query_returns_flat_float_list_with_exact_encode_args():
    model = FakeModel()

    vec = retriever_embedder.embed_query(model, "hello world")

    assert len(vec) == 384
    assert all(isinstance(x, float) for x in vec)
    ((texts, kwargs),) = model.encode_calls
    assert texts == "hello world"
    assert kwargs == {"show_progress_bar": False, "convert_to_numpy": True}


def test_query_and_document_paths_drive_encode_identically():
    """The parity property: same model object, same encode kwargs.

    A future max_seq_length/normalisation edit to either path breaks this.
    The old string-equality check could not catch that; this does.
    """
    ingestion_embedder = _load_ingestion_embedder()
    model = FakeModel()

    q_vec = retriever_embedder.embed_query(model, "shared sentence")
    (d_vec,) = ingestion_embedder.embed_chunks(model, ["shared sentence"])

    assert len(model.encode_calls) == 2
    (_, q_kwargs), (_, d_kwargs) = model.encode_calls
    assert q_kwargs == d_kwargs == {
        "show_progress_bar": False,
        "convert_to_numpy": True,
    }
    assert np.allclose(q_vec, d_vec)
