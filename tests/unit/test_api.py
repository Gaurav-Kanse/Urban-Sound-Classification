"""API handler and invalid-upload tests without opening local sockets."""

import asyncio
from tempfile import SpooledTemporaryFile

import pytest
from fastapi import HTTPException
from starlette.datastructures import UploadFile


def upload(content: bytes, filename: str) -> UploadFile:
    file = SpooledTemporaryFile()
    file.write(content)
    file.seek(0)
    return UploadFile(file=file, filename=filename)


def test_health_and_info():
    from soundsense.api.app import health, model_info, root

    assert health()["status"] == "ok"
    assert root()["name"] == "SoundSense"
    assert len(model_info()["classes"]) == 10


def test_predict_rejects_unsupported_extension():
    from soundsense.api.app import predict

    test_upload = upload(b"abc", "note.txt")
    with pytest.raises(HTTPException) as error:
        asyncio.run(predict(test_upload))
    assert error.value.status_code == 415


def test_predict_reports_missing_model(tmp_path, monkeypatch):
    from soundsense.api import app as api

    monkeypatch.setattr(api, "CHECKPOINT", tmp_path / "missing.pt")
    monkeypatch.setattr(api, "_predictor", None)
    test_upload = upload(b"audio", "recording.wav")
    with pytest.raises(HTTPException) as error:
        asyncio.run(api.predict(test_upload))
    assert error.value.status_code == 503


def test_batch_validates_extensions_before_loading_model():
    from soundsense.api.app import predict_batch

    uploads = [upload(b"x", "note.txt")]
    with pytest.raises(HTTPException) as error:
        asyncio.run(predict_batch(uploads))
    assert error.value.status_code == 415


def test_batch_rejects_more_than_ten_uploads():
    from soundsense.api.app import predict_batch

    uploads = [upload(b"x", f"{index}.wav") for index in range(11)]
    with pytest.raises(HTTPException) as error:
        asyncio.run(predict_batch(uploads))
    assert error.value.status_code == 400
