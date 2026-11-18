"""Offline tests for UrbanSound8K download and verification."""

import csv
import hashlib
import io
import json
import tarfile

import pytest

from scripts.download_dataset import (
    ARCHIVE_NAME,
    RECORD_URL,
    ArchiveInfo,
    DatasetDownloadError,
    download_archive,
    extract_dataset,
    run_download,
)
from scripts.verify_dataset import EXPECTED_CLASSES, EXPECTED_ROWS, validate_dataset


class MockResponse(io.BytesIO):
    def __init__(self, content: bytes, status: int = 200, headers: dict | None = None):
        super().__init__(content)
        self.status = status
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def getcode(self):
        return self.status


def build_archive() -> bytes:
    """Create a small compressed archive with the official metadata row count."""
    metadata = io.StringIO(newline="")
    writer = csv.writer(metadata)
    writer.writerow(["slice_file_name", "fold", "class"])

    files: list[tuple[str, bytes]] = []
    classes = sorted(EXPECTED_CLASSES)
    for index in range(EXPECTED_ROWS):
        filename = f"clip_{index:05}.wav"
        fold = index % 10 + 1
        writer.writerow([filename, fold, classes[index % len(classes)]])
        files.append((f"UrbanSound8K/audio/fold{fold}/{filename}", b""))

    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        metadata_bytes = metadata.getvalue().encode()
        metadata_info = tarfile.TarInfo("UrbanSound8K/metadata/UrbanSound8K.csv")
        metadata_info.size = len(metadata_bytes)
        archive.addfile(metadata_info, io.BytesIO(metadata_bytes))
        for fold in range(1, 11):
            folder = tarfile.TarInfo(f"UrbanSound8K/audio/fold{fold}/")
            folder.type = tarfile.DIRTYPE
            archive.addfile(folder)
        for filename, contents in files:
            member = tarfile.TarInfo(filename)
            member.size = len(contents)
            archive.addfile(member, io.BytesIO(contents))
    return output.getvalue()


def test_verifier_reports_bad_row_count_and_missing_audio(tmp_path):
    dataset = tmp_path / "UrbanSound8K"
    metadata_dir = dataset / "metadata"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "UrbanSound8K.csv").write_text(
        "slice_file_name,fold,class\nmissing.wav,1,air_conditioner\n", encoding="utf-8"
    )
    for fold in range(1, 11):
        (dataset / "audio" / f"fold{fold}").mkdir(parents=True)

    result = validate_dataset(dataset)

    assert not result.valid
    assert result.metadata_rows == 1
    assert len(result.missing_audio) == 1
    assert any("Expected 8,732 metadata rows" in error for error in result.errors)


def test_verifier_accepts_complete_temporary_dataset(tmp_path):
    archive_path = tmp_path / ARCHIVE_NAME
    archive_path.write_bytes(build_archive())
    dataset = tmp_path / "UrbanSound8K"
    extract_dataset(archive_path, dataset, tmp_path)

    result = validate_dataset(dataset)

    assert result.valid
    assert result.metadata_rows == EXPECTED_ROWS
    assert result.missing_audio == []


def test_download_uses_zenodo_metadata_and_installs_archive(tmp_path):
    archive_bytes = build_archive()
    checksum = hashlib.md5(archive_bytes).hexdigest()
    record = {
        "files": [
            {
                "key": ARCHIVE_NAME,
                "size": len(archive_bytes),
                "checksum": f"md5:{checksum}",
                "links": {
                    "self": "https://zenodo.org/api/records/1203745/files/UrbanSound8K.tar.gz/content"
                },
            }
        ]
    }
    calls = []

    def open_url(request, timeout):
        url = request.full_url
        calls.append(url)
        if url == RECORD_URL:
            return MockResponse(json.dumps(record).encode())
        return MockResponse(archive_bytes)

    dataset = tmp_path / "data/raw/UrbanSound8K"
    downloads = tmp_path / "data/downloads"
    run_download(dataset, downloads, open_url)

    assert validate_dataset(dataset).valid
    assert (downloads / ARCHIVE_NAME).is_file()
    assert calls == [
        RECORD_URL,
        "https://zenodo.org/api/records/1203745/files/UrbanSound8K.tar.gz/content",
    ]

    def should_not_download(*args, **kwargs):
        pytest.fail("A valid extracted dataset should not be downloaded again.")

    run_download(dataset, downloads, should_not_download)


def test_download_archive_resumes_partial_response(tmp_path):
    payload = b"abcdef"
    info = ArchiveInfo(
        name=ARCHIVE_NAME,
        url="https://zenodo.org/test",
        size=len(payload),
        checksum=hashlib.md5(payload).hexdigest(),
    )
    archive_path = tmp_path / ARCHIVE_NAME

    class InterruptedResponse(MockResponse):
        def __init__(self):
            super().__init__(b"")
            self.read_count = 0

        def read(self, size=-1):
            self.read_count += 1
            if self.read_count == 1:
                return b"abc"
            raise OSError("connection interrupted")

    with pytest.raises(DatasetDownloadError, match="partial data was kept"):
        download_archive(archive_path, info, lambda *_args, **_kwargs: InterruptedResponse())
    assert archive_path.with_name(f"{ARCHIVE_NAME}.part").read_bytes() == b"abc"

    def resume(request, timeout):
        assert request.get_header("Range") == "bytes=3-"
        return MockResponse(b"def", status=206, headers={"Content-Range": "bytes 3-5/6"})

    download_archive(archive_path, info, resume)
    assert archive_path.read_bytes() == payload
    assert not archive_path.with_name(f"{ARCHIVE_NAME}.part").exists()


def test_extraction_rejects_path_traversal(tmp_path):
    archive_path = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        member = tarfile.TarInfo("UrbanSound8K/../../outside.txt")
        member.size = 4
        archive.addfile(member, io.BytesIO(b"safe"))

    with pytest.raises(DatasetDownloadError, match="Unsafe or unexpected archive path"):
        extract_dataset(archive_path, tmp_path / "raw/UrbanSound8K", tmp_path / "raw")
    assert not (tmp_path / "outside.txt").exists()
