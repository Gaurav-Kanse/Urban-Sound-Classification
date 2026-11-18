"""Download and safely extract UrbanSound8K from the official Zenodo record."""

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    from .verify_dataset import DEFAULT_DATASET_DIR, validate_dataset
except ImportError:
    from verify_dataset import DEFAULT_DATASET_DIR, validate_dataset

ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS_DIR = ROOT / "data/downloads"
RECORD_URL = "https://zenodo.org/api/records/1203745"
ARCHIVE_NAME = "UrbanSound8K.tar.gz"
BUFFER_SIZE = 1024 * 1024
SPACE_MARGIN = 64 * 1024 * 1024

OpenUrl = Callable[..., object]


class DatasetDownloadError(RuntimeError):
    """An expected dataset download, validation, or extraction failure."""


@dataclass(frozen=True)
class ArchiveInfo:
    """Official archive details read from the Zenodo record."""

    name: str
    url: str
    size: int
    checksum: str


def get_archive_info(open_url: OpenUrl = urlopen) -> ArchiveInfo:
    """Resolve the archive URL, size, and checksum directly from Zenodo."""
    request = Request(RECORD_URL, headers={"User-Agent": "SoundSense-dataset-setup/1.0"})
    try:
        with open_url(request, timeout=60) as response:
            record = json.load(response)
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise DatasetDownloadError(f"Could not read the official Zenodo record: {exc}") from exc

    for file_info in record.get("files", []):
        if file_info.get("key") == ARCHIVE_NAME:
            checksum = file_info.get("checksum", "")
            url = file_info.get("links", {}).get("self")
            size = file_info.get("size")
            if not isinstance(url, str) or not url.startswith("https://zenodo.org/"):
                raise DatasetDownloadError(
                    "Zenodo record did not provide the expected HTTPS file URL."
                )
            if not isinstance(size, int) or size <= 0 or not checksum.startswith("md5:"):
                raise DatasetDownloadError(
                    "Zenodo record has incomplete archive size/checksum metadata."
                )
            return ArchiveInfo(ARCHIVE_NAME, url, size, checksum.partition(":")[2])
    raise DatasetDownloadError(f"Zenodo record does not list {ARCHIVE_NAME}.")


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as archive_file:
        for chunk in iter(lambda: archive_file.read(BUFFER_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matches_official_archive(path: Path, info: ArchiveInfo) -> bool:
    return path.is_file() and path.stat().st_size == info.size and _md5(path) == info.checksum


def download_archive(archive_path: Path, info: ArchiveInfo, open_url: OpenUrl = urlopen) -> Path:
    """Download to a resumable partial file and promote it after checksum validation."""
    partial_path = archive_path.with_name(f"{archive_path.name}.part")
    if _matches_official_archive(archive_path, info):
        print(f"Using already downloaded and verified archive: {archive_path}")
        return archive_path
    if archive_path.exists():
        print(f"Removing an incomplete or checksum-invalid archive: {archive_path}")
        archive_path.unlink()
    if partial_path.exists() and partial_path.stat().st_size >= info.size:
        partial_path.unlink()

    offset = partial_path.stat().st_size if partial_path.exists() else 0
    required_space = max(0, info.size - offset) + SPACE_MARGIN
    if shutil.disk_usage(archive_path.parent).free < required_space:
        raise DatasetDownloadError(
            "Insufficient disk space for the archive download: "
            f"need about {required_space:,} bytes free."
        )
    request_headers = {"User-Agent": "SoundSense-dataset-setup/1.0"}
    if offset:
        request_headers["Range"] = f"bytes={offset}-"
        print(f"Resuming download at {offset:,} bytes.")
    request = Request(info.url, headers=request_headers)

    try:
        with open_url(request, timeout=60) as response:
            status = getattr(response, "status", None) or response.getcode()
            if offset and status == 206:
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(f"bytes {offset}-"):
                    raise DatasetDownloadError(
                        "Server returned an invalid range for the resumed download."
                    )
                mode = "ab"
                downloaded = offset
            else:
                mode = "wb"
                downloaded = 0
            next_notice = downloaded + max(info.size // 20, 1)
            with partial_path.open(mode) as output:
                while chunk := response.read(BUFFER_SIZE):
                    output.write(chunk)
                    downloaded += len(chunk)
                    if downloaded >= next_notice or downloaded == info.size:
                        percent = min(100, downloaded * 100 // info.size)
                        print(
                            f"\rDownloading {info.name}: {percent}% "
                            f"({downloaded:,}/{info.size:,} bytes)"
                        )
                        next_notice = downloaded + max(info.size // 20, 1)
                    if downloaded > info.size:
                        partial_path.unlink(missing_ok=True)
                        raise DatasetDownloadError("Download exceeded the size listed by Zenodo.")
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise DatasetDownloadError(
            f"Download failed: {exc}. Any partial data was kept for a later resume."
        ) from exc

    if partial_path.stat().st_size != info.size:
        raise DatasetDownloadError(
            f"Download is incomplete ({partial_path.stat().st_size:,} of {info.size:,} bytes). "
            "Run this command again to resume."
        )
    if _md5(partial_path) != info.checksum:
        partial_path.unlink(missing_ok=True)
        raise DatasetDownloadError(
            "Downloaded archive failed Zenodo's MD5 checksum. The partial file was removed; retry."
        )
    os.replace(partial_path, archive_path)
    print(f"Verified archive saved: {archive_path}")
    return archive_path


def _safe_member_path(stage: Path, member: tarfile.TarInfo) -> Path:
    """Resolve a regular tar member only when it is inside the expected dataset root."""
    name = member.name
    archive_path = PurePosixPath(name)
    path_parts = name.rstrip("/").split("/")
    if (
        not name
        or name.startswith("/")
        or "\\" in name
        or (name.endswith("/") and not member.isdir())
        or any(part in {"", ".", ".."} for part in path_parts)
        or not archive_path.parts
        or archive_path.parts[0] != "UrbanSound8K"
    ):
        raise DatasetDownloadError(f"Unsafe or unexpected archive path: {name!r}")
    if not (member.isdir() or member.isfile()):
        raise DatasetDownloadError(f"Unsupported special file in archive: {name!r}")

    destination = stage.joinpath(*archive_path.parts)
    if not destination.resolve().is_relative_to(stage.resolve()):
        raise DatasetDownloadError(f"Archive member escapes extraction directory: {name!r}")
    return destination


def extract_dataset(archive_path: Path, dataset_dir: Path, raw_dir: Path) -> None:
    """Extract regular files into a temporary directory, then atomically install the dataset."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".UrbanSound8K-extract-", dir=raw_dir))
    try:
        try:
            with tarfile.open(archive_path, mode="r:gz") as archive:
                members = archive.getmembers()
                destinations = [(member, _safe_member_path(stage, member)) for member in members]
                uncompressed_size = sum(
                    member.size for member, _ in destinations if member.isfile()
                )
                free_space = shutil.disk_usage(raw_dir).free
                if free_space < uncompressed_size + SPACE_MARGIN:
                    raise DatasetDownloadError(
                        "Insufficient disk space for extraction: "
                        f"need about {uncompressed_size + SPACE_MARGIN:,} bytes free."
                    )

                for member, destination in destinations:
                    if member.isdir():
                        destination.mkdir(parents=True, exist_ok=True)
                        continue
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    source = archive.extractfile(member)
                    if source is None:
                        raise DatasetDownloadError(f"Could not read archive member: {member.name}")
                    with source, destination.open("xb") as output:
                        shutil.copyfileobj(source, output, length=BUFFER_SIZE)
        except (tarfile.TarError, OSError, EOFError) as exc:
            if isinstance(exc, DatasetDownloadError):
                raise
            raise DatasetDownloadError(f"Could not extract archive: {exc}") from exc

        extracted_root = stage / "UrbanSound8K"
        validation = validate_dataset(extracted_root)
        if not validation.valid:
            details = validation.errors + [
                f"{len(validation.missing_audio):,} metadata-referenced audio files are missing."
            ]
            raise DatasetDownloadError(
                "Extracted archive did not pass dataset validation: " + " ".join(details)
            )
        if dataset_dir.exists():
            raise DatasetDownloadError(
                f"Dataset destination already exists and is not being overwritten: {dataset_dir}"
            )
        os.replace(extracted_root, dataset_dir)
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def run_download(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    downloads_dir: Path = DOWNLOADS_DIR,
    open_url: OpenUrl = urlopen,
) -> None:
    """Download, validate, and install UrbanSound8K without replacing valid data."""
    if dataset_dir.exists():
        existing = validate_dataset(dataset_dir)
        if existing.valid:
            print(f"Valid UrbanSound8K dataset already exists; no download needed: {dataset_dir}")
            return
        raise DatasetDownloadError(
            f"Dataset destination exists but is incomplete or invalid: {dataset_dir}. "
            "Move it aside or repair it before retrying; it will not be overwritten."
        )

    downloads_dir.mkdir(parents=True, exist_ok=True)
    info = get_archive_info(open_url)
    archive_path = downloads_dir / info.name
    print(f"UrbanSound8K archive size: {info.size:,} bytes.")
    archive_path = download_archive(archive_path, info, open_url)
    raw_dir = dataset_dir.parent
    print(f"Extracting verified archive into {raw_dir} ...")
    extract_dataset(archive_path, dataset_dir, raw_dir)
    print(f"SUCCESS: UrbanSound8K is installed and validated at {dataset_dir}")


def main() -> int:
    try:
        run_download()
    except DatasetDownloadError as exc:
        print(f"ERROR: {exc}")
        return 1
    except OSError as exc:
        print(f"ERROR: File or disk operation failed: {exc}")
        return 1
    except KeyboardInterrupt:
        print("\nDownload interrupted. Any partial archive is kept for a later resume.")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
