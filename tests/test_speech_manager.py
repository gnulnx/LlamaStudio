import io
import os
import sys
import tarfile
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.speech_manager import SpeechError, SpeechManager


class FakeProcess:
    def __init__(self):
        self.returncode = None
        self.command = None
        self.stderr = io.BytesIO()

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = 0

    def kill(self):
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


class FakeTranscriptResponse:
    status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return {"text": "  Hello B2.  "}


class TestSpeechManager(unittest.TestCase):
    def test_extract_release_accepts_internal_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "release.tar.gz"
            destination = Path(tmp) / "output"
            with tarfile.open(archive, "w:gz") as bundle:
                data = b"library"
                library = tarfile.TarInfo("release/libwhisper.so.1")
                library.size = len(data)
                bundle.addfile(library, io.BytesIO(data))
                link = tarfile.TarInfo("release/libwhisper.so")
                link.type = tarfile.SYMTYPE
                link.linkname = "libwhisper.so.1"
                bundle.addfile(link)

            SpeechManager._extract_release(archive, destination)

            self.assertEqual((destination / "release/libwhisper.so").read_bytes(), b"library")

    def test_extract_release_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "unsafe.tar.gz"
            with tarfile.open(archive, "w:gz") as bundle:
                member = tarfile.TarInfo("../escape")
                member.size = 1
                bundle.addfile(member, io.BytesIO(b"x"))

            with self.assertRaises(SpeechError):
                SpeechManager._extract_release(archive, Path(tmp) / "output")

    def test_start_builds_cpu_server_command(self):
        manager = SpeechManager()
        fake_process = FakeProcess()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary_dir = root / "bin" / "v1.9.1" / "release"
            binary_dir.mkdir(parents=True)
            binary = binary_dir / "whisper-server"
            binary.touch()
            model_path = root / "models" / "ggml-small.en.bin"
            model_path.parent.mkdir()
            model_path.touch()
            config = {
                "install_dir": str(root),
                "model": "small.en",
                "language": "auto",
                "use_gpu": False,
            }

            def fake_popen(command, **kwargs):
                fake_process.command = command
                return fake_process

            with (
                patch(
                    "app.speech_manager.config_loader.get_speech_config",
                    return_value=config,
                ),
                patch("app.speech_manager.config_loader.save_speech_config"),
                patch.object(manager, "_port_in_use", side_effect=[False, False, True]),
                patch.object(manager, "_wait_for_ready", return_value=True),
                patch("app.speech_manager.subprocess.Popen", side_effect=fake_popen),
            ):
                status = manager.start()

        self.assertTrue(status["running"])
        self.assertIn("--no-gpu", fake_process.command)
        self.assertIn(str(model_path), fake_process.command)

    def test_transcribe_normalizes_audio_and_returns_trimmed_text(self):
        manager = SpeechManager()
        with (
            patch.object(
                SpeechManager,
                "is_running",
                new_callable=PropertyMock,
                return_value=True,
            ),
            patch.object(manager, "_normalize_audio", return_value=b"wav") as normalize,
            patch("app.speech_manager.httpx.post", return_value=FakeTranscriptResponse()) as post,
        ):
            result = manager.transcribe_bytes(b"browser audio")

        self.assertEqual(result["text"], "Hello B2.")
        normalize.assert_called_once_with(b"browser audio")
        self.assertEqual(post.call_args.kwargs["files"]["file"][2], "audio/wav")

    def test_normalize_audio_builds_seekable_wav_header(self):
        raw_pcm = b"\x00\x00\x01\x00"
        completed = SimpleNamespace(returncode=0, stdout=raw_pcm, stderr=b"")
        with (
            patch("app.speech_manager.shutil.which", return_value="/usr/bin/ffmpeg"),
            patch("app.speech_manager.subprocess.run", return_value=completed),
        ):
            normalized = SpeechManager._normalize_audio(b"webm")

        with wave.open(io.BytesIO(normalized), "rb") as recording:
            self.assertEqual(recording.getnchannels(), 1)
            self.assertEqual(recording.getframerate(), 16_000)
            self.assertEqual(recording.readframes(2), raw_pcm)

    def test_transcribe_rejects_empty_recording(self):
        with self.assertRaisesRegex(SpeechError, "empty"):
            SpeechManager().transcribe_bytes(b"")

    def test_system_recording_uses_pulse_default_and_transcribes(self):
        manager = SpeechManager()
        fake_process = FakeProcess()
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)

            def fake_popen(command, **kwargs):
                fake_process.command = command
                recording_path = Path(command[-1])
                with wave.open(str(recording_path), "wb") as recording:
                    recording.setnchannels(1)
                    recording.setsampwidth(2)
                    recording.setframerate(16_000)
                    recording.writeframes(b"\x00\x00" * 100)
                return fake_process

            with (
                patch("app.speech_manager.check_path_safe", return_value=workspace),
                patch("app.speech_manager.shutil.which", return_value="/usr/bin/arecord"),
                patch("app.speech_manager.subprocess.Popen", side_effect=fake_popen),
                patch("app.speech_manager.time.sleep"),
                patch.object(
                    manager,
                    "transcribe_bytes",
                    return_value={"text": "System capture worked."},
                ) as transcribe,
            ):
                started = manager.start_system_recording()
                result = manager.stop_system_recording()

        self.assertTrue(started["recording"])
        self.assertEqual(result["text"], "System capture worked.")
        self.assertIn("pulse", fake_process.command)
        transcribe.assert_called_once()


if __name__ == "__main__":
    unittest.main()
