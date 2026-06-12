import os
import re
import numpy as np
import onnxruntime as ort
import sentencepiece as spm
import soundfile as sf
import kaldi_native_fbank as knf


class STTEngine:
    def __init__(
        self,
        model_dir="/home/lerobot/aicapstone/lerobot/capstone/gengen/models/sensevoice_ko",
        model_name="model.onnx",
        language="ko",
        device="cpu",
    ):
        self.model_dir = os.path.expanduser(model_dir)
        self.model_name = model_name
        self.language = language
        self.device = device

        model_path = os.path.join(self.model_dir, self.model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"model not found: {model_path}")

        if "cuda" in self.device.lower():
            providers = ["CUDAExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        try:
            self.sess = ort.InferenceSession(model_path, providers=providers)
        except Exception:
            self.sess = ort.InferenceSession(
                model_path,
                providers=["CPUExecutionProvider"],
            )

        sp_model = os.path.join(
            self.model_dir,
            "chn_jpn_yue_eng_ko_spectok.bpe.model",
        )

        if not os.path.exists(sp_model):
            for f in os.listdir(self.model_dir):
                if f.endswith(".model") and "bpe" in f:
                    sp_model = os.path.join(self.model_dir, f)
                    break

        if not os.path.exists(sp_model):
            raise FileNotFoundError(f"tokenizer not found in: {self.model_dir}")

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(sp_model)

        self.frontend_opts = knf.FbankOptions()
        self.frontend_opts.frame_opts.dither = 0.0
        self.frontend_opts.frame_opts.snip_edges = True
        self.frontend_opts.frame_opts.samp_freq = 16000.0
        self.frontend_opts.mel_opts.num_bins = 80

        self.lang_dict = {
            "zh": 3,
            "en": 4,
            "ko": 5,
            "ja": 6,
            "yue": 7,
            "auto": 0,
        }
        self.lang_id = self.lang_dict.get(self.language, 0)

    def apply_lfr(self, inputs, lfr_m=7, lfr_n=6):
        if inputs is None or inputs.size == 0:
            return None

        _, t, _ = inputs.shape
        if t < lfr_m:
            return None

        arr = inputs[0]
        frames = []
        t_max = t - lfr_m + 1

        for i in range(0, t_max, lfr_n):
            frames.append(arr[i:i + lfr_m].flatten())

        if not frames:
            return None

        return np.array(frames, dtype=np.float32)[None, :, :]

    def extract_feat(self, waveform: np.ndarray):
        if waveform is None or waveform.size == 0:
            return None

        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        waveform = waveform.astype(np.float32)

        fbank = knf.OnlineFbank(self.frontend_opts)
        fbank.accept_waveform(16000, (waveform * 32768.0).tolist())

        frames = []
        for i in range(fbank.num_frames_ready):
            frames.append(fbank.get_frame(i))

        if not frames:
            return None

        feat = np.array(frames, dtype=np.float32)
        feat = (feat - np.mean(feat, axis=0)) / (np.std(feat, axis=0) + 1e-5)

        return feat[None, :, :]

    def text_postprocess(self, text: str):
        if not text:
            return ""

        text = re.sub(r"<\|.*?\|>", "", text)
        text = re.sub(r"[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9\s.,?!]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def transcribe(self, wav_path: str):
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"wav not found: {wav_path}")

        speech, sr = sf.read(wav_path)

        if sr != 16000:
            raise ValueError(f"expected 16000Hz, got {sr}")

        if speech is None or len(speech) < 160:
            return ""

        feats_80 = self.extract_feat(speech)
        if feats_80 is None:
            return ""

        feats_560 = self.apply_lfr(feats_80)
        if feats_560 is None:
            return ""

        feats_len = np.array([feats_560.shape[1]], dtype=np.int32)

        input_feed = {
            self.sess.get_inputs()[0].name: feats_560,
            self.sess.get_inputs()[1].name: feats_len,
            self.sess.get_inputs()[2].name: np.array([self.lang_id], dtype=np.int32),
            self.sess.get_inputs()[3].name: np.array([15], dtype=np.int32),
        }

        outputs = self.sess.run(None, input_feed)
        if not outputs:
            return ""

        logits = outputs[0]
        token_ids = np.argmax(logits[0], axis=-1).tolist()

        text_tokens = []
        prev = -1
        for t in token_ids:
            if t != 0 and t != prev:
                text_tokens.append(t)
            prev = t

        text = self.sp.decode(text_tokens)
        return self.text_postprocess(text)
