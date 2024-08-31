import os

import fairseq
import pytorch_lightning as pl
import requests
import torch
import torch.nn as nn
from tqdm import tqdm

UTMOS_CKPT_URL = "https://huggingface.co/spaces/sarulab-speech/UTMOS-demo/resolve/main/epoch%3D3-step%3D7459.ckpt"
WAV2VEC_URL = "https://huggingface.co/spaces/sarulab-speech/UTMOS-demo/resolve/main/wav2vec_small.pt"

"""
UTMOS score, automatic Mean Opinion Score (MOS) prediction system, 
adapted from https://huggingface.co/spaces/sarulab-speech/UTMOS-demo
"""


class UTMOSScore:
    """Predicting score for each audio clip."""

    def __init__(self, device, ckpt_path="epoch=3-step=7459.ckpt"):
        self.device = device
        filepath = os.path.join(os.path.dirname(__file__), ckpt_path)
        if not os.path.exists(filepath):
            download_file(UTMOS_CKPT_URL, filepath)
        self.model = BaselineLightningModule.load_from_checkpoint(filepath).eval().to(device)

    def score(self, wavs: torch.tensor) -> torch.tensor:
        """
        Args:
            wavs: audio waveform to be evaluated. When len(wavs) == 1 or 2,
                the model processes the input as a single audio clip. The model
                performs batch processing when len(wavs) == 3.
        """
        if len(wavs.shape) == 1:
            out_wavs = wavs.unsqueeze(0).unsqueeze(0)
        elif len(wavs.shape) == 2:
            out_wavs = wavs.unsqueeze(0)
        elif len(wavs.shape) == 3:
            out_wavs = wavs
        else:
            raise ValueError("Dimension of input tensor needs to be <= 3.")
        bs = out_wavs.shape[0]
        batch = {
            "wav": out_wavs,
            "domains": torch.zeros(bs, dtype=torch.int).to(self.device),
            "judge_id": torch.ones(bs, dtype=torch.int).to(self.device) * 288,
        }
        with torch.no_grad():
            output = self.model(batch)

        return output.mean(dim=1).squeeze(1).cpu().detach() * 2 + 3


def download_file(url, filename):
    """
    Downloads a file from the given URL

    Args:
        url (str): The URL of the file to download.
        filename (str): The name to save the file as.
    """
    print(f"Downloading file {filename}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size_in_bytes = int(response.headers.get("content-length", 0))
    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)

    with open(filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            progress_bar.update(len(chunk))
            f.write(chunk)

    progress_bar.close()


def load_ssl_model(ckpt_path="wav2vec_small.pt"):
    filepath = os.path.join(os.path.dirname(__file__), ckpt_path)
    if not os.path.exists(filepath):
        download_file(WAV2VEC_URL, filepath)
    SSL_OUT_DIM = 768
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([filepath])
    ssl_model = model[0]
    ssl_model.remove_pretraining_modules()
    return SSL_model(ssl_model, SSL_OUT_DIM)


class BaselineLightningModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.construct_model()
        self.save_hyperparameters()

    def construct_model(self):
        self.feature_extractors = nn.ModuleList(
            [load_ssl_model(ckpt_path="wav2vec_small.pt"), DomainEmbedding(3, 128),]
        )
        output_dim = sum([feature_extractor.get_output_dim() for feature_extractor in self.feature_extractors])
        output_layers = [LDConditioner(judge_dim=128, num_judges=3000, input_dim=output_dim)]
        output_dim = output_layers[-1].get_output_dim()
        output_layers.append(
            Projection(hidden_dim=2048, activation=torch.nn.ReLU(), range_clipping=False, input_dim=output_dim)
        )

        self.output_layers = nn.ModuleList(output_layers)

    def forward(self, inputs):
        outputs = {}
        for feature_extractor in self.feature_extractors:
            outputs.update(feature_extractor(inputs))
        x = outputs
        for output_layer in self.output_layers:
            x = output_layer(x, inputs)
        return x


class SSL_model(nn.Module):
    def __init__(self, ssl_model, ssl_out_dim) -> None:
        super(SSL_model, self).__init__()
        self.ssl_model, self.ssl_out_dim = ssl_model, ssl_out_dim

    def forward(self, batch):
        wav = batch["wav"]
        wav = wav.squeeze(1)  # [batches, audio_len]
        res = self.ssl_model(wav, mask=False, features_only=True)
        x = res["x"]
        return {"ssl-feature": x}

    def get_output_dim(self):
        return self.ssl_out_dim


class DomainEmbedding(nn.Module):
    def __init__(self, n_domains, domain_dim) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_domains, domain_dim)
        self.output_dim = domain_dim

    def forward(self, batch):
        return {"domain-feature": self.embedding(batch["domains"])}

    def get_output_dim(self):
        return self.output_dim


class LDConditioner(nn.Module):
    """
    Conditions ssl output by listener embedding
    """

    def __init__(self, input_dim, judge_dim, num_judges=None):
        super().__init__()
        self.input_dim = input_dim
        self.judge_dim = judge_dim
        self.num_judges = num_judges
        assert num_judges != None
        self.judge_embedding = nn.Embedding(num_judges, self.judge_dim)
        # concat [self.output_layer, phoneme features]

        self.decoder_rnn = nn.LSTM(
            input_size=self.input_dim + self.judge_dim,
            hidden_size=512,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )  # linear?
        self.out_dim = self.decoder_rnn.hidden_size * 2

    def get_output_dim(self):
        return self.out_dim

    def forward(self, x, batch):
        judge_ids = batch["judge_id"]
        if "phoneme-feature" in x.keys():
            concatenated_feature = torch.cat(
                (x["ssl-feature"], x["phoneme-feature"].unsqueeze(1).expand(-1, x["ssl-feature"].size(1), -1)), dim=2
            )
        else:
            concatenated_feature = x["ssl-feature"]
        if "domain-feature" in x.keys():
            concatenated_feature = torch.cat(
                (concatenated_feature, x["domain-feature"].unsqueeze(1).expand(-1, concatenated_feature.size(1), -1),),
                dim=2,
            )
        if judge_ids != None:
            concatenated_feature = torch.cat(
                (
                    concatenated_feature,
                    self.judge_embedding(judge_ids).unsqueeze(1).expand(-1, concatenated_feature.size(1), -1),
                ),
                dim=2,
            )
            decoder_output, (h, c) = self.decoder_rnn(concatenated_feature)
        return decoder_output


class Projection(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation, range_clipping=False):
        super(Projection, self).__init__()
        self.range_clipping = range_clipping
        output_dim = 1
        if range_clipping:
            self.proj = nn.Tanh()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), activation, nn.Dropout(0.3), nn.Linear(hidden_dim, output_dim),
        )
        self.output_dim = output_dim

    def forward(self, x, batch):
        output = self.net(x)

        # range clipping
        if self.range_clipping:
            return self.proj(output) * 2.0 + 3
        else:
            return output

    def get_output_dim(self):
        return self.output_dim
