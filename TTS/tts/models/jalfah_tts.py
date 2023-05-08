from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union

import torch
from coqpit import Coqpit
from torch import nn
from torch.cuda.amp.autocast_mode import autocast

from TTS.tts.layers.feed_forward.decoder import Decoder
from TTS.tts.layers.feed_forward.encoder import Encoder
from TTS.tts.layers.align_tts.mdn import MDNBlock #from align_tts
from TTS.tts.layers.generic.pos_encoding import PositionalEncoding
from TTS.tts.layers.glow_tts.duration_predictor import DurationPredictor as GlowDurationPredictor
from TTS.tts.layers.feed_forward.duration_predictor import DurationPredictor as FFDurationPredictor
from TTS.tts.models.base_tts import BaseTTS
from TTS.tts.utils.helpers import average_over_durations, generate_path, maximum_path, sequence_mask
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.utils.visual import plot_alignment, plot_avg_energy, plot_avg_pitch, plot_spectrogram
from TTS.utils.io import load_fsspec
from TTS.vocoder.models.hifigan_generator import HifiganGenerator #from HigiGan Vocoder
from TTS.tts.layers.vits.discriminator import VitsDiscriminator #from VITS
from TTS.tts.utils.helpers import rand_segments, segment #from VITS
from librosa.filters import mel as librosa_mel_fn #from VITS
from trainer.trainer_utils import get_optimizer, get_scheduler #from VITS
from itertools import chain #from VITS
import torchaudio #from VITS
from TTS.vocoder.utils.generic_utils import plot_results #from VITS


##############################
# IO / Feature extraction
##############################

# pylint: disable=global-statement
hann_window = {}
mel_basis = {}

def _amp_to_db(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def amp_to_db(magnitudes):
    output = _amp_to_db(magnitudes)
    return output

def wav_to_mel(y, n_fft, num_mels, sample_rate, hop_length, win_length, fmin, fmax, center=False):
    """
    Args Shapes:
        - y : :math:`[B, 1, T]`

    Return Shapes:
        - spec : :math:`[B,C,T]`
    """
    y = y.squeeze(1)

    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global mel_basis, hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    fmax_dtype_device = str(fmax) + "_" + dtype_device
    wnsize_dtype_device = str(win_length) + "_" + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sample_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=y.dtype, device=y.device)
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_length).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    spec = amp_to_db(spec)
    return spec

def wav_to_spec(y, n_fft, hop_length, win_length, center=False):
    """
    Args Shapes:
        - y : :math:`[B, 1, T]`

    Return Shapes:
        - spec : :math:`[B,C,T]`
    """
    y = y.squeeze(1)
    if torch.isnan(y).any():
        print("y is nan")
        y=torch.nan_to_num(y)

    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    wnsize_dtype_device = str(win_length) + "_" + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_length).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)
    if torch.isnan(y).any():
        print("padded y is nan")
        y=torch.nan_to_num(y)    

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )
    if torch.isnan(spec).any():
        print("spec is nan")
        spec=torch.nan_to_num(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec

def spec_to_mel(spec, n_fft, num_mels, sample_rate, fmin, fmax):
    """
    Args Shapes:
        - spec : :math:`[B,C,T]`

    Return Shapes:
        - mel : :math:`[B,C,T]`
    """
    global mel_basis
    dtype_device = str(spec.dtype) + "_" + str(spec.device)
    fmax_dtype_device = str(fmax) + "_" + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sample_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=spec.dtype, device=spec.device)
    mel = torch.matmul(mel_basis[fmax_dtype_device], spec)
    mel = amp_to_db(mel)
    return mel



@dataclass
class JalfahTTSArgs(Coqpit):

    #for wav generator only
    hifigan_input: int = 80

    num_chars: int = None
    out_channels: int = 80
    hidden_channels: int = 256
    use_aligner: bool = True
    # pitch params
    use_pitch: bool = True
    pitch_predictor_hidden_channels: int = 256
    pitch_predictor_kernel_size: int = 3
    pitch_predictor_dropout_p: float = 0.1
    pitch_embedding_kernel_size: int = 3

    # energy params
    use_energy: bool = False
    energy_predictor_hidden_channels: int = 256
    energy_predictor_kernel_size: int = 3
    energy_predictor_dropout_p: float = 0.1
    energy_embedding_kernel_size: int = 3

    # duration params
    duration_predictor_hidden_channels: int = 256
    duration_predictor_kernel_size: int = 3
    duration_predictor_dropout_p: float = 0.1

    positional_encoding: bool = True
    poisitonal_encoding_use_scale: bool = True
    length_scale: int = 1
    encoder_type: str = "fftransformer"
    encoder_params: dict = field(
        default_factory=lambda: {"hidden_channels_ffn": 1024, "num_heads": 2, "num_layers": 6, "dropout_p": 0.1}
    )
    decoder_type: str = "fftransformer"
    decoder_params: dict = field(
        default_factory=lambda: {"hidden_channels_ffn": 1024, "num_heads": 2, "num_layers": 6, "dropout_p": 0.1}
    )
    detach_duration_predictor: bool = True
    max_duration: int = 75
    num_speakers: int = 1
    use_speaker_embedding: bool = False
    speakers_file: str = None
    use_d_vector_file: bool = False
    d_vector_dim: int = None
    d_vector_file: str = None

    #ssim loss?
    use_ssim: bool = False

    #VITS param
    spec_segment_size: int = 32
    resblock_type_decoder: str = "1"
    resblock_kernel_sizes_decoder: List[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes_decoder: List[List[int]] = field(default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]])
    upsample_kernel_sizes_decoder: List[int] = field(default_factory=lambda: [16, 16, 4, 4])
    upsample_initial_channel_decoder: int = 512
    upsample_rates_decoder: List[int] = field(default_factory=lambda: [8, 8, 2, 2])
    init_discriminator: bool = True
    periods_multi_period_discriminator: List[int] = field(default_factory=lambda: [2, 3, 5, 7, 11])
    use_spectral_norm_disriminator: bool = False
    encoder_sample_rate: int = None


class JalfahTTS(BaseTTS):

    # pylint: disable=dangerous-default-value
    def __init__(
        self,
        config: Coqpit,
        ap: "AudioProcessor" = None,
        tokenizer: "TTSTokenizer" = None,
        speaker_manager: SpeakerManager = None,
    ):
        super().__init__(config, ap, tokenizer, speaker_manager)
        self._set_model_args(config)

        self.init_multispeaker(config)
        #self.init_upsampling()

        self.max_duration = self.args.max_duration
        self.use_aligner = self.args.use_aligner
        self.use_pitch = self.args.use_pitch
        self.use_energy = self.args.use_energy
        self.length_scale = (
            float(self.args.length_scale) if isinstance(self.args.length_scale, int) else self.args.length_scale
        )

        self.emb = nn.Embedding(self.args.num_chars, self.args.hidden_channels)

        self.encoder = Encoder(
            self.args.hidden_channels,
            self.args.hidden_channels,
            self.args.encoder_type,
            self.args.encoder_params,
            self.embedded_speaker_dim,
        )

        if self.args.positional_encoding:
            self.pos_encoder = PositionalEncoding(self.args.hidden_channels)

        self.mel_decoder = Decoder(
            self.args.out_channels,
            self.args.hidden_channels,
            self.args.decoder_type,
            self.args.decoder_params,
        )

        #from VITS
        self.waveform_decoder = HifiganGenerator(
            self.args.hifigan_input,
            1,
            self.args.resblock_type_decoder,
            self.args.resblock_dilation_sizes_decoder,
            self.args.resblock_kernel_sizes_decoder,
            self.args.upsample_kernel_sizes_decoder,
            self.args.upsample_initial_channel_decoder,
            self.args.upsample_rates_decoder,
            inference_padding=0,
            cond_channels=self.embedded_speaker_dim,
            conv_pre_weight_norm=False,
            conv_post_weight_norm=False,
            conv_post_bias=False,
        )

        self.duration_predictor = FFDurationPredictor(
            config.model_args.duration_predictor_hidden_channels
            )
       
        if self.args.use_pitch:
            self.pitch_predictor = GlowDurationPredictor(
                self.args.hidden_channels + self.embedded_speaker_dim,
                self.args.pitch_predictor_hidden_channels,
                self.args.pitch_predictor_kernel_size,
                self.args.pitch_predictor_dropout_p,
            )
            self.pitch_emb = nn.Conv1d(
                1,
                self.args.hidden_channels,
                kernel_size=self.args.pitch_embedding_kernel_size,
                padding=int((self.args.pitch_embedding_kernel_size - 1) / 2),
            )

        if self.args.use_energy:
            self.energy_predictor = GlowDurationPredictor(
                self.args.hidden_channels + self.embedded_speaker_dim,
                self.args.energy_predictor_hidden_channels,
                self.args.energy_predictor_kernel_size,
                self.args.energy_predictor_dropout_p,
            )
            self.energy_emb = nn.Conv1d(
                1,
                self.args.hidden_channels,
                kernel_size=self.args.energy_embedding_kernel_size,
                padding=int((self.args.energy_embedding_kernel_size - 1) / 2),
            )

        if self.args.use_aligner:
            self.mdn_block = MDNBlock(
                config.model_args.hidden_channels, 
                2 * config.model_args.out_channels
            )
        
        #from VITS
        if self.args.init_discriminator:
            self.disc = VitsDiscriminator(
                periods=self.args.periods_multi_period_discriminator,
                use_spectral_norm=self.args.use_spectral_norm_disriminator,
            )

        #from VITS
        self.spec_segment_size=self.args.spec_segment_size

    def init_multispeaker(self, config: Coqpit):
        """Init for multi-speaker training.

        Args:
            config (Coqpit): Model configuration.
        """
        self.embedded_speaker_dim = 0
        # init speaker manager
        if self.speaker_manager is None and (config.use_d_vector_file or config.use_speaker_embedding):
            raise ValueError(
                " > SpeakerManager is not provided. You must provide the SpeakerManager before initializing a multi-speaker model."
            )
        # set number of speakers
        if self.speaker_manager is not None:
            self.num_speakers = self.speaker_manager.num_speakers
        # init d-vector embedding
        if config.use_d_vector_file:
            self.embedded_speaker_dim = config.d_vector_dim
            if self.args.d_vector_dim != self.args.hidden_channels:
                self.proj_g = nn.Conv1d(self.args.d_vector_dim, self.args.hidden_channels, 1)
        # init speaker embedding layer
        if config.use_speaker_embedding and not config.use_d_vector_file:
            print(" > Init speaker_embedding layer.")
            self.emb_g = nn.Embedding(self.num_speakers, self.args.hidden_channels)
            nn.init.uniform_(self.emb_g.weight, -0.1, 0.1)

    def init_upsampling(self):
        """
        Initialize upsampling modules of a model.
        """
        if self.args.encoder_sample_rate:
            self.interpolate_factor = self.config.audio["sample_rate"] / self.args.encoder_sample_rate
            self.audio_resampler = torchaudio.transforms.Resample(
                orig_freq=self.config.audio["sample_rate"], new_freq=self.args.encoder_sample_rate
            )  # pylint: disable=W0201

    @staticmethod
    def generate_attn(dr, x_mask, y_mask=None):
        """Generate an attention mask from the durations.

        Shapes
           - dr: :math:`(B, T_{en})`
           - x_mask: :math:`(B, T_{en})`
           - y_mask: :math:`(B, T_{de})`
        """
        # compute decode mask from the durations
        if y_mask is None:
            y_lengths = dr.sum(1).long()
            y_lengths[y_lengths < 1] = 1
            y_mask = torch.unsqueeze(sequence_mask(y_lengths, None), 1).to(dr.dtype)
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)
        attn = generate_path(dr, attn_mask.squeeze(1)).to(dr.dtype)
        return attn

    def expand_encoder_outputs(self, en, dr, x_mask, y_mask):
        """Generate attention alignment map from durations and
        expand encoder outputs

        Shapes:
            - en: :math:`(B, D_{en}, T_{en})`
            - dr: :math:`(B, T_{en})`
            - x_mask: :math:`(B, T_{en})`
            - y_mask: :math:`(B, T_{de})`

        Examples::

            encoder output: [a,b,c,d]
            durations: [1, 3, 2, 1]

            expanded: [a, b, b, b, c, c, d]
            attention map: [[0, 0, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 1, 1, 0],
                            [0, 1, 1, 1, 0, 0, 0],
                            [1, 0, 0, 0, 0, 0, 0]]
        """
        attn = self.generate_attn(dr, x_mask, y_mask)
        o_en_ex = torch.matmul(attn.squeeze(1).transpose(1, 2).to(en.dtype), en.transpose(1, 2)).transpose(1, 2)
        return o_en_ex, attn

    def format_durations(self, o_dr_log, x_mask):
        """Format predicted durations.
        1. Convert to linear scale from log scale
        2. Apply the length scale for speed adjustment
        3. Apply masking.
        4. Cast 0 durations to 1.
        5. Round the duration values.

        Args:
            o_dr_log: Log scale durations.
            x_mask: Input text mask.

        Shapes:
            - o_dr_log: :math:`(B, T_{de})`
            - x_mask: :math:`(B, T_{en})`
        """
        #o_dr = (torch.exp(o_dr_log) - 1) * x_mask * self.length_scale
        o_dr = (torch.exp(o_dr_log)) * x_mask * self.length_scale
        o_dr[o_dr < 1] = 1.0
        o_dr = torch.round(o_dr)
        return o_dr

    def _forward_encoder(
        self, x: torch.LongTensor, x_mask: torch.FloatTensor, g: torch.FloatTensor = None
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        # [B, T, C]
        x_emb = self.emb(x)
        # encoder pass
        o_en = self.encoder(torch.transpose(x_emb, 1, -1), x_mask)
        # speaker conditioning
        o_en_nospeaker=o_en
        # TODO: try different ways of conditioning
        if g is not None:
            o_en = o_en + g
        return o_en,o_en_nospeaker, x_mask, x_emb

    def _forward_mel_decoder(
        self,
        o_en: torch.FloatTensor,
        dr: torch.IntTensor,
        x_mask: torch.FloatTensor,
        y_lengths: torch.IntTensor,
        g: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        y_mask = torch.unsqueeze(sequence_mask(y_lengths, None), 1).to(o_en.dtype)
        # expand o_en with durations
        o_en_ex, attn = self.expand_encoder_outputs(o_en, dr, x_mask, y_mask)

        # positional encoding
        if hasattr(self, "pos_encoder"):
            o_en_ex = self.pos_encoder(o_en_ex, y_mask)

        # decoder pass
        o_de = self.mel_decoder(o_en_ex, y_mask, g=g)

        #return o_de.transpose(1, 2), attn.transpose(1, 2)
        return o_de, attn
    
    def _forward_pitch_predictor(
        self,
        o_en: torch.FloatTensor,
        x_mask: torch.IntTensor,
        pitch: torch.FloatTensor = None,
        dr: torch.IntTensor = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        o_pitch = self.pitch_predictor(o_en, x_mask)
        if pitch is not None:
            avg_pitch = average_over_durations(pitch, dr)
            o_pitch_emb = self.pitch_emb(avg_pitch)
            return o_pitch_emb, o_pitch, avg_pitch
        o_pitch_emb = self.pitch_emb(o_pitch)
        return o_pitch_emb, o_pitch

    def _forward_energy_predictor(
        self,
        o_en: torch.FloatTensor,
        x_mask: torch.IntTensor,
        energy: torch.FloatTensor = None,
        dr: torch.IntTensor = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        o_energy = self.energy_predictor(o_en, x_mask)
        if energy is not None:
            avg_energy = average_over_durations(energy, dr)
            o_energy_emb = self.energy_emb(avg_energy)
            return o_energy_emb, o_energy, avg_energy
        o_energy_emb = self.energy_emb(o_energy)
        return o_energy_emb, o_energy

    @staticmethod
    def compute_log_probs(mu, log_sigma, y):
        # pylint: disable=protected-access, c-extension-no-member
        y = y.transpose(1, 2).unsqueeze(1)  # [B, 1, T1, D]
        mu = mu.transpose(1, 2).unsqueeze(2)  # [B, T2, 1, D]
        log_sigma = log_sigma.transpose(1, 2).unsqueeze(2)  # [B, T2, 1, D]
        expanded_y, expanded_mu = torch.broadcast_tensors(y, mu)
        exponential = -0.5 * torch.mean(
            torch._C._nn.mse_loss(expanded_y, expanded_mu, 0) / torch.pow(log_sigma.exp(), 2), dim=-1
        )  # B, L, T
        logp = exponential - 0.5 * log_sigma.mean(dim=-1)
        return logp

    def compute_align_path(self, mu, log_sigma, y, x_mask, y_mask):
        # find the max alignment path
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)
        log_p = self.compute_log_probs(mu, log_sigma, y)
        # [B, T_en, T_dec]
        attn = maximum_path(log_p, attn_mask.squeeze(1)).unsqueeze(1)
        dr_mas = torch.sum(attn, -1)
        return dr_mas.squeeze(1), log_p

    def _forward_mdn(self, o_en, y, y_lengths, x_mask):
        # MAS potentials and alignment
        mu, log_sigma = self.mdn_block(o_en)
        y_mask = torch.unsqueeze(sequence_mask(y_lengths, None), 1).to(o_en.dtype)
        dr_mas, logp = self.compute_align_path(mu, log_sigma, y, x_mask, y_mask)
        return dr_mas, mu, log_sigma, logp

    def _set_speaker_input(self, aux_input: Dict):
        d_vectors = aux_input.get("d_vectors", None)
        speaker_ids = aux_input.get("speaker_ids", None)

        if d_vectors is not None and speaker_ids is not None:
            raise ValueError("[!] Cannot use d-vectors and speaker-ids together.")

        if speaker_ids is not None and not hasattr(self, "emb_g"):
            raise ValueError("[!] Cannot use speaker-ids without enabling speaker embedding.")

        g = speaker_ids if speaker_ids is not None else d_vectors
        return g

    def _forward_waveform_decoder(
        self,
        z: torch.FloatTensor,
        y_lengths: torch.LongTensor,
        waveform: torch.Tensor,
        g: torch.tensor = None
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        
        # select a random feature segment for the waveform decoder
        z_slice, slice_ids = rand_segments(z, y_lengths, self.spec_segment_size, let_short_samples=True, pad_short=True)

        # interpolate z if needed
        z_slice, spec_segment_size, slice_ids, _ = self.upsampling_z(z_slice, slice_ids=slice_ids)

        o = self.waveform_decoder(z_slice, g=g)

        wav_seg = segment(
            waveform,
            slice_ids * self.config.audio.hop_length,
            spec_segment_size * self.config.audio.hop_length,
            pad_short=True,
        )

        return o, wav_seg, slice_ids

    def upsampling_z(self, z, slice_ids=None, y_lengths=None, y_mask=None):
        spec_segment_size = self.spec_segment_size
        if self.args.encoder_sample_rate:
            # recompute the slices and spec_segment_size if needed
            slice_ids = slice_ids * int(self.interpolate_factor) if slice_ids is not None else slice_ids
            spec_segment_size = spec_segment_size * int(self.interpolate_factor)
            # interpolate z if needed
            if self.args.interpolate_z:
                z = torch.nn.functional.interpolate(z, scale_factor=[self.interpolate_factor], mode="linear").squeeze(0)
                # recompute the mask if needed
                if y_lengths is not None and y_mask is not None:
                    y_mask = (
                        sequence_mask(y_lengths * self.interpolate_factor, None).to(y_mask.dtype).unsqueeze(1)
                    )  # [B, 1, T_dec_resampled]

        return z, spec_segment_size, slice_ids, y_mask

    def _forward_speaker_embedding(self, aux_input: Dict):
        g = self._set_speaker_input(aux_input)
        if hasattr(self, "emb_g"):
            g = self.emb_g(g)  # [B, C, 1]
        if g is not None:
            g = g.unsqueeze(-1) #[B,C]
        return g

    def forward(
        self,
        x: torch.LongTensor,
        x_lengths: torch.LongTensor,
        waveform: torch.tensor, #for VITS
        y_lengths: torch.LongTensor,
        y: torch.FloatTensor = None,
        pitch: torch.FloatTensor = None,
        energy: torch.FloatTensor = None,
        aux_input: Dict = {"d_vectors": None, "speaker_ids": None},  # pylint: disable=unused-argument
    ) -> Dict:

        #we proper compute the embedding of g 
        g=self._forward_speaker_embedding(aux_input)

        # compute sequence masks
        y_mask = torch.unsqueeze(sequence_mask(y_lengths, None), 1).float()
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.shape[1]), 1).float()

        # encoder pass
        o_en,o_en_nospeaker, x_mask, _ = self._forward_encoder(x, x_mask, g)

        # duration predictor pass
        if self.args.detach_duration_predictor:
            o_dr_log = self.duration_predictor(o_en.detach(), x_mask)
        else:
            o_dr_log = self.duration_predictor(o_en, x_mask)

        #o_dr = torch.clamp(torch.exp(o_dr_log) - 1, 0, self.max_duration)
        o_dr = torch.clamp(torch.exp(o_dr_log), 0, self.max_duration)

        # generate attn mask from predicted durations

        o_attn = self.generate_attn(o_dr.squeeze(1), x_mask)

        # aligner
        if self.use_aligner:
            dr_mas, _, _, logp = self._forward_mdn(o_en_nospeaker, y.transpose(1, 2), y_lengths, x_mask)
            dr_mas_log = torch.log(dr_mas + 1).squeeze(1)

        # pitch predictor pass
        o_pitch = None
        avg_pitch = None
        if self.args.use_pitch:
            o_pitch_emb, o_pitch, avg_pitch = self._forward_pitch_predictor(o_en, x_mask, pitch, dr_mas)

        # energy predictor pass
        o_energy = None
        avg_energy = None
        if self.args.use_energy:
            o_energy_emb, o_energy, avg_energy = self._forward_energy_predictor(o_en, x_mask, energy, dr_mas)

        #add pitch and/or energy embedding to o_en
        if self.args.use_pitch : o_en=o_en+o_pitch_emb
        if self.args.use_energy : o_en=o_en+o_energy_emb

        # mel decoder pass
        o_de, attn = self._forward_mel_decoder(
            o_en, dr_mas, x_mask, y_lengths, g=None
        )

        # waveform decoder pass
        o_wav,wav_seg, slice_ids=self._forward_waveform_decoder(
            z = o_de,
            y_lengths = y_lengths,
            waveform = waveform,
            g=g,
        )

        outputs = {
            #"model_outputs": o_de,  # [B, T, C]
            "durations_log": o_dr_log.squeeze(1),  # [B, T]
            "durations": o_dr.squeeze(1),  # [B, T]
            "attn_durations": o_attn,  # for visualization [B, T_en, T_de']
            "pitch_avg": o_pitch,
            "pitch_avg_gt": avg_pitch,
            "energy_avg": o_energy,
            "energy_avg_gt": avg_energy,
            "alignments": attn,  # [B, T_de, T_en]
            "alignment_mas": dr_mas, #alignment_mas,
            "o_alignment_dur": dr_mas,
            "alignment_logprob": logp, #alignment_logprob,
            "x_mask": x_mask,
            "y_mask": y_mask,
            "durations_mas_log": dr_mas_log,
            #addition for hifigan
            "model_outputs": o_wav,
            "waveform_seg": wav_seg,
            "slice_ids": slice_ids
        }
        return outputs

    @torch.no_grad()
    def inference(self, x, aux_input={"d_vectors": None, "speaker_ids": None}):  # pylint: disable=unused-argument
        #we proper compute the embedding of g 
        g=self._forward_speaker_embedding(aux_input)

        x_lengths = torch.tensor(x.shape[1:2]).to(x.device)
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.shape[1]), 1).to(x.dtype).float()
 
        # encoder pass
        o_en, _, x_mask, _ = self._forward_encoder(x, x_mask, g)
  
        # duration predictor pass
        o_dr_log = self.duration_predictor(o_en, x_mask)
        o_dr = self.format_durations(o_dr_log, x_mask).squeeze(1)
        y_lengths = o_dr.sum(1)

        # pitch predictor pass
        o_pitch = None
        if self.args.use_pitch:
            o_pitch_emb, o_pitch = self._forward_pitch_predictor(o_en, x_mask)

        # energy predictor pass
        o_energy = None
        if self.args.use_energy:
            o_energy_emb, o_energy = self._forward_energy_predictor(o_en, x_mask)

        #if pitch of energy was used, we add the results to o_en
        if self.args.use_pitch : o_en=o_en+o_pitch_emb
        if self.args.use_energy : o_en=o_en+o_energy_emb
  
        # mel decoder pass
        o_de, attn = self._forward_mel_decoder(o_en, o_dr, x_mask, y_lengths, g=g)

        #wav decoder pass
        o_wav = self.waveform_decoder(o_de, g=g)
        #o_wav = self.waveform_decoder((z * y_mask)[:, :, : self.max_inference_len], g=g)

        outputs = {
            "model_outputs": o_wav,
            "alignments": attn,
            "pitch": o_pitch,
            "energy": o_energy,
            "durations_log": o_dr_log,
            #"y_mask": y_mask,#from VITS
        }
        return outputs

    def transform_batch(self,batch_waveform):
        wav_lens = [w.shape[1] for w in batch_waveform]
        wav_lens = torch.LongTensor(wav_lens)
        wav_lens_max = torch.max(wav_lens)
        wav_rel_lens = wav_lens / wav_lens_max

        if self.args.encoder_sample_rate:
            wav = self.audio_resampler(batch_waveform)
        else:
            wav = batch_waveform

        # compute spectrograms    
        spec = wav_to_spec(
            wav,
            self.config.audio.fft_size,
            self.config.audio.hop_length,
            self.config.audio.win_length,
            center=False
        )

        if self.args.encoder_sample_rate:
            # recompute spec with high sampling rate to the loss
            spec_mel = wav_to_spec(
                wav, 
                self.config.audio.fft_size,
                self.config.audio.hop_length,
                self.config.audio.win_length,
                center=False
            )
            # remove extra stft frames if needed
            if spec_mel.size(2) > int(spec.size(2) * self.interpolate_factor):
                spec_mel = spec_mel[:, :, : int(spec.size(2) * self.interpolate_factor)]
            else:
                spec = spec[:, :, : int(spec_mel.size(2) / self.interpolate_factor)]
        else:
            spec_mel = spec

        mel= spec_to_mel(
            spec=spec_mel,
            n_fft=self.config.audio.fft_size,
            num_mels=self.config.audio.num_mels,
            sample_rate=self.config.audio.sample_rate,
            fmin=self.config.audio.mel_fmin,
            fmax=self.config.audio.mel_fmax,
        )

        # compute spectrogram frame lengths
        mel_lengths= (mel.shape[2] * wav_rel_lens).int()

        # zero the padding frames
        mel= mel* sequence_mask(mel_lengths.to('cuda')).unsqueeze(1)
        #mel= mel* sequence_mask(mel_lengths).unsqueeze(1)

        return mel.transpose_(1, 2).to('cuda'), mel_lengths.to('cuda')
        #return mel.transpose_(1, 2), mel_lengths

    def train_step(self, batch: dict, criterion: nn.Module, optimizer_idx: int):
        text_input = batch["text_input"]
        text_lengths = batch["text_lengths"]
        mel_input = batch["mel_input"]
        mel_lengths = batch["mel_lengths"]
        pitch = batch["pitch"] if self.args.use_pitch else None
        energy = batch["energy"] if self.args.use_energy else None
        d_vectors = batch["d_vectors"]
        speaker_ids = batch["speaker_ids"]
        durations = batch["durations"]
        aux_input = {"d_vectors": d_vectors, "speaker_ids": speaker_ids}
        #addition for VITS
        waveform=batch["waveform"].transpose_(1, 2)

         #Generator's pass & Discriminator's Loss
        if optimizer_idx==0:
            #mel_input, mel_lengths=self.transform_batch(waveform)

            # forward pass
            outputs = self.forward(
                x=text_input,
                x_lengths=text_lengths,
                waveform=waveform,
                y_lengths=mel_lengths,
                #y_lengths=batch["mel_transformed_lens"],
                y=mel_input,
                #y=batch["mel_transformed"],
                pitch=pitch,
                energy=energy,
                aux_input=aux_input,
            )
           # cache tensors for the generator pass
            self.model_outputs_cache = outputs  # pylint: disable=attribute-defined-outside-init

            # compute scores and features
            scores_disc_fake, _, scores_disc_real, _ = self.disc(
                outputs["model_outputs"].detach(), outputs["waveform_seg"]
            )

            # compute discriminator loss 
            with autocast(enabled=False):  # use float32 for the criterion
                loss_dict = criterion[optimizer_idx](
                    scores_disc_real=scores_disc_real,
                    scores_disc_fake=scores_disc_fake,
                )
            return outputs, loss_dict

        #Discriminator's pass & Generator's Loss
        if optimizer_idx==1:

            # compute melspec segment
            with autocast(enabled=False):
                if self.args.encoder_sample_rate:
                    spec_segment_size = self.spec_segment_size * int(self.interpolate_factor)
                else:
                    spec_segment_size = self.spec_segment_size

                mel_slice = segment(
                    mel_input.transpose(1,2).float(), self.model_outputs_cache["slice_ids"], spec_segment_size, pad_short=True
                )

                mel_slice_hat = wav_to_mel(
                    y=self.model_outputs_cache["model_outputs"].float(),
                    n_fft=self.config.audio.fft_size,
                    sample_rate=self.config.audio.sample_rate,
                    num_mels=self.config.audio.num_mels,
                    hop_length=self.config.audio.hop_length,
                    win_length=self.config.audio.win_length,
                    fmin=self.config.audio.mel_fmin,
                    fmax=self.config.audio.mel_fmax,
                    center=False,
                )

            # compute discriminator scores and features
            scores_disc_fake, feats_disc_fake, _, feats_disc_real = self.disc(
                self.model_outputs_cache["model_outputs"], self.model_outputs_cache["waveform_seg"]
            )

            # use aligner's output as the duration target
            if self.use_aligner:
                durations = self.model_outputs_cache["o_alignment_dur"]
            # use float32 in AMP
            with autocast(enabled=False):
                # compute loss
                loss_dict = criterion[optimizer_idx](
                    dur_output=self.model_outputs_cache["durations_log"],
                    dur_target=self.model_outputs_cache["durations_mas_log"],
                    pitch_output=self.model_outputs_cache["pitch_avg"] if self.use_pitch else None,
                    pitch_target=self.model_outputs_cache["pitch_avg_gt"] if self.use_pitch else None,
                    energy_output=self.model_outputs_cache["energy_avg"] if self.use_energy else None,
                    energy_target=self.model_outputs_cache["energy_avg_gt"] if self.use_energy else None,
                    input_lens=text_lengths,
                    alignment_logprob=self.model_outputs_cache["alignment_logprob"] if self.use_aligner else None,
                    #addition for VITS
                    mel_lens_target=mel_lengths,
                    mel_slice=mel_slice.float(),
                    mel_slice_hat=mel_slice_hat.float(),
                    feats_disc_real=feats_disc_real,
                    feats_disc_fake = feats_disc_fake,
                    scores_disc_fake = scores_disc_fake
                )
                # compute duration error
                durations_pred = self.model_outputs_cache["durations"]
                duration_error = torch.abs(durations - durations_pred).sum() / text_lengths.sum()
                loss_dict["duration_error"] = duration_error

            return self.model_outputs_cache, loss_dict

    def _create_logs(self, batch, outputs, ap):
        #adding VITS logs
        y_hat = outputs[1]["model_outputs"]
        y = outputs[1]["waveform_seg"]
        figures = plot_results(y_hat, y, ap)
        sample_voice = y_hat[0].squeeze(0).detach().cpu().numpy()
        #train_audio = {"audio": sample_voice}

        alignments = outputs[1]["alignments"]
        align_img = alignments[0].data.cpu().numpy().T
        figures.update(
            {
                "alignment": plot_alignment(align_img, output_fig=False),
            }
        )

        # plot pitch figures
        if self.args.use_pitch:
            pitch_avg = abs(outputs[1]["pitch_avg_gt"][0, 0].data.cpu().numpy())
            pitch_avg_hat = abs(outputs[1]["pitch_avg"][0, 0].data.cpu().numpy())
            chars = self.tokenizer.decode(batch["text_input"][0].data.cpu().numpy())
            pitch_figures = {
                "pitch_ground_truth": plot_avg_pitch(pitch_avg, chars, output_fig=False),
                "pitch_avg_predicted": plot_avg_pitch(pitch_avg_hat, chars, output_fig=False),
            }
            figures.update(pitch_figures)

        # plot energy figures
        if self.args.use_energy:
            energy_avg = abs(outputs[1]["energy_avg_gt"][0, 0].data.cpu().numpy())
            energy_avg_hat = abs(outputs[1]["energy_avg"][0, 0].data.cpu().numpy())
            chars = self.tokenizer.decode(batch["text_input"][0].data.cpu().numpy())
            energy_figures = {
                "energy_ground_truth": plot_avg_energy(energy_avg, chars, output_fig=False),
                "energy_avg_predicted": plot_avg_energy(energy_avg_hat, chars, output_fig=False),
            }
            figures.update(energy_figures)

        # plot the attention mask computed from the predicted durations
        if "attn_durations" in outputs:
            alignments_hat = outputs[1]["attn_durations"][0].data.cpu().numpy()
            figures["alignment_hat"] = plot_alignment(alignments_hat.T, output_fig=False)
    
        return figures, {"audio": sample_voice}

    def train_log(
        self, batch: dict, outputs: dict, logger: "Logger", assets: dict, steps: int
    ) -> None:  # pylint: disable=no-self-use
        figures, audios = self._create_logs(batch, outputs, self.ap)
        logger.train_figures(steps, figures)
        logger.train_audios(steps, audios, self.ap.sample_rate)

    def eval_step(self, batch: dict, criterion: nn.Module, optimizer_idx: int):
        return self.train_step(batch, criterion,optimizer_idx)

    def eval_log(self, batch: dict, outputs: dict, logger: "Logger", assets: dict, steps: int) -> None:
        figures, audios = self._create_logs(batch, outputs, self.ap)
        logger.eval_figures(steps, figures)
        logger.eval_audios(steps, audios, self.ap.sample_rate)

    def load_checkpoint(
        self, config, checkpoint_path, eval=False, cache=False
    ):  # pylint: disable=unused-argument, redefined-builtin
        state = load_fsspec(checkpoint_path, map_location=torch.device("cpu"), cache=cache)
        self.load_state_dict(state["model"])
        if eval:
            self.eval()
            assert not self.training

    def get_criterion(self):
        from TTS.tts.layers.losses import ( # pylint: disable=import-outside-toplevel
            VitsDiscriminatorLoss,
            JalfahTTSLoss
        )
        return [VitsDiscriminatorLoss(self.config),JalfahTTSLoss(self.config)]
    
    #from VITS
    def get_lr(self) -> List:
        return [self.config.lr_disc, self.config.lr_gen]
    
    #from VITS
    def get_optimizer(self) -> List:
        optimizer0 = get_optimizer(self.config.optimizer, self.config.optimizer_params, self.config.lr_disc, self.disc)
        
        # select generator parameters
        gen_parameters = chain(params for k, params in self.named_parameters() if not k.startswith("disc."))
        optimizer1 = get_optimizer(
            self.config.optimizer, self.config.optimizer_params, self.config.lr_gen, parameters=gen_parameters
        )

        return [optimizer0, optimizer1]

    #from VITS
    def get_scheduler(self, optimizer) -> List:
        scheduler_D = get_scheduler(self.config.lr_scheduler_disc, self.config.lr_scheduler_disc_params, optimizer[0])
        scheduler_G = get_scheduler(self.config.lr_scheduler_gen, self.config.lr_scheduler_gen_params, optimizer[1])
        return [scheduler_D, scheduler_G]

    @staticmethod
    def init_from_config(config: "JalfahConfig", samples: Union[List[List], List[Dict]] = None):
        from TTS.utils.audio import AudioProcessor

        ap = AudioProcessor.init_from_config(config)
        tokenizer, new_config = TTSTokenizer.init_from_config(config)
        speaker_manager = SpeakerManager.init_from_config(config, samples)
        return JalfahTTS(new_config, ap, tokenizer, speaker_manager)
