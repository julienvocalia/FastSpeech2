from dataclasses import dataclass, field
from pickle import FALSE
from typing import Dict, List, Tuple, Union
from itertools import chain #from VITS

import torch
from coqpit import Coqpit
from torch import nn
from torch.cuda.amp.autocast_mode import autocast

#from TTS.tts.layers.feed_forward.decoder import Decoder
from TTS.tts.layers.feed_forward.encoder import Encoder
from TTS.tts.layers.align_tts.mdn import MDNBlock #from align_tts
from TTS.tts.layers.generic.pos_encoding import PositionalEncoding
from TTS.tts.layers.glow_tts.duration_predictor import DurationPredictor as GlowDurationPredictor
from TTS.tts.layers.feed_forward.duration_predictor import DurationPredictor as FFDurationPredictor
from TTS.tts.models.base_tts import BaseTTS
from TTS.tts.utils.helpers import average_over_durations, generate_path, maximum_path, sequence_mask
from TTS.tts.utils.helpers import rand_segments, segment #from VITS
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.utils.visual import plot_alignment, plot_avg_energy, plot_avg_pitch, plot_spectrogram
from TTS.utils.io import load_fsspec
from TTS.tts.layers.vits.discriminator import VitsDiscriminator #from VITS
from TTS.vocoder.models.hifigan_generator import HifiganGenerator #from HigiGan Vocoder
from trainer.trainer_utils import get_optimizer, get_scheduler #from VITS
from TTS.tts.models.vits import wav_to_mel #from VITS
from TTS.vocoder.utils.generic_utils import plot_results #from VITS
from TTS.tts.utils.synthesis import synthesis #from VITS

##########################################
# IO / Feature extraction for VITS part  #
##########################################

# pylint: disable=global-statement
hann_window = {}
mel_basis = {}
from librosa.filters import mel as librosa_mel_fn
def wav_to_spec(y, n_fft, hop_length, win_length, center=False):
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
def amp_to_db(magnitudes):
    output = _amp_to_db(magnitudes)
    return output
def _amp_to_db(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

@dataclass
class FAHTTSArgs(Coqpit):
    """ForwardTTS Model arguments.

    Args:

        num_chars (int):
            Number of characters in the vocabulary. Defaults to 100.

        out_channels (int):
            Number of output channels. Defaults to 80.

        hidden_channels (int):
            Number of base hidden channels of the model. Defaults to 512.

        use_aligner (bool):
            Whether to use aligner network to learn the text to speech alignment or use pre-computed durations.
            If set False, durations should be computed by `TTS/bin/compute_attention_masks.py` and path to the
            pre-computed durations must be provided to `config.datasets[0].meta_file_attn_mask`. Defaults to True.

        use_pitch (bool):
            Use pitch predictor to learn the pitch. Defaults to True.

        use_energy (bool):
            Use energy predictor to learn the energy. Defaults to True.

        duration_predictor_hidden_channels (int):
            Number of hidden channels in the duration predictor. Defaults to 256.

        duration_predictor_dropout_p (float):
            Dropout rate for the duration predictor. Defaults to 0.1.

        duration_predictor_kernel_size (int):
            Kernel size of conv layers in the duration predictor. Defaults to 3.

        pitch_predictor_hidden_channels (int):
            Number of hidden channels in the pitch predictor. Defaults to 256.

        pitch_predictor_dropout_p (float):
            Dropout rate for the pitch predictor. Defaults to 0.1.

        pitch_predictor_kernel_size (int):
            Kernel size of conv layers in the pitch predictor. Defaults to 3.

        pitch_embedding_kernel_size (int):
            Kernel size of the projection layer in the pitch predictor. Defaults to 3.

        energy_predictor_hidden_channels (int):
            Number of hidden channels in the energy predictor. Defaults to 256.

        energy_predictor_dropout_p (float):
            Dropout rate for the energy predictor. Defaults to 0.1.

        energy_predictor_kernel_size (int):
            Kernel size of conv layers in the energy predictor. Defaults to 3.

        energy_embedding_kernel_size (int):
            Kernel size of the projection layer in the energy predictor. Defaults to 3.

        positional_encoding (bool):
            Whether to use positional encoding. Defaults to True.

        positional_encoding_use_scale (bool):
            Whether to use a learnable scale coeff in the positional encoding. Defaults to True.

        length_scale (int):
            Length scale that multiplies the predicted durations. Larger values result slower speech. Defaults to 1.0.

        encoder_type (str):
            Type of the encoder module. One of the encoders available in :class:`TTS.tts.layers.feed_forward.encoder`.
            Defaults to `fftransformer` as in the paper.

        encoder_params (dict):
            Parameters of the encoder module. Defaults to ```{"hidden_channels_ffn": 1024, "num_heads": 1, "num_layers": 6, "dropout_p": 0.1}```

        decoder_type (str):
            Type of the decoder module. One of the decoders available in :class:`TTS.tts.layers.feed_forward.decoder`.
            Defaults to `fftransformer` as in the paper.

        decoder_params (str):
            Parameters of the decoder module. Defaults to ```{"hidden_channels_ffn": 1024, "num_heads": 1, "num_layers": 6, "dropout_p": 0.1}```

        detach_duration_predictor (bool):
            Detach the input to the duration predictor from the earlier computation graph so that the duraiton loss
            does not pass to the earlier layers. Defaults to True.

        max_duration (int):
            Maximum duration accepted by the model. Defaults to 75.

        num_speakers (int):
            Number of speakers for the speaker embedding layer. Defaults to 0.

        speakers_file (str):
            Path to the speaker mapping file for the Speaker Manager. Defaults to None.

        speaker_embedding_channels (int):
            Number of speaker embedding channels. Defaults to 256.

        use_d_vector_file (bool):
            Enable/Disable the use of d-vectors for multi-speaker training. Defaults to False.

        d_vector_dim (int):
            Number of d-vector channels. Defaults to 0.

    """

    num_chars: int = None
    out_channels: int = 80
    hidden_channels: int = 256 #384
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
        default_factory=lambda: {"hidden_channels_ffn": 1024, "num_heads": 1, "num_layers": 6, "dropout_p": 0.1}
    )
    decoder_type: str = "fftransformer"
    decoder_params: dict = field(
        default_factory=lambda: {"hidden_channels_ffn": 1024, "num_heads": 1, "num_layers": 6, "dropout_p": 0.1}
    )
    detach_duration_predictor: bool = True
    max_duration: int = 75
    num_speakers: int = 1
    use_speaker_embedding: bool = False
    speakers_file: str = None
    use_d_vector_file: bool = False
    d_vector_dim: int = None
    d_vector_file: str = None

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



class FahTTS(BaseTTS):
    """General forward TTS model implementation that uses an encoder-decoder architecture with an optional alignment
    network and a pitch predictor.

    If the alignment network is used, the model learns the text-to-speech alignment
    from the data instead of using pre-computed durations.

    If the pitch predictor is used, the model trains a pitch predictor that predicts average pitch value for each
    input character as in the FastPitch model.

    `ForwardTTS` can be configured to one of these architectures,

        - FastPitch
        - SpeedySpeech
        - FastSpeech
        - FastSpeech2 (requires average speech energy predictor)

    Args:
        config (Coqpit): Model coqpit class.
        speaker_manager (SpeakerManager): Speaker manager for multi-speaker training. Only used for multi-speaker models.
            Defaults to None.

    Examples:
        >>> from TTS.tts.models.fast_pitch import ForwardTTS, ForwardTTSArgs
        >>> config = ForwardTTSArgs()
        >>> model = ForwardTTS(config)
    """

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

        self.max_duration = self.args.max_duration
        self.use_aligner = self.args.use_aligner
        self.use_pitch = self.args.use_pitch
        self.use_energy = self.args.use_energy
        self.binary_loss_weight = 0.0

        self.length_scale = (
            float(self.args.length_scale) if isinstance(self.args.length_scale, int) else self.args.length_scale
        )

        #from VITS
        self.spec_segment_size=self.args.spec_segment_size

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

        #self.decoder = Decoder(
        #    self.args.out_channels,
        #    self.args.hidden_channels,
        #    self.args.decoder_type,
        #    self.args.decoder_params,
        #)

        #we replace the mel decoder by a waveform decoder
        self.waveform_decoder = HifiganGenerator(
            self.args.hidden_channels,
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
            #config.model_args.hidden_channels_dp
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
        o_dr = (torch.exp(o_dr_log) - 1) * x_mask * self.length_scale
        o_dr[o_dr < 1] = 1.0
        o_dr = torch.round(o_dr)
        return o_dr

    def _forward_encoder(
        self, x: torch.LongTensor, x_mask: torch.FloatTensor, g: torch.FloatTensor = None
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Encoding forward pass.

        1. Embed speaker IDs if multi-speaker mode.
        2. Embed character sequences.
        3. Run the encoder network.
        4. Sum encoder outputs and speaker embeddings

        Args:
            x (torch.LongTensor): Input sequence IDs.
            x_mask (torch.FloatTensor): Input squence mask.
            g (torch.FloatTensor, optional): Conditioning vectors. In general speaker embeddings. Defaults to None.

        Returns:
            Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
                encoder output, encoder output for the duration predictor, input sequence mask, speaker embeddings,
                character embeddings

        Shapes:
            - x: :math:`(B, T_{en})`
            - x_mask: :math:`(B, 1, T_{en})`
            - g: :math:`(B, C)`
        """
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

    #def _forward_decoder(
    #    self,
    #    o_en: torch.FloatTensor,
    #    dr: torch.IntTensor,
    #    x_mask: torch.FloatTensor,
    #    y_lengths: torch.IntTensor,
    #    g: torch.FloatTensor,
    #) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    #    """Decoding forward pass.
    #
    #    1. Compute the decoder output mask
    #    2. Expand encoder output with the durations.
    #    3. Apply position encoding.
    #    4. Add speaker embeddings if multi-speaker mode.
    #    5. Run the decoder.
    #
    #    Args:
    #        o_en (torch.FloatTensor): Encoder output.
    #        dr (torch.IntTensor): Ground truth durations or alignment network durations.
    #        x_mask (torch.IntTensor): Input sequence mask.
    #        y_lengths (torch.IntTensor): Output sequence lengths.
    #        g (torch.FloatTensor): Conditioning vectors. In general speaker embeddings.
    #
    #    Returns:
    #        Tuple[torch.FloatTensor, torch.FloatTensor]: Decoder output, attention map from durations.
    #    """
    #    y_mask = torch.unsqueeze(sequence_mask(y_lengths, None), 1).to(o_en.dtype)
    #    # expand o_en with durations
    #    o_en_ex, attn = self.expand_encoder_outputs(o_en, dr, x_mask, y_mask)
    #    # positional encoding
    #    if hasattr(self, "pos_encoder"):
    #        o_en_ex = self.pos_encoder(o_en_ex, y_mask)
    #    # decoder pass
    #    o_de = self.decoder(o_en_ex, y_mask, g=g)
    #    return o_de.transpose(1, 2), attn.transpose(1, 2)

    def _forward_pitch_predictor(
        self,
        o_en: torch.FloatTensor,
        x_mask: torch.IntTensor,
        pitch: torch.FloatTensor = None,
        dr: torch.IntTensor = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Pitch predictor forward pass.

        1. Predict pitch from encoder outputs.
        2. In training - Compute average pitch values for each input character from the ground truth pitch values.
        3. Embed average pitch values.

        Args:
            o_en (torch.FloatTensor): Encoder output.
            x_mask (torch.IntTensor): Input sequence mask.
            pitch (torch.FloatTensor, optional): Ground truth pitch values. Defaults to None.
            dr (torch.IntTensor, optional): Ground truth durations. Defaults to None.

        Returns:
            Tuple[torch.FloatTensor, torch.FloatTensor]: Pitch embedding, pitch prediction.

        Shapes:
            - o_en: :math:`(B, C, T_{en})`
            - x_mask: :math:`(B, 1, T_{en})`
            - pitch: :math:`(B, 1, T_{de})`
            - dr: :math:`(B, T_{en})`
        """
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
        """Energy predictor forward pass.

        1. Predict energy from encoder outputs.
        2. In training - Compute average pitch values for each input character from the ground truth pitch values.
        3. Embed average energy values.

        Args:
            o_en (torch.FloatTensor): Encoder output.
            x_mask (torch.IntTensor): Input sequence mask.
            energy (torch.FloatTensor, optional): Ground truth energy values. Defaults to None.
            dr (torch.IntTensor, optional): Ground truth durations. Defaults to None.

        Returns:
            Tuple[torch.FloatTensor, torch.FloatTensor]: Energy embedding, energy prediction.

        Shapes:
            - o_en: :math:`(B, C, T_{en})`
            - x_mask: :math:`(B, 1, T_{en})`
            - pitch: :math:`(B, 1, T_{de})`
            - dr: :math:`(B, T_{en})`
        """
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
    
    def _forward_speaker_embedding(self, aux_input: Dict):
        g = self._set_speaker_input(aux_input)
        if hasattr(self, "emb_g"):
            g = self.emb_g(g)  # [B, C, 1]
        if g is not None:
            g = g.unsqueeze(-1) #[B,C]
        return g
    

    #to fix padding with zeros and log operations
    def _remove_inf(t: torch.Tensor):
        mask=torch.isinf(t)
        t[mask]=0
        return t

    def forward(
        self,
        x: torch.LongTensor,
        x_lengths: torch.LongTensor,
        y_lengths: torch.LongTensor,
        waveform: torch.tensor, #for VITS
        y: torch.FloatTensor = None,
        dr: torch.IntTensor = None,
        pitch: torch.FloatTensor = None,
        energy: torch.FloatTensor = None,
        aux_input: Dict = {"d_vectors": None, "speaker_ids": None},  # pylint: disable=unused-argument
    ) -> Dict:
        """Model's forward pass.

        Args:
            x (torch.LongTensor): Input character sequences.
            x_lengths (torch.LongTensor): Input sequence lengths.
            y_lengths (torch.LongTensor): Output sequnce lengths. Defaults to None.
            y (torch.FloatTensor): Spectrogram frames. Only used when the alignment network is on. Defaults to None.
            dr (torch.IntTensor): Character durations over the spectrogram frames. Only used when the alignment network is off. Defaults to None.
            pitch (torch.FloatTensor): Pitch values for each spectrogram frame. Only used when the pitch predictor is on. Defaults to None.
            energy (torch.FloatTensor): energy values for each spectrogram frame. Only used when the energy predictor is on. Defaults to None.
            aux_input (Dict): Auxiliary model inputs for multi-speaker training. Defaults to `{"d_vectors": 0, "speaker_ids": None}`.

        Shapes:
            - x: :math:`[B, T_max]`
            - x_lengths: :math:`[B]`
            - y_lengths: :math:`[B]`
            - y: :math:`[B, T_max2]`
            - dr: :math:`[B, T_max]`
            - g: :math:`[B, C]`
            - pitch: :math:`[B, 1, T]`
        """
        #we proper compute the embedding of g 
        g=self._forward_speaker_embedding(aux_input)
        g=self._remove_inf(g)



        # compute sequence masks
        y_mask = torch.unsqueeze(sequence_mask(y_lengths, None), 1).float()
        y_mask=self._remove_inf(y_mask)
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.shape[1]), 1).float()
        x_mask=self._remove_inf(x_mask)
        # encoder pass
        o_en,o_en_nospeaker, x_mask, _ = self._forward_encoder(x, x_mask, g)
        o_en=self._remove_inf(o_en)
        o_en_nospeaker=self._remove_inf(o_en_nospeaker)
        x_mask=self._remove_inf(x_mask)
        # duration predictor pass
        if self.args.detach_duration_predictor:
            o_dr_log = self.duration_predictor(o_en.detach(), x_mask)
        else:
            o_dr_log = self.duration_predictor(o_en, x_mask)
        o_dr_log=self._remove_inf(o_dr_log)
        o_dr = torch.clamp(torch.exp(o_dr_log) - 1, 0, self.max_duration)
        o_dr=self._remove_inf(o_dr)
        # generate attn mask from predicted durations
        o_attn = self.generate_attn(o_dr.squeeze(1), x_mask)
        o_attn=self._remove_inf(o_attn)
        # aligner
        o_alignment_dur = None
        alignment_soft = None
        alignment_logprob = None
        alignment_mas = None
        if self.use_aligner:
            dr_mas, mu, log_sigma, logp = self._forward_mdn(o_en_nospeaker, y.transpose(1, 2), y_lengths, x_mask)
            dr_mas=self._remove_inf(dr_mas)
            dr_mas_log = torch.log(dr_mas + 1).squeeze(1)
            dr_mas_log=self._remove_inf(dr_mas_log)
            #o_alignment_dur, alignment_soft, alignment_logprob, alignment_mas = self._forward_aligner(
            #    x_emb, y, x_mask, y_mask
            #)
            #alignment_soft = alignment_soft.transpose(1, 2)
            #alignment_mas = alignment_mas.transpose(1, 2)
            #dr = o_alignment_dur
        # pitch predictor pass
        o_pitch = None
        avg_pitch = None
        if self.args.use_pitch:
            o_pitch_emb, o_pitch, avg_pitch = self._forward_pitch_predictor(o_en, x_mask, pitch, dr_mas)
            #o_en = o_en + o_pitch_emb
        # energy predictor pass
        o_energy = None
        avg_energy = None
        if self.args.use_energy:
            o_energy_emb, o_energy, avg_energy = self._forward_energy_predictor(o_en, x_mask, energy, dr_mas)
            #o_en_var = o_en + o_energy_emb
        #if pitch of energy was used, we add the results to o_en
        if self.args.use_pitch : o_en=o_en+o_pitch_emb
        if self.args.use_energy : o_en=o_en+o_energy_emb

        #we use here what was previously computed in _forward_decoder
        y_mask = torch.unsqueeze(sequence_mask(y_lengths, None), 1).to(o_en.dtype)
        y_mask=self._remove_inf(y_mask)
        # expand o_en with durations
        o_en_ex, attn = self.expand_encoder_outputs(o_en, dr_mas, x_mask, y_mask)
        o_en_ex=self._remove_inf(o_en_ex)
        attn=self._remove_inf(attn)
        # positional encoding
        if hasattr(self, "pos_encoder"):
            o_en_ex = self.pos_encoder(o_en_ex, y_mask)
            o_en_ex=self._remove_inf(o_en_ex)

        # select a random feature segment for the waveform decoder
        z_slice, slice_ids = rand_segments(o_en_ex, y_lengths, self.spec_segment_size, let_short_samples=True, pad_short=True)
        z_slice=self._remove_inf(z_slice)

        #wav decoder pass
        o_wav = self.waveform_decoder(z_slice, g=g)
    
        wav_seg = segment(
            waveform,
            slice_ids * self.config.audio.hop_length,
            self.spec_segment_size * self.config.audio.hop_length,
            pad_short=True,
        )

        # decoder pass
        #o_de, attn = self._forward_decoder(
        #    o_en, dr_mas, x_mask, y_lengths, g=None
        #)  # TODO: maybe pass speaker embedding (g) too
        outputs = {
            #"model_outputs": o_de,  # [B, T, C]
            "model_outputs": o_wav,
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
            "waveform_seg": wav_seg,
            "slice_ids": slice_ids,
        }
        return outputs

    @torch.no_grad()
    def inference(self, x, aux_input={"d_vectors": None, "speaker_ids": None}):  # pylint: disable=unused-argument
        """Model's inference pass.

        Args:
            x (torch.LongTensor): Input character sequence.
            aux_input (Dict): Auxiliary model inputs. Defaults to `{"d_vectors": None, "speaker_ids": None}`.

        Shapes:
            - x: [B, T_max]
            - x_lengths: [B]
            - g: [B, C]
        """
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
            #o_en = o_en + o_pitch_emb
        # energy predictor pass
        o_energy = None
        if self.args.use_energy:
            o_energy_emb, o_energy = self._forward_energy_predictor(o_en, x_mask)
            #o_en = o_en + o_energy_emb
        #if pitch of energy was used, we add the results to o_en
        if self.args.use_pitch : o_en=o_en+o_pitch_emb
        if self.args.use_energy : o_en=o_en+o_energy_emb
        # decoder pass
        #o_de, attn = self._forward_decoder(o_en, o_dr, x_mask, y_lengths, g=None)

        y_mask = torch.unsqueeze(sequence_mask(y_lengths, None), 1).to(o_en.dtype)
        # expand o_en with durations
        o_en_ex, attn = self.expand_encoder_outputs(o_en, o_dr, x_mask, y_mask)
        # positional encoding
        if hasattr(self, "pos_encoder"):
            o_en_ex = self.pos_encoder(o_en_ex, y_mask)

        #wav decoder pass
        o_wav = self.waveform_decoder(o_en_ex, g=g)

        outputs = {
            "model_outputs": o_wav,
            "alignments": attn,
            "pitch": o_pitch,
            "energy": o_energy,
            "durations_log": o_dr_log,
            "y_mask": y_mask,#from VITS
        }
        return outputs

    def train_step(self, batch: dict, criterion: nn.Module, optimizer_idx: int) -> Tuple[Dict, Dict]:
        #Generator's optimizer
        if optimizer_idx==0:
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
            #Addition for VITS
            waveform = batch["waveform"]

            # generator forward pass
            outputs = self.forward(
                x=text_input,
                x_lengths=text_lengths,
                y_lengths=mel_lengths,
                waveform=waveform.transpose_(1, 2),
                y=mel_input,
                dr=durations,
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


        #Discriminator's optimizer
        if optimizer_idx==1:


            if self.args.encoder_sample_rate:
                wav = self.audio_resampler(batch["waveform"])
            else:
                wav = batch["waveform"]

            # compute spectrograms
            batch["spec"] = wav_to_spec(
                wav,
                self.config.audio.fft_size,
                self.config.audio.hop_length,
                self.config.audio.win_length,
                center=False
            )

            if self.args.encoder_sample_rate:
                # recompute spec with high sampling rate to the loss
                spec_mel = wav_to_spec(
                    batch["waveform"], 
                    self.config.audio.fft_size,
                    self.config.audio.hop_length,
                    self.config.audio.win_length,
                    center=False
                )
                # remove extra stft frames if needed
                if spec_mel.size(2) > int(batch["spec"].size(2) * self.interpolate_factor):
                    spec_mel = spec_mel[:, :, : int(batch["spec"].size(2) * self.interpolate_factor)]
                else:
                    batch["spec"] = batch["spec"][:, :, : int(spec_mel.size(2) / self.interpolate_factor)]
            else:
                spec_mel = batch["spec"]


            batch["mel"] = spec_to_mel(
                spec=spec_mel,
                n_fft=self.config.audio.fft_size,
                num_mels=self.config.audio.num_mels,
                sample_rate=self.config.audio.sample_rate,
                fmin=self.config.audio.mel_fmin,
                fmax=self.config.audio.mel_fmax,
            )



            #mel = batch["mel_input"]
            mel=batch["mel"]
            mel_lengths = batch["mel_lengths"]
            text_lengths = batch["text_lengths"]

            # compute melspec segment
            with autocast(enabled=False):
                if self.args.encoder_sample_rate:
                    spec_segment_size = self.spec_segment_size * int(self.interpolate_factor)
                else:
                    spec_segment_size = self.spec_segment_size

                mel_slice = segment(
                    mel.float(), self.model_outputs_cache["slice_ids"], spec_segment_size, pad_short=True
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
                # compute losses
                loss_dict = criterion[optimizer_idx](
                    mel_lens_target=mel_lengths,
                    mel_slice=mel_slice.float(),
                    mel_slice_hat=mel_slice_hat.float(),
                    dur_output=self.model_outputs_cache["durations_log"],
                    dur_target=self.model_outputs_cache["durations_mas_log"],
                    pitch_output=self.model_outputs_cache["pitch_avg"] if self.use_pitch else None,
                    pitch_target=self.model_outputs_cache["pitch_avg_gt"] if self.use_pitch else None,
                    energy_output=self.model_outputs_cache["energy_avg"] if self.use_energy else None,
                    energy_target=self.model_outputs_cache["energy_avg_gt"] if self.use_energy else None,
                    input_lens=text_lengths,
                    alignment_logprob=self.model_outputs_cache["alignment_logprob"] if self.use_aligner else None,
                    feats_disc_real=feats_disc_real,
                    feats_disc_fake = feats_disc_fake,
                    scores_disc_fake = scores_disc_fake,    
                )
                # compute duration error
                durations_pred = self.model_outputs_cache["durations"]
                duration_error = torch.abs(durations - durations_pred).sum() / text_lengths.sum()
                loss_dict["duration_error"] = duration_error

            return self.model_outputs_cache, loss_dict

        raise ValueError(" [!] Unexpected `optimizer_idx`.")

    
    def _create_logs(self, batch, outputs, ap):
        """Create common logger outputs."""
        
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

        #model_outputs = outputs["model_outputs"]
        #alignments = outputs["alignments"]
        #mel_input = batch["mel_input"]

        #pred_spec = model_outputs[0].data.cpu().numpy()
        #gt_spec = mel_input[0].data.cpu().numpy()
        #align_img = alignments[0].data.cpu().numpy()

        #figures.update = {
        #    "prediction": plot_spectrogram(pred_spec, ap, output_fig=False),
        #    "ground_truth": plot_spectrogram(gt_spec, ap, output_fig=False),
        #    "alignment": plot_alignment(align_img, output_fig=False),
        #}

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

        # Sample audio
        #train_audio = ap.inv_melspectrogram(pred_spec.T)
        #return figures, {"audio": train_audio}
    
        return figures, {"audio": sample_voice}

    def train_log(
        self, batch: dict, outputs: dict, logger: "Logger", assets: dict, steps: int
    ) -> None:  # pylint: disable=no-self-use
        figures, audios = self._create_logs(batch, outputs, self.ap)
        logger.train_figures(steps, figures)
        logger.train_audios(steps, audios, self.ap.sample_rate)

    @torch.no_grad()
    def eval_step(self, batch: dict, criterion: nn.Module, optimizer_idx: int):
        return self.train_step(batch, criterion, optimizer_idx)

    def eval_log(self, batch: dict, outputs: dict, logger: "Logger", assets: dict, steps: int) -> None:
        figures, audios = self._create_logs(batch, outputs, self.ap)
        logger.eval_figures(steps, figures)
        logger.eval_audios(steps, audios, self.ap.sample_rate)

    #Addition for VITS
    def get_aux_input_from_test_sentences(self, sentence_info):
        if hasattr(self.config, "model_args"):
            config = self.config.model_args
        else:
            config = self.config

        # extract speaker and language info
        text, speaker_name, style_wav, language_name = None, None, None, None

        if isinstance(sentence_info, list):
            if len(sentence_info) == 1:
                text = sentence_info[0]
            elif len(sentence_info) == 2:
                text, speaker_name = sentence_info
            elif len(sentence_info) == 3:
                text, speaker_name, style_wav = sentence_info
            elif len(sentence_info) == 4:
                text, speaker_name, style_wav, language_name = sentence_info
        else:
            text = sentence_info

        # get speaker  id/d_vector
        speaker_id, d_vector, language_id = None, None, None
        if hasattr(self, "speaker_manager"):
            if config.use_d_vector_file:
                if speaker_name is None:
                    d_vector = self.speaker_manager.get_random_embedding()
                else:
                    d_vector = self.speaker_manager.get_mean_embedding(speaker_name, num_samples=None, randomize=False)
            elif config.use_speaker_embedding:
                if speaker_name is None:
                    speaker_id = self.speaker_manager.get_random_id()
                else:
                    speaker_id = self.speaker_manager.name_to_id[speaker_name]

        # get language id
        if hasattr(self, "language_manager") and hasattr(config, 'use_language_embedding') and config.use_language_embedding and language_name is not None:
            language_id = self.language_manager.name_to_id[language_name]

        return {
            "text": text,
            "speaker_id": speaker_id,
            "style_wav": style_wav,
            "d_vector": d_vector,
            "language_id": language_id,
            "language_name": language_name,
        }
    
    #Addition for VITS
    @torch.no_grad()
    def test_run(self, assets) -> Tuple[Dict, Dict]:
        """Generic test run for `tts` models used by `Trainer`.

        You can override this for a different behaviour.

        Returns:
            Tuple[Dict, Dict]: Test figures and audios to be projected to Tensorboard.
        """
        print(" | > Synthesizing test sentences.")
        test_audios = {}
        test_figures = {}
        test_sentences = self.config.test_sentences
        for idx, s_info in enumerate(test_sentences):
            aux_inputs = self.get_aux_input_from_test_sentences(s_info)
            wav, alignment, _, _ = synthesis(
                self,
                aux_inputs["text"],
                self.config,
                "cuda" in str(next(self.parameters()).device),
                speaker_id=aux_inputs["speaker_id"],
                d_vector=aux_inputs["d_vector"],
                style_wav=aux_inputs["style_wav"],
                language_id=aux_inputs["language_id"],
                use_griffin_lim=True,
                do_trim_silence=False,
            ).values()
            test_audios["{}-audio".format(idx)] = wav
            test_figures["{}-alignment".format(idx)] = plot_alignment(alignment.T, output_fig=False)
        return {"figures": test_figures, "audios": test_audios}

    #from VITS
    def get_optimizer(self) -> List:
        """Initiate and return the GAN optimizers based on the config parameters.
        It returnes 2 optimizers in a list. First one is for the generator and the second one is for the discriminator.
        Returns:
            List: optimizers.
        """
        # select generator parameters
        optimizer0 = get_optimizer(self.config.optimizer, self.config.optimizer_params, self.config.lr_disc, self.disc)

        gen_parameters = chain(params for k, params in self.named_parameters() if not k.startswith("disc."))
        optimizer1 = get_optimizer(
            self.config.optimizer, self.config.optimizer_params, self.config.lr_gen, parameters=gen_parameters
        )
        return [optimizer0, optimizer1]

    #from VITS
    def get_lr(self) -> List:
        """Set the initial learning rates for each optimizer.

        Returns:
            List: learning rates for each optimizer.
        """
        return [self.config.lr_disc, self.config.lr_gen]
    
    #from VITS
    def get_scheduler(self, optimizer) -> List:
        """Set the schedulers for each optimizer.

        Args:
            optimizer (List[`torch.optim.Optimizer`]): List of optimizers.

        Returns:
            List: Schedulers, one for each optimizer.
        """
        scheduler_D = get_scheduler(self.config.lr_scheduler_disc, self.config.lr_scheduler_disc_params, optimizer[0])
        scheduler_G = get_scheduler(self.config.lr_scheduler_gen, self.config.lr_scheduler_gen_params, optimizer[1])
        return [scheduler_D, scheduler_G]
    
    #from VITS
    def get_criterion(self):
        """Get criterions for each optimizer. The index in the output list matches the optimizer idx used in
        `train_step()`"""
        from TTS.tts.layers.losses import (  # pylint: disable=import-outside-toplevel
            VitsDiscriminatorLoss,
            FAHTTSLoss,
        )

        return [VitsDiscriminatorLoss(self.config), FAHTTSLoss(self.config)]

    def load_checkpoint(
        self, config, checkpoint_path, eval=False, cache=False
    ):  # pylint: disable=unused-argument, redefined-builtin
        state = load_fsspec(checkpoint_path, map_location=torch.device("cpu"), cache=cache)
        self.load_state_dict(state["model"])
        if eval:
            self.eval()
            assert not self.training


    @staticmethod
    def init_from_config(config: "FAHConfig", samples: Union[List[List], List[Dict]] = None):
        """Initiate model from config

        Args:
            config (ForwardTTSConfig): Model config.
            samples (Union[List[List], List[Dict]]): Training samples to parse speaker ids for training.
                Defaults to None.
        """
        from TTS.utils.audio import AudioProcessor

        ap = AudioProcessor.init_from_config(config)
        tokenizer, new_config = TTSTokenizer.init_from_config(config)
        speaker_manager = SpeakerManager.init_from_config(config, samples)
        return FahTTS(new_config, ap, tokenizer, speaker_manager)
