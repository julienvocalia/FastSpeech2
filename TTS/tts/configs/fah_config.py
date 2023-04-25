from dataclasses import dataclass, field
from typing import List

from TTS.tts.configs.shared_configs import BaseTTSConfig
from TTS.tts.models.fah_tts import FAHTTSArgs


@dataclass
class FahConfig(BaseTTSConfig):
    """Configure `ForwardTTS` as FastPitch model.

    Example:

        >>> from TTS.tts.configs.fastspeech2_config import FastSpeech2Config
        >>> config = FastSpeech2Config()

    Args:
        model (str):
            Model name used for selecting the right model at initialization. Defaults to `fast_pitch`.

        base_model (str):
            Name of the base model being configured as this model so that 🐸 TTS knows it needs to initiate
            the base model rather than searching for the `model` implementation. Defaults to `forward_tts`.

        model_args (Coqpit):
            Model class arguments. Check `FastPitchArgs` for more details. Defaults to `FastPitchArgs()`.

        data_dep_init_steps (int):
            Number of steps used for computing normalization parameters at the beginning of the training. GlowTTS uses
            Activation Normalization that pre-computes normalization stats at the beginning and use the same values
            for the rest. Defaults to 10.

        speakers_file (str):
            Path to the file containing the list of speakers. Needed at inference for loading matching speaker ids to
            speaker names. Defaults to `None`.

        use_speaker_embedding (bool):
            enable / disable using speaker embeddings for multi-speaker models. If set True, the model is
            in the multi-speaker mode. Defaults to False.

        use_d_vector_file (bool):
            enable /disable using external speaker embeddings in place of the learned embeddings. Defaults to False.

        d_vector_file (str):
            Path to the file including pre-computed speaker embeddings. Defaults to None.

        d_vector_dim (int):
            Dimension of the external speaker embeddings. Defaults to 0.

        optimizer (str):PositionalEncoding
            Initial learning rate. Defaults to `1e-3`.

        grad_clip (float):
            Gradient norm clipping value. Defaults to `5.0`.

        spec_loss_type (str):
            Type of the spectrogram loss. Check `ForwardTTSLoss` for possible values. Defaults to `mse`.

        duration_loss_type (str):
            Type of the duration loss. Check `ForwardTTSLoss` for possible values. Defaults to `mse`.

        use_ssim_loss (bool):
            Enable/disable the use of SSIM (Structural Similarity) loss. Defaults to True.

        wd (float):
            Weight decay coefficient.Fastalignedictor's loss. If set 0, disables the huber loss. Defaults to 1.0.

        spec_loss_alpha (float):
            Weight for the L1 spectrogram loss. If set 0, disables the L1 loss. Defaults to 1.0.

        pitch_loss_alpha (float):
            Weight for the pitch predictor's loss. If set 0, disables the pitch predictor. Defaults to 1.0.

        energy_loss_alpha (float):
            Weight for the energy predictor's loss. If set 0, disables the energy predictor. Defaults to 1.0.

        binary_align_loss_alpha (float):
            Weight for the binary loss. If set 0, disables the binary loss. Defaults to 1.0.

        binary_loss_warmup_epochs (float):
            Number of epochs to gradually increase the binary loss impact. Defaults to 150.

        min_seq_len (int):
            Minimum input sequence length to be used at training.

        max_seq_len (int):
            Maximum input sequence length to be used at    lr_scheduler_disc: str = "ExponentialLR"  # one of the schedulers from https:#pytorch.org/docs/stable/optim.html
    lr_scheduler_disc_params: dict = field(default_factory=lambda: {"gamma": 0.999, "last_epoch": -1})
    scheduler_after_epoch: bool = True
            energy cache path. defaults to None
    """

    model: str = "fah"
    base_model: str = "fah_tts"

    # model specific params
    model_args: FAHTTSArgs = FAHTTSArgs(use_pitch=True, use_energy=True)

    # multi-speaker settings
    num_speakers: int = 0
    speakers_file: str = None
    use_speaker_embedding: bool = False
    use_d_vector_file: bool = False
    d_vector_file: str = False
    d_vector_dim: int = 0

    # optimizer parameters for FastAlign 
    #optimizer: str = "Adam"
    #optimizer_params: dict = field(default_factory=lambda: {"betas": [0.9, 0.998], "weight_decay": 1e-6})
    
    optimizer: str = "AdamW"
    optimizer_params: dict = field(default_factory=lambda: {"betas": [0.8, 0.99], "eps": 1e-9, "weight_decay": 0.01})
    
    
    #grad clip updated to have 5.0 for generator (as per FastAlign initial config) and 1000 for discriminator (as per VITS)
    grad_clip: List[float] = field(default_factory=lambda: [1000, 5.0])

    #Learning Rate parameters for FastAlign and HifiGan Generator, as per vits config
    lr_scheduler_gen: str = "NoamLR"
    #actually warmup_epochs, so its 20*6500= 13k steps instead of 4000 steps
    lr_scheduler_gen_params: dict = field(default_factory=lambda: {"warmup_steps": 20}) 
    lr_gen: float = 1e-5 #1e-4
    #lr_gen: float = 0.0002
    #lr_scheduler_gen: str = "ExponentialLR"
    #lr_scheduler_gen_params: dict = field(default_factory=lambda: {"gamma": 0.999875, "last_epoch": -1})

    #Learning Rate parameters for Hifigan Discriminator, as per vits config
    lr_disc: float = 0.0002
    lr_scheduler_disc: str = "ExponentialLR"
    lr_scheduler_disc_params: dict = field(default_factory=lambda: {"gamma": 0.999875, "last_epoch": -1})


    # loss params for FastALigne
    spec_loss_type: str = "mse"
    duration_loss_type: str = "mse"
    use_ssim_loss: bool = False
    #ssim_loss_alpha: float = 1.0
    #spec_loss_alpha: float = 1.0
    pitch_loss_alpha: float = 0.2 #0.1
    energy_loss_alpha: float = 0.2 #0.1 
    dur_loss_alpha: float = 2.0 #0.1
    use_mdn_loss: bool = True
    mdn_loss_alpha: float = 2.0 #1.0

    # loss params for Vits waveform generator and discriminator
    disc_loss_alpha: float = 1.0
    gen_loss_alpha: float = 0.5 #1.0
    feat_loss_alpha: float = 0.5 #1.0
    mel_loss_alpha: float = 15.0 #45 puis 30.0

    # overrides
    min_seq_len: int = 13
    max_seq_len: int = 200
    r: int = 1  # DO NOT CHANGE

    # dataset configs
    compute_f0: bool = True
    f0_cache_path: str = None

    # dataset configs
    compute_energy: bool = True
    energy_cache_path: str = None

    #VITS dataset config
    return_wav: bool = True

    # testing
    test_sentences: List[str] = field(
        default_factory=lambda: [
            "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
            "Be a voice, not an echo.",
            "I'm sorry Dave. I'm afraid I can't do that.",
            "This cake is great. It's so delicious and moist.",
            "Prior to November 22, 1963.",
        ]
    )

    def __post_init__(self):
        # Pass multi-speaker parameters to the model args as `model.init_multispeaker()` looks for it there.
        if self.num_speakers > 0:
            self.model_args.num_speakers = self.num_speakers

        # speaker embedding settings
        if self.use_speaker_embedding:
            self.model_args.use_speaker_embedding = True
        if self.speakers_file:
            self.model_args.speakers_file = self.speakers_file

        # d-vector settings
        if self.use_d_vector_file:
            self.model_args.use_d_vector_file = True
        if self.d_vector_dim is not None and self.d_vector_dim > 0:
            self.model_args.d_vector_dim = self.d_vector_dim
        if self.d_vector_file:
            self.model_args.d_vector_file = self.d_vector_file
