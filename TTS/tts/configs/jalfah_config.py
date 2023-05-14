from dataclasses import dataclass, field
from typing import List

from TTS.tts.configs.shared_configs import BaseTTSConfig
from TTS.tts.models.jalfah_tts import JalfahTTSArgs


@dataclass
class JalfahConfig(BaseTTSConfig):
    model: str = "jalfah"
    base_model: str = "jalfah_tts"

    # model specific params
    model_args: JalfahTTSArgs = JalfahTTSArgs(use_pitch=True, use_energy=True)

    # multi-speaker settings
    num_speakers: int = 0
    speakers_file: str = None
    use_speaker_embedding: bool = False
    use_d_vector_file: bool = False
    d_vector_file: str = False
    d_vector_dim: int = 0

    # optimizer parameters
    optimizer: str = "AdamW"
    optimizer_params: dict = field(default_factory=lambda: {"betas": [0.9, 0.998], "weight_decay": 1e-6})
    grad_clip: List[float] = field(default_factory=lambda: [1000, 5.0])

    #generator Lr param
    lr_gen: float = 1e-5 * 2 #we double it
    lr_scheduler_gen: str = "NoamLR"
    lr_scheduler_gen_params: dict = field(default_factory=lambda: {"warmup_steps": 20}) #prev try with 150

    #discriminator Lr param
    lr_disc: float = 0.0002
    lr_scheduler_disc: str = "ExponentialLR"
    lr_scheduler_disc_params: dict = field(default_factory=lambda: {"gamma": 0.999875, "last_epoch": -1})  

    # loss params
    spec_loss_type: str = "mse"
    duration_loss_type: str = "mse"
    use_ssim_loss: bool = False
    pitch_loss_alpha: float = 0.2
    energy_loss_alpha: float = 0.2
    dur_loss_alpha: float = 2.0
    use_mdn_loss: bool = True
    mdn_loss_alpha: float = 2.0
    gen_loss_alpha: float = 1.0 #0.5
    feat_loss_alpha: float = 1.0 #0.5
    mel_loss_alpha: float = 45.0 #15.0
    disc_loss_alpha: float = 1.0

    #VITS dataset config
    return_wav: bool = True #not sure if still useful without datasetconfigs

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
