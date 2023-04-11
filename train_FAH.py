#!/home/jfrisch/.conda/envs/modularvits python3
import os
from glob import glob

from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseAudioConfig,BaseDatasetConfig
from TTS.tts.configs.fah_config import FahConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.models.fah_tts import FahTTS
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.utils.manage import ModelManager

output_path = os.path.dirname(os.path.abspath(__file__))

dataset_config = [
    BaseDatasetConfig(formatter="mailabs", meta_file_train=None, path="/home/datasets/split_datasets_16khz/mailabs_blizzard/", language="fr_FR"),
    BaseDatasetConfig(formatter="mailabs", meta_file_train=None, path="/home/datasets/split_datasets_16khz/mailabs_comvoice/", language="fr_FR"),
    BaseDatasetConfig(formatter="mailabs", meta_file_train=None, path="/home/datasets/split_datasets_16khz/mailabs_sympflex/", language="fr_FR"),
    BaseDatasetConfig(formatter="mailabs", meta_file_train=None, path="/home/datasets/dataset_ezwa_16khz/", language="fr_FR"),
    BaseDatasetConfig(formatter="mailabs", meta_file_train=None, path="/home/datasets/dataset_bunny_16khz/", language="fr_FR"),
    BaseDatasetConfig(formatter="mailabs", meta_file_train=None, path="/home/datasets/dataset_nadine_16khz/", language="fr_FR"),
    BaseDatasetConfig(formatter="mailabs", meta_file_train=None, path="/home/datasets/dataset_bernard_16khz/", language="fr_FR"),
]

audio_config = BaseAudioConfig(
    sample_rate=16000,
    do_trim_silence=True,
    trim_db=45.0, #60
    signal_norm=False, #False
    mel_fmin=0.0,
    mel_fmax=8000,
    spec_gain=1.0,
    log_func="np.log",
    ref_level_db=20,
    preemphasis=0.0,

    #additions for consistency in AP
    resample=False,
    num_mels = 80,
    frame_shift_ms=None,
    frame_length_ms=None,
    fft_size=1024,
    stft_pad_mode="reflect",
    clip_norm=4.0,
    do_sound_norm=False,
    hop_length=256,
    win_length=1024,

    #leftover params from modularvits and vitspitch
    #pitch_fmax=640,
    #pitch_fmin=1,
    min_level_db=-100,
    power=1.5,
    griffin_lim_iters=60,
    symmetric_norm=True,
    max_norm=4.0

)


config = FahConfig(
    audio=audio_config,
    run_name="fah",
    use_speaker_embedding=True,
    batch_size=32,
    eval_batch_size=32,
    #batch_group_size=0,
    num_loader_workers=8,
    num_eval_loader_workers=8,
    precompute_num_workers=48,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=100000, #1000
    text_cleaner="multilingual_cleaners",
    use_phonemes=True,
    phoneme_language="fr-fr",
    phoneme_cache_path="/home/cache/phonem/16khz_full_espeakng_updated/",
    compute_input_seq_cache=True,
    print_step=100,
    save_step=100000,
    #use_language_weighted_sampler=False,
    print_eval=False,
    mixed_precision=False,
    #min_audio_len=32 * 256 * 4,
    max_audio_len=16000*10,
    min_seq_len=3,
    max_seq_len=500000,
    compute_f0=True,
    f0_cache_path="/home/cache/f0/16khz_mailabs_trim45/",
    compute_energy=True,
    energy_cache_path="/home/cache/energy/16khz_mailabs_trim45/",
    output_path=output_path,
    datasets=dataset_config,
    test_sentences=["Il m'a fallu beaucoup de temps pour d\u00e9velopper une voix, et maintenant que je l'ai, je ne vais pas me taire.","Les sanglots longs des vilons de l'automne, blessent mon coeur d'une langueur monotone. Tout suffocant et blème quand sonne l'heure, je me souviens des jours anciens, et je pleure."]
)

#compute alignments
if not config.model_args.use_aligner:
    print("attention use aligner????")

# force the convertion of the custom characters to a config attribute
config.from_dict(config.to_dict())

# init audio processor
ap = AudioProcessor.init_from_config(config)

# load training samples
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

# init speaker manager for multi-speaker training
# it maps speaker-id to speaker-name in the model and data-loader
speaker_manager = SpeakerManager()
speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")
config.model_args.num_speakers = speaker_manager.num_speakers

#language_manager = LanguageManager(config=config)
#config.model_args.num_languages = language_manager.num_languages

# INITIALIZE THE TOKENIZER
# Tokenizer is used to convert text to sequences of token IDs.
# config is updated with the default characters if not defined in the config.
tokenizer, config = TTSTokenizer.init_from_config(config)

# init model
model = FahTTS(config, ap, tokenizer, speaker_manager)

# init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(),
    config, 
    output_path,
    model=model, 
    train_samples=train_samples, 
    eval_samples=eval_samples
)
trainer.fit()
