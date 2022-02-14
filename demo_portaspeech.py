import torch
from modules.tts.portaspeech.portaspeech_flow import PortaSpeechFlow
from tasks.tts.fs import FastSpeechTask
from tasks.tts.ps import PortaSpeechTask
from utils.audio.pitch.utils import denorm_f0
from utils.commons.hparams import hparams


class PortaSpeechFlowTask(PortaSpeechTask):
    def __init__(self):
        super().__init__()
        self.training_post_glow = False

    def build_tts_model(self):
        ph_dict_size = len(self.token_encoder)
        word_dict_size = len(self.word_encoder)
        self.model = PortaSpeechFlow(ph_dict_size, word_dict_size, hparams)