# Modify 2023 Younghu Park
# Modify 2025 Jaeeun Baik

import math
import torch
import torch.nn as nn
import torchaudio
from cosine import WarmupCosineScheduler
from datamodule.transforms import TextTransform
from espnet.nets.pytorch_backend.distill import DistillLoss

# for testing
from espnet.asr.asr_utils import add_results_to_json, get_model_conf, torch_load
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.lm_interface import dynamic_import_lm
from espnet.nets.pytorch_backend.e2e_asr_transformer_pyh import E2E
from espnet.nets.scorers.length_bonus import LengthBonus
from pytorch_lightning import LightningModule


def compute_word_level_distance(seq1, seq2):
    return torchaudio.functional.edit_distance(
        seq1.lower().split(), seq2.lower().split()
    )

def exponential_decay(epoch, start=0.8, end=0.1, max_epochs=100):
    return end + (start - end) * math.exp(-5 * epoch / max_epochs)

class AVModelModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
#        if self.cfg.data.modality == "audio":
#            self.backbone_args = self.cfg.model.audio_backbone
#        elif self.cfg.data.modality == "video":
#            self.backbone_args = self.cfg.model.visual_backbone

        self.teacher_args = self.cfg.model.audiovisual_teacher
        self.student_args = self.cfg.model.audiovisual_student
        self.text_transform = TextTransform()
        self.token_list = self.text_transform.token_list
        self.teacher_model = E2E(len(self.token_list), self.teacher_args)
        self.student_model = E2E(len(self.token_list), self.student_args)
        # -- initialise
        if self.cfg.teacher_ckpt:
            ckpt = torch.load(
                self.cfg.teacher_ckpt, map_location=lambda storage, loc: storage
            )
            self.teacher_model.load_state_dict(ckpt)
            for param in self.teacher_model.parameters():
                param.requires_grad=False
            student_ckpt = {
                k: v
                for k, v in ckpt.items()
                if k.split('.')[1]=='emformer' and int(k.split('.')[3]) < self.student_args.elayers or k.split('.')[1]=='frontend' or k.startswith('encoder.embed') or k.startswith('aux_encoder.embed')
            }
            self.student_model.load_state_dict(student_ckpt, strict=False)
             # print
        self.distill_loss_criterion = DistillLoss(
            l2_weight=self.student_args.l2_weight,
            l1_weight=self.student_args.l1_weight,
            cos_weight=self.student_args.cos_weight,
            cos_type=self.student_args.cos_type,
        )
        self.audio_ratio = self.student_args.audio_weight
        self.pred_head = nn.Linear(self.student_args.adim, self.teacher_args.adim)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [
                {
                    "name": "model",
                    "params": self.student_model.parameters(),
                    "lr": self.cfg.optimizer.lr,
                },
                {
                    "name": "pred_head",
                    "params": self.pred_head.parameters(),
                    "lr": self.cfg.optimizer.lr,
                }
            ],
            weight_decay=self.cfg.optimizer.weight_decay,
            betas=(0.9, 0.98),
        )
        scheduler = WarmupCosineScheduler(
            optimizer,
            self.cfg.optimizer.warmup_epochs,
            self.cfg.trainer.max_epochs,
            len(self.trainer.datamodule.train_dataloader()),
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def forward(self, sample):
        self.beam_search = get_beam_search_decoder(self.student_model, self.token_list)
        student_feat, _ = self.student_model.encoder(sample.unsqueeze(0).to(self.device), None)
        student_feat = student_feat.squeeze(0)
        nbest_hyps = self.beam_search(student_feat)
        nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
        predicted = add_results_to_json(nbest_hyps, self.token_list)
        predicted = predicted.replace("▁", " ").strip().replace("<eos>", "")
        return predicted

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, step_type="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, step_type="val")

    def test_step(self, sample, sample_idx):
        #print(sample.keys()) # (['video', 'audio', 'target']) # audiovisual
        length_v = sample["video"].size(0)
        length_a = sample["audio"].size(0)
        length_a = torch.div(length_a, 640, rounding_mode="trunc")

        student_feat, _ = self.student_model.encoder(
            sample["video"].unsqueeze(0).to(self.device), None, length_v
        )
        student_aux_feat, _ = self.student_model.aux_encoder(
            sample["audio"].unsqueeze(0).to(self.device), None, length_a
        )


        fus_enc_feat = self.student_model.fusion(torch.cat((student_feat, student_aux_feat), dim=-1))

        fus_enc_feat = fus_enc_feat.squeeze(0)
        nbest_hyps = self.beam_search(fus_enc_feat)
        nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
        predicted = add_results_to_json(nbest_hyps, self.token_list)
        predicted = predicted.replace("▁", " ").strip().replace("<eos>", "")

        token_id = sample["target"]
        actual = self.text_transform.post_process(token_id)
        print('label: ', actual)
        print('decoded: ', predicted)

        self.total_edit_distance += compute_word_level_distance(actual, predicted)
        self.total_length += len(actual.split())
        return

    def _step(self, batch, batch_idx, step_type): # train, val step
        #print(batch.keys()) # dict_keys(['videos', 'video_lengths', 'audios', 'audio_lengths', 'targets', 'target_lengths'])
        #print(batch["video_lengths"][0], batch["audio_lengths"][0]) # video: 105, audio: 67200
        self.teacher_model.eval()
        with torch.no_grad():    
            teacher_feat, _ = self.teacher_model.encoder(batch["videos"], None, batch["video_lengths"])
            teacher_aux, _ = self.teacher_model.aux_encoder(batch["audios"], None, batch["audio_lengths"])
    
        student_feat, _ = self.student_model.encoder(
            batch["videos"], None, batch["video_lengths"]
        )
        student_aux, _ = self.student_model.aux_encoder(
            batch["audios"], None, batch["audio_lengths"]
        )
        
        vid_loss, (vid_mse, vid_l1, vid_cos) = self.distill_loss_criterion(self.pred_head(student_feat), teacher_feat)
        aux_loss, (aux_mse, aux_l1, aux_cos) = self.distill_loss_criterion(self.pred_head(student_aux), teacher_aux)
                
        loss_distill = self.audio_ratio * aux_loss + (1 - self.audio_ratio) * vid_loss
        loss_mse = self.audio_ratio * aux_mse + (1 - self.audio_ratio) * vid_mse
        loss_l1 = self.audio_ratio * aux_l1 + (1- self.audio_ratio) * vid_l1
        loss_cos = self.audio_ratio * aux_cos + (1- self.audio_ratio) * vid_cos
        
        
        loss_avsr, loss_ctc, loss_att, acc = self.student_model(
            batch["videos"], batch["audios"], batch["video_lengths"], batch["audio_lengths"], batch["targets"]
        )
        batch_size = len(batch["videos"])
        loss = loss_avsr * (1 - self.distill_ratio) + loss_distill * self.distill_ratio
        
        if step_type == "train":
            self.log("loss", loss, on_step=True, on_epoch=True, batch_size=batch_size)
            self.log(
                "loss_avsr",
                loss_avsr,
                on_step=True,
                on_epoch=True,
                batch_size=batch_size,
            )
            self.log(
                "loss_ctc",
                loss_ctc,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )
            self.log(
                "loss_att",
                loss_att,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )
            self.log(
                "loss_distill",
                loss_distill,
                on_step=True,
                on_epoch=True,
                batch_size=batch_size,
            )
            self.log(
                "loss_mse",
                loss_mse,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )
            self.log(
                "loss_l1",
                loss_l1,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )
            self.log(
                "loss_cos",
                loss_cos,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )
            self.log(
                "decoder_acc", acc, on_step=True, on_epoch=True, batch_size=batch_size
            )
        else:
            self.log("loss_val", loss, batch_size=batch_size)
            self.log("loss_avsr_val", loss_avsr, batch_size=batch_size)
            self.log("loss_ctc_val", loss_ctc, batch_size=batch_size)
            self.log("loss_att_val", loss_att, batch_size=batch_size)
            self.log("loss_distill_val", loss_distill, batch_size=batch_size)
            self.log("loss_mse_val", loss_mse, batch_size=batch_size)
            self.log("loss_l1_val", loss_l1, batch_size=batch_size)
            self.log("loss_cos_val", loss_cos, batch_size=batch_size)
            self.log("decoder_acc_val", acc, batch_size=batch_size)

        if step_type == "train":
            self.log(
                "monitoring_step", torch.tensor(self.global_step, dtype=torch.float32)
            )

        return loss

    def on_train_epoch_start(self):
        self.distill_ratio = exponential_decay(epoch=self.current_epoch, start=self.cfg.optimizer.distill_start, end=self.cfg.optimizer.distill_end, max_epochs=self.cfg.trainer.max_epochs)
        sampler = self.trainer.train_dataloader.batch_sampler
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(self.current_epoch)
        return super().on_train_epoch_start()

    def on_test_epoch_start(self):
        self.total_length = 0
        self.total_edit_distance = 0
        self.text_transform = TextTransform()
        self.beam_search = get_beam_search_decoder(self.student_model, self.token_list)

    def on_test_epoch_end(self):
        self.log("wer", self.total_edit_distance / self.total_length)


def get_beam_search_decoder(
    model,
    token_list,
    # rnnlm="/home/nas4/user/jaeeun/emformer/kd_avsr/LRS3/lm_en_subword/model.pth",
    # rnnlm_conf="/home/nas4/user/jaeeun/emformer/kd_avsr/LRS3/lm_en_subword/model.json",
    rnnlm=None,
    rnnlm_conf=None,
    penalty=1,
    ctc_weight=0.5,
    lm_weight=0.0,
    beam_size=40,
):
    sos = model.odim - 1
    eos = model.odim - 1
    scorers = model.scorers()

    if not rnnlm:
        lm = None
    else:
        lm_args = get_model_conf(rnnlm, rnnlm_conf)
        lm_model_module = getattr(lm_args, "model_module", "default")
        lm_class = dynamic_import_lm(lm_model_module, lm_args.backend)
        lm = lm_class(len(token_list), lm_args)
        torch_load(rnnlm, lm)
        lm.eval()

    scorers["lm"] = lm
    scorers["length_bonus"] = LengthBonus(len(token_list))
    weights = {
        "decoder": 1.0 - ctc_weight,
        "ctc": ctc_weight,
        "lm": lm_weight,
        "length_bonus": penalty,
    }

    return BatchBeamSearch(
        beam_size=beam_size,
        vocab_size=len(token_list),
        weights=weights,
        scorers=scorers,
        sos=sos,
        eos=eos,
        token_list=token_list,
        pre_beam_score_key=None if ctc_weight == 1.0 else "decoder",
    )

