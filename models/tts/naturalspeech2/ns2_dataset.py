# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
import torch
from torch.nn.utils.rnn import pad_sequence
from utils.data_utils import *
from tqdm import tqdm
import pickle
from models.tts.gpt_tts.g2p_old_en import process, PHPONE2ID
from g2p_en import G2p
import librosa


class NS2Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg, dataset, is_valid=False):
        assert isinstance(dataset, str)

        self.cfg = cfg

        # the path of the processed data
        processed_data_dir = os.path.join(cfg.preprocess.processed_dir, dataset)
        # the name of the meta file, for example: "valid.json" and "train.json"
        meta_file = cfg.preprocess.valid_file if is_valid else cfg.preprocess.train_file
        # the path of the meta file
        self.metafile_path = os.path.join(processed_data_dir, meta_file)

        # the metadata of your data, which is a list of dict
        # for example: [{"Uid": "61-70968-0060", "num_frames": 160000, "phone_id": ..., "path": ..., "duration": ..., "text": ...}]
        # uid is the unique identifier of the speech (e.g. the file name of the speech),
        # num_frames is the number of frames of the speech,
        # phone_id is the phone id of the speech,
        # duration is the duration of the phone,
        # path is the path of the speech
        # you can change the content of the metadata according to your data
        self.metadata = self.get_metadata()

        # the sorted list of speech index according to the number of frames, which is used for bucketing
        self.all_num_frames = []
        for i in range(len(self.metadata)):
            self.all_num_frames.append(self.metadata[i]["num_frames"])
        self.num_frame_sorted = np.array(sorted(self.all_num_frames))
        self.num_frame_indices = np.array(
            sorted(
                range(len(self.all_num_frames)), key=lambda k: self.all_num_frames[k]
            )
        )

    def __len__(self):
        return len(self.metadata)

    def get_metadata(self):
        with open(self.metafile_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        print("metadata len: ", len(metadata))

        return metadata

    def __getitem__(self, index):
        utt_info = self.metadata[index]

        single_feature = dict()

        # load speech
        speech = librosa.load(utt_info["path"], sr=self.cfg.preprocess.sample_rate)[0]
        # load phone id
        phone_id = utt_info["phone_id"]
        # load duration
        duration = utt_info["duration"]

        # align length
        speech, duration = self.align_length(speech, duration)

        # get target and reference
        speech, ref_speech, duration, phone_id = self.get_target_and_reference(
            speech, duration, phone_id
        )

        single_feature.update(
            {
                "duration": duration,
                "phone_id": phone_id,
                "speech": speech,
                "ref_speech": ref_speech,
            }
        )

        return single_feature

    def get_num_frames(self, index):
        utt_info = self.metadata[index]
        return utt_info["num_frames"] // self.cfg.preprocess.hop_size

    def align_length(self, speech, duration):

        speech_frames = len(speech) // self.cfg.preprocess.hop_size
        dur_sum = sum(duration)
        min_len = min(speech_frames, dur_sum)
        speech = speech[: min_len * self.cfg.preprocess.hop_size]
        if dur_sum > min_len:
            assert (duration[-1] - (dur_sum - min_len)) >= 0
            duration[-1] = duration[-1] - (dur_sum - min_len)
            assert duration[-1] >= 0

        return speech, duration

    def get_target_and_reference(self, speech, duration, phone_id):
        phone_nums = len(phone_id)
        clip_phone_nums = np.random.randint(
            int(phone_nums * 0.1), int(phone_nums * 0.5) + 1
        )
        if duration[0] == 0 and clip_phone_nums == 1:
            start_idx = 1
        else:
            start_idx = 0
        end_idx = start_idx + clip_phone_nums
        start_frames = sum(duration[:start_idx])
        end_frames = sum(duration[:end_idx])

        ref_speech = speech[
            start_frames
            * self.cfg.preprocess.hop_size : end_frames
            * self.cfg.preprocess.hop_size
        ]

        new_speech = np.append(
            speech[: start_frames * self.cfg.preprocess.hop_size],
            speech[end_frames * self.cfg.preprocess.hop_size :],
        )
        new_duration = np.append(duration[:start_idx], duration[end_idx:])
        new_phone_id = np.append(phone_id[:start_idx], phone_id[end_idx:])

        speech = new_speech
        duration = new_duration
        phone_id = new_phone_id

        return speech, ref_speech, duration, phone_id


class NS2Collator(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, batch):
        packed_batch_features = dict()

        # phone_id
        # phone_id_mask
        # speech
        # mask
        # ref_speech
        # ref_mask
        # duration

        for key in batch[0].keys():
            if key == "phone_id":
                phone_id = [torch.LongTensor(b["phone_id"]) for b in batch]
                phone_id_mask = [torch.ones(len(b["phone_id"])) for b in batch]
                packed_batch_features["phone_id"] = pad_sequence(
                    phone_id,
                    batch_first=True,
                    padding_value=0,
                )
                packed_batch_features["phone_id_mask"] = pad_sequence(
                    phone_id_mask,
                    batch_first=True,
                    padding_value=0,
                )
            if key == "speech":
                speech = [torch.FloatTensor(b["speech"]) for b in batch]
                mask = [
                    torch.ones(int(len(b["speech"]) // self.cfg.preprocess.hop_size))
                    for b in batch
                ]
                packed_batch_features["speech"] = pad_sequence(
                    speech,
                    batch_first=True,
                    padding_value=0,
                )
                packed_batch_features["mask"] = pad_sequence(
                    mask,
                    batch_first=True,
                    padding_value=0,
                )
            if key == "ref_speech":
                ref_speech = [torch.FloatTensor(b["ref_speech"]) for b in batch]
                ref_mask = [
                    torch.ones(
                        int(len(b["ref_speech"]) // self.cfg.preprocess.hop_size)
                    )
                    for b in batch
                ]
                packed_batch_features["ref_speech"] = pad_sequence(
                    ref_speech,
                    batch_first=True,
                    padding_value=0,
                )
                packed_batch_features["ref_mask"] = pad_sequence(
                    ref_mask,
                    batch_first=True,
                    padding_value=0,
                )
            if key == "duration":
                duration = [torch.LongTensor(b["duration"]) for b in batch]
                packed_batch_features["duration"] = pad_sequence(
                    duration,
                    batch_first=True,
                    padding_value=0,
                )

        return packed_batch_features


def _is_batch_full(batch, num_tokens, max_tokens, max_sentences):
    if len(batch) == 0:
        return 0
    if len(batch) == max_sentences:
        return 1
    if num_tokens > max_tokens:
        return 1
    return 0


def batch_by_size(
    indices,
    num_tokens_fn,
    max_tokens=None,
    max_sentences=None,
    required_batch_size_multiple=1,
):
    """
    Yield mini-batches of indices bucketed by size. Batches may contain
    sequences of different lengths.

    Args:
        indices (List[int]): ordered list of dataset indices
        num_tokens_fn (callable): function that returns the number of tokens at
            a given index
        max_tokens (int, optional): max number of tokens in each batch
            (default: None).
        max_sentences (int, optional): max number of sentences in each
            batch (default: None).
        required_batch_size_multiple (int, optional): require batch size to
            be a multiple of N (default: 1).
    """
    bsz_mult = required_batch_size_multiple

    sample_len = 0
    sample_lens = []
    batch = []
    batches = []
    for i in range(len(indices)):
        idx = indices[i]
        num_tokens = num_tokens_fn(idx)
        sample_lens.append(num_tokens)
        sample_len = max(sample_len, num_tokens)

        assert (
            sample_len <= max_tokens
        ), "sentence at index {} of size {} exceeds max_tokens " "limit of {}!".format(
            idx, sample_len, max_tokens
        )
        num_tokens = (len(batch) + 1) * sample_len

        if _is_batch_full(batch, num_tokens, max_tokens, max_sentences):
            mod_len = max(
                bsz_mult * (len(batch) // bsz_mult),
                len(batch) % bsz_mult,
            )
            batches.append(batch[:mod_len])
            batch = batch[mod_len:]
            sample_lens = sample_lens[mod_len:]
            sample_len = max(sample_lens) if len(sample_lens) > 0 else 0
        batch.append(idx)
    if len(batch) > 0:
        batches.append(batch)
    return batches
