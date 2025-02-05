import torch
import torch.nn as nn
from torch.utils.data import Dataset



class BilingualDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        """
        Construct a BilingualDataset.

        Args:
            ds (Dataset): The HuggingFace dataset object.
            tokenizer_src (Tokenizer): The tokenizer for source language.
            tokenizer_tgt (Tokenizer): The tokenizer for target language.
            src_lang (str): The source language.
            tgt_lang (str): The target language.
            seq_len (int): The maximum sequence length of the transformer model.

        """
        super().__init__()
        self.seq_len = seq_len

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.Tensor(
            [tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.Tensor(
            [tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.Tensor(
            [tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.ds)

    def __getitem__(self, idx):
        """
        Gets a single sample from the dataset.

        Args:
            idx (int): The index of the sample to get.

        Returns:
            A dictionary containing the following:

            * encoder_input (torch.tensor): The input to the encoder.
            * decoder_input (torch.tensor): The input to the decoder.
            * encoder_mask (torch.tensor): The mask for the encoder input.
            * decoder_mask (torch.tensor): The mask for the decoder input.
            * label/target (torch.tensor): The label of the sample.
            * src_text (str): The source language text.
            * tgt_text (str): The target language text.
        """
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # Transform the text into tokens
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Add sos, eos and padding to each sentence
        enc_num_padding_tokens = self.seq_len - \
            len(enc_input_tokens) - 2  # We will add <s> and </s> 
            # -2 foe SOS and EOS
        # We will only add <s>, and </s> only on the label
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 #SOS

        # Make sure the number of padding tokens is not negative. If it is, the sentence is too long
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")


        # Add <s> and </s> token
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.Tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.Tensor([self.pad_token] *
                             enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only <s> token
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.Tensor(dec_input_tokens, dtype=torch.int64),
                torch.Tensor([self.pad_token] *
                             dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only </s> token
        label = torch.cat(
            [
                torch.Tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.Tensor([self.pad_token] *
                             dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Double check the size of the tensors to make sure they are all seq_len long
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            # (1, 1, seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            # (1, seq_len) & (1, seq_len, seq_len),
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


# to t
def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0
