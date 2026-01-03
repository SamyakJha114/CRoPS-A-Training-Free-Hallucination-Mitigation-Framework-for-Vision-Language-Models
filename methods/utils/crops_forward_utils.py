import torch
import json
class GetAttentionMaskwithFastVandTextMask:
    def __init__(self,
                attention_mask: torch.Tensor,
                key_position: dict,
                use_fast_v: bool,
                aggregate_layer_fast_v: int,
                minumum_fast_v_tokens: int,
                use_text_mask: bool,
                aggregate_layer_text_mask: int,
                minimum_text_tokens: int,
                ):
        
        self._attention_mask = attention_mask
        
        self._curr_layer_num = 0

        # Fast V
        self._use_fast_v = use_fast_v
        self._aggregate_layer_fast_v = aggregate_layer_fast_v
        # self._minumum_fast_v_tokens = minumum_fast_v_tokens

        # Text Mask
        self._use_text_mask = use_text_mask
        self._aggregate_layer_text_mask = aggregate_layer_text_mask
        self._minimum_text_tokens = 50

        if self._use_fast_v or self._use_text_mask:
            self._image_start = key_position['image_start']
            self._image_token_length = key_position['image_end'] - self._image_start + 1
            # self._minumum_fast_v_tokens = round((0.25)*(self._image_token_length))
            self._minumum_fast_v_tokens = minumum_fast_v_tokens

        if self._use_fast_v:
            assert self._aggregate_layer_fast_v > 0, "aggregate_layer_fast_v must be greater than 0"
        if self._use_text_mask:
            assert self._aggregate_layer_text_mask > 0, "aggregate_layer_text_mask must be greater than 0"
            assert self._minimum_text_tokens > 0, "minimum_text_tokens must be greater than 0"

    def __call__(self, all_self_attns):
        if self._use_fast_v and self._curr_layer_num == self._aggregate_layer_fast_v:
            self._update_fast_v_attention_mask(all_self_attns[-1])
        
        if self._use_text_mask and self._curr_layer_num == self._aggregate_layer_text_mask:
            self._update_text_attention_mask(all_self_attns[-1])

        self._curr_layer_num += 1

        return self._attention_mask

    def _update_fast_v_attention_mask(self, last_layer_attention):
        # compute average attention over different head
        last_layer_attention_avg = torch.mean(last_layer_attention, dim=1)[0]
        # generate new attention mask based on the average attention, 
        # sample the top _minumum_fast_v_tokens tokens with highest attention
        last_layer_attention_avg_last_tok = last_layer_attention_avg[-1]
        # get the attention in image token
        last_layer_attention_avg_last_tok_image = last_layer_attention_avg_last_tok[
            self._image_start: self._image_start+self._image_token_length
        ]
        # get the indexs of the top _minumum_fast_v_tokens tokens
        top_attention_rank_index = last_layer_attention_avg_last_tok_image.topk(self._minumum_fast_v_tokens, largest=False)
        top_attention_rank_index = top_attention_rank_index.indices + self._image_start
        
        # generate fast v attention mask
        fast_v_attention_mask = torch.ones_like(self._attention_mask)
        fast_v_attention_mask[:, self._image_start:self._image_start+self._image_token_length] = False
        fast_v_attention_mask[:, top_attention_rank_index] = True

        self._attention_mask = fast_v_attention_mask
        
    # def _update_fast_v_attention_mask(self, last_layer_attention):
    #     # compute average attention over different head
    #     last_layer_attention_avg = torch.mean(last_layer_attention, dim=1)[0]
    #     # get the attention of the last token
    #     last_layer_attention_avg_last_tok = last_layer_attention_avg[-1]
    #     # get the attention in image token
    #     last_layer_attention_avg_last_tok_image = last_layer_attention_avg_last_tok[
    #         self._image_start: self._image_start+self._image_token_length
    #     ]
        
    #     # Calculate mean and standard deviation of attention values
    #     attention_mean = torch.mean(last_layer_attention_avg_last_tok_image)
    #     attention_std = torch.std(last_layer_attention_avg_last_tok_image)
        
    #     # Define cutoff as mean + standard deviation
    #     attention_cutoff = attention_mean - attention_std
        
    #     # Select tokens with attention values below the cutoff
    #     below_cutoff_indices = torch.where(last_layer_attention_avg_last_tok_image < attention_cutoff)[0]
    #     selected_indices = below_cutoff_indices + self._image_start
        
    #     # generate fast v attention mask
    #     fast_v_attention_mask = torch.ones_like(self._attention_mask)
    #     fast_v_attention_mask[:, self._image_start:self._image_start+self._image_token_length] = False
    #     fast_v_attention_mask[:, selected_indices] = True

    #     self._attention_mask = fast_v_attention_mask

    ## Inputs should only contain text tokens
    def _update_text_attention_mask(self, last_layer_attention):
        last_layer_attention_avg = torch.mean(last_layer_attention, dim=1)[0]
        last_tok_attn = last_layer_attention_avg[-1]

        total_active = int(self._attention_mask.sum().item())
        k = int(self._minimum_text_tokens) if total_active > self._minimum_text_tokens else total_active

        topk = last_tok_attn.topk(k, largest=False)
        keep_idx = topk.indices

        text_mask = torch.zeros_like(self._attention_mask, dtype=torch.bool)
        text_mask[:, keep_idx] = True

        unselected = (text_mask[0]).nonzero(as_tuple=True)[0].cpu().tolist()
        print("selected token indices:", unselected)

        self._attention_mask = text_mask

    # def _update_text_attention_mask(self, last_layer_attention):
    #     last_layer_attention_avg = torch.mean(last_layer_attention, dim=1)[0]
    #     last_layer_attention_avg_last_tok = last_layer_attention_avg[-1]

    #     if self._attention_mask.sum() > self._minimum_text_tokens:
    #         top_attention_rank_index = last_layer_attention_avg_last_tok.topk(self._minimum_text_tokens, largest=False)
    #         top_attention_rank_index = top_attention_rank_index.indices
    #     else:
    #         top_attention_rank_index = last_layer_attention_avg_last_tok.topk(self._attention_mask.sum(), largest=False)
    #         top_attention_rank_index = top_attention_rank_index.indices

    #     # generate text mask
    #     text_mask = torch.ones_like(self._attention_mask)
    #     text_mask[:,:] = False
    #     text_mask[:, top_attention_rank_index] = True

    #     self._attention_mask = text_mask

    # def _update_text_attention_mask(self, last_layer_attention):
    #     # last_layer_attention: [batch, heads, seq_len, seq_len]
    #     last_layer_attention_avg = torch.mean(last_layer_attention, dim=1)[0]    # -> [seq_len, seq_len]
    #     last_layer_attention_avg_last_tok = last_layer_attention_avg[-1]        # -> [seq_len]

    #     current_mask_count = int(self._attention_mask.sum().item())
    #     if current_mask_count > int(self._minimum_text_tokens):
    #         k = int(self._minimum_text_tokens)
    #     else:
    #         k = current_mask_count

    #     seq_len = last_layer_attention_avg_last_tok.size(0)
    #     device = last_layer_attention_avg_last_tok.device

    #     perm = torch.randperm(seq_len, device=device)
    #     top_attention_rank_index = perm[:k]

    #     text_mask = torch.zeros_like(self._attention_mask)

    #     text_mask[:, :] = False
    #     if top_attention_rank_index.numel() > 0:
    #         text_mask[:, top_attention_rank_index] = True

    #     self._attention_mask = text_mask

