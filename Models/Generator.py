''' This module will handle the text generation with beam search. '''

import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.Main import Encoder2Decoder
from Models.Beam import Beam

class Generator(object):
    ''' Load with trained model and handle the beam search '''

    def __init__(self, args, checkpoint_path, vocab_size):
        self.args = args
        # Change to GPU mode if available
        if torch.cuda.is_available():
            self.device = torch.device( 'cuda' )
        else:
            self.device = torch.device( 'cpu' )

        model = Encoder2Decoder( args.embed_size, vocab_size, args.hidden_size, args.use_MIA, args.iteration_times )

        model.load_state_dict( torch.load( checkpoint_path ) )
        print('[Info] Trained model state loaded.')

        model.word_prob_prj = nn.LogSoftmax(dim=1)
        
        model = model.to(self.device)

        self.model = model
        self.model.eval()

    def translate_batch( self, images, image_concepts ):
        ''' Translation work in one batch '''

        def get_inst_idx_to_tensor_position_map(inst_idx_list):
            ''' Indicate the position of an instance in a tensor. '''
            return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}

        def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
            ''' Collect tensor parts associated to active instances. '''
            n_curr_active_inst = len(curr_active_inst_idx)
            if beamed_tensor.dim() == 3:
                _, d_hs, d_hs_2 = beamed_tensor.size()
                new_shape = ( n_curr_active_inst * n_bm, d_hs, d_hs_2 )
            else:
                _, d_hs = beamed_tensor.size()
                new_shape = ( n_curr_active_inst * n_bm, d_hs )

            beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
            beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
            beamed_tensor = beamed_tensor.view(*new_shape)

            return beamed_tensor

        def collate_active_info(
                V, T, inst_idx_to_position_map, active_inst_idx_list):
            # Sentences which are still active are collected,
            # so the decoder will not run on completed sentences.
            n_prev_active_inst = len(inst_idx_to_position_map)
            active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
            active_inst_idx = torch.LongTensor(active_inst_idx).to(self.device)

            active_V = collect_active_part(V, active_inst_idx, n_prev_active_inst, n_bm)
            active_T = collect_active_part(T, active_inst_idx, n_prev_active_inst, n_bm)
            active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            return active_V, active_T, active_inst_idx_to_position_map

        def beam_decode_step(
                inst_dec_beams, len_dec_seq, V, T, basic_model, inst_idx_to_position_map, n_bm):
            ''' Decode and update beam status, and then return active beam idx '''

            def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
                dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
                dec_partial_seq = torch.stack(dec_partial_seq).to(self.device)
                dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
                return dec_partial_seq

            def prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm):
                dec_partial_pos = torch.arange(1, len_dec_seq + 1 , dtype=torch.long, device=self.device)
                dec_partial_pos = dec_partial_pos.unsqueeze(0).repeat(n_active_inst * n_bm, 1)
                return dec_partial_pos

            def predict_word(dec_seq, dec_pos, V, T, basic_model, n_active_inst, n_bm):
                dec_output = self.model.decoder( V, T, dec_seq, basic_model )
                dec_output = dec_output[:, -1, :]  # Pick the last step: (bh * bm) * d_h
                word_prob = F.log_softmax( dec_output , dim=1 )
                word_prob = word_prob.view( n_active_inst, n_bm, -1 )

                return word_prob

            def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map):
                active_inst_idx_list = []
                for inst_idx, inst_position in inst_idx_to_position_map.items():
                    is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])
                    if not is_inst_complete:
                        active_inst_idx_list += [inst_idx]

                return active_inst_idx_list

            n_active_inst = len(inst_idx_to_position_map)
            dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)
            dec_pos = prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm)
            word_prob = predict_word(dec_seq, dec_pos, V, T, basic_model, n_active_inst, n_bm)

            # Update the beam with predicted word prob information and collect incomplete instances
            active_inst_idx_list = collect_active_inst_idx_list(
                inst_dec_beams, word_prob, inst_idx_to_position_map)
            #print not active_inst_idx_list
            
            return active_inst_idx_list

        def collect_hypothesis_and_scores(inst_dec_beams, n_best):
            all_hyp, all_scores = [], []
            for inst_idx in range(len(inst_dec_beams)):
                scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
                all_scores += [scores[:n_best]]

                hyps = [inst_dec_beams[inst_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
                all_hyp += [hyps]
                
            return all_hyp, all_scores


        # -- Main -- 
        # Generate Captions
        with torch.no_grad():
        
            #-- Encode -- V=[ v_1, ..., v_k ]
            V = self.model.encoder_image( images )
            T = self.model.encoder_concept( image_concepts )
            
            # Mutual Iterative Attention (MIA)
            if self.args.use_MIA:
                SGIR, _, _ = self.model.MIA( V, T ) # SGIR: Semantic-Grounded Image Representations
                V = SGIR
                T = SGIR
                
            #-- Repeat data for beam search
            n_bm = self.args.beam_size
            n_inst, len_s, d_h = V.size()
            V = V.repeat( 1, n_bm, 1 ).view(n_inst * n_bm, len_s, d_h )
            T = T.repeat( 1, n_bm, 1 ).view(n_inst * n_bm, len_s, d_h )
            
            #-- Prepare beams
            inst_dec_beams = [Beam(n_bm, device=self.device) for _ in range(n_inst)]

            #-- Bookkeeping for active or not
            active_inst_idx_list = list(range(n_inst))
            inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            #-- Decode
            for len_dec_seq in range(1, self.args.max_caption_length):

                active_inst_idx_list = beam_decode_step(
                    inst_dec_beams, len_dec_seq, V, T, self.args.basic_model, inst_idx_to_position_map, n_bm)

                if not active_inst_idx_list:
                    break  # all instances have finished their path to <EOS>

                V, T, inst_idx_to_position_map = collate_active_info(
                    V, T, inst_idx_to_position_map, active_inst_idx_list)

        batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, self.args.n_best)
        return batch_hyp, batch_scores

