import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import init

from torch.autograd import Variable
import numpy as np
from Models.MIA import MIA

class EncoderCNN( nn.Module ):
    def __init__( self, embed_size, hidden_size ):
        super( EncoderCNN, self ).__init__()
        
        # ResNet-152 backend
        resnet = models.resnet152( pretrained=True )
        modules = list( resnet.children() )[ :-2 ] # delete the last fc layer and avg pool.
        resnet_conv = nn.Sequential( *modules ) # last conv feature
        
        self.resnet_conv = resnet_conv
        self.avgpool = nn.AvgPool2d( 7 )
        self.affine_a = nn.Linear( 2048, hidden_size ) 
        
        # Dropout before affine transformation
        self.dropout = nn.Dropout( 0.5 )
        
        self.init_weights()
        
    def init_weights( self ):
        """Initialize the weights."""
        init.kaiming_uniform_( self.affine_a.weight, mode='fan_in' )
        self.affine_a.bias.data.fill_( 0 )
        
    def forward( self, images ):
        '''
        Input: images
        Output: V=[v_1, ..., v_n]
        '''
        
        # Last conv layer feature map
        A = self.resnet_conv( images )
        
        # a^g, average pooling feature map
        a_g = self.avgpool( A )
        a_g = a_g.view( a_g.size(0), -1 )
        
        # V = [ v_1, v_2, ..., v_49 ]
        V = A.view( A.size( 0 ), A.size( 1 ), -1 ).transpose( 1,2 )
        V = F.relu( self.affine_a( self.dropout( V ) ) )
        
        return V


# Attention Block for C_t calculation
class AttentionBlock( nn.Module ):
    def __init__( self, hidden_size ):
        super( AttentionBlock, self ).__init__()

        self.affine_x = nn.Linear( hidden_size, 49, bias=False ) # W_x
        self.affine_h = nn.Linear( hidden_size, 49, bias=False ) # W_g
        self.affine_alpha = nn.Linear( 49, 1, bias=False ) # w_h
        
        self.dropout = nn.Dropout( 0.5 )
        self.init_weights()
        
    def init_weights( self ):
        """Initialize the weights."""
        nn.init.xavier_uniform_( self.affine_x.weight )
        nn.init.xavier_uniform_( self.affine_h.weight )
        nn.init.xavier_uniform_( self.affine_alpha.weight )
        
    def forward( self, X, h_t):
        '''
        Input: X=[x_1, x_2, ... x_k], h_t from LSTM
        Output: c_t, attentive feature map
        '''
        
        # W_x * X + W_g * h_t * 1^T
        content_x = self.affine_x( self.dropout( X ) ).unsqueeze( 1 ) \
                    + self.affine_h( self.dropout( h_t ) ).unsqueeze( 2 )
        
        # alpha_t = softmax(W_h * tanh( content_x ))
        z_t = self.affine_alpha( self.dropout( torch.tanh( content_x ) ) ).squeeze( 3 )
        alpha_t = F.softmax( z_t.view( -1, z_t.size( 2 ) ) ).view( z_t.size( 0 ), z_t.size( 1 ), -1 )
        
        # Construct attention_context_t: B x seq x hidden_size
        attention_context_t = torch.bmm( alpha_t, X ).squeeze( 2 )

        return attention_context_t

# Caption Decoder
class Decoder( nn.Module ):
    def __init__( self, embed_size, vocab_size, hidden_size ):
        super( Decoder, self ).__init__()

        self.affine_va = nn.Linear( hidden_size, embed_size )

        # word embedding
        self.caption_embed = nn.Embedding( vocab_size, embed_size )
        
        # LSTM decoder
        self.LSTM = nn.LSTM( embed_size, hidden_size, 1, batch_first=True )
        
        # Save hidden_size for hidden variable 
        self.hidden_size = hidden_size
        
        # Attention Block
        self.attention = AttentionBlock( hidden_size )

        # Final Caption generator
        self.mlp = nn.Linear( hidden_size, vocab_size )
        
        self.dropout = nn.Dropout( 0.5 )
        self.init_weights()
        
    def init_weights( self ):
        init.kaiming_uniform_( self.affine_va.weight, mode='fan_in' )
        self.affine_va.bias.data.fill_( 0 )

        '''
        Initialize final classifier weights
        '''
        init.kaiming_normal_( self.mlp.weight, mode='fan_in' )
        self.mlp.bias.data.fill_( 0 )
        
        
    def forward( self, V, T, captions, basic_model, states=None ):

        candidate_model = ['VisualAttention', 'ConceptAttention', 'VisualCondition', 'ConceptCondition']
        assert basic_model in candidate_model, "The %s is not in the candidate list: %s"%(basic_model, candidate_model) 
        
        # Word Embedding
        embeddings = self.caption_embed( captions )

        # Hiddens: Batch x seq_len x hidden_size
        if torch.cuda.is_available():
            hiddens = Variable( torch.zeros( embeddings.size(0), embeddings.size(1), self.hidden_size ).cuda() )
        else:
            hiddens = Variable( torch.zeros( embeddings.size(0), embeddings.size(1), self.hidden_size ) )     

        # The averaged visual features and textual concepts
        v_a = self.affine_va( self.dropout( torch.mean( V , dim=1 ) ) )
        # v_a = torch.mean( V , dim=1 )
        t_a = torch.mean( T , dim=1 )
        
        # x_t = w_t + v_a/t_a
        if "Visual" in basic_model:
            # The first decoding step: Visual Condition
            if "Condition" in basic_model:
                h_t, states = self.LSTM( t_a, states )
            x = embeddings + v_a.unsqueeze( 1 ).expand_as( embeddings )            
        elif "Concept" in basic_model:
            # The first decoding step: Concept Condition
            if "Condition" in basic_model:
                h_t, states = self.LSTM( v_a, states )
            x = embeddings + t_a.unsqueeze( 1 ).expand_as( embeddings )            
        else:
            x = embeddings
            
        # Recurrent Block
        for time_step in range( x.size( 1 ) ):
            
            # Feed in x_t one at a time
            x_t = x[ :, time_step, : ]
            x_t = x_t.unsqueeze( 1 )
            
            h_t, states = self.LSTM( x_t, states )
            
            # Save hidden
            hiddens[ :, time_step, : ] = h_t.squeeze(1)  # Batch_first

        # Data parallelism for Visual/Concept Attention block
        if "Attention" in basic_model:
        
            if basic_model == "VisualAttention":
                attention_input = V
            elif basic_model == "ConceptAttention":
                attention_input = T
            
            if torch.cuda.device_count() > 1:
                device_ids = range( torch.cuda.device_count() )
                attention_block_parallel = nn.DataParallel( self.attention, device_ids=device_ids )
                attention_context = attention_block_parallel( attention_input, hiddens )
            else:
                attention_context = self.attention( attention_input, hiddens )
                
            # Final score along vocabulary
            scores = self.mlp( self.dropout( attention_context + hiddens ) )
        else:
            # Final score along vocabulary
            scores = self.mlp( self.dropout( hiddens ) )
        
        # Return states for Caption Sampling purpose
        return scores
        
# Whole Architecture with Image Encoder and Caption decoder        
class Encoder2Decoder( nn.Module ):
    def __init__( self, embed_size, vocab_size, hidden_size, use_MIA=True, iteration_times=2 ):
        super( Encoder2Decoder, self ).__init__()
        
        # Image CNN encoder
        self.encoder_image = EncoderCNN( embed_size, hidden_size )
        
        # Concept encoder
        self.encoder_concept = nn.Embedding( vocab_size, embed_size )
        
        # Caption Decoder      
        self.decoder = Decoder( embed_size, vocab_size, hidden_size )
        
        # Share the weight matrix between caption & concept word embeddings
        self.encoder_concept.weight = self.decoder.caption_embed.weight
        
        # Mutual Iterative Attention (MIA)
        self.use_MIA = use_MIA
        
        if self.use_MIA:
            assert iteration_times > 0, "The value of iteration_times should be greater than 0"
            
            self.MIA = MIA(
                d_model=hidden_size, N=iteration_times,
                d_inner=2048, n_head=8, d_k=64, d_v=64,
                dropout=0.1)
        
        assert embed_size == hidden_size, "The values of embed_size and hidden_size should be equal."
        
    def forward( self, images, captions, image_concepts, lengths, basic_model ):
        
        # Data parallelism for V encoder if multiple GPUs are available
        # V=[ v_1, ..., v_k ] 
        if torch.cuda.device_count() > 1:
            device_ids = range( torch.cuda.device_count() )
            encoder_image_parallel = torch.nn.DataParallel( self.encoder_image, device_ids=device_ids )
            V = encoder_image_parallel( images ) 
        else:
            V = self.encoder_image( images )
            
        # Data parallelism for T encoder if multiple GPUs are available
        # T=[ t_1, ..., t_k ] 
        if torch.cuda.device_count() > 1:
            device_ids = range( torch.cuda.device_count() )
            encoder_concept_parallel = torch.nn.DataParallel( self.encoder_concept, device_ids=device_ids )
            T = encoder_concept_parallel( image_concepts ) 
        else:
            T = self.encoder_concept( image_concepts )
        
        # Mutual Iterative Attention (MIA)
        if self.use_MIA:
            if torch.cuda.device_count() > 1:
                device_ids = range(torch.cuda.device_count())
                MIA_Parallel = torch.nn.DataParallel( self.MIA, device_ids=device_ids )
                SGIR, Refine_V, Refine_T = MIA_Parallel( V, T )
            else:
                # SGIR: Semantic-Grounded Image Representations
                SGIR, _, _ = self.MIA( V, T )
            
            # Update the original visual features and textual concepts.
            V = SGIR
            T = SGIR

        # Language Modeling on word prediction
        scores = self.decoder( V, T, captions, basic_model )
        
        # Pack it to make criterion calculation more efficient
        packed_scores = pack_padded_sequence( scores, lengths, batch_first=True )
        
        return packed_scores
